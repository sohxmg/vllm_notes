---
part: 5
files: [nanovllm/engine/scheduler.py]
tags: [nano-vllm]
---

# Scheduling: Continuous Batching

> [!abstract] In one line
> Before every forward pass, the `Scheduler` picks which sequences run and how many tokens each one processes, within a token budget and a sequence budget. It does this by moving sequences between a `waiting` queue and a `running` queue, splitting long prompts into chunks, and evicting sequences when KV-cache memory runs out.

<br>

## Why this exists

The engine loop from [[02 The Main Loop]] does the same thing over and over: `schedule()`, run the model once, `postprocess()`. Something has to decide, at each of those steps, *which* requests go into the batch. That decision is the whole job of `scheduler.py`, and it matters more for throughput than almost anything else in the engine.

**Static batching** is the naive approach: take 8 requests, run them together until *all* 8 are done, then take the next 8. The problem is that generation lengths differ a lot. If one request wants 1000 tokens and the other seven stop after 20, then for 980 steps the GPU runs a batch of 1 while seven slots sit empty and new requests wait outside.

**Continuous batching** (also called iteration-level scheduling) fixes this by re-forming the batch at *every step*. A request that emits EOS leaves the batch immediately and its KV-cache blocks are freed, and a waiting request can join on the very next step. The batch is a stream that sequences flow through, not a fixed group.

<br>

### Prefill and decode cost very different things

A request goes through two phases, and the scheduler treats them differently because they stress the GPU in opposite ways.

- **Prefill**: process the whole prompt (say 2000 tokens) in one forward pass, fill the KV cache for every position, and sample the first output token.
- **Decode**: process exactly one new token per sequence, reading the KV cache for all earlier positions, and sample the next token.

Each forward pass has to stream all the model weights from GPU memory once. For a model with $P$ parameters stored in bf16, processing $T$ tokens in one pass costs roughly

$$
\text{FLOPs} \approx 2PT, \qquad \text{bytes of weights read} \approx 2P, \qquad \text{arithmetic intensity} \approx \frac{2PT}{2P} = T \ \text{FLOPs/byte}.
$$

An H100 does roughly 1000 TFLOP/s against about 3.35 TB/s of memory bandwidth, so it needs about 300 FLOPs per byte before compute becomes the bottleneck.

- A prefill with $T = 4000$ tokens is far above that line, so it is **compute-bound**: the GPU is busy doing matmuls.
- A decode step with 16 sequences has $T = 16$, far below it, so it is **memory-bound**: the GPU spends most of its time waiting for weights and KV cache to arrive, and the matmul units sit mostly idle.

This has two consequences you will see in the code. First, the cost of a prefill step grows with *tokens*, so prefill is limited by a token budget, `max_num_batched_tokens`. Second, the cost of a decode step barely changes between 1 sequence and 64, since you read the weights once either way, so decode should pack in as many sequences as memory allows, up to `max_num_seqs`. nano-vllm never mixes the two phases in a single step: each `schedule()` call returns either a prefill batch or a decode batch, plus a flag saying which one it is.

<br>

## Walkthrough: `nanovllm/engine/scheduler.py`

The whole file is 93 lines. It uses three things from other parts: `Sequence` and its counters ([[03 Tracking a Request]]), and `BlockManager` for KV-cache memory ([[04 Memory - Paged KV Cache]]).

### Imports

*`nanovllm/engine/scheduler.py` L1–5*
```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
```

`deque` matters: the scheduler pops from and pushes to both ends of its queues, and a `deque` does that in O(1) where a `list` would be O(n) at the front.

<br>

### State: two queues and two budgets

*`nanovllm/engine/scheduler.py` L8–17*
```python
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

- `max_num_seqs` (default 512): the most sequences allowed in one batch, prefill or decode. This is what limits decode batches.
- `max_num_batched_tokens` (default 16384): the most tokens processed in one step. This is what limits prefill batches. In decode each sequence contributes one token, so this budget never binds there.
- `eos`: the end-of-sequence token id. `Config` defaults it to `-1`; `LLMEngine.__init__` sets it from the tokenizer before building the scheduler.
- `block_size`: tokens per KV-cache block (default 256). Needed to convert "cached blocks" into "cached tokens".
- `block_manager`: the scheduler owns the one `BlockManager`. Every allocation and free of KV memory goes through the scheduler.
- `waiting`: sequences whose prompt has not been fully prefilled. That includes brand new requests, a sequence that is part-way through a chunked prefill, and preempted sequences.
- `running`: sequences whose prompt is fully in the KV cache and which are now generating one token per decode step.

> [!important] Invariant
> A sequence is in `running` exactly when its whole current token list, except possibly the newest sampled token, has KV entries in the cache. Everything else lives in `waiting`. Finished sequences are in neither queue.

<br>

### `is_finished` and `add`

*`nanovllm/engine/scheduler.py` L19–23*
```python
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
```

The engine's `generate()` loops `while not self.is_finished()`, so the run ends when both queues are empty. New requests join at the back of `waiting`, so admission is first-come, first-served.

<br>

### `schedule()`: the prefill path

`schedule()` returns `(scheduled_seqs, is_prefill)`. It tries prefill first. Only if no prefill work can be scheduled does it fall through to decode.

*`nanovllm/engine/scheduler.py` L25–34*
```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
```

- The loop always looks at `self.waiting[0]`, the head of the queue. It *peeks* rather than pops, because the sequence might not fit, and because a chunked sequence stays in `waiting` after being scheduled.
- `remaining` is how much of the token budget is left for this step. Once it hits 0 the batch is full.
- Admission is strictly in order. If the head does not fit, the loop stops; it never skips ahead to a smaller request behind it.

<br>

#### How many tokens does this sequence still need?

*`nanovllm/engine/scheduler.py` L35–41*
```python
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
```

There are two cases, and `seq.block_table` tells them apart.

- **Empty block table**: this sequence has no KV memory yet. It is either brand new or was preempted, which cleared its table. Ask the block manager whether its blocks can be allocated. `can_allocate` returns `-1` if there are not enough free blocks, and in that case the loop stops: no more prefills this step. Otherwise it returns how many leading blocks are already in the prefix cache (see [[04 Memory - Paged KV Cache]]). Those tokens do not need computing, so `num_tokens` is the total minus the cached part.
  Example: block size 256, prompt of 600 tokens, first 2 blocks cached → `num_tokens = 600 - 512 = 88`.
- **Non-empty block table**: this sequence already went through part of its prefill on an earlier step (chunked prefill, below). Its blocks are already allocated, and `seq.num_cached_tokens` says how far it got. The work left is `num_tokens - num_cached_tokens`.

**`num_cached_tokens`** always means "the number of leading tokens of this sequence whose K and V are already written into the cache", whether they got there from the prefix cache or from an earlier chunk.

<br>

#### Chunked prefill: only the first sequence may be split

*`nanovllm/engine/scheduler.py` L42–45*
```python
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
```

If the sequence needs more tokens than the budget has left, there are two options.

- If other sequences are already in this batch (`scheduled_seqs` is non-empty), stop. This sequence waits for the next step, where it will be first in line and get the full budget.
- If it is the *first* sequence of the batch, schedule it anyway, and it will get only part of its tokens (the `min` on the next line). This is **chunked prefill**: a 40,000-token prompt with a 16,384-token budget gets processed over three steps as 16,384 + 16,384 + 7,232.

The check is placed *before* `allocate` on purpose, so a sequence that will not be scheduled this step does not grab memory.

Why allow chunking only for the first sequence? Because the first sequence starts with the full budget, a chunked one always consumes all of it, so `remaining` becomes 0 and the loop ends. A chunked sequence is therefore always alone in its batch, and there is at most one half-prefilled sequence in the whole system, always sitting at `waiting[0]`. Without the "first only" rule, a long prompt would never be admitted whenever small prompts kept filling part of the budget ahead of it. With the rule, it waits at most one step to become first, and then it makes progress.

`allocate` reserves blocks for the *entire* sequence up front, even if only one chunk will be computed now. It also sets `seq.num_cached_tokens = num_cached_blocks * block_size`. So a chunked sequence holds all its KV memory from its first chunk on, and later chunks never need to allocate.

<br>

#### Commit the sequence to the batch

*`nanovllm/engine/scheduler.py` L46–52*
```python
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
```

- **`num_scheduled_tokens`**: how many tokens of this sequence the model will process in this step. The model runner computes positions `num_cached_tokens` up to `num_cached_tokens + num_scheduled_tokens` ([[06 Getting Data onto the GPU]]). It is `num_tokens` if everything fits, otherwise the rest of the budget.
- If this step finishes the prompt (cached + scheduled covers every token), the sequence moves from `waiting` to the back of `running` and becomes `RUNNING` now, before the forward pass has actually happened. That is safe because `postprocess` runs right after the pass.
- If it does not finish (a chunk), the sequence stays at `waiting[0]`, and the next `schedule()` call finds it there with a non-empty block table.
- In both cases it goes into this step's batch.

Worked example with budget 8: `waiting = [A(5 tokens), B(10 tokens)]`.
1. A: `remaining = 8`, `num_tokens = 5`, scheduled 5, moves to `running`. `num_batched_tokens = 5`.
2. B: `remaining = 3 < 10` and the batch already has A → `break`.
Batch = `[A]`, 5 tokens.

<br>

#### Prefill takes priority

*`nanovllm/engine/scheduler.py` L54–55*
```python
        if scheduled_seqs:
            return scheduled_seqs, True
```

If *any* prefill was scheduled, return immediately with `is_prefill=True`. Decode only runs on steps where nothing in `waiting` could be admitted, either because `waiting` is empty or because the head does not fit in memory.

> [!warning] Decode stalls during prefill
> While new requests keep arriving, running sequences get no new tokens on prefill steps. That is a deliberate trade: prefill-first gets new requests into the (efficient, large) decode batch as fast as possible, at the cost of latency between tokens for the requests already running. Production vLLM mixes prefill chunks and decode tokens in the same batch to smooth this out; nano-vllm keeps them separate for simplicity.

<br>

### `schedule()`: the decode path

*`nanovllm/engine/scheduler.py` L57–70*
```python
        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
```

Take sequences from the front of `running`, oldest first, up to `max_num_seqs`. Each gets exactly one token of work: its newest sampled token, whose KV has not been computed yet.

The only thing that can go wrong is memory. The newest token needs a slot in the KV cache. If it is the first token of a new block, a fresh block is needed. `can_append` checks exactly that: `len(seq) % block_size == 1` means "the last token opens a new block", so one free block is required; otherwise zero are required.

The inner `while ... else` is the part to read slowly. In Python, the `else` of a `while` loop runs only if the loop ended because its condition became false, *not* if it ended with `break`.

- **Enough memory** (`can_append` is true on the first check): the inner loop body never runs, the `else` runs, and the sequence is scheduled. `may_append` allocates the new block if one is needed.
- **Not enough memory, and other sequences are in `running`**: evict the one at the *back* of `running` with `preempt`, which frees its blocks, then check again. Repeat until there is room.
- **Not enough memory, and nothing left to evict**: preempt `seq` itself and `break`. The `break` skips the `else`, so `seq` is not scheduled.

The victim comes from the back because `running` is ordered by admission time: prefills append to the back, and the scheduled decode batch is put back at the front (see the next section). The back therefore holds the most recently admitted sequences, which have the least work invested. The oldest requests keep their memory and finish.

`is_prefill = False` marks the sequence as decoding. It changes how `Sequence.__getstate__` serializes it for the tensor-parallel worker processes: only `last_token` is sent instead of the full token list ([[03 Tracking a Request]]).

<br>

#### Putting the batch back in order

*`nanovllm/engine/scheduler.py` L71–73*
```python
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
```

- `assert scheduled_seqs`: `schedule()` is only called when `is_finished()` is false, so something is in `waiting` or `running`. If nothing could be scheduled, a single sequence cannot fit in the entire KV cache, and the engine cannot make progress. The assert turns that into a crash instead of an infinite loop.
- The scheduled sequences were popped off the front of `running`. `extendleft` pushes items one at a time onto the left, which reverses them, so reversing first restores the original order. Example: `running` after the loop is `[D]` (D was left out by `max_num_seqs`), `scheduled_seqs = [A, B, C]` → `extendleft([C, B, A])` → `running = [A, B, C, D]`.
- Return with `is_prefill=False`.

<br>

### `preempt()`: evict and start over later

*`nanovllm/engine/scheduler.py` L75–79*
```python
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```

nano-vllm handles memory pressure by **recomputation**: throw the evicted sequence's KV cache away, and later prefill it again from scratch.

- `status = WAITING`, `is_prefill = True`: it goes back to being a prefill job. Its "prompt" for the re-prefill is its full current token list, prompt *plus* the tokens generated so far. `num_prompt_tokens` is unchanged, so `num_completion_tokens` keeps counting correctly.
- `deallocate`: decrements each block's reference count, frees blocks that reach 0, clears `block_table` and sets `num_cached_tokens = 0`. An empty block table is exactly what the prefill path uses to recognise "needs `can_allocate` / `allocate`".
- `appendleft`: the evicted sequence goes to the *front* of `waiting`, ahead of brand-new requests, so it is the first to be re-admitted once memory frees up.

Recomputation is cheaper than it sounds. Freed blocks keep their hash in `hash_to_block_id` until they are handed out again, so if nobody has reused them, the re-prefill finds them through the prefix cache and skips most of the work.

<br>

### `postprocess()`: record the results of the step

After the forward pass, the model runner returns one sampled token per scheduled sequence. `postprocess` updates each sequence's bookkeeping.

*`nanovllm/engine/scheduler.py` L81–87*
```python
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
```

- `hash_blocks(seq)`: any block that just became completely full is hashed and registered in the prefix cache. It reads `num_cached_tokens` and `num_scheduled_tokens` to find which blocks were filled in this step, so it has to run *before* the next line changes them.
- `num_cached_tokens += num_scheduled_tokens`: the KV for those tokens is now in the cache.
- `num_scheduled_tokens = 0`: reset for the next step.
- The `continue`: if this was a prefill step and the sequence still has uncached tokens, it was a chunk. Skip the rest of the loop: no token is appended and no stop condition is checked. The sequence is still in `waiting`, and next step's `schedule()` resumes it from `num_cached_tokens`.

The model runner samples one token for every sequence in the batch, so a chunk also produces a token here. That token is thrown away (see the question at the end).

*`nanovllm/engine/scheduler.py` L88–92*
```python
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

- `append_token`: add the new token. `num_tokens` grows by one, and this token has no KV yet. That is the one token the next decode step will process.
- Stop if the token is EOS (unless `ignore_eos` is set, which benchmarks use to force fixed-length outputs), or if the sequence has generated `max_tokens` tokens.
- A finished sequence gives its blocks back right away and leaves `running`. It is in `running` even if it finished on its prefill step, because `schedule()` already moved it there. The engine then reports it as done, because `seq.is_finished` checks the status.

> [!important] The counting invariant
> Between steps, every sequence in `running` has `num_tokens == num_cached_tokens + 1`: everything except the newest sampled token is in the cache. Prefill establishes it (cached == num_tokens, then one token appended), and each decode step keeps it (one token cached, one appended). That is why decode always schedules exactly one token.

<br>

## A timeline: three requests, five steps

To keep the numbers small, pretend `max_num_batched_tokens = 8`, `max_num_seqs = 4`, plenty of free blocks, and no prefix-cache hits. (The real minimum block size is 256; small numbers just make the arithmetic easy to check.) Three requests arrive at once: A with a 5-token prompt, B with 10, C with 3. The notation `B 8/10` means 8 of B's 10 tokens are cached.

| Step | `waiting` before | `running` before | What `schedule()` does | Batch | After `postprocess()` |
|---|---|---|---|---|---|
| 1 | A, B, C | – | A needs 5 ≤ 8: scheduled in full, moves to `running`. B needs 10 > 3 left and the batch is not empty → break. | prefill: A×5 | A 5/5, appends token → A has 6 tokens |
| 2 | B, C | A | B is first, needs 10 > 8 → **chunked**: 8 tokens, stays in `waiting`. Budget is 0 → break. A gets nothing. | prefill: B×8 | B 8/10, **no token appended** |
| 3 | B, C | A | B has a block table, needs 10 − 8 = 2 → finishes, to `running`. C needs 3 ≤ 6 → to `running`. | prefill: B×2, C×3 | B 10/10 → 11 tokens; C 3/3 → 4 tokens |
| 4 | – | A, B, C | `waiting` is empty → decode. Each gets 1 token. | decode: A, B, C | A samples EOS → finished, blocks freed, removed |
| 5 | – | B, C | decode | decode: B, C | ... |

The same flow as a picture:

```text
step 1  prefill  [A:5]            waiting: B C     running: A
step 2  prefill  [B:8]  (chunk)   waiting: B C     running: A      (A stalls)
step 3  prefill  [B:2 C:3]        waiting: -       running: A B C
step 4  decode   [A B C]          waiting: -       running: B C    (A done)
step 5  decode   [B C]            ...
```

Notice three things. A waited through steps 2 and 3 without generating, because prefill has priority. B was the only sequence in its chunked step. And A left the batch at step 4 without anyone else waiting on it, which is continuous batching at work.

<br>

### A preemption, briefly

Suppose later `running = [B, C, D]`, block size 4, zero free blocks, and B has 9 tokens (its newest, 9th token starts a third block since `9 % 4 == 1`). The decode loop pops B, `can_append(B)` is false, so it preempts D (from the back): D's blocks are freed and D goes to the front of `waiting`. Now there is a free block, the `else` runs, B gets its block. C is popped next; if it does not need a new block it is scheduled. The batch is `[B, C]`. On a later step, once memory frees up, D is at `waiting[0]` and gets re-prefilled with its prompt plus everything it had generated.

<br>

> [!question] Check yourself
> 1. Why does a partly-prefilled sequence not get a token appended in `postprocess`?
> 2. Why can only the first sequence in a prefill batch be chunked?
> 3. In the decode loop, why is the preemption victim taken from the back of `running`, and why does a preempted sequence go to the *front* of `waiting`?
>
> > [!success]- Answer
> > 1. The token sampled for a chunk is the model's prediction of the token that comes right after the chunk's last position. That position is still inside the prompt, so the "next token" is already known: it is the next prompt token. Appending the sampled token would put a guess into the middle of the prompt, add a bogus completion token (throwing off `num_completion_tokens`), and break the invariant that `num_tokens - num_cached_tokens` is the prompt work left. The sequence is also still in `waiting`, not `running`, so the EOS and `max_tokens` checks (and `running.remove`) must not run for it. Only the step that caches the *last* prompt token produces a real first output token.
> > 2. The first sequence starts with the full budget, so if it gets chunked it uses up the whole budget and ends the loop. That guarantees at most one half-prefilled sequence at a time, always at `waiting[0]`. It also means a long prompt is never starved: at worst it waits one step to become first. If a later sequence were allowed to chunk, many sequences could be half-done at once, each holding KV blocks for its whole prompt (`allocate` reserves everything up front).
> > 3. The back of `running` holds the most recently admitted sequences, which have the least computed KV to throw away. Older sequences keep going and finish. Putting the victim at the front of `waiting` means it is re-admitted before any new request, so a preempted request is not pushed behind a stream of newcomers.

<br>

> [!tip] Build-it-yourself
> Write the queues, `add` and `is_finished` first. Then write `schedule()` with only whole-prompt prefill and decode (no chunking, no preemption), plus `postprocess`, and check that generation works. Next add preemption in the decode loop. Add chunked prefill last: it is the part that introduces `num_scheduled_tokens` and the "non-empty block table in `waiting`" state.

<br>

## Build it yourself

- [ ] `Scheduler.__init__` holding `max_num_seqs`, `max_num_batched_tokens`, `eos`, `block_size`, a `BlockManager`, and `waiting` / `running` as `deque`s.
- [ ] `add` (append to `waiting`) and `is_finished` (both queues empty).
- [ ] Prefill loop: peek `waiting[0]`, compute tokens still needed (via `can_allocate` for a new sequence, via `num_cached_tokens` for a resumed one), stop on no memory or no budget, chunk only the first sequence, allocate, set `num_scheduled_tokens`, move to `running` when the prompt is complete.
- [ ] Return early with `is_prefill=True` if any prefill was scheduled.
- [ ] Decode loop: pop from the front, `can_append` / preempt-from-back loop with `while ... else`, `may_append`, one token each, then put the batch back at the front of `running` in order.
- [ ] `preempt`: status `WAITING`, `is_prefill = True`, `deallocate`, `appendleft` onto `waiting`.
- [ ] `postprocess`: `hash_blocks` first, advance `num_cached_tokens`, reset `num_scheduled_tokens`, skip unfinished chunks, append the token, check EOS / `max_tokens`, free and remove finished sequences.

Invariants your version must keep:
- A batch is all prefill or all decode, never mixed.
- Admission from `waiting` is in order; never skip the head.
- At most one sequence is partly prefilled, and it sits at `waiting[0]` with a non-empty block table.
- A sequence in `running` satisfies `num_tokens == num_cached_tokens + 1` between steps.
- A sequence with an empty block table has `num_cached_tokens == 0`; a finished sequence holds no blocks and is in no queue.

<br>

← [[04 Memory - Paged KV Cache]] | [[06 Getting Data onto the GPU]] →
