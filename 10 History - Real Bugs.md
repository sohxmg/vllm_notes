---
part: 10
files: [nanovllm/engine/block_manager.py, nanovllm/engine/scheduler.py, nanovllm/engine/model_runner.py, nanovllm/engine/sequence.py]
tags: [nano-vllm]
---

# History: Real Bugs in the KV Cache and Scheduler

> [!abstract] In one line
> Seven real bugs from nano-vllm's git history, each one a broken invariant between the scheduler's bookkeeping (`num_cached_tokens`, `num_scheduled_tokens`, block tables, block hashes) and what is actually sitting in GPU memory. Each invariant becomes a test case for your own engine.



## Why this exists

The code you read in parts 02–08 is the *fixed* version. It looks calm and obvious, and that hides how easy it is to get wrong. An inference engine keeps two copies of the truth: the CPU-side bookkeeping (how many tokens of a sequence are cached, which physical blocks it owns, which hash names which block) and the GPU-side reality (which KV slots actually hold which keys and values). Almost every bug below is a moment where those two copies disagreed.

Most of these bugs don't crash. A wrong position, a key length that runs past the chunk, or a hash that points at the wrong block all produce text that looks plausible and is subtly wrong. The ones that do crash usually crash far from the cause, e.g. an `IndexError` in `model_runner.py` that was really a scheduler bug.

Here is the timeline. Notice that most of them arrived together with **chunked prefill** (`8d63a98`), which broke one quiet assumption that the old code relied on everywhere: *a prefill always runs from `num_cached_tokens` to the end of the sequence*.

| # | Commit | Date | Where | Bug |
|---|---|---|---|---|
| 1 | `f5b4840` (PR #67) | 2025-07 | `model_runner.py` | decode position off by one |
| 2 | `77dd709` (PR #207), then `f64d821` | 2026-04 | `scheduler.py` | token budget computed before prefix-cache lookup |
| 3 | `25794a1` (PR #213) | 2026-04 | `model_runner.py` | chunk attends to keys past its end |
| 4 | `f64d821` | 2026-04 | `block_manager.py`, `model_runner.py` | full prefix hit leaves nothing to compute |
| 5 | `82f5ca2` (PR #145), then `f64d821` | 2025-12 / 2026-04 | `sequence.py`, `scheduler.py`, `block_manager.py` | re-prefill after preemption |
| 6 | `f64d821` | 2026-04 | `block_manager.py` | stale hash entry for an evicted block |
| 7 | `9fa256a` | 2026-04 | `block_manager.py` | a cache hit wipes the block's identity |

To read any of them yourself: `git show <hash>` in the repo. The diffs below are trimmed to the lines that matter.



## Bug 1: The decode token got the wrong position

### Symptom

No crash and no error. Generation works and reads fine, but every generated token is rotated by RoPE as if it sat one position further right than it does. Greedy outputs differ from Hugging Face `transformers` on the same prompt, and quality is slightly worse, especially on long generations.

### The fix

*`f5b4840`, `nanovllm/engine/model_runner.py`, `prepare_decode`*
```diff
         for seq in seqs:
             input_ids.append(seq.last_token)
-            positions.append(len(seq))
+            positions.append(len(seq) - 1)
             context_lens.append(len(seq))
             slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
```

### Root cause

During decode, the sampled token has *already been appended* to the sequence by `postprocess` ([[05 Scheduling - Continuous Batching]]). So the token you feed in is the last one, at index `len(seq) - 1`.

Take a 5-token prompt `t0..t4`, prefilled at positions 0..4. The model samples `t5`, `postprocess` appends it, and now `len(seq) = 6`. The decode step feeds `t5`:

- its true position is 5
- the old code gave position `len(seq) = 6`
- its KV slot was index `last_block_num_tokens - 1 = 5`, which was correct

So the slot said "index 5" and the position said "6". RoPE encodes relative distance ($q_m \cdot k_n$ depends on $m-n$), so from the model's point of view there was a phantom gap of one token between the prompt and the completion. Every decode token after that was consistently shifted, so nothing looked broken locally. And if the sequence was ever preempted and re-prefilled, the same token's KV would be recomputed with position 5, so the cache would change under it.

> [!important] Invariant
> For every token, **position = index in the sequence = slot index within the sequence's blocks**. Prefill uses `range(start, end)` for both positions and slots. Decode must use `len(seq) - 1` for both.



## Bug 2: The token budget was computed before the prefix-cache lookup

### Symptom

`IndexError: list index out of range` inside `prepare_prefill` in `model_runner.py`, at `seq.block_table[i]`. It only happens when a prompt partially hits the prefix cache.

### The first fix (PR #207)

*`77dd709`, `nanovllm/engine/scheduler.py`, `schedule`*
```diff
             seq = self.waiting[0]
-            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
             remaining = self.max_num_batched_tokens - num_batched_tokens
             if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
                 break
-            if remaining < num_tokens and scheduled_seqs:    # only allow chunked prefill for the first seq
-                break
+
             if not seq.block_table:
                 self.block_manager.allocate(seq)
             ...   # (lines elided)
+            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
+
+            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
+                break
+
             seq.num_scheduled_tokens = min(num_tokens, remaining)
```

### Root cause

`num_cached_tokens` is *set by* `allocate()`: that is where the block manager walks the prefix hashes and counts hits ([[04 Memory - Paged KV Cache]]). The scheduler read it one line too early, while it was still 0.

Concrete case, `block_size = 256`, a 600-token prompt whose first 512 tokens (2 full blocks) are already cached:

1. Before `allocate`: `num_cached_tokens = 0`, so `num_tokens = 600`, and `num_scheduled_tokens = 600`.
2. `allocate` finds the 2 hits: `num_cached_tokens = 512`, and `block_table` gets 3 entries (2 shared + 1 new).
3. `prepare_prefill`: `start = 512`, `end = start + seqlen_q = 512 + 600 = 1112`.
4. `end_block = ceil(1112 / 256) = 5`, so the slot-mapping loop runs over `i = 2, 3, 4`, and `seq.block_table[3]` raises `IndexError`.

The scheduler had claimed that 600 new tokens needed computing when only 88 did.

### The second fix: plan first, allocate last (`f64d821`)

PR #207 fixed the crash by moving `allocate()` *above* the "does this sequence fit in the batch?" check. That opens a different hole: if the check then says `break`, the sequence already owns blocks (and bumped `ref_count` on shared ones) without being scheduled this step. `f64d821` fixes this properly by making `can_allocate` a pure planning function that returns the number of hits, so the scheduler can do all its arithmetic before touching any state:

*`f64d821`, `nanovllm/engine/scheduler.py`, `schedule`*
```diff
-            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
+            if remaining == 0:
                 break
-
             if not seq.block_table:
-                self.block_manager.allocate(seq)
             ...   # (lines elided)
-            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
-
+                num_cached_blocks = self.block_manager.can_allocate(seq)
+                if num_cached_blocks == -1:
+                    break
+                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
+            else:
+                num_tokens = seq.num_tokens - seq.num_cached_tokens
             if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                 break
-
+            if not seq.block_table:
+                self.block_manager.allocate(seq, num_cached_blocks)
             seq.num_scheduled_tokens = min(num_tokens, remaining)
```

*`f64d821`, `nanovllm/engine/block_manager.py`*
```diff
-    def can_allocate(self, seq: Sequence) -> bool:
-        return len(self.free_block_ids) >= seq.num_blocks
+    def can_allocate(self, seq: Sequence) -> int:
         h = -1
+        num_cached_blocks = 0
+        num_new_blocks = seq.num_blocks
+        for i in range(seq.num_blocks - 1):
+            token_ids = seq.block(i)
+            h = self.compute_hash(token_ids, h)
+            block_id = self.hash_to_block_id.get(h, -1)
+            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
+                break
+            num_cached_blocks += 1
+            if block_id in self.used_block_ids:
+                num_new_blocks -= 1
+        if len(self.free_block_ids) < num_new_blocks:
+            return -1
+        return num_cached_blocks
```

A bonus: the old `can_allocate` demanded `seq.num_blocks` free blocks even when most of them would be shared. The new one subtracts hits on blocks that are already in use, because those cost nothing. Hits on *free* cached blocks still count, because taking one removes it from the free list.

> [!important] Invariant
> Anything derived from the cache lookup (how many tokens to compute, whether the sequence fits in the budget) must use the lookup's result. **The scheduler decides first and allocates last**, and it allocates only for sequences it actually schedules. `can_allocate` and `allocate` must walk exactly the same hash chain, so the plan and the allocation agree.



## Bug 3: A prefill chunk attended to keys past its own end

### Symptom

No crash. With chunked prefill (a prompt longer than `max_num_batched_tokens`), the output is wrong: early tokens in the chunk "see the future", and later tokens attend to KV slots that were never written (leftover data from whatever sequence used those blocks before, or uninitialized memory).

### The fix (PR #213)

*`25794a1`, `nanovllm/engine/model_runner.py`, `prepare_prefill`*
```diff
             seqlen = len(seq)
             start = min(seq.num_cached_tokens, seqlen - 1)
             seqlen_q = seq.num_scheduled_tokens
-            seqlen_k = seqlen
             end = start + seqlen_q
+            seqlen_k = end
             input_ids.extend(seq[start:end])
```

### Root cause

`seqlen_k = len(seq)` was correct *before* chunked prefill, because a prefill always ran to the end of the sequence. With chunking, the keys that exist after this step are those for positions `[0, end)`, not `[0, len(seq))`.

Why is the output wrong rather than just padded? When `seqlen_k > seqlen_q`, FlashAttention's `causal=True` aligns the causal mask to the **bottom-right** corner: query row $r$ may attend to keys $0 \ldots r + (\text{seqlen\_k} - \text{seqlen\_q})$. The kernel assumes the queries are the *last* `seqlen_q` tokens of the key sequence.

Tiny example: a 10-token prompt, chunk budget 4, first chunk is tokens 0..3.

```text
wrong: seqlen_q=4, seqlen_k=10  (offset 6)      right: seqlen_q=4, seqlen_k=4
          keys 0 1 2 3 4 5 6 7 8 9                        keys 0 1 2 3
query t0       x x x x x x x . . .              query t0       x . . .
query t1       x x x x x x x x . .              query t1       x x . .
query t2       x x x x x x x x x .              query t2       x x x .
query t3       x x x x x x x x x x              query t3       x x x x
               ^ ^^^^^ ^^^^^^^^^^^
               | |     keys 4..9: slots allocated but never written (garbage)
               | keys 1..3 for t0: future tokens (leak)
```

Since `cu_seqlens_k[-1] > cu_seqlens_q[-1]`, the prefix-cache path is taken and keys are read straight from the paged cache through `block_table`, so those garbage slots really do get read.

> [!important] Invariant
> For a prefill chunk covering `[start, end)`: `seqlen_q = end - start` and **`seqlen_k = end`**, never `len(seq)`. That is, **cached tokens + scheduled tokens**. With a prefix hit, `seqlen_k` includes the cached tokens (that is why it exceeds `seqlen_q`), and it must stop exactly where this chunk stops.



## Bug 4: A full prefix hit left nothing to compute

### Symptom

No crash, but the fix was a hack. When a prompt's length is an exact multiple of `block_size` and *all* its blocks are cached, there are zero tokens left to run, so there are no logits to sample the first output token from. The old code handled this by recomputing the last token anyway, which wrote into a **shared** cached block and made the bookkeeping lie.

### The fix

*`8d63a98` introduced the clamp, and `f64d821` removed it, `nanovllm/engine/model_runner.py`*
```diff
         for seq in seqs:
-            seqlen = len(seq)
-            start = min(seq.num_cached_tokens, seqlen - 1)
+            start = seq.num_cached_tokens
             seqlen_q = seq.num_scheduled_tokens
```

*`f64d821`, `nanovllm/engine/block_manager.py`: the old `allocate` could match every block, including a full last block. The new lookup never tries the last block.*
```diff
-        for i in range(seq.num_blocks):
+        for i in range(seq.num_blocks - 1):
             token_ids = seq.block(i)
-            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
+            h = self.compute_hash(token_ids, h)
```

### Root cause

A 512-token prompt, `block_size = 256`, identical to a request that ran earlier (both blocks cached). Under the pre-`f64d821` code:

1. `allocate` hits both blocks, so `num_cached_tokens = 512 = len(seq)`.
2. The scheduler forces `num_tokens = max(512 - 512, 1) = 1`.
3. `prepare_prefill` clamps `start = min(512, 511) = 511`, `end = 512`, so it recomputes token 511 and **writes its KV into slot 255 of block 1**. That block is a cached, possibly shared block that another running sequence may be reading.
4. `postprocess` has to clamp again: `num_cached_tokens = min(512 + 1, 512)`.

The values written are the same in exact arithmetic, but a cached block is supposed to be read-only, and the bookkeeping claimed "1 token after position 512" when it computed position 511.

The new rule is simple: `can_allocate`/`allocate` only look up the first `num_blocks - 1` blocks. The last block (full or partial) is always recomputed, so `num_cached_tokens < num_tokens` always holds for a fresh prefill, at least one real token produces logits, and `start = num_cached_tokens` needs no clamp. The price is recomputing up to `block_size` tokens on a full hit, which is cheap compared with the complexity it removes.

> [!important] Invariant
> A prefill always computes **at least one token**, and it computes it honestly: the computed range is exactly `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)`. **Never write into a block you got from the prefix cache.**



## Bug 5: Re-prefill after preemption, where "am I in prefill?" had three different answers

When the KV cache runs out during decode, the scheduler **preempts** a running sequence: it frees all its blocks and puts it back at the front of `waiting` ([[05 Scheduling - Continuous Batching]]). Later that sequence is prefilled again over *prompt + everything generated so far*. This path is rare, so it rots easily. Here it broke twice.

### 5a. Tensor parallel workers got no tokens (`82f5ca2`, PR #145)

**Symptom.** With `tensor_parallel_size > 1`, the first preemption crashes worker ranks with `AttributeError: 'Sequence' object has no attribute 'token_ids'`, and rank 0 hangs in NCCL.

Rank 0 sends sequences to workers by pickling them through shared memory ([[06 Getting Data onto the GPU]]). To save bandwidth, `__getstate__` sends only `last_token` once decoding has started:

*`82f5ca2`, `nanovllm/engine/sequence.py`*
```diff
     def __getstate__(self):
-        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
-                self.token_ids if self.num_completion_tokens == 0 else self.last_token)
+        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table, self.prefilled,
+                self.last_token if self.prefilled else self.token_ids)
```

The old test was "has it generated anything?", but the question the worker needs answered is "is this step a prefill?". A preempted sequence has completion tokens *and* needs a prefill, so workers got a single int and then tried `seq[start:end]`. The fix added a flag that `deallocate` resets to `False`.

### 5b. The sampled token was thrown away, then block bookkeeping broke at block boundaries (`f64d821`)

**Symptom.** `AssertionError` in `may_append` on the first decode step after a re-prefill, but only when the sequence length hits a block boundary (`len % block_size` is 0 or 1). With asserts off (`python -O`), the block table silently gets an extra block and later tokens' KV lands in the wrong block.

The chunked-prefill commit gave `postprocess` a special case that discarded the token sampled after a re-prefill:

*pre-`f64d821`, `nanovllm/engine/scheduler.py`, `postprocess`*
```diff
     def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
         for seq, token_id in zip(seqs, token_ids):
-            if is_prefill:
-                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
-                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:    # chunked prefill or re prefill after preemption
-                    seq.num_scheduled_tokens = 0
-                    continue
-            seq.append_token(token_id)
-            seq.num_cached_tokens += 1
+            self.block_manager.hash_blocks(seq)
+            seq.num_cached_tokens += seq.num_scheduled_tokens
             seq.num_scheduled_tokens = 0
+            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
+                continue
+            seq.append_token(token_id)
```

The old `may_append` also did hashing and trusted strict assumptions about which block was already hashed:

*`f64d821`, `nanovllm/engine/block_manager.py`*
```diff
     def may_append(self, seq: Sequence):
-        block_table = seq.block_table
-        last_block = self.blocks[block_table[-1]]
         if len(seq) % self.block_size == 1:
-            assert last_block.hash != -1
             block_id = self.free_block_ids[0]
             self._allocate_block(block_id)
-            block_table.append(block_id)
-        elif len(seq) % self.block_size == 0:
-            assert last_block.hash == -1
             ...   # (lines elided)
-        else:
-            assert last_block.hash == -1
+            seq.block_table.append(block_id)
+
+    def hash_blocks(self, seq: Sequence):
+        start = seq.num_cached_tokens // self.block_size
+        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
+        if start == end: return
+        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
+        for i in range(start, end):
+            block = self.blocks[seq.block_table[i]]
+            token_ids = seq.block(i)
+            h = self.compute_hash(token_ids, h)
+            block.update(h, token_ids)
+            self.hash_to_block_id[h] = block.block_id
```

Plus one explicit flag, set in exactly two places:

*`f64d821`, `nanovllm/engine/scheduler.py` and `sequence.py`*
```diff
                 seq.num_scheduled_tokens = 1
+                seq.is_prefill = False
                 self.block_manager.may_append(seq)
 ...
     def preempt(self, seq: Sequence):
         seq.status = SequenceStatus.WAITING
+        seq.is_prefill = True
 ...
-        last_state = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
+        last_state = self.last_token if not self.is_prefill else self.token_ids
```

**Root cause, with numbers.** `block_size = 256`, a sequence preempted at `len = 513` (a 300-token prompt plus 213 generated, where the last one, token 512, was sampled but its KV is not yet written):

1. Re-prefill (old code): `allocate` gives 3 blocks, and blocks 0 and 1 are full, so they are hashed immediately. Block 2 holds only token 512 and has `hash = -1`. The prefill computes all 513 tokens, including token 512.
2. `postprocess`: `num_completion_tokens > 0`, so `continue`. The freshly sampled token 513 is **discarded**. `len` stays 513.
3. Next decode: `may_append` sees `513 % 256 == 1` and concludes "token 512 starts a new block". But token 512 already lives in block 2, which the re-prefill allocated. `assert last_block.hash != -1` fails on block 2.

With `len = 512`, it breaks the other way: `allocate` hashed the full block 1 at allocation time, and then `may_append` sees `512 % 256 == 0` and asserts `last_block.hash == -1`.

Both crashes come from `may_append` inferring cache state from `len(seq) % block_size`, while the old `num_cached_tokens` did not mean one thing: after decode, `postprocess` did `num_cached_tokens += 1` for a token whose KV would only be written *next* step.

The fix gives `num_cached_tokens` one meaning: **the number of tokens whose KV is actually in the cache**. After a (re-)prefill, `num_cached_tokens == num_tokens`, so the sampled token is kept like any other. In decode, `num_cached_tokens == len(seq) - 1`, because the fed token's KV is written during the step. Hashing moves to one place, `hash_blocks`, which runs *after* the forward pass and publishes exactly the blocks that the step just completed. In the 513 case, the re-prefill now appends token 513 (`len = 514`), `514 % 256 == 2`, and no new block is needed, which is correct.

> [!warning] Publish after write
> The old `allocate` registered a block's hash in `hash_to_block_id` *before* any KV was written into it. In this codebase, scheduling order (prefill first, only the first sequence may be chunked) seems to keep anyone from reading such a block too early. That is luck, not design. In your engine, a block becomes findable by hash only after its KV has been written.

> [!important] Invariant
> `num_cached_tokens` = tokens with KV in cache, and it only advances in `postprocess`, by exactly `num_scheduled_tokens`. "Is this a prefill?" is an explicit per-sequence flag, set by the scheduler, not inferred from counts. A preempted sequence re-prefills prompt + all generated tokens, keeps the token sampled at the end, and ends up in exactly the same state as if it had never been preempted.



## Bug 6: An evicted block kept its old name in the hash table

### Symptom

No crash. Rarely and silently, a request gets a prefix-cache "hit" on a block whose KV was computed for a **different context**, so its output is wrong from the first token.

### The fix

*`f64d821`, `nanovllm/engine/block_manager.py`, `_allocate_block`*
```diff
     def _allocate_block(self, block_id: int) -> Block:
         block = self.blocks[block_id]
         assert block.ref_count == 0
+        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
+            del self.hash_to_block_id[block.hash]
         block.reset()
```

### Root cause

Freed blocks keep their contents and their hash-table entry, which is the whole point of prefix caching ([[04 Memory - Paged KV Cache]]). When the free list finally hands such a block to someone new, `reset()` clears the block's own `hash` and `token_ids`, but the old code left `hash_to_block_id[old_hash] → block_id` in the dictionary.

The only guard on a lookup is `self.blocks[block_id].token_ids != token_ids`. That checks the block's **own 256 tokens**, not the context before them. But KV depends on everything before the block (through attention in every earlier layer, and through RoPE positions).

Scenario, `block_size = 4` to keep it small. Let `T = [7, 7, 7, 7]` and `S = [1, 2, 3, 4]`.

1. Request A = `T + ...`. Block X holds `T` at positions 0..3 with hash $h_A = H(T)$. A finishes, and X is freed with `hash_to_block_id[h_A] = X`.
2. X is reallocated to request C = `S + T + ...`, as C's *second* block. `reset()` clears X, but `h_A → X` stays.
3. C's prefill writes `T`'s KV into X, computed after `S` at positions 4..7. `hash_blocks` sets `X.hash = H(H(S), T)`, `X.token_ids = T`.
4. Request D = `T + ...` looks up $H(T) = h_A$, which gives X. `X.token_ids == T` matches, so it **hits** and reuses KV that was computed in context `S` at positions 4..7.

The `get(...) == block_id` guard matters too: two sequences computing the same prefix in the same batch both register the same hash, and the last write wins. If X is evicted but the hash now names block Y, deleting it would throw away Y's valid entry.

> [!important] Invariant
> A hash entry must point at a block whose *current* contents are exactly what that hash describes: the block's tokens **and** all tokens before it. When a block is evicted (reset for new use), remove its entry if the entry still points at it. Chained hashes ($h_i = H(h_{i-1}, \text{tokens}_i)$) make the hash cover the prefix, but only if stale entries can't survive.



## Bug 7: A cache hit on a free block erased that block's identity

### Symptom

Measured with `num_cached_tokens`, prefix caching "works once": the second request with a shared system prompt hits, but the third doesn't. Worse, the next block's chained hash gets computed with a missing prefix, which recreates the poisoning scenario from Bug 6.

### The fix

*`9fa256a`, `nanovllm/engine/block_manager.py`, `allocate` (hit path)*
```diff
             if block_id in self.used_block_ids:
                 block.ref_count += 1
             else:
-                self._allocate_block(block_id)
+                block.ref_count = 1
+                self.free_block_ids.remove(block_id)
+                self.used_block_ids.add(block_id)
             seq.block_table.append(block_id)
```
*and `_allocate_block` now always takes a fresh block from the head of the free list:*
```diff
-    def _allocate_block(self, block_id: int) -> Block:
+    def _allocate_block(self) -> int:
+        block_id = self.free_block_ids.popleft()
         block = self.blocks[block_id]
         ...
-        self.free_block_ids.remove(block_id)
         self.used_block_ids.add(block_id)
-        return block
+        return block_id
```

### Root cause

In `f64d821`, `_allocate_block` came to mean "claim a block for **new** content": delete its hash entry (Bug 6's fix) and `reset()` it. But `allocate` also called it for a **cache hit on a free block**, which is a block we want *because of* its content. So each such hit:

1. deleted `hash_to_block_id[h]`, so the next request with the same prefix misses;
2. `reset()` set `block.hash = -1` and `token_ids = []`, so after this request finishes, the block is back in the free list with no identity at all.

Concrete run, `block_size = 256`, system prompt `P` of 256 tokens:

- R1 = `P + q1`: computes block X, and `hash_blocks` publishes `H(P) → X`. R1 finishes, and X is free but still named.
- R2 = `P + q2` (300 tokens total): hits X, and `_allocate_block(X)` wipes it. Then `hash_blocks` for R2's block 1 uses `h = self.blocks[seq.block_table[0]].hash = -1` as the prefix, so it publishes `H(q2[:256])`, a hash that claims "these tokens at the **start** of a sequence".
- R3 = `P + q3` misses on X, because the entry is gone.
- R4 whose prompt *starts with* `q2[:256]` hits R2's block 1, whose KV was computed after `P`. That is wrong output.

The fix separates the two operations. Reviving a cached free block keeps its hash and tokens: take it out of the free list, set `ref_count = 1`, done. Only fresh allocations go through `_allocate_block`, which pops the *head* of the free deque. That is O(1) instead of the old `deque.remove` (O(n)), and because freed blocks are appended at the tail, the least recently freed cached block is evicted first, which is LRU-like eviction for free.

> [!important] Invariant
> Two different operations must never share one code path: **reuse** a block (keep hash + tokens, bump the ref count) and **recycle** a block (drop hash entry, reset). A block's hash is set in exactly one place (after its KV is written) and cleared in exactly one place (when it is recycled).



> [!question] Check yourself
> 1. After chunked prefill was added, why was `seqlen_k = len(seq)` wrong even though every key slot up to `len(seq)` had been *allocated*?
> 2. A 768-token prompt fully matches three cached blocks. How many tokens does the current code compute, and why not zero?
> 3. Why is checking `blocks[id].token_ids == token_ids` not enough to prove a cache hit is valid?
>
> > [!success]- Answer
> > 1. Allocated is not written. Slots past `end` hold garbage, and FlashAttention's bottom-right causal alignment makes the chunk's queries both see those slots and see future tokens inside the chunk. `seqlen_k` must be `end`.
> > 2. 256. The lookup only tries the first `num_blocks - 1 = 2` blocks, so the last block (tokens 512..767) is recomputed. At least one token has to run to produce logits, and this way nothing is ever written into a shared cached block.
> > 3. It checks only the block's own tokens. KV also depends on every earlier token and on the positions. The chained hash covers that, but only if stale hash entries are removed (Bug 6) and hit blocks keep their hashes (Bug 7).



## Invariant checklist: test cases for your own engine

Each line below is a property to assert, or a test to write. Most are easiest to check with **greedy decoding compared against a reference** (plain Hugging Face `generate`, or your own engine with caching, chunking, and preemption all turned off). Outputs must be token-identical.

- [ ] **Decode position.** Position of the fed token = `len(seq) - 1` = its slot index. *Test:* greedy output matches the HF reference for 100+ generated tokens.
- [ ] **Partial prefix hit.** Tokens to compute = `len - cached_blocks * block_size`, taken from the lookup and never from a stale counter. *Test:* run prompt A, then A's first 512 tokens + 88 new ones. No `IndexError`, and only 88 tokens computed.
- [ ] **Plan before allocate.** A sequence that doesn't fit the budget owns no blocks and holds no references afterward. *Test:* after `schedule()`, every sequence with a non-empty `block_table` is either scheduled or running.
- [ ] **Chunked prefill.** `seqlen_k = num_cached + num_scheduled`, never `len(seq)`. *Test:* a 3000-token prompt with `max_num_batched_tokens = 512` gives output identical to the unchunked run.
- [ ] **Prefix hit + chunked prefill.** `cu_seqlens_k` includes the cached tokens *and* stops at the chunk end. *Test:* shared 1024-token prefix + 2000 new tokens, small budget, identical to the reference.
- [ ] **Full prefix hit.** At least one token is computed, and no write lands in a block with `ref_count > 1` or a block obtained as a hit. *Test:* submit the same 512-token prompt twice (`block_size = 256`), and assert that `slot_mapping` never touches the hit blocks.
- [ ] **`num_cached_tokens` means "KV written".** After every step: prefill done means `cached == len - 1` once the sampled token is appended; decode keeps `cached == len - 1` before each step. *Test:* assert this in `postprocess`.
- [ ] **Preemption is invisible.** *Test:* use a KV cache so small it forces preemptions, with lengths landing on `len % block_size ∈ {0, 1}`. Greedy output is identical to a big-cache run, and the block count per sequence always equals `ceil(len / block_size)` (or `ceil((len - 1) / block_size)` right before `may_append`).
- [ ] **Workers see full tokens on every prefill**, including re-prefill after preemption. *Test:* the preemption test above with `tensor_parallel_size = 2`.
- [ ] **Publish after write.** A block's hash enters the table only after the step that wrote its last slot. *Test:* after each step, every hashed block `i` of a sequence satisfies `(i + 1) * block_size <= num_cached_tokens`.
- [ ] **Eviction removes the name.** *Test:* the Bug 6 scenario. Fill the cache so block X is recycled into a different context, then query the old prefix: it must miss.
- [ ] **Hit keeps the name.** *Test:* three requests share a system prompt. The 2nd **and** 3rd both report `num_cached_tokens > 0`.
- [ ] **Hash chain is unbroken.** For block `i > 0`, the prefix hash is block `i-1`'s hash, and never `-1`. *Test:* assert in `hash_blocks`.
- [ ] **Reference counts balance.** *Test:* after all requests finish, `len(free_block_ids) == num_blocks`, `used_block_ids` is empty, and every `ref_count == 0`.



## Build it yourself

- [ ] Write a reference harness first: greedy HF generation on ~20 prompts of varied lengths, including ones at exact multiples of `block_size`, plus a shared-prefix family. Every feature you add below must keep outputs token-identical.
- [ ] Define `num_cached_tokens` and `num_scheduled_tokens` in one sentence each (see Bug 5), and advance them only in `postprocess`.
- [ ] Make `can_allocate` a pure function that returns the hit count. The scheduler computes budgets from it, and then calls `allocate(seq, num_cached_blocks)`.
- [ ] In `prepare_prefill`, derive everything from `start = num_cached_tokens` and `end = start + num_scheduled_tokens`: `input_ids`, `positions`, `slot_mapping`, and `seqlen_k = end`.
- [ ] Never offer the last block to the prefix cache.
- [ ] Hash blocks in exactly one place, after the forward pass. Keep "reuse a cached block" and "recycle a free block" as separate code paths, and delete stale hash entries on recycle.
- [ ] Add an explicit `is_prefill` flag. Set it to `True` on creation and on preemption, and to `False` when the sequence is scheduled for decode.
- [ ] Turn the checklist above into `assert`s that run in debug mode after every `schedule()` and `postprocess()`.

← [[09 Measuring Speed]] | Back to the start: [[01 Entry Points and Settings]]
