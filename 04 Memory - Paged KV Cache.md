---
part: 4
files: [nanovllm/engine/block_manager.py, nanovllm/engine/sequence.py, nanovllm/engine/scheduler.py]
tags: [nano-vllm]
---

# Memory: the Paged KV Cache

> [!abstract] In one line
> `BlockManager` cuts GPU KV-cache memory into fixed 256-token blocks, gives every sequence a `block_table` that maps its logical blocks to physical ones, and uses a chained hash of each full block to share identical prompt prefixes between requests.

<br>

## Why this exists

A transformer generating text keeps the keys and values of every token it has seen: each new token has to attend over all of them. Recomputing them at every step would turn generation from $O(n)$ into $O(n^2)$ work, so they are stored. This store is the **KV cache**, and in a serving system it is the main thing competing for GPU memory once the weights are loaded.

The question this file answers is: *how do you split one big slab of GPU memory among hundreds of requests whose lengths you don't know in advance, without wasting most of it?* The answer vLLM made famous is **paging**, the same trick an operating system uses for virtual memory. And once memory is in pages, a second win comes nearly free: two requests that start with the same tokens (a shared system prompt, say) can point at the *same* pages. This is **prefix caching**.

`block_manager.py` is 120 lines, and it contains no GPU code at all. It is pure bookkeeping on integers. The GPU tensor it manages lives in `model_runner.py` ([[06 Getting Data onto the GPU]]); the block manager only decides *which block number* each token's K and V go into.

<br>

## Building intuition first

### How big is the KV cache?

For every token, every layer stores one key vector and one value vector per KV head. So the memory per token is

$$
\text{bytes per token} = 2 \times L \times n_{kv} \times d_{head} \times \text{bytes per element}
$$

The $2$ is for K and V. With **Qwen3-0.6B** (the model the repo's examples use): $L = 28$ layers, $n_{kv} = 8$ KV heads, $d_{head} = 128$, bf16 = 2 bytes:

$$
2 \times 28 \times 8 \times 128 \times 2 = 114{,}688 \text{ bytes} \approx 112 \text{ KiB per token}
$$

One 256-token block is therefore $256 \times 112\text{ KiB} = 28\text{ MiB}$, and a single 4096-token conversation needs $448\text{ MiB}$. Three or four such requests already hold as much memory as the model's own weights (roughly 1.2–1.5 GB in bf16). KV memory, not compute, is what caps how many requests you can run at once.

The repo computes exactly this, per block, when it sizes the cache:

*`nanovllm/engine/model_runner.py` L112–115*
```python
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
```

The whole cache is one tensor of shape `[2, L, num_blocks, block_size, n_kv, d_head]`. Whatever memory is left after weights and activations becomes `num_kvcache_blocks` blocks, and that integer is all `BlockManager` is ever told.

<br>

### Why the naive layout wastes memory

The obvious approach is to give each request one contiguous region, big enough for its maximum possible length (`max_model_len`, since you don't know when it will stop).

```text
Contiguous allocation, capacity = 16 slots per request reserved up front
("#" = token actually stored, "." = reserved but empty)

GPU KV memory
+----------------+----------------+----------------+-------+
|######..........|###########.....|####............| free  |
+----------------+----------------+----------------+-------+
  req A (6 used)   req B (11 used)  req C (4 used)    can't fit
                                                      a 16-slot
                                                      request
```

Three kinds of waste appear:

- **Reservation waste**: request A might stop after 6 tokens, but 16 slots were reserved. Most requests finish far below the maximum, so most reserved memory is never touched.
- **External fragmentation**: when B finishes, it leaves a 16-slot hole. Holes of different sizes end up scattered between live requests, and a new request needing one large contiguous run can fail even though the total free memory would be enough.
- **No sharing**: if A and C start with the same 1000-token system prompt, they each hold their own copy of its K and V.

The vLLM paper measured that systems using contiguous allocation actually used only 20–40% of their KV memory for real tokens.

<br>

### Paging: logical blocks and physical blocks

The fix is the one operating systems settled on decades ago. Chop memory into equal-size **physical blocks** (here, 256 token slots each). Think of a sequence's tokens as a sequence of **logical blocks**: logical block 0 holds tokens 0–255, block 1 holds 256–511, and so on. A per-sequence **block table** (a plain Python list, `seq.block_table`) maps logical block `i` to whichever physical block happens to be free.

```text
            logical view                          physical KV memory (num_blocks = 8)
                                                  +----+----+----+----+----+----+----+----+
 seq A:  [ blk0 | blk1 | blk2 ]                   | A0 | B0 | A1 | -- | B1 | A2 | -- | -- |
 block_table = [0, 2, 5]  -------------------->   +----+----+----+----+----+----+----+----+
                                                    0    1    2    3    4    5    6    7
 seq B:  [ blk0 | blk1 ]
 block_table = [1, 4]

 token t of a sequence lives at:
     physical block = block_table[t // block_size]
     offset         = t %  block_size
     slot           = physical block * block_size + offset
```

Now:

- A sequence only holds blocks it has actually filled, plus at most one partly filled last block. Waste is bounded by `block_size - 1` slots per sequence.
- Any free block works for any sequence, so there is no external fragmentation. Free memory is just a list of block ids.
- Two block tables can contain the same physical block id. That is prefix sharing, and it costs one integer per shared block.

The attention kernel (flash-attn's `block_table=` argument) knows how to follow the table, so the K/V of one sequence never needs to be contiguous. How the table becomes a GPU tensor and a `slot_mapping` is [[06 Getting Data onto the GPU]]; here we only care about who owns which block.

> [!important] The division of labour
> `BlockManager` never touches a tensor. It hands out block **ids** and records them in `seq.block_table`. The model runner turns `block_table` into slot indices, and the attention layer writes K/V there. If you get the bookkeeping right, the GPU side is mechanical.

<br>

### The `Sequence` fields this file relies on

`Sequence` is covered in [[03 Tracking a Request]]. The block manager uses only these pieces:

*`nanovllm/engine/sequence.py` L55–65*
```python
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

- `num_blocks` is a ceiling division: 600 tokens → `(600 + 255) // 256 = 3` blocks.
- `block(i)` returns the token ids of logical block `i`. For 600 tokens, `block(2)` has only 88 ids: the last block is usually partial.
- Also used: `seq.block_table` (the list of physical ids), `seq.num_cached_tokens` (tokens whose K/V are already in the cache) and `seq.num_scheduled_tokens` (tokens being computed this step).

<br>

## Walkthrough: `nanovllm/engine/block_manager.py`

### `Block`: one physical page's metadata

*`nanovllm/engine/block_manager.py` L8–23*
```python
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

A `Block` is a small Python record describing one slice `kv_cache[:, :, block_id]` of the GPU tensor. It holds no K/V data itself.

- `block_id`: its index into the physical cache. Fixed forever.
- `ref_count`: how many live sequences have this block in their `block_table`. 0 means nobody is using it; 2 means two requests are sharing it.
- `hash`: the prefix hash of the block's contents, or `-1` if the block is not (yet) full and so not shareable.
- `token_ids`: a copy of the tokens it holds. Used only to double-check a hash match (see `can_allocate`).

`update` is called when a block becomes full and gets registered for reuse. `reset` is called when a block is handed to a new owner: note that it sets `ref_count = 1`, not 0, because resetting always happens at the moment someone claims the block.

<br>

### `BlockManager.__init__`: the pool

*`nanovllm/engine/block_manager.py` L26–33*
```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

Four structures, and every method below is about keeping them consistent:

| Structure | Holds | Meaning |
|---|---|---|
| `blocks` | `Block` objects, indexed by id | per-block metadata |
| `free_block_ids` | ids with `ref_count == 0` | available to hand out; front = next to be reused |
| `used_block_ids` | ids with `ref_count >= 1` | owned by at least one sequence |
| `hash_to_block_id` | prefix hash → block id | the prefix cache index |

Every block is in exactly one of `free_block_ids` or `used_block_ids`. `hash_to_block_id` is independent of both: a *free* block can still be in the hash table. That is the whole trick behind reusing a prefix after the request that computed it has finished.

The scheduler creates the manager with the number computed in `allocate_kv_cache`:

*`nanovllm/engine/scheduler.py` L15*
```python
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
```

<br>

### Taking and returning a block: `_allocate_block` / `_deallocate_block`

*`nanovllm/engine/block_manager.py` L43–56*
```python
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
```

These two are the low-level "get a fresh page" and "give a page back".

`_allocate_block`:
- `popleft()` takes the block at the **front** of the free queue. `_deallocate_block` `append`s to the **back**. So the free queue is ordered oldest-freed first, and a freshly freed block is the *last* to be overwritten. That makes it an LRU eviction policy for cached prefixes with no extra code.
- If the block still carries a hash from its previous life, its contents are about to be overwritten, so its entry in `hash_to_block_id` must go. Otherwise a future request would "hit" on garbage.
- The `== block_id` check matters: the same hash can have been re-registered to a *different* block since (see `hash_blocks` and the warning under `can_allocate`). In that case the dictionary entry belongs to the other block and must be left alone.
- `reset()` wipes the metadata and sets `ref_count = 1`.

`_deallocate_block` does *not* clear `hash` or `token_ids`. The block goes back into the pool with its contents and its hash entry intact, so it can still be found as a cache hit until someone actually pops it and overwrites it.

```text
free_block_ids (deque)                         used_block_ids (set)
front                                   back
[ 7, 8, 9, ..., 3(h), 1(h), 0(h) ]            { 2, 4, 5, 6 }
  ^                          ^
  popleft(): next victim      append(): just freed, still has hash (h)
             (hash dropped    -> survives longest, still hittable
              only now)
```

<br>

### `compute_hash`: one number that names a whole prefix

*`nanovllm/engine/block_manager.py` L35–41*
```python
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
```

- **xxhash** (`xxh64`) is a fast, non-cryptographic 64-bit hash. We need speed, not security.
- The block's token ids are packed into bytes via NumPy (`np.array(...).tobytes()`, int64 per id) and fed to the hasher.
- If there is a previous block, its hash (8 bytes) is fed in **first**. So

$$
h_0 = H(\text{block}_0), \qquad h_i = H(h_{i-1} \,\|\, \text{block}_i)
$$

This **chaining** is essential. Block $i$'s K/V depend on *every* token before it, not just the 256 in the block (attention mixes them in). Two requests can have an identical block 5 but different blocks 0–4, and then their block-5 K/V are different. Because $h_i$ folds in $h_{i-1}$, which folds in $h_{i-2}$, and so on, $h_i$ identifies the entire prefix `tokens[0 : 256·(i+1)]`. A match on $h_i$ means "the first $i+1$ blocks are identical", which is exactly the condition for the K/V to be reusable.

```text
tokens:   [ blk0 ]        [ blk1 ]            [ blk2 ]
             |               |                   |
             v               v                   v
          H(blk0) = h0 --> H(h0 || blk1) = h1 --> H(h1 || blk2) = h2
                                                      ^
                            h2 names tokens 0..767, not just blk2
```

> [!important] Only full blocks get a hash
> A partial last block is never hashed. Its contents are still changing (decode appends tokens into it), so any hash would be stale one step later. And even at a fixed moment, a block holding 88 tokens cannot be handed to a request that needs 256 tokens there. Hashes are assigned only in `hash_blocks`, and only to blocks that have just become completely full.

<br>

### Asking "is there room?": `can_allocate`

*`nanovllm/engine/block_manager.py` L58–73*
```python
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks
```

Called by the scheduler for a waiting sequence that has no blocks yet. It answers two things at once: how many leading blocks are already cached, and whether the rest fits. It returns the number of cached blocks, or `-1` for "doesn't fit, try later". It changes no state.

Idea by idea:

- **Walk the prompt's blocks in order**, computing the chained hash `h` as it goes, and look each one up in `hash_to_block_id`.
- **`range(seq.num_blocks - 1)`: the last block is never probed.** Even if it is full and cached, the sequence must run at least one token through the model to get logits for sampling the next token. Leaving the whole last block to be computed guarantees that. For a 600-token prompt the loop checks blocks 0 and 1 only; for a 512-token prompt it checks only block 0 (block 1 is recomputed even if cached — a small cost accepted for simplicity).
- **`self.blocks[block_id].token_ids != token_ids`**: a guard against hash collisions. 64-bit collisions are astronomically rare, but serving the wrong K/V would silently corrupt output, and comparing 256 ints is cheap.
- **`break` on the first miss.** Once block $i$ misses, every block after it must miss too: $h_{i+1}$ is computed from $h_i$, so if the prefix up to $i$ was never seen, no longer prefix was either. Cache hits are therefore always a contiguous run from block 0.
- **`num_new_blocks`** counts how many blocks must come out of `free_block_ids`. It starts at `num_blocks` and drops by one only for hits on blocks in `used_block_ids`, because those are already live and just get shared. A hit on a *free* cached block does not reduce the count: `allocate` will pull that block out of the free queue, which consumes a free slot just like a fresh block would.

Example: 3-block prompt, blocks 0 and 1 hit, block 0 is in use by another request, block 1 was freed earlier.

```text
num_new_blocks = 3
  blk0 hit, in used_block_ids  -> 2   (shared, costs nothing)
  blk1 hit, in free_block_ids  -> 2   (revived, still takes it out of the free deque)
  blk2 not probed                    (always freshly allocated)
needs len(free_block_ids) >= 2, returns num_cached_blocks = 2
```

<br>

### Claiming the blocks: `allocate`

*`nanovllm/engine/block_manager.py` L75–92*
```python
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
```

`allocate` does what `can_allocate` promised. The scheduler always calls them back to back, with no other block-manager call in between, so the hash table has not changed and the first `num_cached_blocks` lookups are guaranteed hits (hence the plain `[h]` rather than `.get`).

- **Hit on a live block** (`in used_block_ids`): just `ref_count += 1`. The block now appears in two block tables.
- **Hit on a free cached block**: revive it. `ref_count = 1`, move it from free to used. `deque.remove` is $O(n)$ in the number of free blocks, which is fine at this scale. Crucially, it does *not* call `reset()`, since the hash and token ids are exactly what we want to keep.
- **Remaining blocks** (from the first miss to the end, always including the last): fresh blocks from `_allocate_block`.
- **`seq.num_cached_tokens`** is set to the number of tokens whose K/V are already in the cache. The model runner starts computing from this position, so a prompt with 2 cached blocks only runs `num_tokens - 512` tokens through the model.

This is how the scheduler uses the pair (full scheduler logic in [[05 Scheduling - Continuous Batching]]):

*`nanovllm/engine/scheduler.py` L35–46*
```python
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
```

Note that `allocate` reserves blocks for the *whole prompt* up front, even if chunked prefill will only compute part of it this step. A sequence that already has a `block_table` (a chunked prefill in progress) skips both calls.

> [!warning] Same prefix in the same batch does not share
> Blocks get their hashes in `hash_blocks`, which runs in `postprocess` *after* the forward pass. If two requests with the same system prompt are admitted in the same `schedule()` call, the second one's `can_allocate` finds nothing in `hash_to_block_id` yet, and both compute the prefix separately. Afterwards both register the same hash; the dictionary keeps whichever was written last. That duplicate is why `_allocate_block` checks `hash_to_block_id.get(block.hash) == block_id` before deleting.

<br>

### Growing during decode: `can_append` / `may_append`

*`nanovllm/engine/block_manager.py` L103–108*
```python
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
```

During decode, each step computes K/V for exactly one token: the last one, `seq.last_token`, which `postprocess` appended after the previous step but whose K/V are not in the cache yet. That token sits at position `len(seq) - 1`. It needs a new block exactly when it is the first token of a new block, i.e. when `(len(seq) - 1) % block_size == 0`, which is `len(seq) % block_size == 1`.

With `block_size = 256`:

```text
len(seq) = 256  -> new token at position 255, last slot of block 0     -> no new block
len(seq) = 257  -> new token at position 256, first slot of block 1    -> allocate
len(seq) = 258  -> position 257, block 1 slot 1                        -> no new block

block_table:  [ 12 ]                         before, len 256
              [ 12 | 40 ]                    after may_append, len 257
               #### #...
```

- `can_append` uses a small trick: `(len(seq) % self.block_size == 1)` is a `bool`, and `True >= ...` compares as `1`, `False` as `0`. So it reads "need 1 free block if a new one is required, else 0". It is always true in the common case.
- If it is false, the scheduler **preempts** another running sequence (frees all its blocks and puts it back in the waiting queue) until there is room:

*`nanovllm/engine/scheduler.py` L58–70*
```python
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

<br>

### Registering full blocks: `hash_blocks`

*`nanovllm/engine/block_manager.py` L110–120*
```python
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
```

This is the only place a block gains a hash, and the only place entries are added to `hash_to_block_id`. It runs after every forward pass, for every sequence in the batch, *before* `num_cached_tokens` is advanced:

*`nanovllm/engine/scheduler.py` L82–85*
```python
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
```

- Before this step, tokens `[0, num_cached_tokens)` had K/V in the cache. After it, tokens `[0, num_cached_tokens + num_scheduled_tokens)` do.
- `start` = the index of the first block that was not full before. `end` = the number of blocks that are full now (integer division drops a partial block). Blocks `start .. end-1` have **just become full**, and exactly those get hashed.
- `start == end`: no block completed this step (the typical decode step). Return immediately.
- The chain is resumed from the previous block's stored hash, `blocks[block_table[start - 1]].hash`. That block is full and was hashed earlier (either in a previous step or because it was a cache hit), so its hash is valid.
- `block.update(h, token_ids)` stores the hash and a copy of the tokens (the slice from `seq.block(i)` is a new list) for the collision check.

Three cases, with `block_size = 256`:

```text
full prefill, 600 tokens     cached 0,   scheduled 600  -> start 0, end 2  -> hash blocks 0,1
                                                             (block 2 has 88 tokens: skipped)
chunked prefill, chunk 2     cached 300, scheduled 300  -> start 1, end 2  -> hash block 1
decode, block fills          cached 767, scheduled 1    -> start 2, end 3  -> hash block 2
decode, typical              cached 700, scheduled 1    -> start 2, end 2  -> nothing
```

So blocks filled by generated tokens are registered too. That is why a preempted request, when it is rescheduled, usually finds most of its own blocks still in the cache.

<br>

### Releasing a sequence: `deallocate`

*`nanovllm/engine/block_manager.py` L94–101*
```python
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
```

Called when a sequence finishes (in `postprocess`) or is preempted (in `preempt`).

- `ref_count -= 1` for every block in the table. Shared blocks survive as long as anyone else still references them; only blocks reaching 0 go back to the free queue.
- **Reverse order** is deliberate. Freed blocks are `append`ed to the back of `free_block_ids`, and `_allocate_block` recycles from the front. Freeing the tail first means the **prefix blocks land at the very back and are evicted last**. That is the right priority: block 0 of a system prompt is shared by far more future requests than block 9 of one particular conversation. And because of the chaining, block 9 is useless without blocks 0–8 anyway, so evicting from the tail loses nothing that could still be hit.
- The freed blocks keep their `hash` and `token_ids`, and their entries stay in `hash_to_block_id`. A later request with the same prefix can revive them in `allocate`, until `_allocate_block` finally recycles them.
- `num_cached_tokens = 0` and an empty `block_table` put the sequence back into its "never allocated" state, which is what the scheduler checks (`if not seq.block_table`) when a preempted sequence comes back.

<br>

## A worked example: two requests sharing a prefix

To keep the drawing small, pretend `block_size = 4` (the real code asserts a multiple of 256; the logic is identical). Letters are token ids.

```text
A = [ a b c d | e f g h | i j ]        10 tokens, 3 blocks
B = [ a b c d | e f g h | x y z ]      11 tokens, 3 blocks  (shares 8 tokens with A)
free_block_ids = [0 1 2 3 4 5 ...]     hash_to_block_id = {}
```

**Step 1: A is admitted.** `can_allocate(A)` probes blocks 0 and 1 (not 2). Block 0 misses, so it breaks and returns 0. `allocate` gives A three fresh blocks.

```text
A.block_table = [0, 1, 2]      num_cached_tokens = 0     -> prefill computes all 10 tokens

phys   0          1          2
     [abcd]     [efgh]     [ij..]
 ref   1          1          1
hash   -          -          -
```

**Step 2: after A's prefill, `hash_blocks(A)`**: `start = 0`, `end = 10 // 4 = 2`. Blocks 0 and 1 are hashed and registered; block 2 is partial and gets nothing.

```text
h0 = H(abcd)       h1 = H(h0 || efgh)
hash_to_block_id = { h0: 0, h1: 1 }
```

**Step 3: B arrives while A is still decoding.** `can_allocate(B)`: block 0 → `h0` hits block 0 (token ids match, in use) → `num_new_blocks` 3 → 2. Block 1 → `h1` hits block 1 → 2 → 1. Block 2 is not probed. Returns 2. `allocate(B, 2)`:

```text
A.block_table = [0, 1, 2]
B.block_table = [0, 1, 3]      num_cached_tokens = 8     -> prefill computes only x y z

                 A ----+-----------+----------.
                       |           |           \
phys   0          1          2          3
     [abcd]     [efgh]     [ijk.]     [xyz.]
 ref   2          2          1          1
hash   h0         h1         -          -
                       |           |          /
                 B ----+-----------+---------'
```

B's prefill runs 3 tokens instead of 11. Attention for those 3 tokens reads keys/values of `a..h` from physical blocks 0 and 1, which A computed.

**Step 4: A finishes.** `deallocate(A)` walks `[2, 1, 0]`: block 2 → ref 0 → freed; block 1 → ref 1; block 0 → ref 1. B is unaffected.

**Step 5: B finishes.** Walks `[3, 1, 0]`: all reach 0 and are freed in that order.

```text
free_block_ids = [ 4 5 6 ... | 2 | 3 | 1 | 0 ]
                   ^ next to be recycled        ^ evicted last
hash_to_block_id = { h0: 0, h1: 1 }             still there
```

**Step 6: C = `[a b c d | e f g h | q r s t | u]` arrives later.** `can_allocate(C)` probes blocks 0–2: `h0` hits free block 0, `h1` hits free block 1, block 2 misses. Both hits are on *free* blocks, so `num_new_blocks` stays 4. `allocate` pulls 0 and 1 out of the free queue with `ref_count = 1`, and C skips 8 tokens of prefill, even though no request holding them is alive any more.

<br>

> [!question] Check yourself
> Two prompts share their first 600 tokens, and `block_size = 256`. The first prompt has already been prefilled. When the second is admitted, how many blocks are reused?

> [!success]- Answer
> **2 full blocks = 512 tokens.** Blocks 0 (tokens 0–255) and 1 (256–511) are full and identical, so they were registered by `hash_blocks` after the first prefill (`end = 600 // 256 = 2` for a 600-token prompt) and the second prompt's `can_allocate` hits both. The remaining 88 shared tokens (512–599) sit in block 2, which was only partly filled when the first prompt's prefill ended, so it has no hash and cannot be found. Even if the first request has since decoded enough to fill block 2, its hash covers `88 shared tokens + 168 generated tokens`, which differs from the second prompt's block 2, so it misses (after `break`, nothing later is checked). The second request gets `num_cached_tokens = 512` and prefills the rest. One more edge: if the second prompt were *exactly* 512 tokens long, only block 0 would be reused, since `can_allocate` never probes the last block.

> [!question] Check yourself
> Why can a cache hit happen on a block that has `ref_count == 0`, and why must `can_allocate` still count it as needing a free block?

> [!success]- Answer
> `deallocate` returns a block to `free_block_ids` without clearing its hash, so it stays findable in `hash_to_block_id` until `_allocate_block` pops and resets it. Reviving it removes it from `free_block_ids`, so it consumes one of the free slots that `can_allocate` is budgeting; only hits on blocks already in `used_block_ids` are free of charge.

> [!question] Check yourself
> A sequence has 512 tokens after `postprocess` appended the last sampled one. Does `may_append` allocate a block on the next decode step? What about at 513?

> [!success]- Answer
> At 512: `512 % 256 == 0`, no. The token being computed is at position 511, the last slot of block 1. At 513: `513 % 256 == 1`, yes. Position 512 is the first slot of block 2.

<br>

## Build it yourself

> [!tip] Build-it-yourself
> Write the plain allocator first (no hashing), check that the model produces the same output as a contiguous cache, then add prefix caching as a separate step. Nearly every bug here is a counter out of sync, so assert your invariants liberally.

- [ ] Size the cache: $\text{block\_bytes} = 2 \cdot L \cdot \text{block\_size} \cdot n_{kv} \cdot d_{head} \cdot \text{itemsize}$; allocate one `[2, L, num_blocks, block_size, n_kv, d_head]` tensor.
- [ ] `Block` with `block_id`, `ref_count`, `hash`, `token_ids`; `reset()` sets `ref_count = 1`.
- [ ] `BlockManager` with `blocks`, a `deque` of free ids, a `set` of used ids, and `hash_to_block_id`.
- [ ] `_allocate_block` (popleft, drop stale hash entry only if it points to this block, reset) and `_deallocate_block` (append to the back).
- [ ] `allocate` without caching: one fresh block per logical block; `deallocate` in reverse with `ref_count`.
- [ ] `can_append` / `may_append` with the `len(seq) % block_size == 1` rule; wire preemption into your scheduler.
- [ ] `compute_hash` chained on the previous block's hash; `hash_blocks` over `[num_cached_tokens // bs, (num_cached_tokens + num_scheduled_tokens) // bs)`, called before `num_cached_tokens` advances.
- [ ] `can_allocate` probing blocks `0 .. num_blocks - 2`, breaking on first miss, verifying `token_ids`, and discounting only hits on used blocks; `allocate` reviving free hits without `reset()`; set `num_cached_tokens`.

Invariants your version must preserve:

- Every block id is in exactly one of `free_block_ids` / `used_block_ids`, and `ref_count == 0` iff it is free.
- `ref_count` equals the number of block tables containing that id.
- A block has `hash != -1` only if it is completely full, and its hash covers the entire prefix up to and including it.
- A hash-table entry pointing at a block is removed before that block's contents can be overwritten.
- `len(seq.block_table) == seq.num_blocks` for every running sequence after `allocate` / `may_append`.
- At least one prompt token is always computed during prefill (the last block is never taken from the cache).

<br>

← [[03 Tracking a Request]] | [[05 Scheduling - Continuous Batching]] →
