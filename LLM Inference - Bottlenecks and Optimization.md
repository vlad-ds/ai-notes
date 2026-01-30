## The Two Phases of Inference

When you send a prompt to an LLM and get a response, there are two distinct phases:

### Prefill (Processing the Prompt)

The model processes your entire prompt at once, in parallel.

```
Input:  "What is the capital of France?"
        [tok1, tok2, tok3, tok4, tok5, tok6]
                      │
                      ▼
              Single forward pass
                      │
                      ▼
        KV cache populated for all 6 positions
        Logits for position 6 → sample first output token
```

**Characteristics:**
- All prompt tokens processed in ONE forward pass
- Attention is computed for all positions simultaneously
- GPU does lots of matrix multiplies in parallel
- **Compute-bound**: GPU cores are busy doing math

### Decode (Generating the Response)

The model generates tokens one at a time, sequentially.

```
Step 1: [prompt + "The"]     → forward pass → sample "capital"
Step 2: [prompt + "The capital"] → forward pass → sample "of"
Step 3: [prompt + "The capital of"] → forward pass → sample "France"
...
```

With KV cache, each step only processes the NEW token:

```
Step 1: Process 1 token, attend to 6 cached KV  → "capital"
Step 2: Process 1 token, attend to 7 cached KV  → "of"
Step 3: Process 1 token, attend to 8 cached KV  → "France"
```

**Characteristics:**
- One token per forward pass
- Each token depends on the previous (can't parallelize)
- GPU does a small amount of math, but loads entire model
- **Memory-bandwidth-bound**: GPU cores are waiting for data

---

## GPU Resources: Compute vs Memory

A GPU has two main resources:

### 1. Compute (FLOPS)

The number of floating-point operations per second. An A100 can do ~312 TFLOPS (fp16).

**When does compute matter?**
- Large matrix multiplies
- Processing many tokens at once (prefill)
- Batching many sequences together

### 2. Memory Bandwidth

How fast data moves between GPU memory (HBM) and the compute cores. A100 has ~2 TB/s bandwidth.

**When does bandwidth matter?**
- Loading model weights
- Reading/writing KV cache
- Any operation where you load more bytes than you compute

### The Arithmetic Intensity Problem

**Arithmetic intensity** = FLOPS / Bytes loaded

```
For a matrix multiply: Y = X @ W
- X: [batch, input_dim]
- W: [input_dim, output_dim]  (model weights)

FLOPS = 2 × batch × input_dim × output_dim
Bytes loaded = W size = input_dim × output_dim × bytes_per_param

Intensity = 2 × batch
```

If batch = 1 (decode phase), intensity = 2. You load the entire weight matrix to do 2 FLOPS per weight. The GPU is starving for data.

If batch = 512 (prefill), intensity = 1024. Now compute is the bottleneck.

---

## The Bottlenecks During Inference

### During Prefill

**Bottleneck: Compute**

You're doing massive matrix multiplies with large batch sizes (all prompt tokens). The GPU cores are saturated.

**Metric:** Time-to-first-token (TTFT)

**What helps:**
- Faster GPUs (more FLOPS)
- Better matrix multiply implementations (tensor cores)
- Quantization (fewer bits = smaller matrices)

### During Decode

**Bottleneck: Memory Bandwidth**

For each token, you:
1. Load the entire model weights (~140GB for 70B model)
2. Do a tiny amount of compute (one token)
3. Write KV cache updates

```
70B model, fp16:
- Weights: 140 GB
- Bandwidth: 2 TB/s
- Time to load weights: 140 GB / 2 TB/s = 70 ms

That's your floor: ~14 tokens/second, no matter how fast your GPU computes.
```

**Metric:** Tokens per second, latency

**What helps:**
- Higher memory bandwidth (H100 > A100)
- Quantization (smaller weights to load)
- Batching multiple sequences (amortize weight loading)

### KV Cache Memory Capacity

**Bottleneck: GPU Memory Size**

Each sequence needs its own KV cache. For a 70B model:

```
KV cache per token = 2 × num_layers × hidden_dim × bytes
                   = 2 × 80 × 8192 × 2 bytes
                   = 2.6 MB per token

8K context = 21 GB per sequence
```

80GB GPU with 140GB model (multi-GPU) leaves maybe 40GB for KV cache. That's ~2 concurrent 8K sequences.

**What helps:**
- GQA/MQA (smaller KV cache)
- Quantized KV cache
- Paged attention (don't waste memory on unused context)

---

## Optimization Techniques

### 1. Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

**Problem:** KV cache is huge because every attention head has its own K and V.

Standard multi-head attention:

```
num_heads = 64
Each head has its own Q, K, V

KV cache shape: [num_layers, 2, seq_len, num_heads, head_dim]
                                        ↑
                                   64 separate K,V
```

**Multi-Query Attention (MQA):** All heads share ONE K and ONE V.

```
Q: 64 different query heads (each head attends differently)
K: 1 shared key (all heads use the same K)
V: 1 shared value (all heads use the same V)

KV cache shrinks by 64x!
```

```python
# Standard MHA
Q = [Q_head1, Q_head2, ..., Q_head64]  # 64 different
K = [K_head1, K_head2, ..., K_head64]  # 64 different
V = [V_head1, V_head2, ..., V_head64]  # 64 different

# MQA
Q = [Q_head1, Q_head2, ..., Q_head64]  # 64 different (expressive)
K = [K_shared, K_shared, ..., K_shared]  # 1 shared (memory efficient)
V = [V_shared, V_shared, ..., V_shared]  # 1 shared (memory efficient)
```

**Grouped-Query Attention (GQA):** Compromise. Groups of heads share K,V.

```
64 query heads, 8 KV heads
Every 8 query heads share 1 K,V pair

KV cache shrinks by 8x (not 64x, but less quality loss)
```

**Used by:** Llama 2 70B (GQA), Llama 3 (GQA), Mistral (GQA), PaLM (MQA)

**Implementation:**

```python
class GroupedQueryAttention:
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads):
        self.num_q_heads = num_q_heads      # e.g., 64
        self.num_kv_heads = num_kv_heads    # e.g., 8
        self.heads_per_group = num_q_heads // num_kv_heads  # e.g., 8

        self.head_dim = hidden_dim // num_q_heads

        # Q has full heads, K/V have fewer
        self.W_q = Linear(hidden_dim, num_q_heads * self.head_dim)
        self.W_k = Linear(hidden_dim, num_kv_heads * self.head_dim)  # Smaller!
        self.W_v = Linear(hidden_dim, num_kv_heads * self.head_dim)  # Smaller!

    def forward(self, x, kv_cache=None):
        Q = self.W_q(x).view(batch, seq, self.num_q_heads, self.head_dim)
        K = self.W_k(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch, seq, self.num_kv_heads, self.head_dim)

        # Repeat K, V to match Q's head count
        # Each KV head is used by `heads_per_group` Q heads
        K = K.repeat_interleave(self.heads_per_group, dim=2)
        V = V.repeat_interleave(self.heads_per_group, dim=2)

        # Now standard attention
        return attention(Q, K, V)
```

**Trade-off:** Slight quality loss (less expressive K,V) for major memory savings.

---

### 2. Quantization

**Problem:** Model weights and KV cache are too big.

**Solution:** Use fewer bits per number.

```
fp16:  16 bits per parameter  →  70B model = 140 GB
int8:   8 bits per parameter  →  70B model =  70 GB
int4:   4 bits per parameter  →  70B model =  35 GB
```

**How it works (simplified):**

```python
# fp16 weight: can be any value
weight_fp16 = 0.0234375

# int8 quantization: map to 256 discrete values
# Store a scale factor and integer
scale = (max_val - min_val) / 255
weight_int8 = round((weight_fp16 - min_val) / scale)  # e.g., 127

# To use: dequantize back
weight_restored = weight_int8 * scale + min_val  # Approximately 0.0234375
```

**Types of quantization:**

| Type | When quantized | Precision loss |
|------|----------------|----------------|
| Weight-only (W8A16) | Weights stored as int8, activations stay fp16 | Low |
| Weight + Activation (W8A8) | Both quantized | Medium |
| KV Cache quantization | KV cache stored in int8 | Low-Medium |
| 4-bit (GPTQ, AWQ, GGML) | Weights in 4-bit | Medium |

**Why it helps both bottlenecks:**
1. **Memory capacity:** Smaller model fits on fewer GPUs
2. **Memory bandwidth:** Less data to load per forward pass
3. **Compute:** int8 matrix multiply is faster than fp16 on modern GPUs

**Trade-off:** Some quality loss, especially at very low bit widths.

---

### 3. Flash Attention

**Problem:** Standard attention is memory-inefficient.

Standard attention:

```python
def standard_attention(Q, K, V):
    # Q, K, V: [batch, heads, seq_len, head_dim]

    # Step 1: Compute full attention matrix
    scores = Q @ K.T / sqrt(d_k)  # [batch, heads, seq_len, seq_len]
    # ^ This is O(seq_len²) memory!

    # Step 2: Softmax
    weights = softmax(scores, dim=-1)  # Still [batch, heads, seq_len, seq_len]

    # Step 3: Apply to values
    output = weights @ V  # [batch, heads, seq_len, head_dim]

    return output
```

For seq_len = 8192, the attention matrix is 8192 × 8192 × 4 bytes = 268 MB per head per batch item. This gets written to GPU memory (slow HBM), then read back.

**Flash Attention insight:** Never materialize the full attention matrix. Compute it in tiles that fit in fast SRAM.

```
GPU memory hierarchy:
- HBM (main memory): 80 GB, slow (~2 TB/s)
- SRAM (on-chip):    20 MB, fast (~19 TB/s)
```

```python
def flash_attention(Q, K, V):
    # Process in tiles that fit in SRAM
    output = zeros_like(Q)

    for q_tile in tiles(Q):
        # Accumulate softmax numerator and denominator
        numerator = 0
        denominator = 0

        for k_tile, v_tile in tiles(K, V):
            # Compute attention for this tile (fits in SRAM)
            scores_tile = q_tile @ k_tile.T / sqrt(d_k)

            # Online softmax: update running sum
            max_score = max(scores_tile)
            exp_scores = exp(scores_tile - max_score)

            numerator = numerator * correction + exp_scores @ v_tile
            denominator = denominator * correction + sum(exp_scores)

        output[q_tile_idx] = numerator / denominator

    return output
```

**Benefits:**
- O(seq_len) memory instead of O(seq_len²)
- 2-4x faster (less HBM traffic)
- Enables much longer context lengths

**Trade-off:** More complex implementation; requires custom CUDA kernels.

---

### 4. Paged Attention (vLLM)

**Problem:** KV cache memory is wasted.

Standard KV cache allocation:

```
Request comes in with max_seq_len = 8192

Allocate: 8192 × 2.6 MB = 21 GB for this request

But maybe the actual generation is only 500 tokens.
You wasted 20 GB.
```

Also, different requests have different lengths. Memory becomes fragmented.

**Paged Attention insight:** Treat KV cache like operating system virtual memory.

```
Physical memory: divided into fixed-size "pages" (e.g., 16 tokens each)
Logical sequence: maps to pages via a page table

Request 1: needs 100 tokens → 7 pages
Request 2: needs 50 tokens  → 4 pages
Request 3: needs 200 tokens → 13 pages

Pages allocated dynamically as generation proceeds.
No pre-allocation. No fragmentation.
```

```python
class PagedKVCache:
    def __init__(self, page_size=16, num_pages=1000):
        self.page_size = page_size

        # Physical pages: [num_pages, 2, page_size, num_heads, head_dim]
        self.physical_pages = allocate(num_pages, ...)

        # Free list
        self.free_pages = list(range(num_pages))

        # Page tables: request_id -> list of page indices
        self.page_tables = {}

    def allocate_page(self, request_id):
        page_idx = self.free_pages.pop()
        self.page_tables[request_id].append(page_idx)
        return page_idx

    def append_token(self, request_id, layer_idx, k, v):
        pages = self.page_tables[request_id]
        current_page = pages[-1]

        # Find position within page
        tokens_in_request = len(pages) * self.page_size - self.free_slots[current_page]
        pos_in_page = tokens_in_request % self.page_size

        if pos_in_page == 0 and tokens_in_request > 0:
            # Page full, allocate new one
            current_page = self.allocate_page(request_id)
            pos_in_page = 0

        # Write to physical page
        self.physical_pages[current_page, 0, pos_in_page] = k
        self.physical_pages[current_page, 1, pos_in_page] = v

    def get_kv(self, request_id, layer_idx):
        # Gather from potentially non-contiguous pages
        pages = self.page_tables[request_id]
        k = concat([self.physical_pages[p, 0] for p in pages])
        v = concat([self.physical_pages[p, 1] for p in pages])
        return k, v
```

**Benefits:**
- Near-zero memory waste
- No fragmentation
- Can swap pages to CPU if needed (like OS paging)
- Higher batch sizes = better throughput

**Used by:** vLLM, TensorRT-LLM, and most modern inference servers

---

### 5. Continuous Batching

**Problem:** Naive batching is inefficient.

Static batching:

```
Batch 3 requests together:
Request A: 100 tokens prompt, generates 50 tokens
Request B: 100 tokens prompt, generates 200 tokens
Request C: 100 tokens prompt, generates 30 tokens

With static batch:
- All start together
- A finishes at step 50... but we keep processing it (padding)
- C finishes at step 30... but we keep processing it (padding)
- Batch ends when B finishes at step 200

Wasted compute: A padded for 150 steps, C padded for 170 steps
```

**Continuous batching:** Requests can join and leave the batch dynamically.

```
Step 0:   [A, B, C] start prefill
Step 30:  C finishes → remove from batch, add new request D
Step 50:  A finishes → remove from batch, add new request E
Step 80:  E finishes → remove from batch, add new request F
Step 120: D finishes → remove from batch, add new request G
...
```

```python
class ContinuousBatcher:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.active_requests = []
        self.waiting_requests = queue()

    def step(self):
        # Remove finished requests
        self.active_requests = [r for r in self.active_requests if not r.done]

        # Add waiting requests up to batch limit
        while len(self.active_requests) < max_batch_size and self.waiting_requests:
            new_req = self.waiting_requests.pop()
            self.prefill(new_req)
            self.active_requests.append(new_req)

        # Decode one token for all active requests (batched)
        if self.active_requests:
            tokens = self.model.decode_batch(self.active_requests)

            for req, token in zip(self.active_requests, tokens):
                req.append_token(token)
                if token == EOS or req.length >= req.max_tokens:
                    req.done = True
                    self.return_response(req)
```

**Benefits:**
- GPU always has work to do
- No padding waste
- Requests don't wait for slow requests to finish
- Much higher throughput

**Complexity:** Need to handle variable sequence lengths in a batch (paged attention helps here).

---

### 6. Speculative Decoding

**Problem:** Decode is sequential; one token per forward pass of the big model.

**Insight:** A smaller model can draft multiple tokens. The big model can verify them in ONE parallel forward pass.

```
Without speculative decoding:
  Big model: [tok1] → [tok2] → [tok3] → [tok4] → [tok5]
             100ms   100ms    100ms    100ms    100ms  = 500ms

With speculative decoding:
  Small model: [tok1, tok2, tok3, tok4, tok5] draft    = 20ms (fast, sequential)
  Big model:   verify all 5 in parallel                = 100ms
  Accept 4 out of 5, reject 1, resample
  Total: ~120ms for 4 tokens instead of 400ms
```

**How verification works:**

```python
def speculative_decode(context, draft_model, target_model, num_draft=5):
    # 1. Draft model generates candidates (fast, sequential)
    draft_tokens = []
    draft_probs = []

    for _ in range(num_draft):
        logits = draft_model(context + draft_tokens)
        probs = softmax(logits)
        token = sample(probs)

        draft_tokens.append(token)
        draft_probs.append(probs[token])

    # 2. Target model scores ALL positions at once (parallel!)
    # This is the key: one forward pass scores all draft positions
    all_logits = target_model(context + draft_tokens)
    # all_logits shape: [num_draft + 1, vocab_size]
    # Position i gives P(token | context + draft_tokens[:i])

    target_probs = softmax(all_logits, dim=-1)

    # 3. Accept/reject using rejection sampling
    accepted = []
    for i, (token, q) in enumerate(zip(draft_tokens, draft_probs)):
        p = target_probs[i, token]  # Target model's prob for this token

        # Accept with probability min(1, p/q)
        # This ensures we sample from target distribution, not draft
        if random() < min(1.0, p / q):
            accepted.append(token)
        else:
            # Reject: sample from adjusted distribution
            # This handles the case where draft was "too confident"
            adjusted = torch.clamp(target_probs[i] - draft_probs_full[i], min=0)
            adjusted = adjusted / adjusted.sum()

            new_token = sample(adjusted)
            accepted.append(new_token)
            break  # Stop accepting after first rejection

    return accepted
```

**Why the math works:**

The acceptance probability `min(1, p/q)` ensures:
- If target agrees with draft (p ≈ q): accept
- If target disagrees (p < q): reject with probability proportional to disagreement
- The resulting distribution is exactly the target model's distribution

**Speedup depends on:**
- Draft model quality (higher acceptance rate = more speedup)
- Draft model speed (smaller = faster)
- Typical acceptance rates: 70-90% for well-matched draft models

**Trade-off:** Need a good draft model. No quality loss (mathematically equivalent to target model).

---

## Summary: Matching Optimization to Bottleneck

| Bottleneck | Phase | Optimizations |
|------------|-------|---------------|
| Compute | Prefill | Quantization, better hardware, Flash Attention |
| Memory Bandwidth | Decode | Quantization, batching, speculative decoding |
| Memory Capacity | Both | GQA/MQA, quantized KV cache, paged attention |
| Sequential Dependency | Decode | Speculative decoding |
| Batch Efficiency | Both | Continuous batching, paged attention |

The modern inference stack (vLLM, TensorRT-LLM, etc.) combines all of these:

```
Request arrives
    │
    ▼
Continuous batcher adds to batch
    │
    ▼
Prefill with Flash Attention (compute-optimized)
    │
    ▼
KV cache stored in pages (memory-efficient)
    │
    ▼
Decode with batched requests (bandwidth-amortized)
    │
    ▼
Speculative decoding (optional, latency-optimized)
    │
    ▼
Response returned, pages freed
```
