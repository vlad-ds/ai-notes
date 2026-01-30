[https://poloclub.github.io/transformer-explainer/](https://poloclub.github.io/transformer-explainer/)

# Embeddings

A concrete walkthrough using small numbers.

### Setup

```
Batch size: 4 (chunks)
Block size: 8 (tokens per chunk)
Vocab size: 65 (characters in Karpathy's Shakespeare dataset)
Embedding dim: 32
```

### Step 1: Input Tokens

The input is a tensor of token IDs with shape `(batch, block)` = `(4, 8)`:

```
inputs = tensor([
    [24, 43, 58,  5, 57,  1, 46, 43],   # chunk 0
    [44, 53, 56,  1, 58, 46, 39, 58],   # chunk 1
    [52, 58,  1, 58, 46, 39, 58,  1],   # chunk 2
    [25, 17, 27, 10,  0, 21,  1, 54]    # chunk 3
])
```

Each number is a token ID (0-64 for this character-level model). Shape: **(4, 8)**

### Step 2: Token Embedding

The embedding layer is a lookup table of shape `(vocab_size, embed_dim)` = `(65, 32)`:

```
embedding_table = tensor([
    [0.12, -0.34, 0.56, ...],   # embedding for token 0
    [0.78, 0.91, -0.23, ...],   # embedding for token 1
    ...                          # 65 rows total
    [0.45, -0.67, 0.89, ...]    # embedding for token 64
])
```

Each token ID gets replaced by its 32-dimensional embedding vector:

```
Token 24 → [0.23, -0.45, 0.67, 0.12, ...]  (32 numbers)
Token 43 → [0.89, 0.12, -0.34, 0.56, ...]  (32 numbers)
...
```

Output shape: **(4, 8, 32)** — each of the 32 tokens is now a 32-dim vector.

### Step 3: Positional Embedding

The model needs to know *where* each token is in the sequence. Another lookup table of shape `(block_size, embed_dim)` = `(8, 32)`:

```
position_table = tensor([
    [0.01, 0.02, -0.01, ...],   # position 0
    [0.03, -0.02, 0.04, ...],   # position 1
    [0.02, 0.05, -0.03, ...],   # position 2
    ...                          # 8 rows total
    [0.08, -0.04, 0.02, ...]    # position 7
])
```

These get **added** (not concatenated) to the token embeddings:

```
Position indices for one chunk: [0, 1, 2, 3, 4, 5, 6, 7]

For each chunk:
  token_embed[position 0] + position_embed[0]
  token_embed[position 1] + position_embed[1]
  ...
  token_embed[position 7] + position_embed[7]
```

Output shape: still **(4, 8, 32)** — same shape, but now encodes both token identity and position.

### Summary So Far

| Step | Operation | Shape |
|------|-----------|-------|
| Input | Token IDs | (4, 8) |
| Token embed | Lookup table | (4, 8, 32) |
| + Position embed | Add position vectors | (4, 8, 32) |

The input has gone from 32 integers to a `(4, 8, 32)` tensor of continuous vectors, ready for the transformer blocks.

### The Embedding Tables Are Learned

Both `embedding_table` and `position_table` are **parameters** that get updated during training. The model learns:
- Which tokens should have similar embeddings
- How position should affect the representation

These two tables account for:
- Token embeddings: 65 × 32 = 2,080 parameters
- Position embeddings: 8 × 32 = 256 parameters

# Attention Mechanism

Continuing from the embeddings. Input shape:
**(4, 8, 32)** — 4 chunks, 8 positions, 32 dimensions.

### Setup

```
Batch size: 4
Block size: 8 (positions)
Embedding dim: 32
Number of heads: 4
Head dim: 8 (= 32 / 4)
```

### Step 4: Project to Q, K, V

Three separate linear layers project the input into Query, Key, and Value:

```
Input X:  (4, 8, 32)

W_Q: (32, 32)  →  Q = X @ W_Q  →  (4, 8, 32)
W_K: (32, 32)  →  K = X @ W_K  →  (4, 8, 32)
W_V: (32, 32)  →  V = X @ W_V  →  (4, 8, 32)
```

**How the matmul works:**

The operation `X @ W_Q` is really `(8, 32) @ (32, 32) → (8, 32)` broadcast across the 4 batches.

For a single chunk, we have 8 vectors of 32 dimensions. Each output element is a dot product of a row from X with a column from W_Q.

**Why heads have separate weights:**

The (32, 32) weight matrix is effectively 4 separate (32, 8) matrices stacked:

```
W_Q = [ W_Q_head0 | W_Q_head1 | W_Q_head2 | W_Q_head3 ]
        (32, 8)     (32, 8)     (32, 8)     (32, 8)
```

When we reshape Q from (4, 8, 32) to (4, 4, 8, 8):
- Head 0 gets output from columns 0-7 of W_Q
- Head 1 gets output from columns 8-15 of W_Q
- Head 2 gets output from columns 16-23 of W_Q
- Head 3 gets output from columns 24-31 of W_Q

Each head has its own learnable weights. We just compute them in one matmul for efficiency.

### Step 5: Split into Heads

Reshape Q, K, V to separate the heads:

```
Q: (4, 8, 32) → (4, 4, 8, 8)
   (batch, positions, embed) → (batch, heads, positions, head_dim)

Same for K and V.
```

Each head now works with 8-dimensional vectors independently.

### Step 6: Compute Attention Scores

For each head, compute how much each position should attend to each other position:

```
scores = Q @ K.transpose(-2, -1)

Q: (4, 4, 8, 8)
K transposed: (4, 4, 8, 8)

scores: (4, 4, 8, 8)
        (batch, heads, positions, positions)
```

The result is an 8×8 matrix per head per chunk. Entry (i, j) is how much position i wants to attend to position j.

Example for one head, one chunk:

```
         pos0  pos1  pos2  pos3  pos4  pos5  pos6  pos7
pos0 [   1.2   ...   ...   ...   ...   ...   ...   ... ]
pos1 [   0.8   0.5   ...   ...   ...   ...   ...   ... ]
pos2 [   0.3   1.1   0.7   ...   ...   ...   ...   ... ]
pos3 [   0.1   0.4   0.9   0.6   ...   ...   ...   ... ]
pos4 [   0.2   0.3   0.5   1.3   0.4   ...   ...   ... ]
pos5 [   0.4   0.2   0.1   0.8   0.7   0.9   ...   ... ]
pos6 [   0.1   0.5   0.3   0.2   0.6   1.1   0.8   ... ]
pos7 [   0.3   0.4   0.2   0.5   0.4   0.3   0.7   0.6 ]
```

### Step 7: Scale

Divide by sqrt(head_dim) to prevent dot products from getting too large:

```
scores = scores / sqrt(8)
       = scores / 2.83
```

Without scaling, large values would push softmax into saturation (all attention on one position).

### Step 8: Causal Mask

For autoregressive models, position i can only attend to positions ≤ i. Mask out the future:

```
         pos0  pos1  pos2  pos3  pos4  pos5  pos6  pos7
pos0 [   1.2   -inf  -inf  -inf  -inf  -inf  -inf  -inf ]
pos1 [   0.8   0.5   -inf  -inf  -inf  -inf  -inf  -inf ]
pos2 [   0.3   1.1   0.7   -inf  -inf  -inf  -inf  -inf ]
pos3 [   0.1   0.4   0.9   0.6   -inf  -inf  -inf  -inf ]
pos4 [   0.2   0.3   0.5   1.3   0.4   -inf  -inf  -inf ]
pos5 [   0.4   0.2   0.1   0.8   0.7   0.9   -inf  -inf ]
pos6 [   0.1   0.5   0.3   0.2   0.6   1.1   0.8   -inf ]
pos7 [   0.3   0.4   0.2   0.5   0.4   0.3   0.7   0.6  ]
```

Setting future positions to -inf ensures they become 0 after softmax.

### Step 9: Softmax

Convert scores to probabilities (each row sums to 1):

```
attention_weights = softmax(scores, dim=-1)

         pos0  pos1  pos2  pos3  pos4  pos5  pos6  pos7
pos0 [   1.0   0     0     0     0     0     0     0    ]
pos1 [   0.57  0.43  0     0     0     0     0     0    ]
pos2 [   0.19  0.43  0.38  0     0     0     0     0    ]
pos3 [   0.16  0.21  0.35  0.28  0     0     0     0    ]
...
```

Row 0 can only attend to position 0 (100%). Row 1 splits attention between positions 0 and 1. And so on.

Shape: still **(4, 4, 8, 8)**

### Step 10: Weighted Sum of Values

Multiply attention weights by V to get the output:

```
output = attention_weights @ V

attention_weights: (4, 4, 8, 8)
V:                 (4, 4, 8, 8)
output:            (4, 4, 8, 8)
```

Each position's output is a weighted combination of the value vectors it attends to.

Example for position 2 (attends to positions 0, 1, 2 with weights 0.19, 0.43, 0.38):

```
output[pos2] = 0.19 * V[pos0] + 0.43 * V[pos1] + 0.38 * V[pos2]
```

### Step 11: Concatenate Heads

Reshape back to combine all heads:

```
output: (4, 4, 8, 8) → (4, 8, 32)
        (batch, heads, positions, head_dim) → (batch, positions, embed)
```

### Step 12: Output Projection

One final linear layer mixes information across heads:

```
W_O: (32, 32)

output = output @ W_O
```

Shape: **(4, 8, 32)** — same as the input to attention.

### Summary

| Step | Operation | Shape |
|------|-----------|-------|
| Input | From embeddings | (4, 8, 32) |
| Q, K, V projection | Three matmuls | (4, 8, 32) each |
| Split heads | Reshape | (4, 4, 8, 8) each |
| Attention scores | Q @ K^T | (4, 4, 8, 8) |
| Scale | Divide by sqrt(8) | (4, 4, 8, 8) |
| Causal mask | Set future to -inf | (4, 4, 8, 8) |
| Softmax | Normalize rows | (4, 4, 8, 8) |
| Weighted sum | weights @ V | (4, 4, 8, 8) |
| Concat heads | Reshape | (4, 8, 32) |
| O projection | Matmul | (4, 8, 32) |

The attention block transforms a (4, 8, 32) tensor into another (4, 8, 32) tensor, but now each position contains information gathered from the positions it attended to.

# Full Transformer Architecture

What we covered in the attention document was **one attention block**. A full transformer stacks many of these sequentially, with additional components.

### One Transformer Block

A single block contains more than just attention:

```
Input (4, 8, 32)
    ↓
LayerNorm
    ↓
Attention (what we covered in detail)
    ↓
+ Residual connection (add input back)
    ↓
LayerNorm
    ↓
Feed-Forward Network
    ↓
+ Residual connection (add input back)
    ↓
Output (4, 8, 32)
```

### LayerNorm

Normalizes each 32-dim vector independently to mean=0, std=1:

```
For each position:
  [0.5, -0.3, 1.2, ...] → normalize → [-0.1, -0.8, 1.4, ...]
```

This stabilizes training. Has 2 learnable parameters per dimension (scale gamma and shift beta), so 64 parameters total for our 32-dim model.

![LayerNorm diagram](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/c8e2b61e-65e7-4f3f-b854-88504f385c1e/image.png)

### Residual Connections

Instead of just passing data through each component, we **add the input back** to the output:

```
x = input                          # (4, 8, 32)
x = x + attention(layernorm(x))    # attention output added to x
x = x + ffn(layernorm(x))          # ffn output added to x
output = x                         # (4, 8, 32)
```

Step by step:

| Step | Operation | What happens |
|------|-----------|--------------|
| 1 | Start with x | Original input |
| 2 | layernorm(x) | Normalize x (doesn't change x itself) |
| 3 | attention(...) | Compute attention on normalized x |
| 4 | x + attention(...) | Add result back to original x |
| 5 | layernorm(x) | Normalize the new x |
| 6 | ffn(...) | Compute FFN on normalized x |
| 7 | x + ffn(...) | Add result back to x |

The "add back" part is crucial. It means the network can learn to make **small adjustments** to the input rather than completely replacing it. It also helps gradients flow during backpropagation; without residuals, deep networks are hard to train.

### Feed-Forward Network (FFN)

A simple two-layer network applied to each position independently:

```
Input: (4, 8, 32)
    ↓
Linear: (32 → 128)     # expand by 4x
    ↓
GELU activation
    ↓
Linear: (128 → 32)     # contract back
    ↓
Output: (4, 8, 32)
```

FFN parameters:
- W1: 32 × 128 = 4,096
- W2: 128 × 32 = 4,096
- Total: 8,192 parameters per block

### Stacking Blocks Sequentially

Karpathy's small model uses 6 blocks:

```
Embeddings (4, 8, 32)
         ↓
    [ Block 0 ]  ← parallel inside, sequential between blocks
         ↓
    [ Block 1 ]
         ↓
    [ Block 2 ]
         ↓
    [ Block 3 ]
         ↓
    [ Block 4 ]
         ↓
    [ Block 5 ]
         ↓
Final LayerNorm
         ↓
    Output (4, 8, 32)
```

GPT-3 has 96 blocks. Each block refines the representations, building from low-level patterns to high-level understanding.

### Final Output: Predicting Tokens

After all blocks, we need to convert (4, 8, 32) back to token probabilities:

```
Hidden states: (4, 8, 32)
    ↓
Linear layer (unembedding): (32 → 65)
    ↓
Logits: (4, 8, 65)
    ↓
Softmax (during inference)
    ↓
Probabilities: (4, 8, 65)
```

The output (4, 8, 65) means: for each of the 4 chunks, for each of the 8 positions, a probability distribution over 65 possible next tokens.

### Weight Tying (Optional)

Often the unembedding matrix is the **transpose** of the embedding matrix:

```
Embedding:   (65, 32)  — token ID → vector
Unembedding: (32, 65)  — vector → token logits

Unembedding = Embedding.T
```

This reduces parameters and works because similar tokens should have similar embeddings and similar output probabilities.

### Complete Forward Pass

```
tokens (4, 8)
    ↓
Token embedding lookup → (4, 8, 32)
    ↓
+ Position embedding → (4, 8, 32)
    ↓
Block 0: LayerNorm → Attention → +Residual → LayerNorm → FFN → +Residual
    ↓
Block 1: LayerNorm → Attention → +Residual → LayerNorm → FFN → +Residual
    ↓
... (repeat for all blocks)
    ↓
Final LayerNorm → (4, 8, 32)
    ↓
Unembedding → (4, 8, 65)
    ↓
logits for next token prediction
```

### Summary: What's Parallel vs Sequential

| Component | Parallel? |
|-----------|-----------|
| Positions within a chunk | Yes |
| Chunks within a batch | Yes |
| Attention heads | Yes |
| Blocks/layers | **No** (sequential) |

The sequential nature of blocks is the main bottleneck. Each block must wait for the previous one to finish.
