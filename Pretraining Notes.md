[Karpathy Lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3382s).

## Architecture

GPT is a decoder-only transformer.

(The terms encoder/decoder come from the [original transformer paper](https://arxiv.org/abs/1706.03762). The transformer was introduced for translation tasks. The encoder module had bidirectional attention on the sentence to be translated; the decoder model had masked self-attention on the generated translation, plus cross-attention with the encoder. OpenAI kept only the decoder module with masked self-attention for autoregressive language generation.)

![GPT architecture](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/e0f13899-970c-4a4c-9b2e-011ba5c7be3d/image.png)

*GPT architecture ([Wikipedia](https://www.wikiwand.com/en/articles/Generative_pre-trained_transformer)).*

## (Hyper)parameters

- `L` (layers). Number of transformer blocks stacked on top of each other.
- `d_model` (embedding dimension). Size of the vector that represents each token.
- `n_heads`. Parallel attention heads within a single layer.
- `d_head`. Size of the vector processed by each head. Typically it's `d_model` / `n_heads`.
- `T` (context window). The max. sequence that the model can process at once.
- `B` (batch size). How many independent sequences you can process in a forward / backward pass.

![GPT model sizes](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/4bf12231-e773-4083-96e7-3688e2066783/image.png)

A quick formula to approximate the parameter count is `12Ld²` (12 * Layers * embedding_dimension_squared). [See this explainer](https://claude.ai/public/artifacts/9f6731c9-fb1a-4003-ad30-f3237d858552).

How are the model's parameters distributed across the various sections? In the original GPT-3, ~2/3 of parameters are in the feed-forward networks; 1/3 of parameters are doing self-attention. The other parts use a negligible number of parameters.

![Parameter distribution](https://prod-files-secure.s3.us-west-2.amazonaws.com/1909c126-f8a8-40ce-ba59-10d834889388/ef3e731b-2c80-4335-bb60-d43af0b48191/image.png)

Karpathy's [$1,000 nanochat run](https://github.com/karpathy/nanochat/blob/master/runs/run1000.sh) had the following params:
- Tokenizer vocab size is 65,536 trained on 4B tokens
- 32 layers
- 2048 model dimension
- 16 heads
- Batch size 524,288
- Parameters 1,879,048,192
- Estimated FLOPs per token: 1.207960e+10
- Total number of training tokens: 37,580,963,840
- Tokens : Params ratio: 20.00 (Cinchilla optimal)
- Total training FLOPs estimate: 4.539628e+20

## Preparing the inputs

You have a text dataset and a trained tokenizer. You can use scaling laws (like Cinchilla) to determine how many tokens you should train your model on.

Large LLMs typically train for **~1 epoch** on their data. Sometimes less (if data is abundant), rarely more than 2-4 (diminishing returns, overfitting risk). Meaning, they try to read each document only once. That being said, the [FineWeb blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) shows that it's not possible to fully prevent duplication in the dataset, so several documents will be seen multiple times.

A concrete walkthrough from raw text to training batches.

### Our Setup

```
Context length: 8 tokens
Batch size: 2 sequences
Special tokens:
  <|end|>  → marks document boundaries (ID: 50256)
  <|pad|>  → fills incomplete chunks (ID: 50257)
```

### Step 1: Start with Raw Documents

```
Document A: "The cat sat on the mat."
Document B: "Hello world."
Document C: "Machine learning is a subset of artificial intelligence."
Document D: "It rained yesterday."
```

### Step 2: Tokenize Each Document

Each document becomes a list of token IDs. I'll show both words and IDs:

```
Doc A: ["The", "cat", "sat", "on", "the", "mat", "."]
       → [464, 3797, 3332, 319, 262, 2603, 13]

Doc B: ["Hello", "world", "."]
       → [15496, 995, 13]

Doc C: ["Machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence", "."]
       → [22203, 4673, 318, 257, 850, 286, 11666, 4430, 13]

Doc D: ["It", "rained", "yesterday", "."]
       → [1026, 26163, 7415, 13]
```

### Step 3: Concatenate with End Tokens

Join all documents into one stream, with `<|end|>` between them:

```
The cat sat on the mat . <|end|> Hello world . <|end|> Machine learning is a subset of artificial intelligence . <|end|> It rained yesterday . <|end|>
```

Total: 28 tokens

### Step 4: Slice into Fixed-Length Chunks

Chunk the stream into sequences of 8 tokens:

```
Chunk 0: [The, cat, sat, on, the, mat, ., <|end|>]
Chunk 1: [Hello, world, ., <|end|>, Machine, learning, is, a]
Chunk 2: [subset, of, artificial, intelligence, ., <|end|>, It, rained]
...
```

Notice:
- Chunks often contain parts of multiple documents
- The `<|end|>` token marks where one document ends and another begins

### Step 5: Create Training Examples

The model sees a sequence and predicts the next token at each position:

| Step | Input Sequence | Predict |
|------|----------------|---------|
| 0 | The | → cat |
| 1 | The cat | → sat |
| 2 | The cat sat | → on |
| 3 | The cat sat on | → the |
| 4 | The cat sat on the | → mat |
| 5 | The cat sat on the mat | → . |
| 6 | The cat sat on the mat . | → `<|end|>` |
| 7 | The cat sat on the mat . `<|end|>` | → Hello |

**How many training examples per chunk?**

A context_length of N gives N predictions. To create them, we actually read N+1 consecutive tokens from the stream:
- `inputs = tokens[0:8]` → The cat sat on the mat . `<|end|>`
- `targets = tokens[1:9]` → cat sat on the mat . `<|end|>` Hello

All positions are computed in parallel using causal masking, and the loss is summed across all 8 predictions.

Notice:
- Position 6: learns to end documents with `<|end|>`
- Position 7: learns to start fresh after `<|end|>`

### Step 6: Shuffle and Batch

**The goal: see all data exactly once.**

Unlike classical ML where you might run 100 epochs, large LLMs typically do a single pass through the data. Every chunk gets seen exactly once. No repetition, no skipping.

With a real dataset you'd have millions of chunks. Shuffle them:

```
Before shuffle: [Chunk 0, Chunk 1, Chunk 2, Chunk 3, ...]
After shuffle:  [Chunk 847, Chunk 12, Chunk 1893, Chunk 4, ...]
```

**Why shuffle if we're seeing everything once anyway?**

Without shuffling, consecutive batches would contain consecutive chunks from the same documents and topics. This creates two problems:
1. Gradient updates become correlated (bad for optimization)
2. The model might overfit to local patterns rather than learning general language

Shuffling ensures each batch contains a diverse mix of content, while still guaranteeing every chunk is seen exactly once.

Group into batches of 2:

```
Batch 0: [Chunk 847, Chunk 12]
  inputs shape:  (2, 8)
  targets shape: (2, 8)
Batch 1: [Chunk 1893, Chunk 4]
...
```

### Step 7: Training Loop

```python
for epoch in range(1):  # Usually just 1 epoch
    for inputs, targets in batches:
        # inputs shape:  (2, 8)  - batch of 2 chunks, 8 tokens each
        # targets shape: (2, 8)  - same, but shifted by 1 position

        logits = model(inputs)   # → (2, 8, vocab_size)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
```

### What the Special Tokens Do

**`<|end|>` (end of document)**

When the model sees:

```
... mat . <|end|> Hello world ...
```

The model *can* see all previous tokens (standard causal masking). But through training, it *learns* that:
1. After "." in certain contexts, `<|end|>` often comes next
2. After `<|end|>`, a new document begins, so tokens before it aren't useful for prediction

The model learns this naturally from the data; no special masking is needed.

**`<|pad|>` (padding)**

Used to fill incomplete chunks to the required length. During training:
1. Padding tokens are masked out of the loss calculation (we don't penalize the model for padding predictions)
2. In practice, large training runs avoid padding entirely by always having enough documents to fill chunks completely

### Realistic Numbers

For a 7B model training run:

```
Dataset: 2 trillion tokens
Context length: 4,096 tokens
Total chunks: 2T / 4096 ≈ 488 million chunks

Batch size: 2,048 sequences
Tokens per batch: 2,048 × 4,096 = 8.4 million tokens
Total training steps: 488M / 2048 ≈ 238,000 steps

Epochs: 1 (each chunk seen exactly once)
```

One training step processes 8.4 million tokens. After 238,000 steps, you've seen all 2 trillion.

# The Counterintuitive Simplicity of LLM Training

One of the most surprising things about how LLMs learn is how *unlike* human learning it is.

### No Full Documents

A model trained with a 4096-token context never sees a complete book, paper, or even most articles. A novel gets sliced into hundreds of disconnected chunks. The chunk containing Chapter 1's opening might be in batch 50,000. The chunk with Chapter 1's conclusion might be in batch 7,000,000. The model never experiences them together.

### No Curriculum

Humans learn progressively: letters before words, words before sentences, simple books before complex ones. We build scaffolding.

LLMs get everything at once, shuffled randomly. A chunk of a children's book sits next to a chunk of a physics paper sits next to a chunk of Reddit comments. There's no "start with the basics." The first batch might contain graduate-level mathematics.

And yet it learns the basics anyway.

### Chunks Don't Respect Boundaries

A single chunk might contain:
- The last paragraph of a news article
- An `<|end|>` token
- The first two paragraphs of an unrelated blog post

The model learns from this fragment that starts mid-thought and ends mid-thought. It never sees how that news article began or how that blog post concluded.

### Why Does This Work?

A few possible reasons:

**Redundancy.** Language is incredibly redundant. The patterns that make up "good writing" or "logical reasoning" appear over and over across millions of documents. Even if any single chunk is incomplete, the statistical patterns emerge from sheer volume.

**Local structure.** Most of what makes language work is local. Subject-verb agreement, word choice, sentence flow; these don't require seeing a full document. A chunk is enough.

**Scale.** With trillions of tokens, the model sees enough examples of everything: story beginnings, story endings, argument structures, code patterns. No single document matters; the aggregate does.

### The Humbling Part

We spent decades building complex curricula, careful lesson plans, structured learning progressions. Then it turned out that "concatenate everything, shuffle, predict next token" works remarkably well.

It suggests that either:
1. Language has more learnable local structure than we thought
2. Scale compensates for almost everything
3. Our intuitions about learning are wrong

Probably all three.

# Parallelism in LLM Training

Training LLMs is massively parallel at multiple levels. Here's how.

### Level 1: Within a Chunk (Sequence Parallelism)

A chunk of 8 tokens gives 8 predictions. These are computed **in parallel**, not sequentially.

```
Input:   [The, cat, sat, on, the, mat, ., <|end|>]
          ↓    ↓    ↓   ↓    ↓    ↓     ↓       ↓
Predict: [cat, sat, on, the, mat,  ., <|end|>, Hello]
```

All 8 predictions happen simultaneously in a single forward pass. The causal mask ensures each position only "sees" earlier tokens, but the computation itself is parallel.

This is why GPUs are so effective: matrix multiplications process all positions at once.

### Level 2: Across Chunks (Batch Parallelism)

A batch contains multiple chunks, all processed together:

```
Batch of 2048 chunks, each 4096 tokens:
  Chunk 0:    [tokens 0-4095]    → 4096 predictions
  Chunk 1:    [tokens 0-4095]    → 4096 predictions
  ...
  Chunk 2047: [tokens 0-4095]    → 4096 predictions

Total: 2048 × 4096 = 8.4 million predictions per batch
```

Each chunk is independent; they don't interact. This makes batching embarrassingly parallel. Larger batches = better GPU utilization.

### Level 3: Within Attention (Head Parallelism)

Multi-head attention splits work across heads:

```
GPT-3: 96 heads, each working in 128 dimensions

Input (12288 dims)
    ↓
Split into 96 heads
    ↓
  Head 0:  128 dims → attention → 128 dims
  Head 1:  128 dims → attention → 128 dims
  ...
  Head 95: 128 dims → attention → 128 dims
    ↓
Concatenate (96 × 128 = 12288 dims)
    ↓
O projection (12288 → 12288)
    ↓
Output (12288 dims)
```

All 96 heads compute independently and in parallel. Each head can learn to attend to different patterns (syntax, semantics, position, etc.).

### Summary

| Level | What's Parallel | Scale (GPT-3) |
|-------|-----------------|---------------|
| Sequence | Positions within a chunk | 4,096 positions |
| Batch | Chunks within a batch | 2,048 chunks |
| Heads | Attention heads per layer | 96 heads |

**What happens in one training step:**
- 2,048 chunks × 4,096 predictions each = **8.4 million next-token predictions**
- Each prediction produces a loss value (cross-entropy against the true next token)
- These 8.4 million losses are averaged into a single scalar
- One backward pass computes gradients
- One optimizer step updates weights

So each step processes 8.4 million training examples, but the model only updates once.

### What About Layers?

Layers are **not** parallel; they're sequential. Layer 2's input depends on Layer 1's output. This is actually the main bottleneck in transformer training and why techniques like pipeline parallelism exist for distributed training.

## What a Transformer Learns: A Bird's Eye View

A discursive summary of what gets learned during training.

### Token Embeddings

The embedding layer learns what each token means **in isolation**, before seeing any context. It's a general representation: "cat" gets a 32-dimensional vector that captures something about cat-ness, regardless of what comes before or after it.

### Positional Embeddings

The positional embedding learns what it means to be at a certain position in a sequence. Position 0 might learn "I'm the start of something." Position 7 might learn "I'm near the end of this context window" (if the context window is 8). This gets added to the token embedding, so now each vector encodes both *what* and *where*.

### The Attention Block

The hidden state has a direct path to the FFN via the residual connection. But before it gets there, attention adds its contribution.

Inside attention, the model learns to project each hidden state into three different vectors:
- **Query**: what is this token looking for?
- **Key**: what does this token offer for matching?
- **Value**: what content does this token contribute if selected?

Each token uses its own query and compares it against the keys of all previous tokens. This produces attention weights: how much should I listen to each previous position? Then it aggregates the values from those positions using these weights.

This happens in parallel across all attention heads. Each head can learn to look for different things: one might focus on syntax, another on semantic similarity, another on recent positions.

After all heads compute in parallel, the O-projection learns how to combine information from different heads into a single vector.

### The Residual Connection

The original hidden state gets the attention output **added** to it. This means attention makes adjustments to the representation rather than replacing it entirely.

### The Feed-Forward Network

Now the hidden state (shifted by attention) enters the FFN. Here, with its 4x expansion, the model has room to:
- Digest the information gathered from attention
- Store patterns and memories in the weights
- Do computation and transformation
- Prepare the representation for the next layer

The FFN is where most of the parameters live; it's the "thinking" capacity of each layer.

### Repeat for All Blocks

One attention block is: attention → residual → FFN → residual.

This repeats for every block (6 in Karpathy's model, 96 in GPT-3). Each block refines the representation further. Early blocks might capture syntax and local patterns. Later blocks might capture abstract relationships and long-range dependencies.

### Hierarchical Learning Through Depth

The sequential nature of blocks enables hierarchical learning. Early blocks work with raw token representations and tend to learn low-level patterns: syntax, punctuation, local word relationships. Later blocks receive increasingly processed representations and can learn more abstract concepts: sentiment, argument structure, factual relationships, reasoning patterns.

This is similar to how convolutional neural networks learn: early layers detect edges, middle layers detect shapes, late layers detect objects. In transformers, the hierarchy is less visual but still present: early layers might figure out that "the cat sat" is a grammatical unit; later layers might understand that this sentence is describing a scene.

Each block builds on top of what previous blocks computed. Block 50 doesn't see raw tokens; it sees tokens that have already been contextualized and transformed by 49 layers of attention and processing. This depth is what allows transformers to represent complex, abstract patterns that simple shallow networks cannot.

### The Final Prediction

After all blocks, one last linear layer (the unembedding) takes everything the model has computed and produces logits over the vocabulary.

For each position, this layer asks: "Given everything I've learned through 6 (or 96) layers of attention and processing, what token should come next?"

The output is a probability distribution over all possible tokens. During training, we compare this to the actual next token and compute the loss.

### What Gets Learned

Every component has learnable parameters that update during training:

| Component | What it learns |
|-----------|----------------|
| Token embeddings | General meaning of each token |
| Position embeddings | What each position signifies |
| Q, K, V projections | How to create queries, keys, values |
| O projection | How to combine attention heads |
| FFN weights | How to process and transform information |
| Unembedding | How to convert hidden states to token predictions |

All of this updates simultaneously from the same loss signal: "predict the next token."

## Operations and Shapes View

This chart shows all the operations that happen inside the transformer, the dimensions of the tensors, and the parameter counts for GPT-3.

[transformer_shapes.jsx](https://claude.ai/public/artifacts/2f528a9f-67f6-4dc2-b43f-19ed46d1e0c5)
