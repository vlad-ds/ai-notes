# Multimodality: How LLMs See, Hear, and Beyond

*Getting transformers to understand the world beyond text.*

---

An LLM is, at its core, a machine that processes sequences of vectors. When you type "Hello world", the tokenizer converts it to token IDs, and the embedding layer converts those IDs into vectors. The transformer then attends over this sequence of vectors, updating them through layers until the final representation emerges.

The transformer doesn't actually care where those vectors came from. It just sees positions in a sequence, each carrying a d-dimensional vector. So if we could somehow convert an image, or an audio clip, or a video, into a sequence of vectors in the same space as text embeddings—the transformer would process them just like words.

That's the entire trick. Every multimodal model, from GPT-4V to Whisper to video understanding systems, is doing some version of this:

```
Raw modality → Encoder → Sequence of vectors → Feed into LLM alongside text
```

The rest of this chapter is about how to build good encoders for different modalities.

---

## Part 1: Vision — Teaching Models to See

### The Naive Approach

What if we just treated pixels as tokens? A 224×224 RGB image has 224 × 224 × 3 = 150,528 values. We could flatten this into a sequence and feed it to the transformer.

This fails spectacularly, for two reasons. First, quadratic attention on 150K tokens would require 22.5 billion attention computations per layer—completely impractical. Second, individual pixels carry almost no semantic meaning. Knowing that pixel (47, 123) is slightly reddish tells you nothing about what's in the image. You need to see patterns across many pixels to recognize "this is an eye" or "this is a tree."

### The Vision Transformer: Patches as Tokens

The breakthrough came in 2020 with the Vision Transformer (ViT). The idea is simple: instead of treating each pixel as a token, treat each *patch* of the image as a token.

```
Original image (512×512):              Divided into 32×32 patches:

┌──────────────────────────────┐       ┌────┬────┬────┬────┐
│                              │       │ 1  │ 2  │ 3  │ 4  │← 16 patches across
│                              │       ├────┼────┼────┼────┤
│        (a photo of           │  →    │ 5  │ 6  │ .. │    │
│         something)           │       ├────┼────┼────┼────┤
│                              │       │    │    │    │    │← 16 patches down
│                              │       ├────┼────┼────┼────┤
└──────────────────────────────┘       │    │    │    │256 │
                                       └────┴────┴────┴────┘

                                       256 patches total, each 32×32 pixels
```

Let's work through this step by step with a concrete example: a 512×512 RGB image divided into 32×32 pixel patches.

**Step 1: Divide the image into patches**

We tile the image into non-overlapping squares. With a 512×512 image and 32×32 patches:
- 512 ÷ 32 = 16 patches per side
- 16 × 16 = **256 patches** total

Each patch is a small square containing 32 × 32 pixels, and each pixel has 3 color values (RGB). So each patch contains 32 × 32 × 3 = **3072 numbers**.

```
┌────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │   Each patch covers its own 32×32 region.
├────┼────┼────┼────┤   No pixel belongs to more than one patch.
│ 5  │ 6  │ 7  │ 8  │   (This is what "non-overlapping" means.)
├────┼────┼────┼────┤
│ 9  │ 10 │ 11 │ 12 │
└────┴────┴────┴────┘
        ... 256 patches total
```

**Step 2: Flatten each patch into a vector**

Take each 32×32×3 patch and flatten it into a single vector of 3072 numbers. Now we have 256 vectors, each of length 3072.

```python
# Conceptually:
patches = []
for row in range(16):
    for col in range(16):
        # Extract the 32×32×3 region
        patch = image[:, row*32:(row+1)*32, col*32:(col+1)*32]  # shape: (3, 32, 32)
        # Flatten to a vector
        flat = patch.flatten()  # shape: (3072,)
        patches.append(flat)
patches = torch.stack(patches)  # shape: (256, 3072)
```

**Step 3: Project each patch to a smaller embedding dimension**

3072 numbers per patch is unwieldy. We want each patch represented by a fixed-size embedding vector, say 1024 dimensions. This is done with a simple linear transformation: multiply the 3072-dimensional vector by a learnable weight matrix of shape (3072, 1024).

```python
# Linear projection: 3072 → 1024
patch_embed = nn.Linear(3072, 1024)

# Each patch gets projected
embedded_patches = patch_embed(patches)  # shape: (256, 1024)
```

This is **learned compression**. The weight matrix starts random, but during training it learns which combinations of pixels matter. Maybe it learns that certain edge patterns are important, or that color gradients in particular directions indicate texture. The 1024 output dimensions encode these learned features.

**Why you'll see this written as a "convolution"**

In actual ViT code, you'll often see patch embedding implemented like this:

```python
self.proj = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=32, stride=32)
```

This looks different from the `nn.Linear` approach above, but it does exactly the same thing. Here's why.

A 2D convolution is an operation that slides a small window (the "kernel") across an image. At each position, it:
1. Extracts the pixels under the window
2. Multiplies them by learned weights
3. Sums to produce one output number

Normally, convolutions slide the window one pixel at a time, producing many overlapping outputs. But we can control how far the window moves between positions with the "stride" parameter.

When kernel_size=32 and stride=32, the window is 32×32 pixels and moves 32 pixels between positions—meaning it doesn't overlap at all. It lands exactly on each patch once:

```
stride=32 means: move 32 pixels between window positions

┌────┬────┬────┬────┐
│ ■  │    │    │    │  Window at position (0,0)
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
       ↓ move right by 32 pixels
┌────┬────┬────┬────┐
│    │ ■  │    │    │  Window at position (0,1)
├────┼────┼────┼────┤
│    │    │    │    │
└────┴────┴────┴────┘
```

With `out_channels=1024`, the convolution produces 1024 output values at each window position—exactly like multiplying the flattened patch (3072 values) by a weight matrix (3072 × 1024) to get 1024 outputs.

So `nn.Conv2d(3, 1024, kernel_size=32, stride=32)` is mathematically equivalent to "extract each 32×32 patch, flatten it, multiply by a learned matrix." The convolution version is just more efficient because it's implemented as a single optimized operation instead of explicit loops.

Now we have 256 vectors instead of 786K values (512×512×3)—a 3000× reduction. And each vector represents a semantic chunk of the image rather than meaningless individual pixels.

### Building the Full Vision Transformer

We have 256 patch embeddings, each a 1024-dimensional vector. Now we need to process them through a transformer. But transformers have a problem with images: they have no built-in notion of position. When processing text, position matters ("dog bites man" ≠ "man bites dog"), so transformers add positional information. Images need this too—patch 1 (top-left) and patch 256 (bottom-right) contain different spatial information.

**Step 4: Add positional embeddings**

Positional embeddings are vectors that encode "where" each patch came from in the image. We create a learnable embedding for each position (1 through 256), and add it to the corresponding patch embedding.

```python
# Create learnable position embeddings: one 1024-dim vector per position
pos_embed = nn.Parameter(torch.randn(256, 1024))

# Add position information to each patch
# Position 0's embedding gets added to patch 0, position 1's to patch 1, etc.
embedded_patches = embedded_patches + pos_embed  # shape still (256, 1024)
```

This is element-wise addition. Each of the 256 patches gets its own positional embedding added. The model learns during training what these positional embeddings should be—they end up encoding spatial relationships like "top-left corner" or "center of image."

**Step 5: Prepend a CLS token**

Here's something that might seem strange: we add a *new* token at the beginning of the sequence that doesn't correspond to any patch. This is called the CLS (classification) token, borrowed from BERT. Understanding why it exists and how it works requires understanding how attention works in this context.

**The problem: we need a single vector for the whole image**

After the transformer processes our 256 patch tokens, each patch token will contain information about its region plus context from other patches. But for tasks like classification ("is this a cat or a dog?"), we need a single vector representing the entire image to feed to a classifier. Which of the 256 patch vectors should we use?

We could average them all, but that treats all patches equally. We could use a specific patch (like the center), but that's arbitrary. The CLS token is a more elegant solution.

**What is the CLS token?**

The CLS token is a learnable vector—just another parameter of the model, like the weights in the linear layers. It's initialized randomly and updated during training through backpropagation.

```python
# A learnable 1024-dimensional vector
cls_token = nn.Parameter(torch.randn(1, 1024))
```

We prepend it to the sequence of patch embeddings:

```python
# Before: patches has shape (256, 1024)
# After: sequence has shape (257, 1024)
sequence = torch.cat([cls_token, patches], dim=0)

# The sequence is now: [CLS, patch_1, patch_2, ..., patch_256]
# Positions:           [ 0,    1,       2,     ...,    256   ]
```

**Why position 0?** Convention, borrowed from BERT. It could be anywhere—what matters is that it's in the sequence and can attend to all other tokens.

**How does attention work here? (Bidirectional, not masked)**

This is crucial. In ViT, we use a transformer **encoder** with **bidirectional attention**. This is different from GPT-style decoders that use causal (masked) attention.

In causal attention (GPT), each token can only attend to previous tokens—token 5 can see tokens 0-4 but not 6-256. This is for autoregressive generation where you can't peek at future tokens.

In bidirectional attention (BERT, ViT), every token can attend to every other token. Token 5 can see tokens 0-4 AND tokens 6-256. There's no masking. This makes sense for images: there's no "future" in an image, all patches exist simultaneously.

```
Causal attention (GPT):              Bidirectional attention (ViT):

Token 0 sees: [0]                    Token 0 sees: [0,1,2,3,4,5...]
Token 1 sees: [0,1]                  Token 1 sees: [0,1,2,3,4,5...]
Token 2 sees: [0,1,2]                Token 2 sees: [0,1,2,3,4,5...]
Token 3 sees: [0,1,2,3]              Token 3 sees: [0,1,2,3,4,5...]
...                                  ...
(triangular mask)                    (no mask—full attention)
```

**How does the CLS token aggregate information?**

The CLS token starts as a random vector with no image information. But it's in position 0 of the sequence, and in bidirectional attention, it can attend to ALL 256 patch tokens.

In transformer attention, each token computes:
1. "What am I looking for?" (query, from the token itself)
2. "What do others have to offer?" (keys and values, from all tokens)
3. "How relevant is each other token to me?" (attention weights = softmax of query·key)
4. "What should I take from them?" (weighted sum of values)

So in each layer, the CLS token:
1. Produces a query vector based on its current state
2. Computes attention weights over all 256 patches (and itself)
3. Takes a weighted combination of all patch values
4. Updates its representation

**The information flow in ViT**

Who talks to whom:

```
In each transformer layer, ALL tokens attend to ALL tokens:

        CLS ←→ patch_1 ←→ patch_2 ←→ ... ←→ patch_256
         ↑        ↑          ↑                  ↑
         └────────┴──────────┴──────────────────┘
                   (everyone talks to everyone)
```

But at the OUTPUT, only CLS matters for classification:

```
After all 12 layers:

  [CLS']  [patch_1']  [patch_2']  ...  [patch_256']
    │         │            │               │
    │         └────────────┴───────────────┘
    │                      │
    │              (thrown away)
    ↓
classifier → prediction → loss
```

So the patches influence the output, but only **indirectly through CLS**:
- Patch 47 contains info about a cat's ear
- Patch 47 attends to neighboring patches, refining its representation
- CLS attends to patch 47 and absorbs the "cat ear" information
- CLS output goes to classifier

The patches are like a team of researchers gathering and discussing information among themselves, but only one person (CLS) presents to the board. The presenters' insights come from the team's discussions, but only the presentation affects the final decision.

Let's trace through concretely:

**Layer 1:**
- CLS token (random vector) attends to all patches
- Patches attend to each other (and to CLS, though CLS has nothing useful yet)
- CLS absorbs a blurry average of all patch information
- Neighboring patches start sharing information (patch 5 learns about patch 6)

**Layer 2:**
- CLS token (now containing some image info) attends to all patches
- But now the patches have also been updated—each patch attended to its neighbors in layer 1
- CLS is now attending to "patches that know about their neighborhoods"

**Layer 6:**
- CLS has refined its query: "I'm looking for high-level features"
- Patches have been mixing information for 5 layers—each contains broad context
- CLS attention weights become more selective: high weights on informative patches, low on background

**Layer 12:**
- CLS token has attended to all patches, directly and indirectly, many times
- Its representation now encodes global image features: "this image contains a cat, facing left, on a couch"

**What makes the CLS token special? Why does IT become the aggregator?**

The CLS token isn't inherently special. It becomes the aggregator because **we only use it for the final prediction, and we throw away all other tokens.**

Look at how classification works:

```python
# After transformer processing, we have 257 output vectors
output = transformer(sequence)  # shape: (257, 1024)

# We ONLY take the CLS token (position 0) and ignore the other 256
cls_output = output[0]  # shape: (1024,)

# Only CLS goes to the classifier
logits = classifier(cls_output)  # shape: (1000,) for 1000 classes

# Loss is computed only from this prediction
loss = cross_entropy(logits, true_label)
```

The 256 patch token outputs? **We throw them away.** They don't contribute to the loss. They don't affect the prediction.

This is why CLS becomes the aggregator: it's the only token whose output matters. During backpropagation:

1. Loss depends on classifier output
2. Classifier output depends on CLS token output
3. CLS token output depends on what CLS attended to
4. So gradients push CLS to attend to whatever helps classification

The patch tokens don't have this pressure. Their outputs aren't used, so there's no gradient signal saying "patch 47, you need to aggregate global information." They just need to be useful things for CLS to attend *to*.

**Could we use a different token?**

Yes! We could use patch token 128 (the center) instead:

```python
center_output = output[128]  # Use center patch instead of CLS
logits = classifier(center_output)
```

If we trained with this setup, patch 128 would learn to aggregate information, because now IT'S the one whose output matters. The CLS token would become useless (no gradient signal to shape it).

We could even use the average of all tokens:

```python
avg_output = output.mean(dim=0)  # Average all 257 tokens
logits = classifier(avg_output)
```

This would spread the "aggregation pressure" across all tokens.

**So why use a dedicated CLS token?**

The advantage of a dedicated CLS token is separation of concerns:
- Patch tokens can focus on representing their local image regions
- CLS token can focus purely on aggregation

If we used patch 128 for classification, that patch would have two jobs: represent its local region AND aggregate global info. These might conflict.

The CLS token has no local region—it's not tied to any patch. Its only job is aggregation, so it can specialize fully for that purpose.

**How does the CLS token learn what to aggregate?**

Through training. The loss signal flows backward from the classification task:

1. CLS token output → classifier → prediction "cat" → loss (cross-entropy with true label)
2. Gradients flow back through classifier → back through all 12 transformer layers → back to CLS token's initial value

The gradients adjust:
- The CLS token's initial value (so it starts as a better "query" for useful image features)
- The attention weights in each layer (so CLS learns to focus on informative patches)
- Everything else in the network

Over millions of images, the CLS token learns to be an effective "information aggregator." Its initial value becomes a learned query meaning something like "tell me the high-level content of this image," and the attention patterns learn to route relevant information to it.

**Code with positional embedding**

```python
# Create learnable CLS token
cls_token = nn.Parameter(torch.randn(1, 1024))

# Prepend to sequence
sequence = torch.cat([cls_token, embedded_patches], dim=0)  # (257, 1024)

# Positional embeddings: one for each position including CLS at position 0
pos_embed = nn.Parameter(torch.randn(257, 1024))
sequence = sequence + pos_embed
```

The sequence is now: `[CLS, patch_1, patch_2, ..., patch_256]` at positions `[0, 1, 2, ..., 256]`.

**Step 6: Run through transformer layers**

Now we have a sequence of 257 vectors, each 1024-dimensional. This gets fed through standard transformer encoder layers—the same architecture used in BERT or GPT, with multi-head self-attention and feed-forward networks.

```python
# Standard transformer encoder
transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=1024, nhead=16),
    num_layers=12
)

output = transformer(sequence)  # shape: (257, 1024)
```

Each layer:
1. Computes self-attention: every token attends to every other token, learning which patches are relevant to which
2. Applies a feed-forward network: two linear layers with a non-linearity, processing each token independently

After 12 layers, the CLS token (position 0) has attended to all patches multiple times and accumulated information about the entire image.

**Step 7: Extract the image representation**

The final output is 257 vectors of dimension 1024. For image classification, we typically use just the CLS token:

```python
image_representation = output[0]  # The CLS token, shape: (1024,)
```

For multimodal LLMs, we often use all 257 tokens (or just the 256 patch tokens), giving the language model access to spatially-localized information.

### The Complete Forward Pass

Putting it all together:

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size=512, patch_size=32, embed_dim=1024, num_layers=12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 256
        patch_dim = patch_size * patch_size * 3         # 3072

        # Learnable parameters
        self.patch_embed = nn.Linear(patch_dim, embed_dim)     # 3072 → 1024
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(num_patches + 1, embed_dim))

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=16, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, image):
        # image shape: (batch, 3, 512, 512)
        batch_size = image.shape[0]

        # Step 1-2: Extract and flatten patches
        # Reshape to (batch, 16, 16, 32, 32, 3), then to (batch, 256, 3072)
        patches = image.unfold(2, 32, 32).unfold(3, 32, 32)  # extract patches
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, 256, -1)

        # Step 3: Project patches to embedding dimension
        x = self.patch_embed(patches)  # (batch, 256, 1024)

        # Step 5: Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 257, 1024)

        # Step 4: Add positional embeddings
        x = x + self.pos_embed

        # Step 6: Transformer layers
        x = self.transformer(x)  # (batch, 257, 1024)

        return x  # Full sequence, or x[:, 0] for just CLS token
```

### How ViT is Trained

The Vision Transformer is typically trained for image classification. The setup:

1. **Dataset**: Images with labels (e.g., ImageNet with 1000 classes: "cat", "dog", "car", etc.)
2. **Task**: Given an image, predict which class it belongs to
3. **Output head**: A linear layer that maps the CLS token (1024-dim) to class logits (1000-dim for ImageNet)

```python
class ViTForClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer()
        self.classifier = nn.Linear(1024, 1000)  # 1000 ImageNet classes

    def forward(self, image):
        features = self.vit(image)       # (batch, 257, 1024)
        cls_output = features[:, 0]      # (batch, 1024) — just the CLS token
        logits = self.classifier(cls_output)  # (batch, 1000)
        return logits
```

**The loss function** is standard cross-entropy:

```python
# Training step
logits = model(image)                    # (batch, 1000)
loss = F.cross_entropy(logits, labels)   # labels is (batch,) of integers 0-999
loss.backward()
optimizer.step()
```

Cross-entropy loss pushes the model to assign high probability to the correct class and low probability to incorrect classes. The gradients flow back through the classifier, through the transformer layers, through the patch embedding, all the way to the learned weights. Over millions of images, the model learns:
- What patch features matter (the patch embedding weights)
- Where patches are spatially (the positional embeddings)
- How patches relate to each other (the transformer attention weights)
- How to aggregate information into the CLS token
- What visual patterns correspond to which classes (the classifier weights)

### Connecting Vision to Language

Here's where multimodality actually happens. We have:

- Vision encoder output: 257 vectors of dimension 1024
- LLM embedding space: vectors of dimension 4096 (for a typical 7B model)

What do those 257 vectors represent?

- **Vector 0 (CLS)**: The whole image, aggregated. After 12 layers of attending to all patches, this vector encodes global information like "a cat sitting on a blue couch."
- **Vectors 1-256 (patches)**: Each represents a specific spatial region of the image, enriched by context from other patches. Vector 47 might encode "cat's ear, top-left area, with fur texture continuing into neighboring regions." These vectors know about their local content AND have context from what's nearby (through attention in earlier layers).

For classification, we only used the CLS vector. But for multimodal LLMs, we often pass **all 257 vectors** to the language model. This gives the LLM access to spatially-localized information—it can "look at" specific regions when answering questions like "what color is the object in the top-right corner?"

These don't match the LLM's embedding dimension. We need a projection layer that maps vision space into language space. The simplest version is just a linear layer:

```python
vision_proj = nn.Linear(1024, 4096)
```

More sophisticated versions use an MLP or even cross-attention. But the principle is the same: learn a mapping that translates "what the vision encoder thinks this patch means" into "something the LLM can understand."

Once projected, the vision tokens are concatenated with the text tokens:

```
Sequence fed to LLM:

[img_1] [img_2] ... [img_257] [What] [is] [in] [this] [image] [?]
   ↑                              ↑
   └─ projected vision tokens     └─ regular text embeddings
```

The LLM processes this combined sequence with its normal attention mechanism. It can attend to image tokens when predicting text, effectively "looking at" different parts of the image to form its response.

### LLaVA's Training Recipe

LLaVA (Large Language and Vision Assistant) provides a clean example of how to train a multimodal model. You don't need to train everything from scratch. You can take an already-trained vision encoder, an already-trained LLM, and just learn to connect them.

LLaVA uses:
- **Vision encoder**: A pretrained ViT from CLIP (we'll cover CLIP later—it's a vision model trained to understand images in relation to text)
- **Language model**: Vicuna (a fine-tuned version of LLaMA)
- **Projection layer**: A simple linear layer or MLP connecting them

The training happens in two stages.

**Stage 1: Feature Alignment**

In this stage, we "freeze" the vision encoder and the LLM—meaning we don't update their weights during training. We only train the projection layer. (Freezing is just setting `requires_grad=False` on those parameters, so gradients don't flow through them.)

The training data is image-caption pairs: an image and a sentence describing it ("a cat sitting on a couch").

The training process:
1. Pass the image through the frozen vision encoder → get 257 vectors
2. Pass those vectors through the projection layer (which we ARE training) → get 257 vectors in LLM space
3. Feed those vectors to the frozen LLM, followed by a start token
4. The LLM predicts the next token, then the next, then the next... generating the caption
5. Compute cross-entropy loss between predicted tokens and actual caption tokens
6. Backpropagate—but gradients only update the projection layer (since everything else is frozen)

```python
# Simplified training loop for stage 1
for image, caption in dataset:
    # Forward pass
    vision_features = frozen_vit(image)          # no gradients here
    projected = projection_layer(vision_features) # gradients flow here
    logits = frozen_llm(projected, caption_tokens) # no gradients in LLM

    # Loss: how well did we predict the caption?
    loss = cross_entropy(logits, caption_tokens)

    # Backward pass: only projection_layer weights update
    loss.backward()
    optimizer.step()
```

After this stage, the projection layer has learned to produce vectors that the LLM interprets as meaningful image content. When the LLM sees these vectors, it can "read" them as if they described the image.

**Stage 2: Instruction Tuning**

Now we unfreeze the LLM (or use LoRA, a technique that adds small trainable adapters without modifying the original weights). The vision encoder stays frozen.

The training data changes: instead of just image-caption pairs, we use instruction-following conversations:

```
User: <image> What's in this image?
Assistant: This image shows a cat sitting on a blue couch...

User: <image> How many people are in this photo?
Assistant: There are three people in the photo...
```

The model learns to follow instructions that involve visual reasoning. It's the same training process (predict next token, compute loss, backprop), but now the LLM weights also update, so it learns new capabilities specific to visual tasks.

After both stages, the model can take an image and a question, and generate a relevant answer—it has become a vision-language model.

---

## Part 2: Audio — How Whisper Works

Audio presents different challenges than images. It's inherently temporal—a 30-second clip has 30 seconds of sequential information. It's dense—standard audio is 16,000 samples per second. And it's variable length—unlike images which we can resize to 224×224, audio clips range from milliseconds to hours.

### From Waveform to Mel Spectrogram

Raw audio is just amplitude values over time. If you plot it, you see a wiggly line:

```
Raw waveform (amplitude vs time):

     ↑
     │    ╱╲      ╱╲
     │   ╱  ╲    ╱  ╲    ╱╲
  0  │──╱────╲──╱────╲──╱──╲──────
     │        ╲╱      ╲╱    ╲╱
     │
     └────────────────────────────→ time
```

This representation isn't great for neural networks. A slight time shift changes all the values even though the sound is perceptually identical. And there's no clear structure—just a sequence of numbers.

The mel spectrogram transforms audio into a time-frequency representation. Think of it as a "picture" of the sound:

```
Mel spectrogram:

Frequency ↑  ┌─────────────────────────────────┐
(mel scale)  │░░▓▓░░░░░▓▓▓░░░░░░▓░░░░░░░░░░░░░│← high frequencies
             │░▓▓▓░░░░▓▓▓▓▓░░░░▓▓▓░░░░░░░░░░░░│
             │▓▓▓▓▓░░▓▓▓▓▓▓▓░░▓▓▓▓▓░░░░░░░░░░░│
             │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░│← low frequencies
             └─────────────────────────────────┘
                    time →

             Darker = more energy at that frequency/time
```

Each column is a "snapshot" of which frequencies are present at that moment. The y-axis uses a mel scale, which compresses frequencies in a way that matches human perception (we're more sensitive to differences in low frequencies than high frequencies).

The conversion involves:
1. Split audio into overlapping windows (typically 25ms windows, 10ms apart)
2. Compute FFT on each window to get frequency content
3. Apply mel filterbank to compress to 80 frequency bins
4. Take log to compress dynamic range

For 30 seconds of audio at 16kHz, with 10ms hop length, we get 3000 time steps × 80 frequency bins.

### Whisper's Architecture

Whisper is an encoder-decoder transformer, like the original "Attention Is All You Need" architecture:

```
                    ┌──────────────┐
Mel spectrogram →   │   Encoder    │ → Hidden states (1500 × dim)
(80 × 3000)         │ (transformer)│        │
                    └──────────────┘        │ cross-attention
                                            ↓
                    ┌──────────────┐   ┌──────────────┐
                    │   Decoder    │ → │  Text tokens │
                    │ (transformer)│   │  (output)    │
                    └──────────────┘   └──────────────┘
                           ↑
                    Previous tokens
                    (autoregressive)
```

The encoder processes the mel spectrogram. It starts with two convolutional layers that halve the time dimension (3000 → 1500 time steps), then runs transformer encoder layers. The output is 1500 vectors, each representing roughly 20ms of audio.

The decoder generates text autoregressively, attending to both its own previous outputs (causal self-attention) and the encoder outputs (cross-attention). This is how it "reads" the audio while generating the transcription.

### What Makes Whisper Special

Three things make Whisper exceptional:

**Massive weakly-supervised training.** OpenAI trained Whisper on 680,000 hours of audio with transcriptions scraped from the internet—YouTube subtitles, podcast transcripts, etc. This data is noisy (subtitles often have errors), but the scale overcomes the noise. The model learns to ignore labeling mistakes because they're inconsistent, while correct patterns are reinforced.

**Multitask training with special tokens.** Whisper learns multiple tasks through prompt tokens:

```
English transcription:
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> The quick brown fox...

Spanish → English translation:
<|startoftranscript|><|es|><|translate|><|notimestamps|> The quick brown fox...

Transcription with timestamps:
<|startoftranscript|><|en|><|transcribe|><|0.00|> The <|2.50|> quick <|3.10|> brown...
```

The decoder learns to look at these special tokens to determine what task it's performing. Same model, different behavior based on the prompt.

**Fixed 30-second chunks.** Whisper always processes exactly 30 seconds of audio. Shorter clips are padded with silence; longer recordings are processed in 30-second windows. This simplifies the architecture (fixed-size inputs) at the cost of some context loss at chunk boundaries.

### Using Whisper

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large
result = model.transcribe("audio.mp3")
print(result["text"])
```

That's it. The model handles language detection, transcription, and even translation (any language → English) automatically.

---

## Part 3: Connecting Audio to LLMs

You've probably noticed the pattern by now. Whisper's encoder produces a sequence of vectors (1500 × 512 for the base model). To connect this to an LLM, we just need a projection layer:

```
Whisper encoder output     Projection       LLM embedding space
   (1500 × 512)        →   (linear)     →     (1500 × 4096)
```

Then concatenate with text embeddings, exactly like we did for images:

```
[audio_1] [audio_2] ... [audio_1500] [What] [did] [they] [say] [?]
```

Models like Qwen-Audio and SALMONN follow this pattern. The main question is whether to keep the Whisper encoder frozen or fine-tune it alongside the projection layer.

### GPT-4o: Native Multimodality

GPT-4o takes a fundamentally different approach. Instead of bolting specialized encoders onto an LLM, it was trained from scratch on all modalities simultaneously.

The difference in architecture:

```
Traditional (GPT-4V, LLaVA, etc.):

Image → [CLIP ViT] → [projection] → ┐
                                    ├─→ [LLM] → text
Audio → [Whisper]  → [projection] → ┘
Text  → [tokenizer] ────────────────┘


Native multimodality (GPT-4o):

Image → [learned tokenizer] → ┐
                              │
Audio → [learned tokenizer] → ├─→ [Single unified model] → any modality
                              │
Text  → [learned tokenizer] → ┘
```

In the traditional approach, the LLM was pretrained on text, and vision/audio understanding is "bolted on" through projection layers. The modalities live in somewhat separate spaces.

In GPT-4o, all modalities are processed by the same weights from the beginning. The model learns unified representations where images, audio, and text naturally interrelate. This is why GPT-4o has such low latency for voice—it's not running a pipeline (ASR → LLM → TTS), it's processing audio directly and generating audio directly.

The downside: native multimodality requires vastly more compute to train. You're learning vision, audio, and language understanding all at once, rather than leveraging pretrained specialists.

---

## Part 4: Video — Time Makes Everything Harder

Video is images plus time. A 10-second clip at 30fps has 300 frames. If each frame becomes 256 tokens (like ViT), that's 76,800 tokens per video—far too many for practical attention.

The core challenge is how to capture temporal information without exploding the token count.

### Temporal Sampling

The simplest approach: don't use every frame. Sample 8-16 frames uniformly across the video:

```
Original video: 300 frames
                ▼
Sample uniformly: frames 0, 37, 75, 112, 150, 187, 225, 262
                ▼
Result: 8 frames × 256 patches = 2,048 tokens ✓
```

This works surprisingly well for many tasks. Most videos have redundancy—consecutive frames are nearly identical. Sampling every ~40 frames (at 30fps, that's ~1.3 seconds between samples) often captures the essential information.

### Temporal Modeling

Sampling reduces tokens, but loses fine-grained temporal information. What if something important happens between sampled frames? Or what if the order of events matters?

Video-specific architectures like TimeSformer handle this with "divided space-time attention":

```
Standard attention: every token attends to every other token
                   O(N²) where N = frames × patches

Divided attention:
  1. Spatial attention: patches within each frame attend to each other
     O(F × P²) where F = frames, P = patches per frame

  2. Temporal attention: same patch position across frames attend to each other
     O(P × F²)

Total: O(F × P² + P × F²) << O((F × P)²)
```

In practice: first let each frame understand itself (spatial), then let the model track how things change over time (temporal). This factorization makes video attention tractable.

---

## Part 5: The Universal Pattern

Every modality follows the same recipe:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Modality         Encoder              Projection         LLM          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Image     →    ViT (patches)      →    Linear/MLP    →               │
│  (H×W×3)        (256 × 1024)            (256 × 4096)      │           │
│                                                            │           │
│  Audio     →    Whisper encoder    →    Linear/MLP    →   │ Unified   │
│  (mel spec)     (1500 × 512)            (1500 × 4096)     │ Transformer│
│                                                            │           │
│  Video     →    ViT + temporal     →    Linear/MLP    →   │           │
│  (T×H×W×3)      (T×256 × 1024)          (T×256 × 4096)    │           │
│                                                            │           │
│  Text      →    Tokenizer          →    (identity)    →               │
│  (string)       (S × 4096)              (S × 4096)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The LLM sees everything as tokens in the same embedding space. It doesn't know which tokens came from images versus audio versus text—they're all just positions in the sequence to attend over.

The design decisions are:

1. **Encoder choice:** Pretrained (CLIP, Whisper) vs trained from scratch
2. **Projection complexity:** Linear vs MLP vs cross-attention (Q-Former)
3. **Frozen vs fine-tuned:** Freeze encoders for speed, fine-tune for alignment
4. **Training strategy:** Two-stage (align then instruct-tune) vs joint

---

## Part 6: CLIP — The Foundation

Most vision-language models use CLIP's vision encoder. Understanding CLIP helps explain why.

CLIP (Contrastive Language-Image Pre-training) learns to align images and text in a shared embedding space. Given a batch of (image, caption) pairs, it learns to:

1. Pull matching pairs together
2. Push non-matching pairs apart

```
Batch of images:              Batch of captions:
[img_1] → embed → v_1         [cap_1] → embed → t_1    "a photo of a cat"
[img_2] → embed → v_2         [cap_2] → embed → t_2    "sunset over ocean"
[img_3] → embed → v_3         [cap_3] → embed → t_3    "person riding bike"
...                           ...

Similarity matrix (dot products):
                 cap_1   cap_2   cap_3   cap_4
         img_1  [ 0.9    0.1     0.05    0.02 ]  ← want diagonal high
         img_2  [ 0.1    0.85    0.1     0.05 ]
         img_3  [ 0.05   0.1     0.88    0.03 ]
         img_4  [ 0.02   0.05    0.03    0.92 ]

Loss: cross-entropy pushing diagonal toward 1, off-diagonal toward 0
```

After training on 400 million image-text pairs, CLIP learns rich visual representations. The image embedding for a cat photo is close to the text embedding for "a photo of a cat", but far from "a photo of a dog." This works for arbitrary concepts, not just predefined categories.

Why this matters for multimodal LLMs: CLIP's vision encoder already "speaks language" to some extent. Its representations are organized by semantic meaning (cat photos cluster together, not just visually similar images). This gives multimodal LLMs a head start—the projection layer doesn't have to learn visual semantics from scratch, just translate CLIP's already-semantic representations into the LLM's space.

---

## Part 7: Why Open Source Lags on Audio and Video

You mentioned that open-source models are slower on audio and video. There are two main reasons.

**Compute cost.** Text pretraining processes sequences of ~2048 tokens. Adding vision adds ~200 tokens per image (10% overhead). Adding audio adds ~1500 tokens per 30-second clip (75% overhead). Video is worse. And attention is quadratic in sequence length, so longer sequences are disproportionately expensive.

**Data availability.** ImageNet has 14 million labeled images. LAION has 5 billion image-text pairs. These datasets enabled CLIP, which enabled all downstream vision-language models.

High-quality audio and video datasets are much scarcer. YouTube has lots of video, but getting accurate transcripts or descriptions is hard. Most subtitles have errors. Descriptions are often missing or low-quality. Whisper overcame this with brute scale (680K hours), but training Whisper required OpenAI-scale resources.

Until there's a "CLIP for audio" that the community can build on, open-source audio-language models will lag behind. Video is even further back—we don't yet have a dominant pretrained video encoder that everyone can use as a foundation.

---

## Summary

The core idea of multimodality is simple: convert every modality into sequences of vectors that live in the LLM's embedding space. The transformer doesn't care where vectors come from—it just attends over positions.

For vision, ViT converts images to patch sequences. For audio, we first transform to mel spectrograms, then encode with transformers (Whisper). For video, we sample frames and add temporal modeling.

The practical recipe is: take a good pretrained encoder (CLIP for vision, Whisper for audio), add a projection layer, freeze the encoder initially, and train the projection to align with the LLM. Then optionally fine-tune everything together for better integration.

Native multimodality (GPT-4o style) trains all modalities jointly from scratch—more elegant, but requires immense compute. The encoder-projection approach lets you bootstrap from existing specialists.

The open-source gap exists because audio and video are more expensive (longer sequences) and lack the large-scale pretrained encoders that vision has (CLIP). This will likely close as more compute becomes available and better datasets emerge.
