# Multimodality: How LLMs See, Hear, and Beyond

## The Core Insight

**Goal**: Get an LLM (which only understands sequences of vectors) to process images, audio, video, and any other modality.

**The fundamental principle**: Every modality must be converted into the same format the LLM already understands—a sequence of embedding vectors. That's it. The entire field of multimodal AI is about finding good ways to do this conversion.

```
Image  → [encoder] → sequence of vectors → LLM
Audio  → [encoder] → sequence of vectors → LLM
Video  → [encoder] → sequence of vectors → LLM
Text   → [tokenizer + embedding] → sequence of vectors → LLM
```

The LLM doesn't know or care where the vectors came from. To the transformer, they're all just positions in a sequence to attend to.

---

## Part 1: Vision — Teaching Models to See

### The Simplest Approach That Could Work

What if we just... flattened the image into pixels and fed them as tokens?

```python
# Naive approach: each pixel is a token
image = load_image("cat.jpg")  # shape: (224, 224, 3)
pixels = image.flatten()        # shape: (150528,)
# Feed 150,528 tokens to the LLM...
```

**Why this breaks**:
- 150K tokens for one small image (quadratic attention = dead)
- Individual pixels carry almost no semantic meaning
- No spatial structure preserved

### The Solution: Vision Transformer (ViT)

The breakthrough idea (2020): treat an image as a sequence of patches, not pixels.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into a sequence of patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 196 patches for 224x224

        # Single conv layer does patch extraction + linear projection in one step
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.proj(x)           # (batch, 768, 14, 14)
        x = x.flatten(2)           # (batch, 768, 196)
        x = x.transpose(1, 2)      # (batch, 196, 768)
        return x

# Now we have 196 tokens instead of 150K
patch_embed = PatchEmbedding()
image = torch.randn(1, 3, 224, 224)
patches = patch_embed(image)
print(patches.shape)  # (1, 196, 768) — manageable!
```

**What each patch captures**:
- A 16x16 region of the image
- Each patch token represents a semantic chunk (part of an eye, edge of a table, etc.)
- 196 tokens is comparable to a short text sequence

### Full Vision Transformer

```python
class VisionTransformer(nn.Module):
    """Simplified ViT for understanding the architecture."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        n_layers=12,
        n_heads=12
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2

        # CLS token: a learnable vector prepended to the sequence
        # Used to aggregate information for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings: learned, not sinusoidal (works better for images)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, 196, 768)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 197, 768)

        # Add position embeddings
        x = x + self.pos_embed

        # Transform
        x = self.transformer(x)  # (batch, 197, 768)

        return x  # Full sequence, or x[:, 0] for just CLS token

vit = VisionTransformer()
image = torch.randn(1, 3, 224, 224)
output = vit(image)
print(output.shape)  # (1, 197, 768)
```

### Connecting Vision to Language: The Projection Layer

Here's where multimodality happens. We have:
- Vision encoder output: (batch, 197, 768)
- LLM expects: (batch, seq_len, 4096) for a typical 7B model

**Solution**: Learn a projection that maps vision space → language space.

```python
class VisionLanguageConnector(nn.Module):
    """Projects vision embeddings into the LLM's embedding space."""

    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()
        # Simple: single linear projection
        # More complex: MLP, cross-attention, etc.
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features):
        return self.proj(vision_features)

# Vision tokens now live in the same space as text tokens
connector = VisionLanguageConnector()
vision_tokens = connector(output)  # (1, 197, 4096)
```

### Putting It Together: A Multimodal LLM

```python
class SimpleVisionLLM(nn.Module):
    """Conceptual architecture of models like LLaVA, GPT-4V."""

    def __init__(self, vision_encoder, connector, llm):
        super().__init__()
        self.vision_encoder = vision_encoder  # Frozen or fine-tuned
        self.connector = connector             # Trained
        self.llm = llm                         # Frozen or fine-tuned

    def forward(self, image, text_tokens):
        # 1. Encode the image
        vision_features = self.vision_encoder(image)  # (batch, 197, 768)

        # 2. Project to LLM space
        vision_tokens = self.connector(vision_features)  # (batch, 197, 4096)

        # 3. Embed the text
        text_embeddings = self.llm.embed_tokens(text_tokens)  # (batch, text_len, 4096)

        # 4. Concatenate: [vision tokens] [text tokens]
        combined = torch.cat([vision_tokens, text_embeddings], dim=1)

        # 5. Run through LLM
        output = self.llm(inputs_embeds=combined)

        return output
```

**The key insight**: The LLM sees `[img_token_1, img_token_2, ..., img_token_197, "What", "is", "in", "this", "image", "?"]`. It attends to all positions equally. The vision tokens are just... tokens.

### Training Strategy: LLaVA as Case Study

LLaVA (Large Language and Vision Assistant) shows a practical training recipe:

**Stage 1: Feature Alignment (Pretraining)**
- Freeze vision encoder AND LLM
- Only train the connector
- Use image-caption pairs
- Goal: Teach the connector to produce embeddings the LLM can understand

```python
# Pseudocode for stage 1
for image, caption in image_caption_dataset:
    vision_tokens = connector(frozen_vision_encoder(image))
    text_tokens = tokenizer(caption)

    # Standard language modeling loss
    # Given vision tokens, predict the caption
    loss = llm.forward(
        inputs_embeds=cat([vision_tokens, text_tokens]),
        labels=text_tokens
    )
    loss.backward()
    optimizer.step()  # Only updates connector
```

**Stage 2: Instruction Tuning**
- Freeze vision encoder
- Train connector AND LLM
- Use instruction-following data with images
- Goal: Teach the model to follow instructions about images

---

## Part 2: Audio — How Whisper Works

### The Challenge

Audio is fundamentally different from images:
- Temporal: unfolds over time
- Variable length: a clip can be 1 second or 1 hour
- Dense information: 16,000 samples per second for standard audio

### Step 1: Convert Audio to Mel Spectrogram

Raw audio is just amplitude over time—not very useful for neural networks. The mel spectrogram converts audio into a time-frequency representation.

```python
import torch
import torchaudio

def audio_to_mel_spectrogram(waveform, sample_rate=16000):
    """
    Convert raw audio waveform to mel spectrogram.

    What this does:
    1. Split audio into overlapping windows
    2. Compute FFT on each window → frequency content
    3. Apply mel filterbank → human-perceptual frequency scale
    4. Take log → compress dynamic range
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,           # Window size for FFT (25ms at 16kHz)
        hop_length=160,      # Step between windows (10ms at 16kHz)
        n_mels=80            # Number of mel frequency bins
    )

    mel_spec = mel_transform(waveform)

    # Log scale (standard for audio)
    log_mel = torch.log(mel_spec + 1e-10)

    return log_mel

# Example
waveform = torch.randn(1, 16000 * 30)  # 30 seconds of audio
mel = audio_to_mel_spectrogram(waveform)
print(mel.shape)  # (1, 80, 3000) — 80 frequency bins, 3000 time steps
```

**Intuition**: The mel spectrogram is like a "picture" of the audio where:
- X-axis = time
- Y-axis = frequency (mel-scaled to match human hearing)
- Brightness = energy at that time-frequency point

### Step 2: Whisper Architecture

Whisper is an encoder-decoder transformer (like the original "Attention Is All You Need" architecture).

```python
class WhisperEncoder(nn.Module):
    """
    Encodes mel spectrogram into a sequence of hidden states.
    """

    def __init__(self, n_mels=80, n_ctx=1500, d_model=512, n_heads=8, n_layers=6):
        super().__init__()

        # Two conv layers to downsample time dimension
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Positional embedding for the sequence
        self.positional_embedding = nn.Embedding(n_ctx, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel):
        # mel: (batch, 80, time_steps) — e.g., (batch, 80, 3000)

        # Conv layers: extract features and downsample
        x = torch.gelu(self.conv1(mel))   # (batch, 512, 3000)
        x = torch.gelu(self.conv2(x))     # (batch, 512, 1500) — stride=2 halves time

        x = x.transpose(1, 2)  # (batch, 1500, 512) — sequence format

        # Add positional embeddings
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.positional_embedding(positions)

        # Transform
        x = self.transformer(x)

        return x  # (batch, 1500, 512)


class WhisperDecoder(nn.Module):
    """
    Autoregressively generates text tokens from encoder output.
    """

    def __init__(self, vocab_size=51865, n_ctx=448, d_model=512, n_heads=8, n_layers=6):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(n_ctx, d_model)

        # Transformer decoder with cross-attention to encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, encoder_output):
        # tokens: (batch, seq_len) — previously generated tokens
        # encoder_output: (batch, 1500, 512) — from encoder

        # Embed tokens
        x = self.token_embedding(tokens)  # (batch, seq_len, 512)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = x + self.positional_embedding(positions)

        # Causal mask for autoregressive decoding
        seq_len = tokens.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Decode with cross-attention to encoder
        x = self.transformer(
            x,
            encoder_output,
            tgt_mask=causal_mask.to(x.device)
        )

        x = self.ln_final(x)
        logits = self.proj(x)  # (batch, seq_len, vocab_size)

        return logits


class Whisper(nn.Module):
    """Complete Whisper model."""

    def __init__(self):
        super().__init__()
        self.encoder = WhisperEncoder()
        self.decoder = WhisperDecoder()

    def forward(self, mel, tokens):
        encoder_output = self.encoder(mel)
        logits = self.decoder(tokens, encoder_output)
        return logits

    def transcribe(self, mel, max_len=448):
        """Generate transcription autoregressively."""
        encoder_output = self.encoder(mel)

        # Start with start-of-transcript token
        tokens = torch.tensor([[50258]])  # <|startoftranscript|>

        for _ in range(max_len):
            logits = self.decoder(tokens, encoder_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == 50257:  # <|endoftext|>
                break

        return tokens
```

### What Makes Whisper Special

**1. Massive Weakly-Supervised Training**
- 680,000 hours of audio-text pairs from the internet
- No manual transcription needed—scraped from subtitles, captions, etc.
- The scale overcomes noise in the data

**2. Multitask Training with Special Tokens**

Whisper uses special tokens to specify the task:

```python
# Token sequence for English transcription:
# <|startoftranscript|><|en|><|transcribe|><|notimestamps|> ... text ...

# Token sequence for translation to English:
# <|startoftranscript|><|es|><|translate|><|notimestamps|> ... english text ...

# Token sequence with timestamps:
# <|startoftranscript|><|en|><|transcribe|><|0.00|> Hello <|2.50|> world <|4.00|>

SPECIAL_TOKENS = {
    "<|startoftranscript|>": 50258,
    "<|endoftext|>": 50257,
    "<|transcribe|>": 50359,
    "<|translate|>": 50358,
    "<|notimestamps|>": 50363,
    # Language tokens: <|en|>, <|es|>, <|fr|>, etc.
}
```

**3. Input Normalization**
- Fixed 30-second audio chunks
- Shorter audio is padded with silence
- Longer audio is processed in chunks

```python
def pad_or_trim(mel, target_length=3000):
    """Ensure consistent input size."""
    if mel.shape[-1] > target_length:
        mel = mel[..., :target_length]
    elif mel.shape[-1] < target_length:
        pad = target_length - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad))
    return mel
```

### Using Whisper in Practice

```python
import whisper

# Load model (tiny, base, small, medium, large)
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("audio.mp3")
print(result["text"])

# With options
result = model.transcribe(
    "audio.mp3",
    language="en",           # Force language (or auto-detect)
    task="transcribe",       # or "translate" for translation to English
    fp16=True,               # Use half precision for speed
    word_timestamps=True     # Get word-level timing
)
```

---

## Part 3: Connecting Audio to LLMs

### The Pattern Should Look Familiar

Just like vision, we need to project audio representations into the LLM's embedding space:

```python
class AudioLanguageConnector(nn.Module):
    """Projects Whisper encoder output into LLM space."""

    def __init__(self, whisper_dim=512, llm_dim=4096):
        super().__init__()
        # Could also use a more complex adapter (Q-Former, etc.)
        self.proj = nn.Linear(whisper_dim, llm_dim)

    def forward(self, audio_features):
        return self.proj(audio_features)


class AudioLLM(nn.Module):
    """Conceptual audio-language model."""

    def __init__(self, whisper_encoder, connector, llm):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.connector = connector
        self.llm = llm

    def forward(self, mel, text_tokens):
        # 1. Encode audio
        audio_features = self.whisper_encoder(mel)  # (batch, 1500, 512)

        # 2. Project to LLM space
        audio_tokens = self.connector(audio_features)  # (batch, 1500, 4096)

        # 3. Embed text
        text_embeddings = self.llm.embed_tokens(text_tokens)

        # 4. Concatenate
        combined = torch.cat([audio_tokens, text_embeddings], dim=1)

        # 5. Run through LLM
        return self.llm(inputs_embeds=combined)
```

### GPT-4o's Approach: Native Multimodality

GPT-4o (Omni) took a different approach than the "encoder + projection" pattern:

**Traditional approach** (GPT-4V, most open-source):
- Separate specialized encoders for each modality
- Projection layers to map into LLM space
- LLM trained primarily on text, adapted to other modalities

**Native multimodality** (GPT-4o):
- Single model trained from scratch on all modalities
- All modalities tokenized similarly
- Joint embedding space learned during pretraining

This is why GPT-4o has such low latency for voice—it's not running separate ASR → LLM → TTS pipeline, it's processing audio natively.

---

## Part 4: Video — Adding the Time Dimension

### The Challenge

Video = sequence of images + temporal relationships

```python
# A 10-second video at 30fps
frames = 300
# If each frame becomes 196 tokens...
total_tokens = 300 * 196  # = 58,800 tokens per video
```

### Solutions

**1. Temporal Sampling**: Don't use every frame

```python
def sample_frames(video, n_frames=8):
    """Sample n frames uniformly from video."""
    total_frames = len(video)
    indices = torch.linspace(0, total_frames - 1, n_frames).long()
    return video[indices]

# 8 frames x 196 tokens = 1,568 tokens — manageable
```

**2. Temporal Pooling**: Compress across time

```python
class TemporalPooling(nn.Module):
    """Pool vision features across time dimension."""

    def __init__(self, n_frames=8, pool_size=4):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, frame_features):
        # frame_features: (batch, n_frames, n_patches, dim)
        # Reshape and pool across frames
        b, t, n, d = frame_features.shape
        x = frame_features.transpose(1, 2)  # (batch, n_patches, n_frames, dim)
        x = x.reshape(b * n, t, d)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(b, n, -1, d)
        return x.transpose(1, 2)  # (batch, n_frames/pool_size, n_patches, dim)
```

**3. Video-Specific Architectures**: ViViT, TimeSformer

```python
# TimeSformer approach: divided space-time attention
class TimeSformerBlock(nn.Module):
    """Alternates between spatial and temporal attention."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, n_frames, n_patches):
        # x: (batch, n_frames * n_patches, dim)
        b, _, d = x.shape

        # Temporal attention: attend across frames for each patch position
        x_t = x.view(b, n_frames, n_patches, d)
        x_t = x_t.permute(0, 2, 1, 3).reshape(b * n_patches, n_frames, d)
        x_t = self.temporal_attn(x_t, x_t, x_t)[0]
        x_t = x_t.view(b, n_patches, n_frames, d).permute(0, 2, 1, 3).reshape(b, -1, d)
        x = self.norm1(x + x_t)

        # Spatial attention: attend across patches for each frame
        x_s = x.view(b, n_frames, n_patches, d)
        x_s = x_s.reshape(b * n_frames, n_patches, d)
        x_s = self.spatial_attn(x_s, x_s, x_s)[0]
        x_s = x_s.view(b, n_frames, n_patches, d).reshape(b, -1, d)
        x = self.norm2(x + x_s)

        return x
```

---

## Part 5: The Universal Pattern

Every modality follows the same recipe:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Raw Input        Encoder           Projection      LLM       │
│   ──────────────────────────────────────────────────────────   │
│                                                                 │
│   Image      →   ViT           →   Linear/MLP   →             │
│   (H,W,C)        (N, d_vision)     (N, d_llm)      Shared      │
│                                                     Transformer │
│   Audio      →   Whisper Enc   →   Linear/MLP   →  (with       │
│   (mel)          (T, d_audio)      (T, d_llm)      cross-modal │
│                                                     attention)  │
│   Video      →   ViViT         →   Linear/MLP   →             │
│   (T,H,W,C)      (T*N, d_vid)      (T*N, d_llm)               │
│                                                                 │
│   Text       →   Tokenizer+Emb →   (identity)   →             │
│   (string)       (S, d_llm)        (S, d_llm)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Encoder: Frozen vs. Fine-tuned**
- Frozen: Faster training, leverages pretrained features
- Fine-tuned: Better alignment but risk of catastrophic forgetting

**2. Projection: Linear vs. Complex**
- Linear: Simple, fast, often works well
- MLP: More expressive
- Q-Former (BLIP-2): Learnable queries compress information
- Cross-attention: Most flexible but expensive

**3. Training: Joint vs. Staged**
- Joint: Train everything together (expensive, potentially better)
- Staged: First align, then instruction-tune (practical, works well)

---

## Part 6: Open Source Landscape

### Vision-Language Models

| Model | Architecture | Training | Notes |
|-------|-------------|----------|-------|
| LLaVA | CLIP ViT + Linear + Vicuna | 2-stage | Clean, simple, effective |
| InternVL | InternViT + MLP + InternLM | Joint | Strong performance |
| Qwen-VL | ViT + Cross-attn + Qwen | Joint | Good multilingual |
| CogVLM | EVA-CLIP + MLP + Vicuna | 2-stage | Visual expert approach |

### Audio-Language Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| Qwen-Audio | Whisper + Qwen | Multiple audio tasks |
| SALMONN | Whisper + BEATs + Vicuna | Dual audio encoders |
| LLaMA-Omni | Whisper + LLaMA | Direct speech input/output |

### The Compute Problem

Why open-source is slower on audio/video:

```python
# Compute comparison (rough estimates)

# Text pretraining
text_tokens_per_example = 2048
examples_per_batch = 4

# Vision pretraining
vision_tokens_per_example = 2048 + 197  # text + image
# ~10% more tokens, but images need encoding (extra forward pass)

# Audio pretraining
audio_tokens_per_example = 2048 + 1500  # text + 30s audio
# ~73% more tokens, longer sequences = quadratic attention cost

# Video pretraining
video_tokens_per_example = 2048 + (8 * 196)  # text + 8 frames
# ~76% more tokens, plus video loading/decoding is slow
```

The bottleneck isn't just compute—it's also data. High-quality audio/video + text pairs are much harder to curate than image-text pairs.

---

## Part 7: CLIP — The Foundation of Modern Vision-Language

Before we had multimodal LLMs, there was CLIP (Contrastive Language-Image Pre-training). Understanding CLIP is essential because most vision-language models use CLIP's vision encoder.

### The CLIP Insight

**Goal**: Learn a shared embedding space where images and their descriptions are close together.

```python
# CLIP training: match images with their captions
# Given a batch of (image, text) pairs, learn to:
# 1. Pull matching pairs together
# 2. Push non-matching pairs apart
```

### Contrastive Learning

```python
import torch
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder  # ViT
        self.text_encoder = text_encoder      # Transformer

        # Project both to shared dimension
        self.vision_proj = nn.Linear(vision_encoder.embed_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.embed_dim, embed_dim)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        # Encode both modalities
        image_features = self.vision_encoder(images)[:, 0]  # CLS token
        text_features = self.text_encoder(texts)[:, 0]

        # Project to shared space
        image_embeds = self.vision_proj(image_features)
        text_embeds = self.text_proj(text_features)

        # Normalize (crucial for contrastive learning)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return image_embeds, text_embeds

    def compute_loss(self, image_embeds, text_embeds):
        # Similarity matrix: (batch, batch)
        # Entry [i,j] = similarity between image i and text j
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_embeds.T

        # Labels: diagonal should be highest (matching pairs)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric loss: image→text and text→image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
```

**What CLIP learns**:
- `image_embed · text_embed` = semantic similarity
- "A photo of a cat" will be close to cat images
- "A photo of a dog" will be close to dog images
- This works for arbitrary concepts, not just predefined classes

### Why CLIP Matters for Multimodal LLMs

1. **Zero-shot classification**: No fine-tuning needed for new tasks
2. **Strong visual features**: CLIP's ViT learns rich representations
3. **Aligned spaces**: Vision and language already somewhat aligned

Most multimodal LLMs start with CLIP's vision encoder because it already "speaks language" to some extent.

---

## Part 8: Beyond Perception — Generation

So far we've focused on understanding modalities. But what about generating them?

### Image Generation from LLMs

Two approaches:

**1. Discrete tokens (like DALL-E 1, Parti)**
```python
# Learn a visual vocabulary (like BPE for images)
# Image → discrete tokens → LLM generates tokens → tokens → image

class VQ_VAE(nn.Module):
    """Learn discrete image tokens."""
    def __init__(self, n_codes=8192, code_dim=256):
        super().__init__()
        self.encoder = ...  # Image → continuous latent
        self.codebook = nn.Embedding(n_codes, code_dim)  # Discrete codes
        self.decoder = ...  # Discrete codes → image

    def encode(self, image):
        z = self.encoder(image)  # (batch, h, w, dim)
        # Find nearest code for each spatial position
        distances = torch.cdist(z, self.codebook.weight)
        indices = distances.argmin(dim=-1)  # (batch, h, w) of integers
        return indices

    def decode(self, indices):
        z = self.codebook(indices)
        return self.decoder(z)

# Now LLM can generate images as token sequences!
# "A cat sitting on a mat" → [img_token_1, img_token_2, ...]
```

**2. Continuous generation (diffusion-based)**
```python
# LLM outputs conditioning embeddings
# Diffusion model generates image conditioned on those embeddings

class LLMWithDiffusion(nn.Module):
    def __init__(self, llm, diffusion_model):
        super().__init__()
        self.llm = llm
        self.diffusion = diffusion_model

    def generate_image(self, prompt_tokens):
        # LLM produces conditioning
        llm_output = self.llm(prompt_tokens)
        conditioning = llm_output[:, -1, :]  # Or pooled output

        # Diffusion generates image
        return self.diffusion.sample(conditioning)
```

### Audio Generation

Similar pattern—either discrete tokens (like AudioLM) or continuous generation:

```python
# AudioLM approach: hierarchical tokens
# Semantic tokens (from w2v-BERT) → Coarse acoustic → Fine acoustic → Audio

# 1. LLM generates semantic tokens (captures "what" is said)
# 2. Separate model generates acoustic tokens (captures "how" it sounds)
# 3. Decoder reconstructs waveform
```

---

## Summary

1. **Core principle**: Convert all modalities to sequences of vectors that live in the LLM's embedding space

2. **Vision**: Patch an image into tokens with ViT, project to LLM dimension

3. **Audio**: Convert to mel spectrogram, encode with transformer, project to LLM dimension. Whisper showed massive scale overcomes weak supervision.

4. **Video**: Sample frames, encode each, handle temporal relationships, project to LLM dimension

5. **The pattern is universal**: `Raw → Encoder → Projection → LLM`

6. **CLIP**: Foundation for vision-language, learns aligned embedding spaces via contrastive learning

7. **Native multimodality** (GPT-4o style) trains all modalities jointly from scratch—more elegant but requires immense compute

8. **Open-source gap**: Not just compute, but also data availability for audio/video

9. **Generation**: Either discretize modalities into tokens or use continuous generation (diffusion)
