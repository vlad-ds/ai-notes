# Multimodality: How LLMs See, Hear, and Beyond

*Getting transformers to understand the world beyond text.*

---

An LLM is, at its core, a machine that processes sequences of vectors. When you type "Hello world", the tokenizer converts it to token IDs, and the embedding layer converts those IDs into vectors. The transformer then attends over this sequence of vectors, updating them through layers until the final representation emerges.

Here's the key insight that unlocks multimodality: the transformer doesn't actually care where those vectors came from. It just sees positions in a sequence, each carrying a d-dimensional vector. So if we could somehow convert an image, or an audio clip, or a video, into a sequence of vectors in the same space as text embeddings—the transformer would process them just like words.

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

The breakthrough came in 2020 with the Vision Transformer (ViT). The idea is simple: instead of treating each pixel as a token, treat each patch of the image as a token.

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

Let's use a concrete example. Take a 512×512 image with 32×32 patches:

```python
self.proj = nn.Conv2d(
    in_channels=3,       # RGB
    out_channels=1024,   # embedding dimension
    kernel_size=32,      # patch size
    stride=32            # same as kernel_size = non-overlapping
)

# Input: (batch, 3, 512, 512)
# Output: (batch, 1024, 16, 16) → reshape to (batch, 256, 1024)
```

Working through the numbers:
- 512 ÷ 32 = 16 patches per side, so 16 × 16 = **256 patches** total
- Each patch contains 32 × 32 × 3 = **3072 raw pixel values**
- These get projected down to **1024 dimensions**
- That's 3:1 compression per patch

The convolution learns which combinations of those 3072 pixel values matter for understanding image content, and compresses them into 1024 semantic features. This is real work—not just reshaping, but learning to extract meaningful representations from raw pixels.

Notice that stride equals kernel size. This means the patches are **non-overlapping**—each 32×32 region belongs to exactly one patch:

```
┌────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │   Each patch covers its own 32×32 region.
├────┼────┼────┼────┤   No pixel belongs to more than one patch.
│ 5  │ 6  │ 7  │ 8  │
├────┼────┼────┼────┤
│ 9  │ 10 │ 11 │ 12 │
└────┴────┴────┴────┘
        ... 256 patches total
```

Some variants use overlapping patches (stride < kernel size) to better capture information at boundaries, but this increases the sequence length. The original ViT used non-overlapping for simplicity.

Now we have 256 tokens instead of 786K values (512×512×3)—a 3000× reduction. And each token represents a semantic chunk of the image rather than a meaningless single pixel.

The rest of the Vision Transformer is just a standard transformer encoder. We prepend a learnable CLS token (like BERT), add positional embeddings, and run transformer layers. The output is a sequence of 257 vectors (256 patches + 1 CLS), each 1024-dimensional.

### Connecting Vision to Language

Here's where multimodality actually happens. We have:

- Vision encoder output: 257 vectors of dimension 1024
- LLM embedding space: vectors of dimension 4096 (for a typical 7B model)

These don't match. We need a projection layer that maps vision space into language space. The simplest version is just a linear layer:

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

LLaVA (Large Language and Vision Assistant) provides a clean example of how to train such a model. It uses a two-stage approach.

**Stage 1: Feature Alignment.** Freeze both the vision encoder (CLIP ViT) and the LLM (Vicuna). Only train the projection layer. Use image-caption pairs: given the image tokens, predict the caption with standard language modeling loss. This teaches the projection layer to output vectors the LLM can interpret as meaningful content.

**Stage 2: Instruction Tuning.** Keep the vision encoder frozen, but now also train the LLM (or fine-tune it with LoRA). Use instruction-following data that includes images: "Describe what's happening in this image", "What color is the car?", etc. This teaches the model to follow instructions that involve visual understanding.

The key insight is that you don't need to train everything end-to-end from scratch. You can bootstrap from a good vision encoder (CLIP) and a good LLM (Vicuna), then just learn the connection between them.

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
