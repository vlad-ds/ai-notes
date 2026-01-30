Notes based on [Karpathy's nanochat discussion](https://github.com/karpathy/nanochat/discussions/1) and code analysis.

---

## What is Midtraining?

**Midtraining** is an intermediate finetuning stage between pretraining and SFT (Supervised Fine-Tuning). It's algorithmically identical to pretraining—same training loop, same optimizer—but the dataset shifts from raw internet documents to structured conversations.

### The Modern LLM Training Pipeline

```
Traditional (InstructGPT era):
Pretraining → SFT → RLHF

Modern pipelines:
Pretraining → Midtraining → SFT → RLHF/DPO
```

The "midtraining" stage goes by different names in the industry:
- **Continued pretraining**
- **Annealing** (used by Meta in Llama 3)
- **Domain adaptation**
- **Stage 2 pretraining**

Karpathy's framing makes this stage explicit and pedagogically clean, but in production the boundaries between stages are often blurry.

---

## Purpose of Midtraining

### 1. Learn Conversation Structure

The model learns special tokens that structure multi-turn conversations:
- `<|bos|>` — Beginning of sequence (document delimiter, used in pretraining too)
- `<|user_start|>`, `<|user_end|}` — Wrap user messages
- `<|assistant_start|>`, `<|assistant_end|>` — Wrap assistant messages
- `<|python_start|>`, `<|python_end|>` — Wrap Python tool calls
- `<|output_start|>`, `<|output_end|>` — Wrap tool outputs

Example conversation format:

```
<|bos|>
<|user_start|>What is 5 * 7 + 3?<|user_end|>
<|assistant_start|>Let me calculate that.
<|python_start|>5 * 7 + 3<|python_end|>
<|output_start|>38<|output_end|>
The answer is 38.<|assistant_end|>
```

### 2. Domain Shift

Adapt from internet document distribution to conversation distribution.

### 3. Skill Injection

Teach specific capabilities:
- **Multiple choice quiz-taking** — Small models don't naturally learn this from web data
- **Tool use** — Python interpreter via special tokens
- **Potentially context length expansion**

### 4. Knowledge vs Interface

A key insight from Karpathy:

> "The issue is not that the model doesn't have the knowledge, it's that it doesn't understand how Multiple Choice works to surface that knowledge."

Midtraining is partly about **building interfaces to existing knowledge**, not just adding new knowledge. The model knows Paris is the capital of France from pretraining, but doesn't know that when shown "A) London B) Paris C) Berlin D) Madrid" it should output "B".

---

## Nanochat's Midtraining Data Mixture

```python
train_dataset = TaskMixture([
    SmolTalk(split="train"),                          # 460K rows
    MMLU(subset="auxiliary_train", split="train"),    # 100K rows
    GSM8K(subset="main", split="train"),              # 8K rows
    CustomJSON(filepath=identity_conversations),       # 1K rows (x2 epochs)
    SimpleSpelling(size=200000, split="train"),       # 200K rows
    SpellingBee(size=80000, split="train"),           # 80K rows
])
# Total: ~850K rows ≈ 400-500M tokens
```

### Dataset Details

| Dataset | Rows | Purpose | Link |
|---------|------|---------|------|
| **SmolTalk** | 460K | General conversations | [HuggingFace](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) |
| **MMLU auxiliary_train** | 100K | Multiple choice problems (from ARC, MC_TEST, OBQA, RACE) | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| **GSM8K** | 8K | Grade school math + calculator tool use | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) |
| **SimpleSpelling** | 200K | Synthetic: "spell the word 'apple'" | Generated |
| **SpellingBee** | 80K | Synthetic: "how many 'r' in 'strawberry'?" | Generated |
| **Identity conversations** | 1K | Synthetic: model identity | Generated |

**Note on synthetic data**: The spelling tasks are interesting given the famous "strawberry" problem. They inject programmatically generated data to teach specific skills.

---

## Scale Comparison

| Stage | Tokens | Ratio |
|-------|--------|-------|
| Pretraining | ~11B | 100% |
| Midtraining | ~400-500M | ~4% |

The midtraining stage is much smaller than pretraining—roughly 4% of the token count.

---

## Special Tokens: Implementation Details

### Reserved from the Start

Special tokens are in the tokenizer vocabulary and embedding table from the beginning:

```python
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]
```

### How They're Added

1. BPE training happens with `vocab_size - len(SPECIAL_TOKENS)` tokens
2. Special tokens are added at the END of the vocabulary after BPE training
3. During pretraining, only `<|bos|>` is used (to delimit documents)
4. Other special tokens exist with random embeddings but receive no gradients
5. During midtraining, these tokens first appear and their embeddings get trained

This is the standard approach—reserve token IDs upfront to avoid resizing the embedding matrix later.

---

## Tool Use: Output Tokens

The `<|output_start|>` and `<|output_end|>` tokens wrap Python interpreter output:

```python
elif part["type"] == "python_output":
    # none of these tokens are supervised because they come from Python at test time
    add_tokens(output_start, 0)  # mask=0, NOT trained
    add_tokens(value_ids, 0)
    add_tokens(output_end, 0)
```

**Key point**: The mask is `0` for output tokens—the model is NOT trained to predict them. At inference, the actual Python interpreter runs the code and injects real output. The model only learns to *read and use* the output, not *generate* it.

---

## How Midtraining Differs from Pretraining

### What's the Same

- Same Muon + AdamW optimizer combo
- Same forward/backward/step loop
- Same gradient accumulation
- All parameters trainable (no layer freezing)

### What's Different

| Aspect | Pretraining | Midtraining |
|--------|-------------|-------------|
| Data | Web documents | Conversations |
| Duration | 21K iterations, 11B tokens | Single epoch, ~500M tokens |
| Learning rates | Higher | Lower (typical for finetuning) |
| LR schedule | Varies | Constant 80%, then linear decay to 0 |
| Starting point | Random init | Loads pretrained checkpoint |

---

## How Midtraining Differs from SFT

| Aspect | Midtraining | SFT |
|--------|-------------|-----|
| Data quality | Bulk data | Cherry-picked "beautiful" data |
| Data packing | Packs multiple conversations per row | Pads each example separately |
| Loss masking | Trains on all tokens | Uses mask (only supervise assistant) |
| Purpose | Bulk adaptation | Final polish, safety training |

### Packing vs Padding

Midtraining still **packs** conversations for efficiency:

```
[conv1][conv2][conv3]  <- packed into one training row
```

SFT **pads** to match inference exactly:

```
[conv1][padding...]    <- single conversation per row
```

This is a domain mismatch that SFT corrects.

---

## Catastrophic Forgetting

### The Concern

With no layer freezing and full parameter updates, the model might forget pretraining knowledge.

### Implicit Mitigations in Nanochat

- **Low data ratio** — midtraining is ~4% of pretraining tokens
- **Lower learning rates** — defaults are lower than pretraining
- **Content overlap** — SmolTalk contains world knowledge, MMLU has facts
- **Short duration** — single epoch

### Production Mitigations (not used in nanochat)

- **Replay**: Mix pretraining data into finetuning
- **LoRA/Adapters**: Only train a small subset of parameters
- **EWC (Elastic Weight Consolidation)**: Penalize changes to "important" weights
- **Layer freezing**: Freeze early layers

Some forgetting almost certainly happens. The bet is that format capabilities are worth the tradeoff.

---

## Why Staged Training?

### Why Not Mix Conversation Data into Pretraining?

**Dilution problem**: Midtraining data is ~4% of pretraining. If mixed in, special tokens would appear very rarely—they'd be noise in the gradient signal.

**Curriculum effects**: Research suggests data ordering matters:
1. First, learn *what* (knowledge from web)
2. Then, learn *how to present it* (conversation format)

Mixing asks the model to learn both simultaneously from a much noisier signal.

### Practical Benefits of Stages

- Reusable base model for different finetuning purposes
- Easier to debug and iterate (swap midtraining recipes without re-running pretraining)
- Clear separation of concerns

### Counter-argument

Llama 3's approach suggests data mixing schedules during pretraining matter a lot. They upweight high-quality data toward the end ("annealing"). The line between "late pretraining" and "midtraining" is blurry.

---

## How Much Data?

**There's no principled formula.** It's found empirically through ablations:
- Too little → model doesn't learn the format well
- Too much → catastrophic forgetting

Industry labs probably run extensive sweeps. For a pedagogical project like nanochat, it's "good enough" tuning.

---

## Open Questions

1. **Knowledge vs Interface**: How much of post-training is "teaching knowledge" vs "teaching how to express/access knowledge"?
2. **Packing/Padding Mismatch**: How much does this actually hurt? Is it measurable?
3. **Synthetic Data Tradeoffs**: Cheap and targeted vs expensive and diverse—what's the right balance?
4. **Teaching to the Test**: MMLU auxiliary_train teaches multiple choice, then we evaluate on MMLU. Legitimate skill transfer or teaching to the test?
5. **Special Token Learning Dynamics**: How quickly do special token embeddings converge? Phase transition or gradual?
6. **What Breaks Without Midtraining?**: If you went directly base → SFT:
   - Would special tokens be learned?
   - Would it take longer?
   - Would final quality suffer?
   Intuition: It would probably work, but less efficiently. Midtraining is a "bulk" adaptation that makes SFT more effective.

---

## Key Takeaways

1. **Midtraining is format training, not (primarily) knowledge training** — the knowledge is already there from pretraining
2. **Algorithmically identical to pretraining** — same optimizer, same loop, no freezing, just different data
3. **Empiricism**: Much of this is empirically discovered, not theoretically understood. The "right" recipe depends on model scale, data quality, compute budget, and target capabilities.
4. **Karpathy's contribution**: Making this stage explicit and pedagogically clean, when in production it's often implicit or blurred with other stages.

---

## References

- [Nanochat GitHub Discussion](https://github.com/karpathy/nanochat/discussions/1)
- [SmolTalk Dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [FineWeb-EDU Dataset](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [DCLM / CORE Metric](https://arxiv.org/abs/2406.11794)
