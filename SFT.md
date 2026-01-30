## What is SFT?

Supervised Fine-Tuning is the process of continuing to train a pretrained language model on curated (instruction, response) pairs. The model learns to generate appropriate responses given instructions or prompts.

**Core idea**: Take a model that can predict next tokens well (from pretraining) and adapt it to follow instructions and produce helpful outputs.

---

## Where SFT Fits in the LLM Training Pipeline

```javascript
Pretraining
    │  Objective: Next-token prediction on massive web corpus
    │  Data: Trillions of tokens (web pages, books, code)
    │  Result: Strong language model, "fancy autocomplete"
    │
    ▼
[Optional] Midtraining / Continued Pretraining
    │  Objective: Domain adaptation, format learning
    │  Data: Conversations, specific formats, tool use examples
    │
    ▼
Supervised Fine-Tuning (SFT)
    │  Objective: Learn to follow instructions
    │  Data: (instruction, response) pairs; high quality
    │  Result: Instruction-following model
    │
    ▼
[Optional] RLHF / Preference Optimization
    │  Objective: Align with human preferences
    │  Data: Human feedback, preference rankings
    │
    ▼
Deployed Model
```

---

## How SFT Differs from Pretraining

| Aspect | Pretraining | SFT |
|--------|-------------|-----|
| **Objective** | Predict next token | Generate good responses to instructions |
| **Data scale** | Trillions of tokens | Thousands to millions of examples |
| **Data quality** | Web-scale (noisy) | Curated, high-quality |
| **Loss computation** | All tokens | Usually only response tokens (masked loss) |
| **Learning rate** | Higher | Lower (avoid catastrophic forgetting) |
| **Duration** | Weeks/months | Hours/days |

---

## Special Tokens: The Bridge Between Pretraining and SFT

### The Problem

A pretrained model knows nothing about conversation structure. It's just been predicting next tokens in documents. How does it learn to distinguish between "user talking" and "assistant responding"?

### The Solution: Reserve Tokens Early

During tokenizer training (before pretraining), special tokens are reserved in the vocabulary:

```javascript
Token ID 0:     <|bos|>           # Beginning of sequence
Token ID 1:     <|eos|>           # End of sequence
Token ID 2:     <|pad|>           # Padding
Token ID 3:     <|user_start|>    # User turn begins
Token ID 4:     <|user_end|>      # User turn ends
Token ID 5:     <|assistant_start|>
Token ID 6:     <|assistant_end|>
Token ID 7:     <|system_start|>
Token ID 8:     <|system_end|>
```

These tokens exist in the vocabulary from the start but are essentially **dormant** during pretraining. The model encounters them rarely (or never) in web text, so their embeddings remain near-random initialization.

### SFT "Activates" These Tokens

During SFT, the model finally sees these tokens in context:

```javascript
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4.<|assistant_end|>
```

The model learns:
- `<|user_start|>` signals "instruction coming"
- `<|assistant_start|>` signals "now I should generate a response"
- `<|assistant_end|>` signals "stop generating"

### At Inference: APIs Handle This Automatically

When you call an API like OpenAI's:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
)
```

The API internally converts this to the model's expected format with special tokens. The model then generates until it produces the end token, at which point the API stops and returns the response.

**You never see these tokens** in the API response; they're stripped out. But they're essential for the model to know when to start and stop.

---

## The SFT Loss Function

### SFT is Mechanically Identical to Pretraining

This is important: SFT uses the **exact same training mechanism** as pretraining. The model processes the entire sequence in a single forward pass, all positions are computed in parallel, and each position predicts the next token. The only difference is which tokens contribute to the loss.

**Pretraining**: Loss computed on all tokens
**SFT**: Loss computed only on response tokens

That's it. Same forward pass, same backpropagation, same optimizer. We just zero out the loss for instruction positions.

### Concrete Example

**Instruction**: "What is the capital of France?"
**Response**: "The capital of France is Paris."

After tokenization, we have tokens for the instruction (positions 0-8) and tokens for the response (positions 9-17).

### Single Forward Pass

The entire sequence enters the transformer at once. The causal mask ensures each position only attends to previous positions.

At the output, **every position** produces a probability distribution over the vocabulary, predicting what comes next. But we only compute loss on the response positions.

### What "Predicts" Means

At position 14, after seeing `[BOS] What is the capital of France ? The capital of France is`, the model outputs a probability distribution over all ~50,000 tokens in its vocabulary:

```javascript
P(Paris)  = 0.72
P(Lyon)   = 0.08
P(Berlin) = 0.03
P(the)    = 0.02
...
```

The loss at this position is `-log P(Paris)`. High confidence in the correct token = low loss.

### Why Zero the Instruction Loss?

The model *does* make predictions at instruction positions. We simply ignore them:

1. **We don't care about predicting the instruction.** At inference, the instruction is given to us.
2. **Learning signal should focus on response quality.** Including instruction tokens would spend model capacity learning patterns like "What is the capital of..." which doesn't improve response quality.

### Teacher Forcing

Because we feed the entire correct sequence as input, each position always sees the ground truth previous tokens, never the model's own predictions. This is called "teacher forcing."

This is efficient for training but creates a train-test mismatch (see Limitations below).

---

## Data Format

### Common Chat Templates

**ChatML (OpenAI-style)**

```javascript
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

**Llama-style**

```javascript
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2+2? [/INST] 2+2 equals 4.
```

**Alpaca format**

```javascript
### Instruction:
What is 2+2?

### Response:
2+2 equals 4.
```

The specific format matters less than consistency: the model must see the same format during training and inference.

---

## Key Technical Considerations

### 1. Learning Rate

SFT uses significantly lower learning rates than pretraining (often 10-100x lower) to avoid catastrophic forgetting of pretrained knowledge.

Typical ranges:
- Pretraining: 1e-4 to 6e-4
- SFT: 1e-5 to 2e-5

### 2. Data Quality vs Quantity

SFT is quality-sensitive. A smaller set of high-quality examples often outperforms a larger set of noisy data. This is unlike pretraining where scale dominates.

Key data properties:
- Diverse instruction types
- Accurate, helpful responses
- Consistent formatting
- Representative of target use cases

### 3. Sequence Packing vs Padding

**Packing** (efficient for pretraining): Multiple conversations concatenated into one sequence.

**Padding** (standard for SFT): One conversation per sequence, padded to length.

**Why padding for SFT?**

From Karpathy's nanochat discussion:

> "One domain adaptation that happens here is that SFT stretches out rows of data and pads them, exactly mimicking the test-time format. In other words, examples are not just randomly concatenated into long rows like in pre/mid-training, where it is done for efficiency of training. Fixing this domain mismatch serves as another little 'tightening the screws' boost."

At inference, each conversation starts fresh at position 0. If training always packs sequences, the model never learns this "fresh start" pattern.

### 4. Number of Epochs

SFT typically runs for 1-3 epochs. More epochs risk:
- Overfitting to training examples
- Losing diversity in outputs
- Memorizing specific phrasings

### 5. Batch Size and Gradient Accumulation

Larger effective batch sizes tend to produce more stable training. If memory-limited, use gradient accumulation.

---

## Common SFT Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| **Alpaca** | 52K | GPT-4 generated instruction-following |
| **ShareGPT** | ~90K | Real user conversations with ChatGPT |
| **OpenAssistant** | 161K | Crowdsourced multi-turn conversations |
| **Dolly** | 15K | Databricks employee-written |
| **FLAN** | Millions | Academic NLP tasks reformatted as instructions |
| **SmolTalk** | 460K | HuggingFace curated conversations |

---

## What SFT Teaches the Model

1. **Instruction following**: Understanding what's being asked
2. **Response format**: How to structure outputs appropriately
3. **Tone and style**: Being helpful, clear, appropriately detailed
4. **Task patterns**: How to approach different types of requests
5. **Refusals**: When and how to decline inappropriate requests

SFT does NOT primarily teach new knowledge. That comes from pretraining. SFT teaches the model how to *use* its knowledge to respond to instructions.

---

## Limitations of SFT

### 1. Catastrophic Forgetting

Perhaps the most significant limitation. When fine-tuning on domain-specific data, the model can "forget" capabilities learned during pretraining.

**What happens**: The model's weights are optimized for the new task, overwriting representations that were useful for general tasks. A model fine-tuned on legal documents might become worse at math, coding, or general knowledge.

**Research findings**:
- Larger models actually suffer *worse* forgetting (counterintuitive)
- Domain knowledge tasks suffer the most forgetting
- Even LoRA and other PEFT methods still exhibit forgetting, just less severely
- Forgetting follows a power law: more training steps = more forgetting

**Mitigation strategies**:
- **Lower learning rates**: Smaller updates preserve more original knowledge
- **Fewer epochs**: 1-3 epochs is typical
- **LoRA/PEFT**: Freeze most parameters, only train small adapter layers
- **Rehearsal/Replay**: Mix original pretraining-style data with SFT data
- **Early stopping**: Monitor validation loss on held-out general tasks

### 2. Exposure Bias (from Teacher Forcing)

During training, each position sees ground truth previous tokens. During inference, each position sees the model's own outputs. This mismatch means:
- Errors don't propagate during training
- But errors compound during inference

The model never learns to recover from its own mistakes because it never experiences them during training.

### 3. No Preference Signal

SFT treats all training responses as equally good. There's no mechanism to express "B is better than A" or vice versa.

### 4. Quality Ceiling

The model can only be as good as the training data. SFT optimizes for imitating the demonstrations; it cannot discover better responses than those in the dataset.

### 5. Overfitting and Mode Collapse

With small datasets or too many epochs:
- Model memorizes specific phrasings rather than learning general patterns
- Output diversity decreases (mode collapse)
- Model becomes brittle to input variations

### 6. Data Imbalance Issues

If 60% of training responses say "I cannot answer this," the model will over-refuse at inference. The training distribution must match the desired inference distribution.

### 7. Safety Guardrail Erosion

Fine-tuning can degrade safety behaviors trained into the base model. Research shows that safety guardrails can be partially "fine-tuned away," even unintentionally.

These limitations motivate RLHF/DPO as follow-up stages, which can optimize for preferences and allow the model to explore beyond the demonstration distribution.

---

## Making SFT Go Well (and What Can Go Wrong)

### Recipe for Success

**1. Start with good evals**

Before fine-tuning, establish clear metrics:
- What does "good" look like for your task?
- How will you measure improvement over the base model?
- What's your baseline performance?

Without evals, you're flying blind.

**2. Data quality over quantity**

- 50-100 high-quality examples often outperform 10,000 noisy ones
- Each example should demonstrate exactly the behavior you want
- Remove contradictory examples
- Ensure consistent formatting across all examples

**3. Match the inference distribution**

Your training data should look like what the model will see in production:
- Same prompt format
- Same types of questions
- Same difficulty distribution
- Same response length expectations

**4. Use validation data**

Split your data into training and validation sets. Monitor validation loss during training. If validation loss increases while training loss decreases, you're overfitting.

**5. Start conservative with hyperparameters**

- Learning rate: 1e-5 to 2e-5
- Epochs: 1-3
- Batch size: as large as memory allows

### Common Failure Modes

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Overfitting** | Great on training examples, poor on new inputs | Fewer epochs, more data, regularization |
| **Catastrophic forgetting** | Lost general capabilities | Lower LR, PEFT methods, rehearsal data |
| **Mode collapse** | All outputs sound the same | More diverse training data, higher temperature at inference |
| **Wrong format** | Model ignores desired output structure | More format examples, explicit format instructions |
| **Over-refusal** | Model refuses too many reasonable requests | Balance refusal examples in training data |
| **Hallucination increase** | Model confidently makes things up | Higher quality data, consider RAG instead |

### Debugging Checklist

When SFT isn't working:
1. **Check your data first**: Are there errors or inconsistencies?
2. **Check for data imbalance**: Does the distribution match desired production distribution?
3. **Evaluate on held-out data**: Is the model overfitting?
4. **Compare to base model**: What specifically got better/worse?

---

## Fine-Tuning via APIs: How OpenAI Does It

### The Process

When you fine-tune through OpenAI's API, here's what happens behind the scenes:

**1. Data Upload**

You upload a JSONL file with conversation examples:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

OpenAI validates the format and checks for issues.

**2. Job Creation**

```python
client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-4o-mini",  # Base model
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 1.0,
        "batch_size": 1
    }
)
```

**3. Training**

OpenAI runs SFT on their infrastructure. Training typically takes minutes to hours depending on dataset size.

**4. Model Hosting**

The result is a new model ID like `ft:gpt-4o-mini-2024-07-18:org:weather:B7R9VjQd`. This is your "fork" of the base model with your fine-tuning applied.

### Pricing Model

- **Training cost**: Per token in your training data x number of epochs
- **Inference cost**: Higher than base models (typically ~2-6x)

This makes fine-tuning economical only when:
- You'll use the model at significant scale
- The improvement justifies the cost premium
- The alternative (longer prompts) would be more expensive

### Key Insights from OpenAI's Recommendations

1. **Minimum 10 examples**, but 50-100 recommended for meaningful improvement
2. **Use the weight field** to disable training on specific assistant messages (set weight=0)
3. **Distillation works**: Fine-tune a smaller model on outputs from a larger model
4. **Evals first**: Don't fine-tune until you can measure if it's working
5. **Iterate on data**: If results are poor, improve data quality rather than hyperparameters

### Fine-Tuning Methods Available

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Supervised (SFT)** | Learn from example outputs | Default; task-specific behavior |
| **Direct Preference Optimization (DPO)** | Learn from preference pairs | Subjective quality, style preferences |
| **Reinforcement Fine-Tuning (RFT)** | Learn from grader scores | Complex multi-step reasoning |
| **Vision Fine-Tuning** | SFT with image inputs | Image understanding tasks |

---

## When to Use SFT vs. Other Approaches

SFT is one tool in a larger toolkit. Here's how to decide what to use:

### The Decision Hierarchy

Start simple, escalate as needed:

```javascript
1. Prompt Engineering (hours/days, $0)
        ↓ Not enough?
2. Few-shot Examples (hours, minimal cost)
        ↓ Not enough?
3. RAG (days/weeks, moderate cost)
        ↓ Not enough?
4. Fine-tuning (weeks/months, high cost)
```

### Prompt Engineering

**Use when:**
- You need quick iteration
- The task is within the model's existing capabilities
- You have limited data

### RAG (Retrieval-Augmented Generation)

**Use when:**
- You need access to specific, current, or private information
- The knowledge changes frequently
- You want to cite sources

### Fine-Tuning (SFT)

**Use when:**
- You need consistent output format/style that prompting can't achieve
- You want to reduce prompt length (and cost) at scale
- You have proprietary patterns the model hasn't seen
- You're distilling from a larger model to a smaller one

**Do NOT use when:**
- You need up-to-date information (use RAG instead)
- Your data changes frequently
- You have limited data (< 50 examples)
- Prompt engineering already works

### Decision Matrix

| Need | Best Approach |
|------|---------------|
| Current/changing information | RAG |
| Consistent output format | Fine-tuning or structured outputs |
| Domain-specific terminology | Fine-tuning |
| Verifiable facts with sources | RAG |
| Specific tone/style | Fine-tuning (or detailed prompting) |
| Reduce inference costs at scale | Fine-tuning (shorter prompts) |
| Quick experimentation | Prompt engineering |

### The Hybrid Approach

In practice, production systems often combine multiple approaches:
1. **Fine-tune** for consistent style and format
2. **RAG** for factual grounding
3. **Tool use** for actions and calculations
4. **Prompt engineering** to guide the overall behavior

---

## SFT vs DPO vs RLHF: Detailed Comparison

All three are post-pretraining alignment methods, but they use fundamentally different training signals.

### Supervised Fine-Tuning (SFT)

**Training signal**: Demonstrations (correct answers)

**What it learns**: "Given this input, produce this output"

**Strengths**:
- Simple and stable
- Direct control; model produces exactly what you demonstrate
- Fast training

**Weaknesses**:
- Requires knowing the "right answer"
- Cannot express "A is better than B"; only "A is correct"
- Bounded by demonstration quality

### Direct Preference Optimization (DPO)

**Training signal**: Preference pairs (chosen vs rejected)

**What it learns**: "Given this input, A is better than B"

**Strengths**:
- No reward model needed
- No RL instability
- Can express nuanced preferences
- Simpler pipeline than RLHF

**Weaknesses**:
- Requires paired preference data (expensive to collect)
- Still bounded by the responses in your data

### RLHF (Reinforcement Learning from Human Feedback)

**Training signal**: Learned reward model + RL

**What it learns**: "Generate responses that maximize learned human preferences"

**Strengths**:
- Model can explore beyond demonstrated responses
- Can optimize complex, hard-to-specify objectives
- Most powerful alignment method in terms of ceiling

**Weaknesses**:
- Complex pipeline (SFT -> Reward Model -> RL)
- Training instability (reward hacking, mode collapse)
- Expensive and finicky to tune

### When to Use Each

| Scenario | Best Method |
|----------|-------------|
| Have demonstrations of correct outputs | SFT |
| Have preference pairs, want stability | DPO |
| Need model to improve beyond training data | RLHF |
| Limited compute/expertise | SFT or DPO |
| Maximum performance, have resources | RLHF |

### Typical Pipeline

```javascript
Pretrained Model
      ↓
    SFT  ──────────▶ Basic instruction-following
      ↓
  DPO or RLHF  ────▶ Preference-aligned model
```

Most production models use all stages: SFT first establishes basic format and capabilities, then preference optimization (DPO or RLHF) refines quality and safety.

---

## LoRA and Parameter-Efficient Fine-Tuning (PEFT)

### The Problem: Fine-Tuning Is Expensive

A 7B parameter model needs roughly 56GB of GPU memory just for training (weights + optimizer states + gradients). And if you want 10 different fine-tuned versions for 10 different tasks, you need to store 10 complete copies of a 14GB model.

This is wasteful. Most of the model stays roughly the same after fine-tuning; you're really just making small adjustments.

### The Core Idea of LoRA

**LoRA (Low-Rank Adaptation)** is based on a key insight: when you fine-tune a model, the changes to the weights are surprisingly "simple" in structure. You don't need to adjust every single parameter independently; the adjustments can be captured with far fewer numbers.

**Analogy**: Imagine you have a 1000x1000 pixel image (1 million pixels). If you want to describe a small modification to that image, you could specify a new value for each pixel (1 million numbers). But if the modification is something simple like "make the whole image slightly warmer," you could describe it with just a few numbers (shift red up, blue down). LoRA exploits this: fine-tuning changes are often "simple" enough to describe with far fewer parameters.

**How it works in practice**:
- Freeze all the original model weights (they don't change)
- Add small "adapter" layers that learn the adjustment
- These adapters have far fewer parameters than the original layers (often 100-1000x fewer)
- At inference: original output + adapter adjustment = fine-tuned output

**The result**: Instead of training 7 billion parameters, you might train only 10-50 million. Memory requirements drop dramatically. You can store dozens of fine-tuned variants by just saving tiny adapter files.

### Practical LoRA Settings

| Setting | What It Controls | Typical Values |
|---------|------------------|----------------|
| **Rank (r)** | Adapter capacity; higher = more expressive but more compute | 8, 16, 32 |
| **Alpha** | Scaling factor; usually set equal to rank | Same as rank |
| **Target modules** | Which layers get adapters | Attention layers (Q, K, V, O projections) |

**Rule of thumb**: Start with rank 8 or 16. If the model isn't learning well enough, increase it. If you're overfitting, decrease it.

### QLoRA: LoRA + Quantization

**Quantization** means storing the model weights with less precision (e.g., 4 bits instead of 16 bits). This shrinks the model by 4x.

**QLoRA** combines both tricks:
1. Load the base model in 4-bit precision (fits in much less memory)
2. Train small LoRA adapters in full precision

**Result**: You can fine-tune a 65B parameter model on a single consumer GPU. This democratized LLM fine-tuning enormously.

### Other PEFT Methods (Brief Overview)

| Method | Core Idea |
|--------|-----------|
| **LoRA** | Add small trainable adapters to frozen layers |
| **Prefix Tuning** | Learn a set of "virtual tokens" prepended to every input |
| **Adapters** | Insert small trainable layers between the frozen layers |
| **QLoRA** | LoRA + 4-bit quantization for memory efficiency |

They all share the same goal: train a small number of parameters while keeping most of the model frozen.

### When to Use PEFT vs Full Fine-Tuning

**Use LoRA/PEFT when**:
- You have limited GPU memory
- You want to store multiple fine-tuned variants cheaply
- You're worried about catastrophic forgetting (PEFT is gentler)
- The base model is already strong and you just need small adjustments

**Use full fine-tuning when**:
- You need maximum possible performance
- You have plenty of compute
- The task requires significant changes to model behavior
- You only need one variant

**In practice**: LoRA gets you 90-95% of full fine-tuning performance at a fraction of the cost. For most use cases, it's the right default choice.

---

## Multi-Turn Conversation Handling

### The Structure

Multi-turn conversations are serialized into a single sequence:

```javascript
<|system|>You are a helpful assistant.<|end|>
<|user|>What's the capital of France?<|end|>
<|assistant|>The capital of France is Paris.<|end|>
<|user|>What's the population?<|end|>
<|assistant|>Paris has about 2.1 million residents in the city proper.<|end|>
```

### Loss Masking

Only compute loss on assistant turns. The model sees the full context but only learns to generate assistant responses.

### Training Data Creation

From one multi-turn conversation, you can create multiple training examples:

**Approach 1: Full conversation only**
- Use entire conversation as one example
- Pro: Most context, realistic
- Con: Longer sequences, one gradient signal per conversation

**Approach 2: Expanding windows**
- Turn 1-2: Train on assistant response to first user message
- Turn 1-4: Train on second assistant response (with prior context)
- Pro: More training signal, teaches model to handle varying context lengths

### Context Length Considerations

Multi-turn conversations can exceed model context length. Strategies:

**Truncation from the beginning**: `[System prompt] + [Last N turns]`. Simple but loses early context.

**Sliding window**: `[System prompt] + [Summary of earlier turns] + [Last N turns]`. More complex but preserves more information.

### Special Considerations

**Turn boundaries**: Model must learn to stop generating at end tokens and understand when a new turn begins.

**Role consistency**: Ensure the model doesn't confuse its role across turns.

**Context utilization**: Model should appropriately reference earlier turns when relevant.

---

## Evaluation Metrics for SFT Models

### Automatic Metrics

**Perplexity / Cross-entropy loss**: Lower is better, but weakly correlated with actual quality. A model can have low perplexity and still produce bad responses. Use for sanity checking, detecting training issues, comparing checkpoints.

**BLEU**: N-gram overlap with reference. Standard for translation.

**ROUGE**: Recall-oriented; measures how much of the reference appears in the output. Standard for summarization.

**BERTScore**: Semantic similarity using BERT embeddings. Better than surface-level matching.

### Benchmark Suites

| Benchmark | Focus | Format |
|-----------|-------|--------|
| **MMLU** | Academic knowledge across 57 subjects | Multiple choice |
| **HellaSwag** | Commonsense reasoning | Multiple choice completion |
| **ARC** | Science reasoning | Multiple choice |
| **TruthfulQA** | Resistance to common misconceptions | Generation + MC |
| **GSM8K** | Grade school math | Open-ended |
| **HumanEval** | Code generation | Function completion |
| **MT-Bench** | Multi-turn conversation | Open-ended, LLM-judged |

**Critical**: Run benchmarks before AND after fine-tuning to check for regression.

### LLM-as-Judge

Use a capable model (GPT-4, Claude) to evaluate outputs. Prompt it to rate on helpfulness, accuracy, clarity, safety.

**Pros**: Scalable, can evaluate open-ended responses, correlates reasonably with human judgment.

**Cons**: Introduces judge model's biases, sensitive to prompt phrasing.

**Best practice**: Use pairwise comparison (A vs B) rather than absolute scores; more reliable.

### Human Evaluation

Still the gold standard. Key dimensions:

| Dimension | Question |
|-----------|----------|
| **Helpfulness** | Did it help with the task? |
| **Accuracy** | Is the information correct? |
| **Coherence** | Does it make sense? |
| **Relevance** | Is it on-topic? |
| **Safety** | Is it appropriate? |

**A/B testing**: Show humans outputs from model A and B (unlabeled), ask which is better. More reliable than absolute ratings.

### Regression Testing

Fine-tuning can break things. Maintain a test suite of:
- Representative task examples
- Edge cases
- Capabilities you want to preserve (math, code, general knowledge)

Run before and after fine-tuning; investigate any degradation.

### The Evaluation Stack

```javascript
1. Automatic metrics (fast, cheap, rough signal)
      ↓
2. Benchmark suites (standardized, comparable)
      ↓
3. LLM-as-judge (scalable quality assessment)
      ↓
4. Human evaluation (gold standard, expensive)
```

Run 1-3 frequently during development. Use 4 for final validation.

---

## Data Generation Techniques for SFT

Getting high-quality training data is often the bottleneck for SFT. Here are the main approaches.

### Human Annotation

**Direct writing**: Humans write both instructions and responses from scratch.

**Pros**: Highest quality, captures real user needs, diverse

**Cons**: Expensive ($15-50+ per example for expert tasks), slow, hard to scale

**Best for**: High-stakes domains (medical, legal), establishing quality baselines, creating seed data for synthetic expansion.

### Distillation from Stronger Models

Use a capable model (GPT-4, Claude) to generate training data for a smaller model.

**Pros**: Cheap, fast, consistent quality

**Cons**: Bounded by teacher model's capabilities, potential licensing issues

**Alpaca dataset**: 52K examples generated by GPT-3.5 from 175 seed tasks. Cost: ~$500. Enabled instruction-following in LLaMA.

### Self-Instruct

The model generates its own training data through a bootstrapping process.

**Pipeline**:
1. Start with small seed set of (instruction, response) pairs
2. Prompt model to generate new instructions similar to seeds
3. Model generates responses for new instructions
4. Filter for quality
5. Add to training set, repeat

### Evol-Instruct (WizardLM)

Evolve simple instructions into more complex ones through iterative rewriting.

**Evolution operations**:
- **Add constraints**: "Write a poem" -> "Write a sonnet about loss that doesn't use the letter 'e'"
- **Deepen**: "Explain X" -> "Explain X, including the historical context and modern implications"
- **Increase reasoning**: "What is 2+2?" -> "If I have 2 apples and buy 2 more, then give half away, how many do I have?"

### Rejection Sampling

Generate multiple responses, keep only the best ones.

**Process**:
1. For each instruction, generate N responses (e.g., N=10)
2. Score each response (using reward model, LLM judge, or heuristics)
3. Keep only top-k responses

**Why it works**: Even if a model produces good responses only 30% of the time, rejection sampling lets you curate a dataset of mostly good responses.

### Quality Filtering

Raw generated data needs filtering:
- **Length filters**: Remove too-short or too-long responses
- **Format filters**: Check for expected structure
- **Deduplication**: Remove near-duplicate instructions
- **LLM-based filtering**: Have a model judge quality
- **Reward model filtering**: Score with a trained reward model
- **Human spot-checking**: Randomly sample and manually verify

### Data Quality Checklist

Before training, verify:
- [ ] Format consistency: All examples follow same structure
- [ ] No contradictions: Similar instructions don't have conflicting responses
- [ ] Factual accuracy: Responses contain correct information
- [ ] Appropriate length: Responses match expected verbosity
- [ ] Task diversity: Cover the range of intended use cases
- [ ] Difficulty range: Include easy, medium, and hard examples
- [ ] Safety examples: Include appropriate refusals and sensitive handling
- [ ] No data leakage: Training data doesn't include test examples

### Cost Comparison

| Method | Cost per 1K examples | Quality | Speed |
|--------|----------------------|---------|-------|
| Expert human annotation | $5,000-50,000 | Highest | Slow |
| Crowdsourced annotation | $500-2,000 | Medium-High | Medium |
| GPT-4 distillation | $10-50 | High | Fast |
| Self-Instruct | $5-20 | Medium | Fast |

### Recommended Approach

**For most projects**:
1. **Start small**: 50-100 hand-crafted examples to establish quality bar
2. **Expand with distillation**: Use strong model to generate 1K-10K examples
3. **Filter aggressively**: Keep only high-quality examples (often discard 30-50%)
4. **Add domain data**: Include domain-specific examples for your use case
5. **Iterate based on evals**: Identify failure modes, generate targeted examples

**Data quantity guidelines**:
- Minimum viable: 50-100 examples
- Good baseline: 1K-10K examples
- Production quality: 10K-100K examples

---

## References

- Karpathy, A. (2025). [Introducing nanochat](https://github.com/karpathy/nanochat/discussions/1). GitHub Discussion.
- Luo, Y. et al. (2023). [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2308.08747). arXiv.
- Kotha, S. et al. (2024). [Scaling Laws for Forgetting When Fine-Tuning Large Language Models](https://arxiv.org/abs/2401.05605). arXiv.
- Hu, E. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv.
- Dettmers, T. et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). arXiv.
- Rafailov, R. et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). arXiv.
- Wang, Y. et al. (2023). [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560). arXiv.
- Xu, C. et al. (2023). [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244). arXiv.
- Taori, R. et al. (2023). [Stanford Alpaca: An Instruction-following LLaMA model](https://github.com/tatsu-lab/stanford_alpaca). GitHub.
- OpenAI. (2024). [Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning). OpenAI Platform.
