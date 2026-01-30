## How Autoregressive Generation Works

LLMs generate text **one token at a time**, where each token depends on all previous tokens. This is the fundamental loop:

```python
def generate(prompt, model, tokenizer, max_tokens):
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)

    for i in range(max_tokens):
        # 1. Feed ALL tokens into the model
        logits = model(tokens)  # Shape: [vocab_size]

        # 2. Convert logits to probabilities
        probs = softmax(logits)

        # 3. Sample the next token
        next_token = sample(probs)

        # 4. Check for end of sequence
        if next_token == EOS_TOKEN:
            break

        # 5. Append to context for next iteration
        tokens.append(next_token)

    return tokenizer.decode(tokens)
```

**Key insight:** In the naive case, the *entire* context is fed through the model on every single step. If your prompt is 1000 tokens and you generate 100 tokens:
- Step 1: Process 1000 tokens → get 1 new token
- Step 2: Process 1001 tokens → get 1 new token
- Step 3: Process 1002 tokens → get 1 new token
- …
- Step 100: Process 1099 tokens → get 1 new token

This is extremely wasteful. The computation for positions 1-1000 is identical every time.

---

## The KV Cache: Making Generation Efficient

### Why Attention is the Bottleneck

In self-attention, each position needs to attend to all previous positions:

```python
def attention(Q, K, V):
    # Q: query for current position(s)
    # K, V: keys and values for ALL positions
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    return weights @ V
```

For a sequence of length n, computing attention is O(n²) because every position attends to every other position.

### The Naive Approach (No Cache)

```python
def generate_naive(prompt_tokens, model, max_tokens):
    tokens = prompt_tokens.copy()

    for i in range(max_tokens):
        # Recompute EVERYTHING from scratch
        hidden = model.embed(tokens)

        for layer in model.layers:
            Q = layer.W_q(hidden)  # All positions
            K = layer.W_k(hidden)  # All positions
            V = layer.W_v(hidden)  # All positions
            hidden = attention(Q, K, V)

        logits = model.lm_head(hidden[-1])  # Only need last position's logits
        next_token = sample(logits)
        tokens.append(next_token)

    return tokens
```

**Cost per generated token:** O(n²) where n is current sequence length.

We compute K and V for all 1000 prompt tokens again and again, even though they never change.

### With KV Cache

**Idea:** Cache the K and V vectors for positions we've already processed. Only compute K, V for the new token.

```python
def generate_with_cache(prompt_tokens, model, max_tokens):
    tokens = prompt_tokens.copy()

    # --- PREFILL PHASE: Process entire prompt at once ---
    hidden = model.embed(tokens)
    kv_cache = []

    for layer in model.layers:
        Q = layer.W_q(hidden)
        K = layer.W_k(hidden)
        V = layer.W_v(hidden)

        # Store K, V for this layer
        kv_cache.append({'K': K, 'V': V})

        hidden = attention(Q, K, V)

    logits = model.lm_head(hidden[-1])
    next_token = sample(logits)
    tokens.append(next_token)

    # --- DECODE PHASE: One token at a time ---
    for i in range(max_tokens - 1):
        # Only embed the NEW token
        hidden = model.embed([tokens[-1]])  # Shape: [1, hidden_dim]

        for layer_idx, layer in enumerate(model.layers):
            # Only compute Q, K, V for the NEW position
            # These projections don't need context. Just matrix multiplies.
            q = layer.W_q(hidden)  # [1, hidden_dim]
            k = layer.W_k(hidden)  # [1, hidden_dim]
            v = layer.W_v(hidden)  # [1, hidden_dim]

            # Append to cache
            kv_cache[layer_idx]['K'] = concat(kv_cache[layer_idx]['K'], k)
            kv_cache[layer_idx]['V'] = concat(kv_cache[layer_idx]['V'], v)

            # The attention is where we need K, V from previous positions.
            # Attend to ALL positions using cached K, V
            hidden = attention(q, kv_cache[layer_idx]['K'], kv_cache[layer_idx]['V'])

        logits = model.lm_head(hidden)
        next_token = sample(logits)
        tokens.append(next_token)

    return tokens
```

**Cost breakdown:**
- Prefill (prompt): O(n²) once for n prompt tokens
- Decode (each new token): O(n) where n is current sequence length

The key difference: We only run the model's linear projections (W_q, W_k, W_v) on ONE token per step, not all tokens.

**Single token flow during decode:**

```
Single token
    │
    ▼
Embedding (1 token)
    │
    ▼
Q, K, V projection (1 token) ──► K, V get appended to cache
    │
    ▼
Attention: this 1 token's Q attends to ALL cached K, V
    │
    ▼
MLP (1 token)
    │
    ▼
 ... repeat for each layer ...
    │
    ▼
LM head → logits → probabilities → sample next token
```

The only place where this single token "sees" the rest of the sequence is in the attention step. Everything else (embedding, projections, MLP, layer norms, LM head) operates on just that one token's hidden state.

### Memory Cost of KV Cache

The cache stores K and V tensors for every layer:

```
KV cache size = 2 × num_layers × seq_len × hidden_dim × bytes_per_param
```

For Llama 70B (80 layers, 8192 hidden, fp16):

```
2 × 80 × 8192 × 8192 × 2 bytes ≈ 21 GB per sequence
```

This is why:
- Long context is expensive (memory scales linearly with seq_len)
- Batch size is limited by GPU memory
- Techniques like GQA (Grouped Query Attention) reduce KV cache by sharing K,V across heads

---

## Why Input Tokens and Output Tokens Cost Differently

If you look at API pricing, output tokens are typically **3-5x more expensive** than input tokens:

| Provider | Input (per 1M) | Output (per 1M) | Ratio |
|----------|----------------|-----------------|-------|
| GPT-4o | $2.50 | $10.00 | 4x |
| Claude Sonnet | $3.00 | $15.00 | 5x |
| Gemini 1.5 Pro | $1.25 | $5.00 | 4x |

### Why the Difference?

**The core asymmetry (fundamental to autoregressive generation):**

```
1000 input tokens  = 1 forward pass (parallel)
1000 output tokens = 1000 forward passes (sequential, one per token)
```

This is true whether you use KV cache or not. Each output token depends on all previous tokens, so you can't parallelize generation. You must do one forward pass per output token.

**What KV cache changes:**

Without KV cache:
```
Output token 1:   forward pass on 1001 tokens
Output token 2:   forward pass on 1002 tokens
Output token 100: forward pass on 1100 tokens

Each pass recomputes everything (all tokens through all layers)
```

With KV cache:
```
Output token 1:   forward pass on 1 token (attend to 1000 cached)
Output token 2:   forward pass on 1 token (attend to 1001 cached)
Output token 100: forward pass on 1 token (attend to 1099 cached)

Each pass is cheaper (only new token through projections)
```

**So:**
- The 1:N forward pass ratio is due to autoregressive generation (fundamental)
- KV cache makes each of those N decode passes cheaper, but doesn't reduce N
- Without KV cache, output tokens would be even MORE expensive relative to input

**Input tokens (Prefill):**
- Processed in **parallel** in a single forward pass
- All tokens computed together; GPU utilization is high
- Compute: O(n²) total, but done once
- 1000 input tokens ≈ 1 forward pass

**Output tokens (Decode):**
- Generated **sequentially**, one at a time
- Each token requires a separate forward pass
- Cannot parallelize across tokens (each depends on the previous)
- 100 output tokens = 100 forward passes

```
Prefill:  [tok1, tok2, tok3, ..., tok1000] → single forward pass → logits
Decode:   [tok1001] → forward pass → logits → sample
          [tok1002] → forward pass → logits → sample
          [tok1003] → forward pass → logits → sample
          ... (100 times)
```

---

## Sampling Parameters

During pretraining, the model learns to predict `P(next_token | context)`. At inference, we **sample** from this distribution. Sampling parameters control **how** we sample.

The model outputs **logits** (raw scores for each token). These become probabilities via softmax:

```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```

### Temperature

[Temperature in AI: Interactive Softmax Visualizer](https://claude.ai/public/artifacts/dced227d-1a17-4fd8-a4fe-8797e9a66ba5)

**What it does:** Scales logits before softmax, controlling distribution sharpness.

```python
def apply_temperature(logits, temperature):
    return logits / temperature
```

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

**Effect:**
- `T = 1.0`: Use the learned distribution as-is
- `T < 1.0`: Sharpen (more deterministic); dividing by <1 amplifies gaps between logits
- `T > 1.0`: Flatten (more random); dividing by >1 compresses gaps
- `T → 0`: Greedy decoding (always pick highest probability)
- `T → ∞`: Uniform random sampling

**Example** with logits `[2.0, 1.0, 0.5]`:

| Temperature | P(A) | P(B) | P(C) |
|-------------|------|------|------|
| 0.5 | 0.84 | 0.11 | 0.05 |
| 1.0 | 0.59 | 0.27 | 0.13 |
| 2.0 | 0.43 | 0.33 | 0.24 |

---

### Top-k Sampling

**What it does:** Only consider the k highest-probability tokens; zero out everything else.

```python
def top_k_filter(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)

    filtered = torch.full_like(logits, -float('inf'))
    filtered[top_k_indices] = top_k_logits

    return filtered
```

**Problem:** Fixed k ignores distribution shape. If model is confident (one token dominates), k=50 includes garbage. If uncertain, k=10 excludes valid options.

---

### Top-p (Nucleus Sampling)

**What it does:** Dynamically select the smallest set of tokens whose cumulative probability ≥ p.

```python
def top_p_filter(logits, p):
    probs = softmax(logits)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens until cumsum exceeds p
    mask = cumsum - sorted_probs > p  # Shift to include the crossing token
    sorted_probs[mask] = 0.0

    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_indices, sorted_probs)

    return filtered_probs / filtered_probs.sum()  # Renormalize
```

**Why preferred over top-k:** Adapts to confidence. Certain → few tokens. Uncertain → many tokens.

---

### min_p Sampling

**What it does:** Keep tokens with probability ≥ min_p × max_probability.

```python
def min_p_filter(logits, min_p):
    probs = softmax(logits)
    max_prob = probs.max()

    threshold = min_p * max_prob
    probs[probs < threshold] = 0.0

    return probs / probs.sum()
```

More intuitive than top-p: "Keep anything at least 10% as likely as the best option."

---

### Presence Penalty

**What it does:** Flat penalty for tokens that have appeared in the output.

```python
seen_tokens = set()

for step in range(max_tokens):
    logits = model(context)

    for token_id in seen_tokens:
        logits[token_id] -= presence_penalty  # Flat subtraction

    next_token = sample(softmax(logits))
    seen_tokens.add(next_token)
```

**Behavior:**
- Binary: Token appeared → penalized. Didn't appear → not penalized.
- Same penalty whether token appeared once or 100 times
- Encourages topic diversity (new words)
- Typical range: 0.0 to 2.0

---

### Frequency Penalty

**What it does:** Penalty proportional to token count.

```python
token_counts = defaultdict(int)

for step in range(max_tokens):
    logits = model(context)

    for token_id, count in token_counts.items():
        logits[token_id] -= frequency_penalty * count  # Scaled by count

    next_token = sample(softmax(logits))
    token_counts[next_token] += 1
```

**Behavior:**
- Linear: Token used 5x → 5x the penalty
- More aggressive than presence_penalty for repeated tokens
- Reduces word-level repetition ("the the the")
- Typical range: 0.0 to 2.0

**Comparison:**

```
Token appeared 1x:  presence=0.5 subtracts 0.5,  frequency=0.5 subtracts 0.5
Token appeared 5x:  presence=0.5 subtracts 0.5,  frequency=0.5 subtracts 2.5
```

---

### Repetition Penalty (HuggingFace)

**What it does:** Multiplicative penalty on seen tokens (including prompt).

```python
seen_tokens = set(prompt_token_ids)  # Includes prompt!

for step in range(max_tokens):
    logits = model(context)

    for token_id in seen_tokens:
        if logits[token_id] > 0:
            logits[token_id] /= repetition_penalty
        else:
            logits[token_id] *= repetition_penalty

    next_token = sample(softmax(logits))
    seen_tokens.add(next_token)
```

**Key differences:**
- Multiplicative, not additive
- Applied to prompt tokens too (won't repeat words from question)
- Typical range: 1.0 to 1.5 (1.0 = no effect)
- Handles positive/negative logits differently to always reduce probability

---

### Logit Bias

**What it does:** Manually adjust specific token logits.

```python
# logit_bias: {token_id: bias_value}

for step in range(max_tokens):
    logits = model(context)

    for token_id, bias in logit_bias.items():
        logits[token_id] += bias

    next_token = sample(softmax(logits))
```

**Use cases:**

```python
logit_bias = {50256: -100}  # Ban EOS token (can't stop)
logit_bias = {1234: 5}      # Strongly boost token 1234
logit_bias = {9876: -2}     # Soft discourage
```

---

### Structured Outputs (JSON Mode, Grammar Constraints)

**What it does:** Force the model to output valid JSON, XML, or any grammar-conforming text.

**How it works:** Same technique as logit_bias, but dynamic. At each step, determine which tokens are valid given the current parse state, and mask out everything else.

```python
def constrained_generate(prompt, model, grammar):
    tokens = tokenizer.encode(prompt)
    parser_state = grammar.initial_state()

    for i in range(max_tokens):
        logits = model(tokens)

        # Ask parser: which tokens are valid next?
        valid_token_ids = grammar.get_valid_tokens(parser_state)

        # Mask out invalid tokens
        mask = torch.full_like(logits, -float('inf'))
        mask[valid_token_ids] = 0
        logits = logits + mask

        next_token = sample(softmax(logits))
        tokens.append(next_token)

        # Update parser state
        parser_state = grammar.advance(parser_state, next_token)

        if grammar.is_complete(parser_state):
            break

    return tokenizer.decode(tokens)
```

**Example: JSON object with specific schema**

Schema: `{"name": string, "age": integer}`

```
Generated so far: {"name": "
Valid next tokens: any string characters, or " to close string

Generated so far: {"name": "Alice", "age":
Valid next tokens: digits, whitespace, or minus sign

Generated so far: {"name": "Alice", "age": 25
Valid next tokens: more digits, or } to close object
```

**Implementation approaches:**

1. **Finite state machine:** For simple grammars (JSON, specific formats), precompute valid tokens for each state.
2. **Incremental parsing:** Use a real parser (like a JSON parser) and at each step try appending each possible token to see if it's still parseable.
3. **Token healing aware:** Handle the complication that tokens don't align with characters. The string `"age"` might be one token or multiple.

**Libraries:** Outlines, Guidance, LMQL, llama.cpp grammars

**API implementations:** OpenAI's JSON mode, Anthropic's tool use with schemas. These use similar techniques server-side, possibly combined with fine-tuning to make the model "want" to output valid JSON.

---

### max_tokens / min_tokens

**max_tokens:** Hard limit on generated tokens.

```python
for i in range(max_tokens):
    next_token = sample(logits)
    if next_token == EOS:
        break
```

**min_tokens:** Force at least n tokens before allowing EOS.

```python
for i in range(max_tokens):
    logits = model(context)

    if i < min_tokens:
        logits[EOS_TOKEN_ID] = -float('inf')  # Ban EOS temporarily

    next_token = sample(softmax(logits))
```

---

### stop_sequences

**What it does:** Stop generation when specific strings appear.

```python
output_text = ""

for i in range(max_tokens):
    next_token = sample(logits)
    output_text += tokenizer.decode(next_token)

    for stop_seq in stop_sequences:
        if output_text.endswith(stop_seq):
            return output_text[:-len(stop_seq)]  # Trim stop sequence
```

**Note:** Checked at string level after decoding. Stop sequences can span multiple tokens.

**Why trim?** The stop sequence is a signal, not content. If your stop sequence is `"\n\nHuman:"` (to prevent the model from simulating the next conversation turn), you don't want the response to end with that string visible to the user. The model emitted it, but we strip it from the output.

---

### seed

**What it does:** Fix RNG for reproducible outputs.

```python
torch.manual_seed(seed)
# or
random.seed(seed)

# Now sampling is deterministic given same inputs
```

**Caveats:**
- Only controls sampling randomness
- Same seed + same prompt + same parameters = same output
- GPU non-determinism can still cause variation

---

### Combining Parameters: Order of Operations

```python
def sample_token(logits, config, seen_tokens, token_counts):
    # 1. Repetition penalties (multiplicative)
    if config.repetition_penalty != 1.0:
        for token_id in seen_tokens:
            if logits[token_id] > 0:
                logits[token_id] /= config.repetition_penalty
            else:
                logits[token_id] *= config.repetition_penalty

    # 2. Presence/frequency penalties (additive)
    for token_id in seen_tokens:
        logits[token_id] -= config.presence_penalty
    for token_id, count in token_counts.items():
        logits[token_id] -= config.frequency_penalty * count

    # 3. Logit bias
    for token_id, bias in config.logit_bias.items():
        logits[token_id] += bias

    # 4. Temperature
    logits = logits / config.temperature

    # 5. Top-k filter
    if config.top_k > 0:
        logits = top_k_filter(logits, config.top_k)

    # 6. Top-p filter
    if config.top_p < 1.0:
        probs = top_p_filter(logits, config.top_p)
    else:
        probs = softmax(logits)

    # 7. Sample
    return torch.multinomial(probs, 1)
```

**Common configurations:**
- Deterministic: `temperature=0.001` (not exactly 0 to avoid division issues)
- Balanced: `temperature=0.7, top_p=0.9`
- Creative: `temperature=1.0, top_p=0.95`
- Code: `temperature=0.2` (precision matters)

---

## Decoding Strategies

Beyond sampling, there are different strategies for selecting tokens.

### Greedy Decoding

**What it does:** Always pick the highest probability token. No sampling.

```python
for i in range(max_tokens):
    logits = model(context)
    next_token = torch.argmax(logits)

    if next_token == EOS:
        break
    context.append(next_token)
```

**Properties:**
- Deterministic
- Fast (no sampling overhead)
- Can get stuck in repetitive loops
- Locally optimal ≠ globally optimal

---

### Beam Search

**What it does:** Track k candidate sequences (beams), expand each, keep top k overall.

```python
def beam_search(prompt, model, beam_width, max_tokens):
    # Each beam: (token_sequence, cumulative_log_prob)
    beams = [(prompt, 0.0)]

    for step in range(max_tokens):
        all_candidates = []

        for seq, score in beams:
            logits = model(seq)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Expand with top tokens
            top_tokens = torch.topk(log_probs, beam_width * 2)

            for token, log_prob in zip(top_tokens.indices, top_tokens.values):
                new_seq = seq + [token]
                new_score = score + log_prob
                all_candidates.append((new_seq, new_score))

        # Keep top beam_width candidates
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0][0]  # Best sequence
```

**Parameters:**
- `num_beams`: Parallel hypotheses to track
- `length_penalty`: Adjust score by length (>1 favors longer, <1 favors shorter)
- `no_repeat_ngram_size`: Ban n-grams that already appeared

**When to use:** Translation, summarization, tasks with "correct" answers.

---

### Contrastive Search

**What it does:** Balance probability with diversity; penalize tokens similar to recent context.

```python
def contrastive_search(context, model, alpha, k):
    for step in range(max_tokens):
        logits = model(context)
        top_k_tokens = torch.topk(logits, k).indices

        scores = []
        for token in top_k_tokens:
            # Probability term
            prob_score = (1 - alpha) * softmax(logits)[token]

            # Degeneration penalty: similarity to recent hidden states
            token_hidden = get_hidden_state(context + [token])
            max_sim = max(cosine_similarity(token_hidden, h) for h in recent_hiddens)

            penalty = alpha * max_sim
            scores.append(prob_score - penalty)

        next_token = top_k_tokens[torch.argmax(scores)]
        context.append(next_token)
```

**Key idea:** High probability but similar to context → probably repetitive. Penalize it.

---

### Speculative Decoding

**What it does:** Small "draft" model proposes tokens; large model verifies in parallel.

```python
def speculative_decode(context, draft_model, target_model, gamma=4):
    while not done:
        # Draft model proposes gamma tokens (fast, sequential)
        draft_tokens = []
        draft_probs = []
        for _ in range(gamma):
            p = softmax(draft_model(context + draft_tokens))
            token = sample(p)
            draft_tokens.append(token)
            draft_probs.append(p[token])

        # Target model scores ALL positions in ONE forward pass
        target_logits = target_model(context + draft_tokens)
        target_probs = softmax(target_logits)

        # Accept/reject each draft token
        accepted = []
        for i, (token, q_prob) in enumerate(zip(draft_tokens, draft_probs)):
            p_prob = target_probs[i][token]

            if random() < min(1, p_prob / q_prob):
                accepted.append(token)
            else:
                # Reject: sample from adjusted distribution
                adjusted = max(0, target_probs[i] - draft_probs[i])
                accepted.append(sample(adjusted / adjusted.sum()))
                break

        context.extend(accepted)
```

**Why it works:**
- Draft model is fast (small)
- Target model verifies multiple tokens in ONE forward pass
- Mathematically equivalent to sampling from target model
- Speedup = tokens_accepted / (1 + draft_model_cost)

---

## Connection to Training

### Exposure Bias

During training, the model always sees **ground truth** previous tokens (teacher forcing). At inference, it sees **its own generations**, which may drift from training distribution.

```
Training: P(token_5 | ground_truth_1, ground_truth_2, ground_truth_3, ground_truth_4)
Inference: P(token_5 | generated_1, generated_2, generated_3, generated_4)
```

If the model makes an error at position 2, it's never seen this situation before. Errors compound.

**Mitigations:**
- Lower temperature (stay "on distribution")
- Scheduled sampling during training (mix ground truth and model predictions)
- RLHF (train on model's own outputs)

---

## Prompt Caching (Prefix Caching)

**Idea:** If multiple requests share the same prefix (system prompt), cache the KV states.

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}  # hash(prefix) -> kv_states

    def get_or_compute(self, tokens, model):
        for prefix_len in range(len(tokens), 0, -1):
            prefix = tuple(tokens[:prefix_len])

            if hash(prefix) in self.cache:
                cached_kv = self.cache[hash(prefix)]
                remaining = tokens[prefix_len:]

                new_kv = model.forward_with_cache(remaining, initial_kv=cached_kv)
                return merge(cached_kv, new_kv)

        kv = model.forward(tokens)
        self.cache[hash(tuple(tokens))] = kv
        return kv
```

**Benefits:**
- System prompts: Compute once, reuse across all requests
- Conversations: Previous turns are cached
- Multiple completions: Prompt processed once

---

## Token Healing

**Problem:** Tokenization depends on context. Prompt ending mid-word causes issues.

```
Prompt: "The URL is http"
Tokenized: ["The", " URL", " is", " http"]

If model generates "s" next:
We want: "https" (single token)
We get: " http" + "s" (two tokens, different than " https")
```

**Solution:** Back up to last complete token, constrain first generation to complete it properly.

```python
def heal_tokens(prompt, tokenizer, model):
    tokens = tokenizer.encode(prompt)
    last_token_text = tokenizer.decode([tokens[-1]])

    if not prompt.endswith(last_token_text):
        # Prompt ends mid-token
        tokens = tokens[:-1]
        partial = prompt[len(tokenizer.decode(tokens)):]

        # Constrain first token to start with partial
        logits = model(tokens)
        for tid in range(vocab_size):
            if not tokenizer.decode([tid]).startswith(partial):
                logits[tid] = -float('inf')
```

---

## Logprobs and Perplexity

### Logprobs

At each position, the model outputs logits, which become probabilities via softmax. The **logprob** is just the logarithm of that probability.

```python
logits = model(context)                    # Raw scores
probs = softmax(logits)                    # Probabilities (sum to 1)
logprobs = log(probs)                      # Log probabilities (all negative or zero)
```

**Why use log probabilities?**

Probabilities of sequences multiply. If you want P(sequence), you compute:

```
P("The cat sat") = P("The") × P("cat"|"The") × P("sat"|"The cat")
                 = 0.01 × 0.002 × 0.05
                 = 0.000001
```

These numbers get tiny fast. Multiplying many small numbers causes underflow.

Logs turn multiplication into addition:

```
log P("The cat sat") = log P("The") + log P("cat"|"The") + log P("sat"|"The cat")
                     = -4.6 + -6.2 + -3.0
                     = -13.8
```

Much more numerically stable. This is why we work in log space.

**What logprobs tell you:**

```
Token       Probability    Logprob
"the"       0.85          -0.16      (very confident)
"Paris"     0.30          -1.20      (moderate)
"xylophone" 0.001         -6.90      (very unlikely)
```

- Logprobs are always ≤ 0 (since probabilities are ≤ 1)
- Higher (closer to 0) = more confident
- Lower (more negative) = less confident

**Use cases:**

1. **Confidence estimation:** Low logprobs on a span might indicate hallucination
2. **Sequence scoring:** Compare total logprob of different completions
3. **Perplexity calculation:** (see below)
4. **Calibration analysis:** Are the model's confidences accurate?

---

### Perplexity

Perplexity measures how "surprised" the model is by a text. Lower = less surprised = better model.

**Concrete example:**

Given the sentence "The capital of France is Paris", we ask at each position: "What probability did the model assign to the token that actually came next?"

```
Step 1: Model sees: <start>
        What prob does it assign to "The"?        → log P("The")

Step 2: Model sees: "The"
        What prob does it assign to "capital"?    → log P("capital" | "The")

Step 3: Model sees: "The capital"
        What prob does it assign to "of"?         → log P("of" | "The capital")

Step 4: Model sees: "The capital of"
        What prob does it assign to "France"?     → log P("France" | "The capital of")

Step 5: Model sees: "The capital of France"
        What prob does it assign to "is"?         → log P("is" | "The capital of France")

Step 6: Model sees: "The capital of France is"
        What prob does it assign to "Paris"?      → log P("Paris" | "The capital of France is")

Sum them all → total log probability of the sequence
Average by N → average log probability
Negate and exponentiate → perplexity
```

You're asking at each position: "Given everything so far, how surprised was the model by what actually came next?"

**Important:** Perplexity uses the raw model probabilities, not any sampling modifications. No temperature, no top-k, no top-p. You're not generating; you're evaluating how well the model predicts existing text.

**Building up the formula:**

**Step 1: Probability of the sequence**

Given tokens [t₁, t₂, …, tₙ], the model assigns:

```
P(sequence) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂) × ... × P(tₙ|t₁,...,tₙ₋₁)
```

**Step 2: Log probability (for numerical stability)**

```
log P(sequence) = Σ log P(tᵢ | t₁, ..., tᵢ₋₁)
```

This is just the sum of all the logprobs.

**Step 3: Average log probability**

Different sequences have different lengths. To compare fairly, normalize by length:

```
Average log prob = (1/N) × Σ log P(tᵢ | context)
```

**Step 4: Negate it**

Log probabilities are negative (since P ≤ 1). By convention, we want "higher = worse", so we negate:

```
Negative average log prob = -(1/N) × Σ log P(tᵢ | context)
```

This is also called **cross-entropy loss** (what we minimize during training).

**Step 5: Exponentiate**

The negative average log prob is in log space. To get back to an interpretable scale:

```
Perplexity = exp( -(1/N) × Σ log P(tᵢ | context) )
```

**The complete formula:**

```
PPL = exp( -1/N × Σᵢ log P(tᵢ | t₁, ..., tᵢ₋₁) )
```

Or equivalently:

```
PPL = ( ∏ᵢ P(tᵢ | context) ) ^ (-1/N)
```

This is the **geometric mean of the inverse probabilities**.

---

### The Intuition: "Branching Factor"

Here's the key insight that makes perplexity intuitive:

**If the model were uniformly uncertain between K options at every position, the perplexity would be K.**

Example: Imagine a model that always assigns equal probability to exactly 10 tokens:

```
P(each of 10 tokens) = 0.1
log P = log(0.1) = -2.3

Average log prob = -2.3
Negative avg log prob = 2.3
Perplexity = exp(2.3) = 10
```

So perplexity answers: **"On average, how many tokens is the model choosing between?"**

**Interpreting perplexity:**

| PPL | Interpretation |
|-----|----------------|
| 1 | Perfect prediction (model always 100% confident and correct) |
| 10 | Like choosing between 10 equally likely options |
| 100 | Like choosing between 100 equally likely options |
| 50,000 | Random guessing over a 50K vocabulary |

**Real-world values:**

- GPT-2 on Wikipedia: ~30 perplexity
- Good language models on typical text: 10-30
- On highly predictable text (code, formulaic writing): lower
- On unpredictable text (poetry, nonsense): higher

---

### Computing Perplexity in Practice

```python
def compute_perplexity(model, tokenizer, text):
    tokens = tokenizer.encode(text)

    total_log_prob = 0.0
    num_tokens = 0

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]

        logits = model(context)
        log_probs = log_softmax(logits, dim=-1)

        # Log prob of the actual next token
        token_log_prob = log_probs[target]
        total_log_prob += token_log_prob
        num_tokens += 1

    avg_log_prob = total_log_prob / num_tokens
    perplexity = torch.exp(-avg_log_prob)

    return perplexity
```

**Note:** In practice, you do this in batches using the parallel forward pass, grabbing the logprob at each position for the token that actually appeared at the next position.

---

### Perplexity vs Loss

Cross-entropy loss (what we train on) is:

```
Loss = -(1/N) × Σ log P(tᵢ | context)
```

Perplexity is:

```
PPL = exp(Loss)
```

They're directly related. Minimizing loss = minimizing perplexity. But perplexity has that nice "branching factor" interpretation, while loss is just a number.

| Loss | Perplexity |
|------|------------|
| 2.3 | 10 |
| 3.0 | 20 |
| 3.9 | 50 |
| 4.6 | 100 |

---

### What is Perplexity Used For?

**1. Comparing models**

The main use case. Given two models, which one is better at predicting text?

```
Model A on Wikipedia test set: PPL = 25
Model B on Wikipedia test set: PPL = 18

Model B is better (less surprised by the text).
```

This is how papers report progress. "We achieved state-of-the-art perplexity on WikiText-103."

**2. Tracking training progress**

During pretraining, perplexity on a held-out validation set tells you if the model is improving.

```
Epoch 1: PPL = 150
Epoch 2: PPL = 80
Epoch 3: PPL = 45
Epoch 4: PPL = 44  ← maybe starting to overfit
```

**3. Evaluating domain fit**

A model trained on code will have low perplexity on code, high perplexity on legal documents.

```
Model trained on code:
  - Perplexity on Python files: 12
  - Perplexity on legal contracts: 85

This model "understands" code much better than legal text.
```

**4. Detecting out-of-distribution text**

If the model is very surprised (high perplexity), the text might be unusual, corrupted, or from a domain the model hasn't seen.

**5. Sanity checking fine-tuning**

After fine-tuning, perplexity on the original domain shouldn't increase dramatically (catastrophic forgetting), and perplexity on the new domain should decrease.

**Limitations:**

- Perplexity doesn't measure *usefulness* or *correctness*, just prediction accuracy
- A model can have low perplexity but still hallucinate facts
- Different tokenizers make perplexity not directly comparable across model families

---

## Streaming

Most LLM APIs support **streaming**, where tokens are sent to the client as they're generated rather than waiting for the complete response.

**How it works:**

```python
# Non-streaming: wait for everything
response = model.generate(prompt, max_tokens=500)  # Blocks until done
print(response)  # See all 500 tokens at once

# Streaming: receive tokens incrementally
for token in model.generate_stream(prompt, max_tokens=500):
    print(token, end='', flush=True)  # See each token as it's generated
```

**Under the hood:**

The server uses Server-Sent Events (SSE) or similar to push each token:

```
Client sends: POST /generate {"prompt": "Hello"}

Server sends: data: {"token": "Hi"}
Server sends: data: {"token": " there"}
Server sends: data: {"token": "!"}
Server sends: data: {"token": " How"}
...
Server sends: data: [DONE]
```

**Why first token is slower:**

```
Non-streaming timeline:
[====== Prefill ======][== Decode tok1 ==][== tok2 ==][== tok3 ==]...[== tokN ==][Return everything]

Streaming timeline:
[====== Prefill ======][tok1 → send][tok2 → send][tok3 → send]...[tokN → send]
```

The user sees nothing during prefill (processing the prompt), then tokens appear rapidly. This is called **time-to-first-token (TTFT)** latency.

For long prompts, TTFT can be significant. For short prompts, it's fast and tokens seem to flow immediately.

**Benefits of streaming:**
- Better UX (user sees progress immediately)
- Can abort early if response is going wrong
- Lower perceived latency

**Downsides:**
- Slightly more complex client code
- Harder to do full-response validation before showing user
- Can't easily "undo" shown tokens if you wanted to re-rank

---

## Context Window: Limits and Overflow

### What is the context window?

The maximum number of tokens the model can process at once. This includes both the input (prompt) and output (generation).

```
Context window = 8192 tokens

Prompt: 7000 tokens
Available for generation: 8192 - 7000 = 1192 tokens max
```

### What happens when you exceed it?

**Option 1: Error**
API returns an error: "Context length exceeded."

**Option 2: Truncation**
Oldest tokens are dropped:

```
Original: [tok1, tok2, tok3, ..., tok8000, tok8001, ..., tok9000]
Truncated: [tok1808, tok1809, ..., tok9000]  # Keep most recent 8192
```

Or sometimes the middle is dropped (keep beginning + end).

**Option 3: Sliding window**
Some models (like Mistral) have sliding window attention: each token only attends to the last N tokens, allowing "infinite" length but losing distant context.

### Why can't we just use longer context at inference?

The model has **positional encodings** that tell it "where" each token is. These were only trained up to the maximum training length.

**Depends on the encoding scheme:**

**Absolute positional embeddings (GPT-2, original transformer):**

```python
position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
# Literally no embedding exists for positions beyond max_seq_len
```

Can't extrapolate at all.

**RoPE (Rotary Position Embeddings) - Llama, Mistral:**

Positions encoded as rotation angles, computed dynamically. Technically *can* compute any position, but the model never saw those angles during training. Attention patterns break down, performance degrades.

**ALiBi (Attention with Linear Biases):**

Designed to extrapolate better. Adds a linear penalty based on token distance. Generalizes further, but still degrades eventually.

### How people extend context length in practice

**1. Position interpolation**

Scale positions to fit the trained range:

```
Trained on: 4K context
Want: 8K context

Position 8000 → mapped to position 4000
Position 4000 → mapped to position 2000
```

Model sees familiar positions, just "compressed." Works reasonably well with some fine-tuning.

**2. NTK-aware scaling / YaRN**

RoPE encodes position using multiple frequency components. High frequencies capture local patterns, low frequencies capture global position.

Naive interpolation breaks high-frequency patterns. NTK-aware methods scale different frequencies differently, preserving local attention while extending range.

**3. Fine-tune on longer sequences**

Take a 4K model, continue training on 16K examples. The model learns the new position encodings. Requires compute and memory (longer sequences = bigger KV cache).

**4. Sliding window attention**

Each token only attends to the last W tokens (e.g., 4096). Distant tokens are "forgotten."

```
Token at position 10000 attends to positions [5904, ..., 10000]
```

Enables infinite generation, but loses long-range dependencies.

**5. Sparse attention patterns**

Attend to nearby tokens + some distant "landmark" tokens. Approximates full attention with lower cost.

**The bottom line:**

Context extension is possible but involves tradeoffs. The model's "sense of distance" was calibrated on training lengths, and extrapolating requires either tricks (interpolation) or additional training.

---

## Practical Tips

1. **Start with defaults**, adjust based on output quality
2. **Temperature and top_p are redundant**; tune one, leave other at default
3. **Factual tasks:** Low temperature (0.1-0.3)
4. **Creative tasks:** Higher temperature (0.7-1.0) with top_p ~0.9
5. **Repetition issues?** Try frequency_penalty before lowering temperature
6. **Need diversity?** Multiple samples at moderate temperature > one sample at high temperature
7. **Debugging:** Request logprobs to understand model confidence
