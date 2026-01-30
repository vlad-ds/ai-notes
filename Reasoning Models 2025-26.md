# Reasoning Models 2025-26

# How Reasoning Models Actually Work (2024-2025)

The previous version of this section gave a surface-level overview of what different labs offer in their APIs. That's not what we're here for. We want to understand *how* these models actually learn to reason; what algorithms run, what the training pipeline looks like, what the hyperparameters are.

The good news: Chinese labs (DeepSeek, Qwen/Alibaba) have been remarkably open about their methods. We can see exactly what they do. The bad news: OpenAI, Anthropic, and Google haven't shared their techniques in the same detail. So we'll focus on what's actually documented.

---

## The Core Algorithmic Innovation: GRPO

GRPO (Group Relative Policy Optimization) is the algorithm that powers DeepSeek R1 and many other reasoning models. It was introduced in the DeepSeekMath paper and became the foundation for modern reasoning model training.

### The Problem GRPO Solves

Remember PPO from Part 2? It requires a **value function** to compute advantages. The value function is another neural network that predicts "how much reward should I expect from this state?" Training it is tricky:

1. It adds memory overhead (another full model to store)
2. It can become stale as the policy changes
3. Getting it wrong destabilizes training
4. You need to keep it synchronized with the policy

GRPO's insight: **we can compute advantages without a value function**.

### The GRPO Algorithm

The idea is beautifully simple. For each prompt, we generate multiple responses (a "group"). We score each response. Then we use the group's statistics as our baseline.

```python
def grpo_training_step(policy, ref_model, prompt, reward_fn, num_samples=64):
    """
    GRPO: Group Relative Policy Optimization

    Key insight: instead of learning a value function to estimate
    expected reward, just generate multiple responses and use
    the mean as your baseline.
    """
    # Step 1: Generate a GROUP of responses for this prompt
    # This is where "Group" in GRPO comes from
    responses = []
    for _ in range(num_samples):
        response = policy.generate(prompt, temperature=1.0)
        responses.append(response)

    # Step 2: Score each response
    # For R1-Zero, this is just: did you get the right answer?
    rewards = [reward_fn(prompt, r) for r in responses]

    # Step 3: Compute group statistics
    mean_reward = sum(rewards) / len(rewards)
    std_reward = compute_std(rewards) + 1e-8  # avoid div by zero

    # Step 4: Normalize advantages using group statistics
    # This is the key innovation - no value function needed!
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Step 5: Compute policy gradient with clipping (like PPO)
    total_loss = 0.0
    for response, advantage in zip(responses, advantages):
        # Log probabilities under current policy and reference
        log_prob_policy = compute_log_prob(policy, prompt, response)
        log_prob_ref = compute_log_prob(ref_model, prompt, response)

        # Importance ratio: how much more likely is this under new policy?
        # Note: in GRPO, "old" model is the policy from start of this step
        log_prob_old = log_prob_policy.detach()  # stop gradient
        ratio = torch.exp(log_prob_policy - log_prob_old)

        # PPO-style clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

        # Take minimum (conservative update)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

        # KL penalty to stay close to reference model
        kl_penalty = log_prob_policy - log_prob_ref

        total_loss += policy_loss + kl_coef * kl_penalty

    return total_loss / num_samples
```

Let me trace through a concrete example to build intuition.

### Concrete Example: Learning to Solve Math

Suppose we have a prompt: "What is 17 × 23?"

**Step 1: Generate 64 responses**

The model generates 64 different attempts:
- Response 1: "Let me calculate... 17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391" ✓
- Response 2: "17 × 23 = 401" ✗
- Response 3: "I'll break this down: 17 × 23 = 391" ✓
- ... (61 more responses)

Say 40 out of 64 get it right (reward = 1) and 24 get it wrong (reward = 0).

**Step 2: Compute group statistics**

```python
rewards = [1, 0, 1, 1, 0, ...]  # 40 ones, 24 zeros
mean_reward = 40/64 = 0.625
std_reward = sqrt(40/64 * (1-0.625)^2 + 24/64 * (0-0.625)^2)
           = sqrt(0.625 * 0.140625 + 0.375 * 0.390625)
           = sqrt(0.234375) ≈ 0.484
```

**Step 3: Compute advantages**

```python
# For a correct response (reward = 1):
advantage = (1 - 0.625) / 0.484 = 0.775  # Positive: reinforce this!

# For an incorrect response (reward = 0):
advantage = (0 - 0.625) / 0.484 = -1.291  # Negative: discourage this!
```

**Why this works:**

The mean reward (0.625) adapts to the prompt difficulty. For this math problem, the model gets it right 62.5% of the time. A correct response is "better than average" and gets positive advantage. An incorrect response is "worse than average" and gets negative advantage.

Now consider a much harder problem where only 5/64 responses are correct:
- mean_reward = 0.078
- For a correct response: advantage = (1 - 0.078) / std ≈ +2.5 (huge positive signal!)
- For an incorrect response: advantage = (0 - 0.078) / std ≈ -0.2 (small negative signal)

**The baseline naturally adapts to difficulty.** On hard problems, even a slightly-correct response gets strong positive reinforcement. On easy problems, you need to be reliably correct to get positive signal. No value function needed to estimate this; it falls out of the group statistics.

### Memory Savings: From 4 Models to 2

Here's where GRPO really shines. Compare the memory requirements:

**Standard PPO (like InstructGPT):**
1. Policy model (the LLM being trained)
2. Reference model (frozen copy for KL penalty)
3. Reward model (scores responses)
4. Value model (estimates expected reward)

That's 4 full models in GPU memory.

**GRPO:**
1. Policy model
2. Reference model

That's 2 models. We eliminated the value model entirely (GRPO's main contribution), and for R1-Zero, they went further...

**DeepSeek R1-Zero (pure RL):**

DeepSeek R1-Zero doesn't use a neural reward model at all. The reward function is just regex/string matching:

```python
def r1_zero_reward(response, ground_truth):
    """
    DeepSeek R1-Zero's reward function.

    No neural network! Just string matching.
    """
    # Extract final answer (they use \boxed{} format)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if not match:
        return 0.0  # No answer in correct format

    extracted = match.group(1).strip()

    # Binary correctness
    if extracted == ground_truth:
        return 1.0
    else:
        return 0.0
```

This is shockingly simple. No learned reward model, no human feedback loop, no preference training. Just: did you get the right answer?

And yet, this produced a model that goes from 15.6% on AIME 2024 to 71.0%.; spontaneously developing chain-of-thought reasoning, backtracking, and "aha moments" along the way.

### DeepSeek R1's Actual Hyperparameters

From the paper, here's what they actually used:

```python
# DeepSeek R1-Zero training config
config = {
    "num_samples_per_prompt": 64,      # Generate 64 responses per prompt
    "epsilon": 10.0,                    # Clipping range (much larger than PPO's 0.2!)
    "kl_coefficient": 0.001,            # Very small; allow divergence from reference
    "learning_rate": 1e-6,              # Standard fine-tuning LR
    "max_response_length": 32768,       # Allow VERY long reasoning chains
    "batch_size": 1024,                 # Number of prompts per batch
}
```

Notice the epsilon = 10.0. Standard PPO uses 0.2. This means DeepSeek allows much larger policy updates. The theory is that for reasoning tasks, you want to explore more aggressively; the model needs to discover new reasoning strategies, not just refine existing ones.

The max_response_length of 32K tokens is also crucial. Reasoning chains can get very long. If you cap at 1K tokens (like some early experiments), the model can't develop extended thinking.

---

## GRPO Variants: The Algorithm Zoo

After GRPO's success, researchers identified limitations and proposed fixes. Let me explain the main variants and why they matter.

### The Token vs Sequence Problem

GRPO applies importance sampling **per-token**. But rewards are **per-sequence**. This mismatch causes problems.

Consider this: your model generates a 100-token response. Token 47 is a rare but crucial insight ("Wait, I should try factoring!"). Under GRPO:

```python
# Per-token importance ratio
for t in range(len(tokens)):
    ratio_t = exp(log_prob_new[t] - log_prob_old[t])
    clipped_ratio_t = clamp(ratio_t, 1-eps, 1+eps)
    # If token 47 is very rare under old policy but common under new,
    # ratio_47 might be huge (say, 50), and gets clipped to 1+eps
```

The problem: that crucial "aha moment" token gets its gradient clipped to nearly zero, even though it was the key to getting the answer right. The model can't learn to generate these insightful pivots.

### GSPO: Group Sequence Policy Optimization

Qwen3 uses GSPO, which fixes this by computing importance ratios at the **sequence level**:

```python
def gspo_loss(policy, ref_model, prompt, response, advantage, epsilon=0.2):
    """
    GSPO: Compute importance ratio over the ENTIRE sequence,
    not per-token.

    This prevents "token dropping" where rare but important
    tokens get their gradients clipped to zero.
    """
    # Sum log probs over all tokens
    log_prob_new_seq = sum(log_prob_new[t] for t in range(len(response)))
    log_prob_old_seq = sum(log_prob_old[t] for t in range(len(response)))

    # Single ratio for the whole sequence
    ratio_seq = torch.exp(log_prob_new_seq - log_prob_old_seq)

    # Clip the sequence-level ratio
    clipped_ratio = torch.clamp(ratio_seq, 1 - epsilon, 1 + epsilon)

    # Loss applies to entire sequence
    loss = -torch.min(ratio_seq * advantage, clipped_ratio * advantage)

    return loss
```

The difference is subtle but important:
- **GRPO**: Each token has its own clipping. Rare tokens get clipped independently.
- **GSPO**: The whole sequence has one clipping threshold. If the sequence as a whole isn't too different from the old policy, all tokens (including rare ones) get gradient signal.

This is especially important for:
1. Long sequences (where per-token clipping compounds)
2. MoE models (where expert routing can cause high variance in token probabilities)
3. Reasoning models (where pivotal insight tokens are rare but crucial)

### DAPO: Decoupled Advantage Policy Optimization

DAPO addresses a different problem: exploration vs exploitation.

Standard GRPO/PPO clips symmetrically: you can't increase or decrease a token's probability by more than epsilon. But for reasoning, we might want **asymmetric** clipping:
- For surprising, exploratory tokens (low probability under old policy): allow larger increases
- For common, exploitative tokens: keep tight clipping

```python
def dapo_clipping(ratio, advantage, eps_low=0.2, eps_high=0.5):
    """
    DAPO: Asymmetric clipping.

    Higher upper bound (eps_high) allows the model to
    "discover" new tokens (reflective behaviors, aha moments).
    """
    if advantage > 0:
        # For positive advantage, use higher upper clip
        # This allows more exploration
        clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    else:
        # For negative advantage, use standard clipping
        clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_low)

    return clipped
```

DAPO also introduces:
- **Dynamic sampling**: Remove flat-reward samples (all correct or all wrong) from batches
- **Per-token loss** instead of per-response (similar to how GRPO works internally)
- **Length penalty management** to prevent reward hacking via verbose responses

### Dr.GRPO: Debiased Relative GRPO

Dr.GRPO addresses a more subtle issue: GRPO's advantage normalization introduces bias.

When you normalize advantages by dividing by std, you're assuming the rewards are roughly Gaussian. But for binary rewards (correct/incorrect), they're not; they're Bernoulli. This creates bias in the gradient estimates.

Dr.GRPO removes these heuristic normalizations for more stable, unbiased updates:

```python
def dr_grpo_advantage(rewards):
    """
    Dr.GRPO: Remove biased normalizations.

    Standard GRPO: advantage = (r - mean) / std
    Dr.GRPO: advantage = r - mean (or uses importance-weighted mean)
    """
    # Use importance-weighted mean as baseline
    # instead of simple arithmetic mean
    mean = weighted_mean(rewards, importance_weights)

    # Don't divide by std; it introduces bias
    advantages = [r - mean for r in rewards]

    return advantages
```

### CISPO: Clipped Importance Sampling Policy Optimization

CISPO takes a different approach: instead of clipping the objective, clip the importance weights directly.

```python
def cispo_loss(log_prob_new, log_prob_old, advantage, clip_weight=5.0):
    """
    CISPO: Clip the importance WEIGHT, not the ratio in the objective.

    Key difference: clipped tokens still contribute gradient,
    just with a capped weight. GRPO gives zero gradient to
    heavily-clipped tokens.
    """
    ratio = torch.exp(log_prob_new - log_prob_old)

    # Clip the weight directly
    clipped_weight = torch.clamp(ratio, max=clip_weight)

    # Loss uses clipped weight
    # Note: still gets SOME gradient even when weight is clipped
    loss = -clipped_weight * advantage * log_prob_new

    return loss
```

The key difference: in GRPO, when a token hits the clip boundary, its gradient goes to zero. In CISPO, it still gets gradient, just with a capped weight. This can be more stable for training.

### Which Variant to Use?

Here's a practical guide:

| Situation | Recommended |
|---|---|
| General reasoning training | GSPO (sequence-level clipping) |
| Need exploration/discovery | DAPO (asymmetric clipping) |
| Training instability | Dr.GRPO (unbiased estimates) |
| MoE models | GSPO (handles expert routing variance) |
| Very long sequences | GSPO (no per-token compound clipping) |

Qwen3 uses GSPO. DeepSeek R1 uses vanilla GRPO with very large epsilon (10.0), which has similar effect to GSPO for allowing exploration.

---

## Qwen3's Training Pipeline: The Full Picture

Qwen3 represents the state of the art in open reasoning models (as of late 2025). Let me walk through their complete training pipeline because it shows how all these pieces fit together.

### The Four Stages

```plain text
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Long-CoT Cold Start                                        │
│  Goal: Teach the format of chain-of-thought reasoning                │
│  Method: SFT on curated reasoning examples                           │
│  Duration: Minimal (avoid limiting RL potential)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Reasoning RL                                               │
│  Goal: Learn to actually reason well                                 │
│  Method: GRPO/GSPO on verifiable tasks (math, code)                 │
│  Duration: 170+ RL steps                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Thinking Mode Fusion                                       │
│  Goal: Support both thinking and non-thinking modes                  │
│  Method: SFT mixing thinking + non-thinking data                     │
│  Innovation: /think and /no_think chat template flags                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: General RL                                                 │
│  Goal: Polish everything (instruction following, safety, agents)     │
│  Method: Multi-task RL with 20+ reward types                         │
│  Duration: Final alignment                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Long-CoT Cold Start

The goal here is to teach the model the *format* of reasoning, not to make it good at reasoning yet. That comes from RL.

```python
def stage1_cold_start():
    """
    Stage 1: Teach the model what chain-of-thought looks like.

    Key insights from Qwen3 paper:
    1. Filter OUT queries that Qwen2.5-72B can answer WITHOUT CoT
       (these might teach superficial pattern matching, not real reasoning)
    2. Generate N candidates with QwQ-32B (a reasoning model)
    3. Keep only correct answers
    4. Human verify when Pass@N fails
    5. Filter out bad patterns: repetition, guessing, inconsistencies
    """
    # Curate dataset: math, code, logic, STEM with verifiable answers
    queries = collect_reasoning_queries()

    # Filter: remove "easy" queries that don't need reasoning
    hard_queries = []
    for q in queries:
        # If base model (no CoT) can solve it, skip it
        if not qwen2_5_72b_can_solve_without_cot(q):
            hard_queries.append(q)

    # Generate reasoning chains using existing reasoning model
    training_data = []
    for q in hard_queries:
        candidates = [qwq_32b.generate(q) for _ in range(N)]
        correct_ones = [c for c in candidates if is_correct(c, q.answer)]

        if correct_ones:
            # Filter for quality (no repetition, guessing, etc.)
            good_ones = filter_quality(correct_ones)
            training_data.extend([(q, c) for c in good_ones])
        else:
            # Pass@N failed; send to human annotators
            human_written = get_human_annotation(q)
            training_data.append((q, human_written))

    # Minimal SFT; don't overtrain!
    # The goal is just format, not capability
    model = sft(base_model, training_data, epochs=1)

    return model
```

The key insight: **minimal training**. If you SFT too much, the model "memorizes" reasoning patterns and becomes harder to improve with RL. You just want it to understand the format: `<think>reasoning here</think><answer>answer here</answer>`.

### Stage 2: Reasoning RL

This is where the magic happens. The model learns to actually reason well.

```python
def stage2_reasoning_rl(model):
    """
    Stage 2: The core reasoning improvement stage.

    Qwen3 specifics:
    - 3,995 query-verifier pairs (disjoint from Stage 1)
    - Criteria: learnable, challenging, broad sub-domains
    - GRPO/GSPO with large batch size, high rollouts
    - Off-policy training for sample efficiency
    - Entropy control for stable training
    """
    # Collect training problems
    # These must be: verifiable, challenging, diverse
    problems = collect_rl_problems(n=3995)

    # GRPO/GSPO training
    for step in range(170):  # Qwen3 used ~170 RL steps
        for batch in sample_batches(problems, batch_size=1024):
            loss = 0
            for prompt in batch:
                # Generate multiple responses (high rollouts for variance reduction)
                responses = [model.generate(prompt) for _ in range(64)]

                # Verify correctness
                rewards = [verify(r, prompt.answer) for r in responses]

                # GSPO update (sequence-level clipping)
                loss += gspo_loss(model, ref_model, prompt, responses, rewards)

            # Update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Monitor: AIME'24 score
        # Qwen3-235B-A22B: 70.1 → 85.1 over 170 steps
        eval_score = evaluate_aime24(model)
        print(f"Step {step}: AIME'24 = {eval_score}")

    return model
```

**Off-policy training:** Qwen3 uses off-policy RL for sample efficiency. Instead of generating fresh responses every step, they maintain a buffer of recent generations and sample from it. This reduces the compute cost of generation (which is expensive for large models).

**Entropy control:** They monitor the entropy of the policy's output distribution. If entropy collapses (model becomes too deterministic), training becomes unstable. They use entropy bonuses or early stopping to maintain diversity.

### Stage 3: Thinking Mode Fusion

This is a Qwen3 innovation: the model supports BOTH thinking and non-thinking modes in a single checkpoint.

```python
def stage3_thinking_mode_fusion(model):
    """
    Stage 3: Fuse thinking and non-thinking capabilities.

    The model learns:
    - /think flag → Use chain-of-thought (default)
    - /no_think flag → Skip reasoning, answer directly

    Emergent capability: Thinking Budget
    The model learns to stop thinking early if user specifies a limit.
    """
    # Chat template design
    template = """
    User: {prompt} {mode_flag}

    Assistant: {response}

    Where:
    - mode_flag = "/think" or "/no_think" or empty (default: /think)
    - For /think: response = <think>reasoning</think>answer
    - For /no_think: response = <think></think>answer (empty think block)
    """

    # Create training data
    thinking_data = []   # From Stage 2 rejection sampling
    non_thinking_data = []  # Diverse general assistant data

    # Mix the two types
    # The model learns to switch modes based on the flag
    mixed_data = interleave(thinking_data, non_thinking_data)

    # SFT on mixed data
    model = sft(model, mixed_data)

    return model
```

**The emergent Thinking Budget capability:**

A surprising discovery: after Stage 3 training, the model develops the ability to stop thinking early if you specify a token budget. If you say "think for at most 1000 tokens", the model will:
1. Start reasoning normally
2. Detect when it's approaching the budget
3. Insert: "Considering the limited thinking budget, I'll give my current best answer."
4. Close the `</think>` tag and generate an answer

This wasn't explicitly trained; it emerged from the mode fusion training. The model learned that the `<think>` block is optional and can be truncated.

### Stage 4: General RL

Finally, a broad RL phase that covers everything else.

```python
def stage4_general_rl(model):
    """
    Stage 4: General alignment and capability polish.

    20+ task types with custom rewards:
    - Instruction following
    - Format compliance (/think vs /no_think tags)
    - Preference alignment (helpfulness, harmlessness)
    - Agent capabilities
    - RAG (retrieval-augmented generation)

    Three reward types:
    1. Rule-based: Correctness, format matching
    2. Model-based with reference: Compare to reference answer
    3. Model-based without reference: Trained reward model
    """
    reward_functions = {
        "instruction_following": rule_based_instruction_reward,
        "format_compliance": check_think_tags,
        "helpfulness": qwen2_5_72b_preference_model,
        "harmlessness": safety_classifier,
        "code_execution": run_tests,
        "math_correctness": verify_math,
        # ... 14+ more
    }

    for task_type, reward_fn in reward_functions.items():
        problems = get_problems(task_type)

        for epoch in range(num_epochs[task_type]):
            for batch in problems:
                responses = [model.generate(p) for p in batch]
                rewards = [reward_fn(r, p) for r, p in zip(responses, batch)]

                loss = gspo_loss(model, ref_model, batch, responses, rewards)
                loss.backward()
                optimizer.step()

    return model
```

The reward types are worth understanding:

**1. Rule-based (correctness, format):**
Simple regex/string matching. Did the model follow instructions? Is the answer in the right format? Does code pass tests?

**2. Model-based with reference (Qwen2.5-72B as judge):**
For tasks without clear correct answers (summarization, explanation), use a stronger model to compare the response against a reference answer.

**3. Model-based without reference (trained reward model):**
For subjective quality judgments (helpfulness, engagement), train a reward model on human preferences, like traditional RLHF.

---

## Strong-to-Weak Distillation

Here's a remarkable finding from Qwen3: you can train small models (0.6B to 14B) to reason almost as well as large models; and it's 10x cheaper than running the full RL pipeline.

```python
def distill_reasoning(teacher_model, student_model):
    """
    Distill reasoning capabilities from large to small model.

    Two approaches:
    1. Off-policy: Student imitates teacher outputs (SFT)
    2. On-policy: Student generates, minimize KL with teacher logits

    Efficiency: 1/10 GPU hours vs full 4-stage RL
    Performance: Often BETTER than RL for small models!
    """
    # Off-policy distillation (simpler)
    teacher_outputs = []
    for problem in problems:
        # Generate reasoning chains from teacher
        response = teacher_model.generate(problem, temperature=0.7)
        if is_correct(response, problem.answer):
            teacher_outputs.append((problem, response))

    # SFT student on teacher outputs
    student = sft(student_model, teacher_outputs)

    # On-policy refinement (optional, better results)
    for step in range(distill_steps):
        for batch in problems:
            # Student generates
            student_responses = [student.generate(p) for p in batch]

            # Get teacher's distribution (not just output)
            with torch.no_grad():
                teacher_logits = teacher_model.get_logits(batch)

            # Minimize KL divergence: student should match teacher's distribution
            student_logits = student.get_logits(batch)
            kl_loss = kl_divergence(student_logits, teacher_logits)

            kl_loss.backward()
            optimizer.step()

    return student
```

**The numbers from Qwen3 paper (Table 21):**

| Model | Training Method | AIME'24 | MATH500 | GPU Hours |
|---|---|---|---|---|
| Qwen3-14B | Full 4-stage RL | 67.6 | 94.8 | X |
| Qwen3-14B | Distillation | **74.4** | **97.0** | **X/10** |

Yes, you read that right: distillation is BETTER and 10x CHEAPER for small models.

Why? The full RL pipeline is designed for frontier-scale models. Smaller models can't explore effectively; they don't have enough capacity to discover novel reasoning strategies. But they CAN imitate reasoning patterns they see. Distillation gives them the patterns directly.

---

## DeepSeek R1: The Complete Pipeline

Let me also document DeepSeek's full pipeline since it's well-documented and influential.

### R1-Zero: Pure RL (No Demonstrations)

```python
def train_r1_zero(base_model):
    """
    DeepSeek R1-Zero: Pure RL from scratch.

    No chain-of-thought demonstrations.
    No reward model.
    Just: did you get the right answer?

    And yet... reasoning emerges spontaneously.
    """
    model = base_model  # DeepSeek-V3 base
    ref_model = base_model.copy()

    # Simple binary reward
    def reward(response, correct_answer):
        extracted = extract_boxed_answer(response)
        return 1.0 if extracted == correct_answer else 0.0

    # GRPO training with high exploration
    for step in range(total_steps):
        for batch in math_and_code_problems:
            responses = [model.generate(p, max_length=32768) for p in batch]
            rewards = [reward(r, p.answer) for r, p in zip(responses, batch)]

            # GRPO update
            loss = grpo_loss(
                model, ref_model,
                batch, responses, rewards,
                epsilon=10.0,  # Very high; allow exploration
                kl_coef=0.001  # Very low; allow divergence
            )
            loss.backward()
            optimizer.step()

    return model
```

**What emerges:**
- Long reasoning chains (thousands of tokens)
- Backtracking ("Wait, that's not right. Let me try again...")
- Self-verification ("Let me check: 17 × 23 = 391. Yes, that's correct.")
- "Aha moments" ("Oh! I should factor this differently...")

None of this was programmed or demonstrated. The model discovered these strategies because they help get correct answers.

### Full R1: The Four-Stage Pipeline

```python
def train_full_r1(base_model):
    """
    Full DeepSeek R1 pipeline.
    """
    # Stage 1: Cold start SFT (~1000 examples)
    # Just to get the format right
    cold_start_data = collect_high_quality_reasoning_chains(n=1000)
    model = sft(base_model, cold_start_data, epochs=1)

    # Stage 2: Large-scale reasoning RL
    # Tens of thousands of math and code problems
    model = grpo_training(
        model,
        problems=reasoning_problems,  # ~50K problems
        reward_fn=correctness_reward,
        steps=many
    )

    # Stage 3: Rejection sampling SFT
    # Generate many, keep correct, fine-tune on filtered data
    good_data = []
    for problem in diverse_problems:
        responses = [model.generate(problem) for _ in range(10)]
        correct = [r for r in responses if is_correct(r, problem.answer)]
        good_data.extend([(problem, c) for c in correct])

    model = sft(model, good_data)

    # Stage 4: Final RL with broad rewards
    # Helpfulness, safety, formatting
    model = grpo_training(
        model,
        problems=all_problems,
        reward_fn=combined_reward,  # Correctness + helpfulness + safety
        steps=final_polish_steps
    )

    return model
```

### Distillation Results

DeepSeek also distilled R1 to smaller models:

| Model | AIME'24 | Notes |
|---|---|---|
| DeepSeek-R1 (671B MoE) | 79.8% | Full model |
| R1-Distill-Qwen-32B | 72.6% | Distilled |
| R1-Distill-Qwen-7B | 55.5% | Distilled |
| R1-Distill-Qwen-1.5B | 28.9% | Distilled |

For comparison, GPT-4o gets 12% on AIME'24. A 7B distilled reasoning model beats it by 4x.

---

## What's Actually Different Between Labs

Let me summarize the key differences we can identify:

### Thinking Visibility

| Lab | Visibility | Reasoning |
|---|---|---|
| OpenAI | Hidden | "Competitive advantage + safety" |
| Anthropic | Summarized | "Transparency for monitoring" |
| Google | Summarized | Similar to Anthropic |
| Chinese labs | Fully visible | Open science ethos |

### Budget Control

| Lab | Mechanism |
|---|---|
| OpenAI | "Reasoning effort" (low/medium/high) |
| Anthropic | Token budget (1K-128K) |
| Google | Token budget + Deep Think mode |
| Qwen3 | `/think` vs `/no_think` flags + token budget |

### Algorithm

| Lab | What we know |
|---|---|
| DeepSeek | GRPO with large ε (10.0), binary rewards |
| Qwen | GSPO (sequence-level clipping), multi-stage |
| OpenAI | "Large-scale RL" (details unknown) |
| Anthropic | Constitutional AI + RL (details unknown) |

### Open Weights

| Lab | Status |
|---|---|
| Chinese labs | Yes (Apache 2.0 license) |
| Western labs | No |

---

## Practical Takeaways

### When to Use Reasoning Models

**Good use cases:**
- Multi-step math problems
- Code requiring planning/debugging
- Scientific reasoning
- Complex analysis
- Tasks where accuracy >> speed

**Bad use cases:**
- Simple factual questions (overkill)
- Creative writing (usually; reasoning can help with plot consistency)
- High-volume, low-stakes queries (too expensive)
- Latency-critical applications

### Cost Model

Thinking tokens cost money. A rough model:

```python
def estimate_cost(task_difficulty, accuracy_requirement):
    """
    Thinking tokens are expensive but reduce retries.
    """
    if task_difficulty == "easy":
        # Non-thinking is fine
        thinking_budget = 0
        expected_accuracy = 0.9
    elif task_difficulty == "medium":
        thinking_budget = 4096
        expected_accuracy = 0.95
    elif task_difficulty == "hard":
        thinking_budget = 32768
        expected_accuracy = 0.98
    else:  # "extreme"
        thinking_budget = 128000
        expected_accuracy = 0.99

    # Cost = tokens × price_per_token
    # But also factor in: fewer retries when accuracy is higher

    base_cost = thinking_budget * price_per_thinking_token
    retry_factor = 1 / expected_accuracy  # Expected retries to get correct

    total_cost = base_cost * retry_factor

    return total_cost, thinking_budget
```

### Training Your Own

If you want to train a reasoning model:

1. **Start with distillation.** It's 10x cheaper and often better for small models.
2. **Use GSPO over vanilla GRPO** for stability, especially for long sequences.
3. **Verifiable rewards are key.** Math and code work because you can check correctness automatically.
4. **Allow exploration.** Use large ε (> 1.0) and low KL coefficient (< 0.01).
5. **Don't overtrain on SFT.** Minimal cold start to preserve RL potential.

---

## Open Questions

Some things we still don't fully understand:

**1. Why does reasoning emerge from binary rewards?**
R1-Zero shows that chain-of-thought reasoning emerges spontaneously from "did you get the right answer?" feedback. But *why*? What's special about RL that makes this happen? SFT on correct answers doesn't produce the same effect.

**2. Is the chain of thought faithful?**
When the model writes "Let me try factoring...", is that actually what it's "thinking"? Or is the written CoT a post-hoc rationalization? Research suggests CoT isn't always faithful; models sometimes use information without verbalizing it.

**3. What are the limits of test-time scaling?**
Performance improves logarithmically with thinking tokens. But does it plateau? At what point does more thinking stop helping?

**4. Can we do process rewards at scale?**
Process Reward Models (PRMs) provide feedback on each reasoning step, not just the final answer. In theory, this should be more efficient. But labeling intermediate steps is expensive. Can we automate PRM training?

**5. What's actually in OpenAI/Anthropic's training?**
We're largely in the dark about Western labs' methods. Are they doing something fundamentally different, or just GRPO with different hyperparameters?

---

## References

**Primary sources:**
- DeepSeekMath paper (GRPO origin): https://arxiv.org/pdf/2402.03300
- DeepSeek R1 paper: https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/pdf/2505.09388

**Deep dives:**
- Cameron Wolfe's GRPO series: https://cameronrwolfe.substack.com/p/grpo
- Nathan Lambert's RL book: https://rlhfbook.com/c/11-policy-gradients.html
- Oxen.ai GRPO explainer: https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/

**GRPO variants:**
- GSPO (arxiv): Sequence-level clipping for stability
- DAPO (arxiv): Asymmetric clipping for exploration
- Dr.GRPO (arxiv): Unbiased advantage estimation
- CISPO (arxiv): Clipped importance weights
