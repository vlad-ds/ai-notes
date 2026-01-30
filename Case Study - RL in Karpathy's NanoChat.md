# Case Study: RL in Karpathy's NanoChat

*Putting theory into practice with the simplest possible implementation that works.*

---

After all the theory we've covered; PPO with value functions, clipping, KL penalties, DPO's mathematical elegance; you might think implementing RL for language models requires a massive engineering effort. Karpathy's NanoChat project proves otherwise. It's a deliberately minimal implementation that strips away everything non-essential while still producing a model that learns to reason.

This chapter is a case study in understanding *why* all those complexity-reducing tricks exist by seeing what happens when you remove them.

## What NanoChat Does

NanoChat trains a small language model to solve GSM8K math problems. The training loop is straightforward: give the model a math problem, let it generate a response, check if the answer is correct, and update the model to be more likely to generate correct responses and less likely to generate incorrect ones.

That's it. No reward model. No value function. No human preferences. Just binary correctness: you got the right number, or you didn't.

If this sounds familiar, it should. It's exactly the setup DeepSeek used for R1-Zero. The difference is that Karpathy's implementation is about 300 lines of Python, designed to be read and understood in an afternoon.

## The Algorithm: REINFORCE with a Mean Baseline

The core of NanoChat is vanilla REINFORCE with one simple variance-reduction trick: subtracting the mean reward.

Remember the policy gradient from Part 1:

```python
loss = -advantage * log_prob(response)
```

And remember how in PPO, the advantage comes from a trained value function that predicts expected reward for each prompt:

```python
advantage = reward - value_function(prompt)
```

NanoChat does something much simpler. For each training step, it generates 16 responses to the same problem. Each response gets a reward (1 for correct, 0 for incorrect). The advantage for each response is just how much better or worse it was than the group average:

```python
def compute_advantages(rewards):
    """
    NanoChat's advantage computation.

    No value function needed; just use the mean of the current batch
    as the baseline. This is the key simplification.
    """
    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]
    return advantages
```

Let's trace through a concrete example. Suppose we generate 16 responses to a math problem, and 4 of them are correct:

```plain text
Rewards:    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
Mean:       4/16 = 0.25

Advantages: [0.75, -0.25, 0.75, -0.25, -0.25, ...]
```

The correct responses get positive advantage (+0.75): "you beat the average, do more of this." The incorrect responses get negative advantage (-0.25): "you underperformed, do less of this."

This is exactly what GRPO does (we covered this in Part 4). The only difference is that NanoChat doesn't normalize by standard deviation. GRPO would compute:

```python
std = compute_std(rewards)
advantages = [(r - mean) / (std + 1e-8) for r in rewards]
```

This z-score normalization makes advantages roughly unit-scale regardless of how easy or hard the problem is. Karpathy drops it for simplicity; the mean baseline alone provides most of the variance reduction.

## Why No PPO?

PPO has two main components beyond basic REINFORCE: the clipped objective and the value function. NanoChat uses neither. Why can it get away with this?

### No Clipping Because On-Policy

Remember what PPO's clipping does: it prevents the policy from changing too much when learning from off-policy data. If you generated responses with model_v1 but you're updating model_v2, the distributions might have diverged, and you need to correct for this mismatch.

```python
# PPO ratio
ratio = prob_new(response) / prob_old(response)
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
```

NanoChat sidesteps this entirely by being strictly on-policy. Every training step looks like:

```python
# 1. Generate responses with the CURRENT model
responses = model.generate(prompt, num_samples=16)

# 2. Compute rewards for those responses
rewards = [task.reward(r) for r in responses]

# 3. Immediately update the SAME model
loss = compute_loss(model, responses, rewards)
loss.backward()
optimizer.step()
```

The responses are always from the current policy. There's no "old model" and "new model"; they're the same. This means:

```python
ratio = prob_current(response) / prob_current(response) = 1.0
```

The ratio is always 1, so clipping would never activate. You can just use vanilla policy gradient:

```python
loss = -advantage * log_prob(response)  # no ratio needed
```

This is a significant simplification. PPO's ratio computation requires keeping the old model around (or at least its log probabilities), and the clipping logic adds complexity. On-policy training makes all of that unnecessary.

### No Value Function Because the Batch Provides the Baseline

The value function in PPO exists to answer: "What reward should I expect for this prompt?" In NanoChat, we answer this empirically by generating multiple responses.

If we generate 16 responses and 4 are correct, the empirical expected reward is 0.25. We don't need a neural network to predict this; we just measured it. The mean of the batch *is* the baseline.

This has an elegant property: the baseline automatically adapts to problem difficulty. For easy problems where most responses are correct (mean ≈ 0.8), only the incorrect responses get strongly negative advantage. For hard problems where few responses are correct (mean ≈ 0.1), even a barely-correct response gets strongly positive advantage.

The cost is sample efficiency. Instead of amortizing the baseline computation across many examples (which a learned value function does), we're estimating it fresh for each prompt using 16 samples. We're trading compute for simplicity.

## No KL Penalty: Trusting the Task

In RLHF, the KL penalty keeps the model from drifting too far from the SFT checkpoint. This prevents reward hacking; finding degenerate ways to maximize the reward that don't correspond to genuinely good behavior.

```python
# RLHF-style reward
adjusted_reward = reward_model(response) - kl_coef * kl_divergence
```

NanoChat drops this entirely. There's no reference model, no KL term, no regularization back to a starting point. The model is free to drift wherever the reward signal takes it.

Why doesn't this cause problems? Two reasons:

First, the reward is *verifiable*. The model either got the right answer or it didn't. There's no reward model that could be fooled by surface patterns. You can't hack "did 2 + 2 equal 4" the way you might hack "does this response seem helpful."

Second, the task is narrow. NanoChat is training on GSM8K math problems, not general instruction following. There's less room for the model to find weird distributional shifts that game the reward but produce nonsense outputs. If the model produces the correct numerical answer, it's doing something right.

This wouldn't work for general RLHF, where the reward comes from an imperfect proxy (the reward model) and the task space is vast. But for verifiable, narrow tasks, you can often skip the KL regularization.

## Token-Level vs Sequence-Level Normalization

One detail in NanoChat worth highlighting: the loss is computed per-token, not per-sequence.

In standard REINFORCE, you'd compute the loss as:

```python
# Sequence-level: one scalar advantage for the whole response
sequence_log_prob = sum(log_prob(token) for token in response)
loss = -advantage * sequence_log_prob
```

NanoChat does something different, which they call "DAPO-style" (from a recent paper on direct alignment):

```python
# Token-level: apply advantage to each token separately
per_token_log_probs = [log_prob(token) for token in response]
per_token_objective = sum(adv * lp for lp, adv in zip(per_token_log_probs, advantages))

# Normalize by number of valid tokens, not sequences
loss = -per_token_objective / num_valid_tokens
```

The practical effect: loss is normalized by the total number of tokens across all responses, not by the number of sequences. This means long responses and short responses contribute proportionally to their length, rather than one long correct response dominating the loss.

## The Mask: Learning Only From What You Generated

NanoChat has a neat feature: the model can use a calculator tool. When it outputs a code block, the code gets executed and the result gets injected back into the response.

This creates a question: should the model learn from the calculator's output? The answer is no. The model didn't generate those tokens; they were forced by the environment. Learning from forced tokens would confuse the credit assignment.

```python
def prepare_targets(tokens, mask):
    """
    Mask out tokens we don't want to learn from.

    mask[i] = 1 means: this token was generated by the model
    mask[i] = 0 means: this token was forced (prompt or tool output)
    """
    targets = tokens.clone()
    targets[mask == 0] = -1  # -1 = ignore in loss computation
    return targets
```

This is conceptually similar to the prompt masking we discussed in Part 2, where SFT only backprops through the response, not the prompt. Here it's extended to also mask out tool execution results within the response.

## Putting It All Together

Here's the complete NanoChat training loop, annotated with references to concepts we've covered:

```python
def nanochat_training_step(model, optimizer, problem, num_samples=16):
    """
    One step of NanoChat training.

    This is REINFORCE with:
    - Binary rewards (correct/incorrect)
    - Mean baseline (no value function)
    - On-policy sampling (no PPO ratio)
    - No KL penalty (no reference model)
    - Token-level normalization (DAPO-style)
    """
    # Step 1: Generate responses from current policy (on-policy)
    responses, masks = model.generate_batch(problem.prompt, num_samples)

    # Step 2: Compute binary rewards
    # Just like DeepSeek R1; did you get the right answer?
    rewards = torch.tensor([
        1.0 if problem.check_answer(r) else 0.0
        for r in responses
    ])

    # Step 3: Compute advantages using mean baseline
    # This is the key simplification; no value function needed
    mean_reward = rewards.mean()
    advantages = rewards - mean_reward

    # Step 4: Compute per-token log probabilities
    # (We covered this computation in detail in Part 1)
    inputs = responses[:, :-1]
    targets = responses[:, 1:].clone()
    targets[masks[:, 1:] == 0] = -1  # Mask non-generated tokens

    # Cross-entropy with ignore_index=-1 skips masked positions
    log_probs = -model.forward(inputs, targets, reduction='none')

    # Step 5: Policy gradient objective
    # Sum of (log_prob × advantage) across tokens and samples
    pg_objective = (log_probs * advantages.unsqueeze(-1)).sum()

    # Normalize by valid tokens (DAPO-style)
    num_valid = (targets >= 0).sum()
    loss = -pg_objective / num_valid

    # Step 6: Update (standard gradient descent)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        'loss': loss.item(),
        'mean_reward': mean_reward.item(),
        'num_correct': (rewards == 1).sum().item()
    }
```

Compare this to the GRPO code from Part 4. The structure is identical; generate multiple samples, compute rewards, subtract mean, do policy gradient. The differences are all simplifications: no std normalization, no clipping, no KL penalty, no reference model.

## What's Missing (And When It Would Matter)

NanoChat deliberately omits several things that would be important for production RL training:

**No reference model or KL penalty**. This works for verifiable tasks like math, but for open-ended tasks where you're optimizing a learned reward model, you'd need KL regularization to prevent reward hacking.

**No PPO clipping**. Works for on-policy, but if you wanted to reuse data from previous iterations (for sample efficiency), you'd need off-policy corrections.

**No value function**. The batch mean baseline works, but a learned value function would reduce variance and might learn faster. The trade-off is simplicity vs sample efficiency.

**No z-score normalization**. Including the standard deviation in the advantage normalization would make training more stable across problems of varying difficulty.

**Single learning rate schedule**. NanoChat uses different learning rates for different parameter groups (embeddings, attention, MLPs), but doesn't do sophisticated things like warmup or adaptive KL coefficients.

## The Broader Lesson

NanoChat demonstrates something important: the core of RL for language models is simple. Generate samples, score them, increase probability of good samples, decrease probability of bad samples. That's really it.

All the complexity in PPO, GRPO, DPO, and friends exists to handle specific failure modes:
- **Value functions** reduce variance when you can't generate many samples per prompt
- **PPO clipping** enables off-policy learning for better sample efficiency
- **KL penalties** prevent reward hacking when optimizing proxy rewards
- **DPO** eliminates the reward model entirely when you have direct preference data

For a narrow task with verifiable rewards and unlimited compute for samples, you don't need any of these. Vanilla REINFORCE with a mean baseline gets you surprisingly far.

The practical takeaway: start simple. If you're training on verifiable tasks (code execution, math, structured outputs), try the NanoChat approach first. You might not need the complexity of full PPO or RLHF. Save that complexity for when you have evidence you need it.

## Connection to DeepSeek R1

NanoChat and DeepSeek R1-Zero are doing essentially the same thing, just at different scales:

| Aspect | NanoChat | DeepSeek R1-Zero |
|---|---|---|
| Task | GSM8K math | Math + coding |
| Reward | Binary correctness | Binary correctness |
| Algorithm | REINFORCE + mean baseline | GRPO (similar, with z-score) |
| KL penalty | None | Small (β=0.001) |
| Clipping | None | Large range (ε=10.0) |
| Samples/prompt | 16 | 16 |
| Model size | Small (~100M) | Large (671B) |

Both prove the same point: you can elicit reasoning behavior from pure RL on verifiable tasks, without any supervised examples of reasoning chains. The model learns that thinking step by step helps it get correct answers, so it learns to think.

NanoChat is the educational version; small enough to run on a laptop, simple enough to read in an hour. R1-Zero is the production version; scaled up and tuned for state-of-the-art performance. The core algorithm is the same.

---

*Code for NanoChat is available at https://github.com/karpathy/nanochat. It's worth reading through; the main training loop is under 300 lines and implements everything we've discussed.*
