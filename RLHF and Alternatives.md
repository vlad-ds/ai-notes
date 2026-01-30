# RLHF and Alternatives

---

## Part 1: The Core Problem We're Trying to Solve

Let's start with the fundamental question: why do we need reinforcement learning for language models at all?

When you pretrain a language model, you're doing next-token prediction. The model sees a sequence of tokens and learns to predict what comes next. This works remarkably well for teaching a model the patterns of language, facts about the world, and even reasoning capabilities. But here's the thing: next-token prediction optimizes for *likelihood*; the model learns to assign high probability to text that looks like the training data.

This creates a problem. The internet contains all kinds of text: helpful explanations, but also misinformation; polite responses, but also rude ones; safe advice, but also dangerous instructions. A model trained purely on likelihood will happily generate any of these. It's just trying to produce plausible-looking text, not *good* text by any human standard.

Reinforcement learning enters the picture because we want to optimize for something different. Instead of asking "what text is most likely given this context?", we want to ask "what text would be most *useful*, *helpful*, or *safe*?". This requires a different training signal; one that comes from some notion of quality rather than mere statistical patterns.

### Mapping the RL Framework onto Language Generation

To use RL, we need to frame language generation as a sequential decision-making problem. Here's how the mapping works.

The **state** at any point in generation is everything the model has seen so far: the original prompt plus all the tokens it has already generated. The **action** is choosing the next token from the vocabulary. The **policy** is the model itself; given a state, it outputs a probability distribution over possible next tokens, and we sample from this distribution to pick an action. Finally, the **reward** is some score that tells us how good the complete response was.

Notice something important here: unlike in games where you might get rewards after each move, in language generation we typically only get a reward at the very end. The model generates an entire response, and then we score that complete response. This is called "sparse reward" and it makes the learning problem harder; we need to figure out which of the many token choices along the way contributed to the final good or bad outcome.

### The Policy: Your Model as a Probability Machine

Let's make this concrete with code. When we talk about the "policy" in RL for LLMs, we're literally just talking about the language model. Given some context, it outputs logits for each token in the vocabulary, we apply softmax to get probabilities, and then we can either sample from this distribution or compute the probability of a specific sequence.

The key question is: what's the probability that this model would generate this specific response, given this prompt? To answer that, we use the chain rule of probability. The probability of a sequence is the product of the probability of each token, conditioned on everything that came before it:

```plain text
P(response | prompt) = P(token_1 | prompt)
                     × P(token_2 | prompt, token_1)
                     × P(token_3 | prompt, token_1, token_2)
                     × ...
```

So if you have a 100-token response, you're multiplying together 100 probabilities. Each probability is typically small (maybe 0.1 or 0.01), so this product becomes astronomically tiny; something like 0.1^100, which underflows to zero in floating point.

The solution is to work in log space. Since log(a × b) = log(a) + log(b), we can convert the product into a sum:

```plain text
log P(response | prompt) = log P(token_1 | prompt)
                         + log P(token_2 | prompt, token_1)
                         + log P(token_3 | prompt, token_1, token_2)
                         + ...
```

Now instead of multiplying tiny numbers, we're adding negative numbers (log of any probability less than 1 is negative). A 100-token response might have a log probability of -230, which is a perfectly reasonable number to work with.

Here's the code:

```python
def compute_log_probability(model, prompt_tokens, response_tokens):
    """
    Compute log P(response | prompt) under the model.

    We compute this by summing log probabilities of each token,
    where each token's probability is conditioned on the prompt
    plus all previous response tokens.
    """
    log_prob_total = 0.0
    context = prompt_tokens.copy()  # Start with just the prompt

    for token in response_tokens:
        # Forward pass: get probability distribution over next token
        # This gives us P(next_token | context)
        logits = model.forward(context)
        probs = softmax(logits)

        # Get log probability of the actual token that was generated
        # This is log P(token | context), and it's negative since probs < 1
        log_prob_total += math.log(probs[token])

        # Extend context: next iteration conditions on this token too
        context.append(token)

    return log_prob_total  # This will be a negative number, e.g., -230
```

Let me trace through a concrete example. Say the prompt is "What is 2+2?" and the response is three tokens: ["The", "answer", "is", "4"].

```plain text
Iteration 1: context = ["What", "is", "2+2?"]
             → model gives P("The" | "What is 2+2?") = 0.3
             → log(0.3) = -1.2
             → log_prob_total = -1.2

Iteration 2: context = ["What", "is", "2+2?", "The"]
             → model gives P("answer" | "What is 2+2? The") = 0.15
             → log(0.15) = -1.9
             → log_prob_total = -1.2 + -1.9 = -3.1

Iteration 3: context = ["What", "is", "2+2?", "The", "answer"]
             → model gives P("is" | ...) = 0.4
             → log(0.4) = -0.9
             → log_prob_total = -3.1 + -0.9 = -4.0

Iteration 4: context = [..., "is"]
             → model gives P("4" | ...) = 0.7
             → log(0.7) = -0.36
             → log_prob_total = -4.0 + -0.36 = -4.36
```

The final log probability is -4.36, which corresponds to a joint probability of exp(-4.36) ≈ 0.013, or about 1.3%. That seems reasonable for a short, fairly predictable response.

Where does the response come from in the first place? It depends on the training stage. During RL training (like PPO), the model itself generates the response by sampling; we then compute its log probability to figure out how to update the weights. During reward model training or DPO, the responses typically come from training data: either human-written ideal responses, or pairs of model outputs that humans compared. In the example above, "The answer is 4" might be a human-written demonstration, or it might be something the model generated during a training rollout. The log probability computation is the same either way; we're just asking "how likely would this model be to produce this exact sequence?"

This function is fundamental. Policy gradients, PPO, DPO; they all rely on being able to compute these log probabilities.

### How the Policy Gradient Works at the Token Level

This is a crucial detail that's easy to misunderstand. When we talk about "the loss for a response," you might imagine computing one scalar per response and then backpropagating that. But that's not quite right. Let's be very precise.

**The policy gradient formula in textbooks:**
```
loss = -advantage × log P(response | prompt)
```

This looks like one scalar times one scalar. But `log P(response | prompt)` is itself a sum of per-token log probabilities. Expanding this:

```
loss = -advantage × [log P(token_1) + log P(token_2) + ... + log P(token_N)]
```

Which distributes to:

```
loss = -advantage × log P(token_1)
     + -advantage × log P(token_2)
     + ...
     + -advantage × log P(token_N)
```

**The key insight:** The same advantage (one scalar for the whole response) gets multiplied by each token's log probability separately. When you call `loss.backward()`, PyTorch computes gradients for each token's contribution independently.

**In practice with batched sequences:**

```python
# Shape: (batch_size, seq_len)
log_probs = compute_per_token_log_probs(model, sequences)

# Shape: (batch_size,) - one advantage per sequence
advantages = rewards - baseline

# Broadcast: advantages[:, None] has shape (batch_size, 1)
# This multiplies each token by its sequence's advantage
per_token_objective = log_probs * advantages[:, None]

# Sum everything into one scalar
loss = -per_token_objective.sum() / num_valid_tokens

# One backward pass updates all weights
loss.backward()
```

**What this means for credit assignment:**

Every token in a "good" response (positive advantage) gets pushed to be more likely. Every token in a "bad" response (negative advantage) gets pushed to be less likely. The model learns which token *patterns* lead to good outcomes, not just which final answers are good.

This is why we say the gradient is "noisy" in REINFORCE: if only one token in a 100-token response caused the bad outcome, all 100 tokens still get the same negative gradient signal. The model has to average over many examples to figure out which tokens actually matter. Value functions and other variance reduction techniques help with this, but the fundamental token-level mechanics remain the same.

---

## Part 2: The RLHF Pipeline

RLHF (Reinforcement Learning from Human Feedback) was the technique that transformed GPT-3 into ChatGPT. It's a three-stage pipeline, and understanding each stage is crucial for understanding everything that came after.

### Stage 1: Supervised Fine-Tuning

Before we even get to RL, we start with supervised fine-tuning (SFT). The idea is simple: collect a dataset of high-quality prompt-response pairs and fine-tune the base model on them using standard cross-entropy loss.

```python
def sft_training_step(model, prompt, ideal_response):
    """
    Standard supervised fine-tuning.

    This is identical to language model pretraining,
    except we're training specifically on curated
    high-quality examples.
    """
    # Concatenate prompt and response
    full_sequence = prompt + ideal_response

    # Cross-entropy loss on the response portion
    # (we don't backprop through the prompt)
    loss = 0.0
    for i, target_token in enumerate(ideal_response):
        context = prompt + ideal_response[:i]
        logits = model.forward(context)
        probs = softmax(logits)
        loss -= math.log(probs[target_token])

    return loss / len(ideal_response)
```

For InstructGPT, OpenAI used about 13,000 demonstrations for this stage. These were written by human labelers who were given prompts and asked to write ideal responses. The resulting model is already much better at following instructions than the base model, but it's limited to imitating the examples it saw; it doesn't generalize well beyond them.

### Stage 2: Training the Reward Model

Here's where things get interesting. We want to teach a model to score responses based on human preferences. But directly asking humans "rate this response from 1 to 10" produces noisy, inconsistent data. Different people have different standards for what constitutes a "7".

The solution is to use comparisons. Show a human two responses to the same prompt and ask: which one is better? This comparative judgment is much more reliable and consistent across annotators.

Given this comparison data, we train a reward model using the Bradley-Terry model. The idea is elegant: if response A is preferred over response B, then the reward model should assign a higher score to A. Specifically, we model the probability that A beats B as:

```python
def bradley_terry_probability(reward_A, reward_B):
    """
    Bradley-Terry model: probability that A is preferred over B.

    If reward_A is much higher than reward_B, this approaches 1.
    If they're equal, this is 0.5.
    If reward_A is much lower, this approaches 0.
    """
    return sigmoid(reward_A - reward_B)
```

This is just a sigmoid of the difference in rewards. The training loss follows directly: we want to maximize the probability of the observed preferences.

```python
def reward_model_loss(reward_model, prompt, response_winner, response_loser):
    """
    Train the reward model on a single preference pair.

    The winner should get a higher score than the loser.
    """
    # Get scalar rewards for both responses
    # Note: the model sees each response independently
    reward_winner = reward_model(prompt + response_winner)
    reward_loser = reward_model(prompt + response_loser)

    # Bradley-Terry loss: maximize probability that winner beats loser
    # This is equivalent to: -log(sigmoid(reward_winner - reward_loser))
    loss = -math.log(sigmoid(reward_winner - reward_loser))

    return loss
```

But what is the reward model, exactly? It's typically the same transformer architecture as the LLM, often initialized from the SFT checkpoint. The only difference is the output head. Instead of a language modeling head that outputs logits over the vocabulary (50,000+ numbers for each position), you replace it with a scalar head: a single linear layer that takes the final hidden state and outputs one number.

```python
class RewardModel:
    def __init__(self, base_model):
        # Same transformer backbone as the LLM
        self.transformer = base_model.transformer

        # Replace the LM head with a scalar head
        # Instead of: hidden_state → [50257] vocab logits
        # We do:      hidden_state → [1] scalar reward
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, prompt_and_response):
        # Run through transformer, same as the LLM would
        hidden_states = self.transformer(prompt_and_response)

        # Take the last token's hidden state
        last_hidden = hidden_states[-1]

        # Project to a single scalar
        reward = self.reward_head(last_hidden)

        return reward  # Just one number, like 2.5 or -1.3
```

Starting from the SFT model makes sense because it already "understands" language; it can read the prompt and response and has internal representations for quality, helpfulness, coherence, and so on. We're just training a small head on top to convert that understanding into a scalar score.

For InstructGPT, OpenAI collected about 33,000 prompts with multiple responses each, yielding around 300,000 pairwise comparisons. The resulting reward model learns to score responses in a way that reflects human preferences.

One crucial detail: the reward model's absolute numbers are meaningless. Only the *differences* matter. If the model gives response A a score of 2.5 and response B a score of 1.3, all we know is that A is preferred. The 2.5 could just as well be 25 or 0.25; what matters is that it's higher than 1.3.

Notice what the loss is actually doing: it's pushing the model to maximize the gap between `reward_winner` and `reward_loser`. The larger that gap, the closer `sigmoid(gap)` gets to 1, the closer `-log(sigmoid(gap))` gets to 0, and the lower the loss. But there's a natural stopping point built in. The sigmoid saturates; once the gap reaches +5 or so, `sigmoid(5) ≈ 0.993`, and making the gap even larger barely changes the loss. The gradient essentially vanishes, and the model stops pushing. This is a soft form of regularization: the model learns to confidently distinguish winners from losers, but it doesn't waste capacity driving the gap to infinity.

### Stage 3: RL Fine-Tuning with PPO

Now comes the actual reinforcement learning. We have a language model and a reward model. The goal is to update the language model so it generates responses that get high rewards.

In RL terminology, the language model is called the **policy**. This is just jargon; it means "the thing that decides what action to take." In our case, the "action" is choosing the next token, and the policy (the language model) outputs a probability distribution over all possible tokens. When we say "update the policy," we mean "update the language model's weights." When we say "the policy generates a response," we mean "the language model samples tokens one by one until it produces a complete response."

The naive approach to RL would be: generate a response, get its reward, multiply the gradient of the log-probability by the reward, and update. This is the REINFORCE algorithm, and it works in principle but has terrible variance in practice. Some responses score high by luck, others score low by luck, and the gradients are extremely noisy.

> 💡 *Why doesn't REINFORCE work in practice?*
>
> Two words: **high variance**.
>
> The reward is a single scalar for the entire sequence. This same scalar (the advantage) gets applied to every token in that sequence. If the response was good (positive advantage), every token gets pushed to be more likely. If bad (negative advantage), every token gets pushed to be less likely. Even though maybe only 3 tokens were actually responsible for the bad output, all 100 tokens get the same "bad response" signal.
>
> (Note: the advantage is the same, but the actual gradient per model parameter is computed token by token; each token's log probability contributes separately to the loss. The problem is that we're applying a sequence-level judgment to token-level updates.)
>
> In practice this means:
> - Training is unstable (one unlucky batch can wreck your model)
> - You need millions of samples to average out the noise
> - Learning is painfully slow
>
> PPO fixes this with three patches:
> 1. **Advantages** (subtract a baseline so you're learning "better than expected" not raw reward)
> 2. **Clipping** (cap how much any single update can change the policy)
> 3. **KL penalty** (don't drift too far from a known-good starting point)
>
> All three are variance reduction / stability tricks. The core algorithm is still REINFORCE underneath; PPO just makes it not explode.

PPO (Proximal Policy Optimization) addresses this with two key ideas.

First, instead of using raw rewards, we use **advantages**. The advantage of a response is how much better (or worse) it was compared to what we expected. If we expected a reward of 3 and got 5, the advantage is +2. If we expected 3 and got 1, the advantage is -2. This centers the learning signal around zero, which dramatically reduces variance.

```python
def compute_advantage(reward, expected_reward):
    """
    Advantage = actual reward minus expected reward.

    Positive advantage: "this was better than usual, do more of this"
    Negative advantage: "this was worse than usual, do less of this"
    """
    return reward - expected_reward
```

But where does the expected reward come from? This is the job of the **value function**, and it's worth understanding in detail before we move on.

### The Value Function: Predicting Reward from Prompts

The value function is a neural network that answers a simple question: "Given this prompt, what reward should I expect if the current policy generates a response?" Notice that it only sees the prompt, not any response. It's trying to predict the *average* reward the policy would get across all the responses it might generate for this prompt.

Why is this useful? Because some prompts are inherently harder than others. If you ask the model to write a haiku, it will probably get a decent reward. If you ask it to solve a complex math problem, it might struggle. The value function learns these differences. For the haiku prompt, it might predict an expected reward of 0.8. For the hard math problem, maybe 0.3.

Architecturally, the value function looks a lot like the reward model: same transformer backbone, scalar output head. But there's a key difference in what they're trained on:
- **Reward model**: Trained on human preferences (winner vs loser comparisons)
- **Value function**: Trained to predict the reward model's scores for policy-generated responses

```python
class ValueFunction:
    def __init__(self, base_model):
        # Same transformer backbone
        self.transformer = base_model.transformer

        # Scalar head, just like reward model
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, prompt):
        # Note: only takes the prompt, not the response
        hidden_states = self.transformer(prompt)
        last_hidden = hidden_states[-1]
        predicted_reward = self.value_head(last_hidden)
        return predicted_reward
```

The value function is trained alongside the policy during RL. Here's how:

```python
def train_value_function(value_fn, prompt, actual_reward):
    """
    Simple regression: predict the reward we actually got.

    Over time, the value function learns to predict
    what reward to expect for different prompts.
    """
    predicted_reward = value_fn(prompt)

    # Mean squared error: how wrong was our prediction?
    loss = (predicted_reward - actual_reward) ** 2

    return loss
```

During each training step, we:
1. Generate a response from the policy
2. Score it with the reward model → `actual_reward`
3. Ask the value function to predict the reward from just the prompt → `predicted_reward`
4. Update the value function to reduce `(predicted_reward - actual_reward)²`

Over thousands of updates, the value function learns which prompts tend to produce high-reward responses and which tend to produce low-reward responses. This lets us compute meaningful advantages: if the value function predicted 0.3 but we got 0.8, that's a big positive surprise (+0.5 advantage), and we should reinforce whatever the policy did.

One important detail: the value function and the policy learn together, simultaneously. At the start of training, both are bad. The policy generates mediocre responses, and the value function's predictions are basically random guesses. But as training progresses, the policy improves (generating better responses), and the value function improves (getting better at predicting what rewards to expect). They bootstrap off each other: better value estimates lead to better advantages, which lead to better policy updates, which generate new data for the value function to learn from.

### PPO: Don't Change Too Much at Once

Now back to PPO. We have advantages from the value function. The second key idea in PPO is **clipping** to prevent the policy from changing too much in a single update. This is where the "proximal" in Proximal Policy Optimization comes from. The idea is to measure how different the new policy is from the old one, and if they're becoming too different, stop the gradient.

```python
def ppo_loss(model, old_model, prompt, response, advantage, epsilon=0.2):
    """
    PPO's clipped surrogate objective.

    The key insight: we compute how much the policy has changed
    (the ratio of new probability to old probability) and clip
    this ratio to prevent too-large updates.
    """
    # Compute log probs under both models
    log_prob_new = compute_log_probability(model, prompt, response)
    log_prob_old = compute_log_probability(old_model, prompt, response)

    # The ratio tells us: how much more (or less) likely is this
    # response under the new policy vs the old policy?
    ratio = math.exp(log_prob_new - log_prob_old)

    # Unclipped objective: ratio * advantage
    # This is what we'd use in vanilla policy gradient
    unclipped = ratio * advantage

    # Clipped objective: don't let ratio go below 1-epsilon or above 1+epsilon
    clipped_ratio = max(1 - epsilon, min(1 + epsilon, ratio))
    clipped = clipped_ratio * advantage

    # Take the minimum of clipped and unclipped
    # This is conservative: it only lets beneficial changes through
    # if they're not too large
    loss = -min(unclipped, clipped)

    return loss
```

Let me explain the clipping logic in more detail because it's subtle. When the advantage is positive (the response was better than expected), we want to increase its probability. The gradient naturally pushes the ratio above 1. But if the ratio goes above 1+ε (say, 1.2 with ε=0.2), we clip it; we stop getting gradient signal. This prevents us from making the response *too* much more likely in a single update.

When the advantage is negative (the response was worse than expected), we want to decrease its probability. The gradient pushes the ratio below 1. But if it goes below 1-ε (0.8), we clip it.

The minimum operation ensures we always take the more conservative option. If the unclipped version would have us make a big change, but the clipped version limits that change, we use the clipped version.

### The KL Penalty: Don't Forget Your Roots

There's one more crucial piece of RLHF that we haven't discussed: the KL penalty. Without it, the language model might find degenerate ways to get high rewards that have nothing to do with being genuinely helpful.

For example, suppose the reward model was trained on data where longer responses tended to be preferred. The policy might learn to generate extremely long, repetitive responses that aren't actually better; they're just gaming the reward model's bias.

The KL penalty prevents this by penalizing the policy for diverging too far from the original supervised fine-tuned model.

```python
def compute_rlhf_reward(response, prompt, reward_model, sft_model, policy, kl_coef=0.1):
    """
    The actual reward used in RLHF training.

    It's not just the reward model score; it's that score
    minus a penalty for diverging from the SFT model.
    """
    # Base reward from the reward model
    reward = reward_model(prompt + response)

    # Compute how different the policy is from the SFT model
    # (using KL divergence, approximated per-token)
    log_prob_policy = compute_log_probability(policy, prompt, response)
    log_prob_sft = compute_log_probability(sft_model, prompt, response)

    # KL divergence: how much more likely is this under policy than SFT?
    kl_divergence = log_prob_policy - log_prob_sft

    # Final reward: base reward minus KL penalty
    adjusted_reward = reward - kl_coef * kl_divergence

    return adjusted_reward
```

The kl_coef hyperparameter controls how strongly we penalize divergence. Too small, and the model might drift far from reasonable behavior. Too large, and the model can't learn anything new. InstructGPT used a value around 0.02, though this was adapted during training.

> 💡 *What is the difference between clipping and KL penalty?*
>
> They solve different problems:
>
> **Clipping** = "Don't change too much *in this update*"
>
> It's a per-step speed limit. Looks at the ratio `new_prob / old_prob` and caps it to [0.8, 1.2]. Prevents a single bad batch from nuking your model.
>
> **KL penalty** = "Don't drift too far *from where you started*"
>
> It's a tether to the original SFT model. Even if every individual step is small (passes clipping), after 10,000 steps you could end up somewhere crazy. The KL penalty keeps pulling you back toward the reference.
>
> ```python
> # Clipping: local constraint (this step)
> ratio = new_prob / old_prob
> clipped_ratio = clip(ratio, 0.8, 1.2)  # "slow down!"
>
> # KL penalty: global constraint (total drift)
> kl = log_prob_policy - log_prob_reference  # how far from SFT?
> adjusted_reward = reward - 0.01 * kl  # "come back!"
> ```
>
> **Car analogy:**
> - Clipping = speed limit (can't go faster than 60 mph at any moment)
> - KL penalty = rubber band attached to your house (no matter how slowly you drive, if you keep going in one direction, it pulls you back)
>
> Without clipping: one update destroys you.
> Without KL: you slowly drift into reward hacking (e.g., model learns to output garbage that somehow scores high on the reward model).

---

## Part 3: DPO; Cutting Out the Middle Man

RLHF works, but it's complicated. You need to train a reward model, then run PPO with a value function, and manage four different models during training. It's unstable, finicky to tune, and there's a fundamental indirection: you train a reward model to approximate human preferences, then optimize against that proxy. What if the reward model is subtly wrong? You might optimize for something that looks good to the reward model but isn't actually what humans wanted.

In 2023, a paper called "Direct Preference Optimization" (DPO) asked: what if we could skip the reward model entirely and train the policy directly on preference data?

### What Do We Actually Want?

Let's think about what we're trying to achieve. We have pairs of responses where humans said "A is better than B." At the end of the day, we want a model that would be more likely to generate A than B. That's it. The reward model in RLHF is just a means to an end; we train it so we have a signal to optimize against. But if we could directly make the policy prefer winners over losers, we wouldn't need the reward model at all.

### The DPO Idea

DPO does exactly this. For each preference pair (winner, loser), we ask: has our policy learned to prefer the winner?

But we can't just ask "is the winner more likely than the loser?" in absolute terms. Maybe the winner is a short response and the loser is a long response; short responses are always more likely, but that doesn't mean the policy has learned anything about quality. We need to measure *relative change*; how much has the policy shifted its preference compared to where it started?

This is why DPO compares the policy to a reference model (usually the SFT checkpoint before preference training). We ask two questions:
1. How much more does our policy like the winner compared to the reference model?
2. How much more does our policy like the loser compared to the reference model?

If the policy has increased its preference for the winner more than it has for the loser, we're on the right track. If not, we have a loss signal that pushes the policy to favor the winner more.

```python
def dpo_loss(policy, ref_model, prompt, winner, loser, beta=0.1):
    """
    Direct Preference Optimization loss.

    We measure: has the policy shifted to prefer the winner over the loser,
    compared to where the reference model started?
    """
    # How likely is the winner under policy vs reference?
    log_prob_winner_policy = compute_log_probability(policy, prompt, winner)
    log_prob_winner_ref = compute_log_probability(ref_model, prompt, winner)
    winner_log_ratio = log_prob_winner_policy - log_prob_winner_ref
    # Positive means: policy likes winner MORE than reference did

    # How likely is the loser under policy vs reference?
    log_prob_loser_policy = compute_log_probability(policy, prompt, loser)
    log_prob_loser_ref = compute_log_probability(ref_model, prompt, loser)
    loser_log_ratio = log_prob_loser_policy - log_prob_loser_ref
    # Positive means: policy likes loser MORE than reference did

    # We want winner_log_ratio to be bigger than loser_log_ratio
    # That means: policy shifted toward winner more than toward loser
    preference_margin = winner_log_ratio - loser_log_ratio

    # Same Bradley-Terry setup as reward model training:
    # sigmoid squashes to 0-1, -log turns into a loss
    loss = -math.log(sigmoid(beta * preference_margin))

    return loss
```

Let's trace through what this does:
- If `preference_margin` is large and positive (policy strongly prefers winner over loser, relative to reference), then `sigmoid(beta * preference_margin)` is close to 1, and `log(...)` is close to 0. Low loss; this is what we want.
- If `preference_margin` is small or negative (policy hasn't learned to prefer winner, or actually prefers loser), then `sigmoid(...)` is closer to 0.5 or below, and `log(...)` is larger. High loss; training signal to fix this.

> 💡 **What DPO actually maximizes:**
>
> ```python
> (policy_winner - policy_loser) - (ref_winner - ref_loser)
> ```
>
> Which rearranges to:
>
> ```python
> (policy_winner - ref_winner) - (policy_loser - ref_loser)
> ```
>
> **Cleaner one-sentence summary:**
>
> "Train the policy to shift toward the winner more than it shifts toward the loser, relative to the reference."
>
> Or even simpler:
>
> "Make the policy prefer winners over losers *more* than the reference model does."
>
> It's not about absolute probabilities. A short winner might have lower absolute probability than a long loser. DPO measures *relative change* from the reference. Did you boost the winner more than you boosted the loser? Good. Did you accidentally boost both equally? No learning signal. Did you boost the loser more? Bad, here's a gradient to fix that.

### What is Beta?

Beta is a hyperparameter you set before training (like a learning rate); it's not learned. It controls how strongly the policy should differentiate between winners and losers:
- **Large beta (e.g., 0.5):** Push hard to prefer winners. The policy will deviate more from the reference model.
- **Small beta (e.g., 0.01):** Mild preferences. The policy stays closer to the reference model.

The name comes from the KL penalty in RLHF. Remember the constraint "stay close to the reference model"? Beta plays a similar role here; it balances quality improvement against staying grounded to the reference.

### Why This Works

Here's the beautiful part: the DPO authors proved mathematically that this simple loss function is *exactly equivalent* to doing full RLHF with an optimal reward model. You get the same result, but without ever explicitly training a reward model or running RL. The reward is implicitly encoded in how the policy differs from the reference.

The formal relationship is: if you think of `beta * (log_prob_policy - log_prob_ref)` as an implicit reward, then maximizing the DPO objective is the same as maximizing expected reward with a KL penalty. But you don't need to understand this math to use DPO; the intuition is simply "train the policy to prefer winners over losers, relative to where it started."

### The Simplicity

What makes DPO so attractive is its simplicity. You just need your policy, a frozen reference model, and preference data. The training loop looks like ordinary supervised learning:

```python
def dpo_training_loop(policy, ref_model, preference_data, epochs=3):
    """
    DPO training is remarkably simple.
    It looks almost like standard fine-tuning.
    """
    for epoch in range(epochs):
        for prompt, winner, loser in preference_data:
            loss = dpo_loss(policy, ref_model, prompt, winner, loser)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

No reward model training. No PPO with its clipping and value functions. No worry about reward hacking (the policy can only move so far from the reference before the loss stops providing signal). The simplicity is real, and it translates to much easier implementation and more stable training.

DPO was quickly adopted after its release. The Zephyr model in October 2023 was one of the first major successes. By 2024, Llama 3 had switched from RLHF to DPO for preference tuning.

> 💡 *Why doesn't DPO have the same problems as RLHF?*
>
> DPO sidesteps the hard parts of RL entirely. It's not that DPO solved the problems; it reformulated the task so those problems don't exist.
>
> **Problem 1: Credit Assignment (the "straw")**
>
> RLHF: Generate sequence → get one scalar → smear it across 100 tokens → noisy gradient
>
> DPO: Here's a winner, here's a loser → directly compute "which do you prefer?" → clean gradient
>
> DPO is supervised learning. You're not exploring and getting sparse feedback. You're being *told* the answer: "this one beats that one." Dense signal, like the firehose.
>
> **Problem 2: Reward Model Can Be Wrong**
>
> RLHF: Train reward model → optimize against it → model finds exploits the reward model didn't anticipate (reward hacking)
>
> DPO: No explicit reward model. The "reward" is implicit in the policy-reference gap. There's nothing to hack.
>
> **Problem 3: Training Instability**
>
> RLHF: Policy generates → reward model scores → value function estimates baseline → PPO clips → KL penalizes. Four models, all interacting, all moving.
>
> DPO: Policy and frozen reference. That's it. Standard supervised learning optimization.
>
> ```python
> # RLHF training: 4 models, complex interaction
> policy, reward_model, value_function, reference  # all in memory
> # generate → score → estimate advantage → clip → update → repeat
>
> # DPO training: 2 models, simple supervised learning
> policy, reference  # reference is frozen
> loss = -log(sigmoid(beta * (winner_gap - loser_gap)))
> loss.backward()  # normal gradient descent
> ```
>
> **The Trade-off**
>
> DPO is *offline*. You learn only from your fixed dataset of preferences. If the dataset doesn't cover some behavior, you won't learn about it.
>
> RLHF is *online*. The policy explores, generates new responses, gets feedback. It can discover good responses that weren't in any dataset.
>
> For most alignment use cases, offline is fine; you have enough preference data. That's why DPO won.

---

## Other Approaches Worth Knowing

Before we move on to test-time compute, let's briefly cover some other methods and distinctions that round out the picture.

### Online vs Offline RL

PPO and DPO represent two fundamentally different approaches to preference learning.

**PPO is "online" RL.** During training, the policy generates new responses, those responses get scored by the reward model, and the scores produce gradients. The policy is constantly generating fresh data for itself to learn from. This means it can explore and discover good responses that weren't in any original dataset. The downside: you need to run the reward model on every generated response, and you need all the PPO machinery (value function, clipping, etc.). It's expensive and complex.

**DPO is "offline" RL.** You have a fixed dataset of preference pairs collected beforehand, and you just train on that. No generation during training, no reward model inference, no value function. Much simpler and cheaper. The downside: you're limited to what's in the dataset. If the dataset doesn't cover some type of response, the model won't learn about it.

In practice, offline methods like DPO have become more popular because they're so much easier to work with. But online methods can achieve better results when you have the compute budget.

### GRPO: Simplifying PPO for Language Models

GRPO (Group Relative Policy Optimization) deserves special attention because it's become the go-to algorithm for RL training of reasoning models. DeepSeek used it for R1, and it's simpler than PPO while being just as effective for LLM training.

The key insight: **you don't need a value function if you generate multiple responses per prompt.**

Recall PPO's problem. You need advantages ("how much better than expected?"), and the "expected" part comes from a value function; a whole separate neural network that predicts what reward you'll get for a given prompt. Training this value function is finicky:
- It's another model to keep in memory
- It needs to stay in sync with your policy
- If it's wrong, your advantages are wrong, and training destabilizes

GRPO's solution is beautifully simple: for each prompt, generate a *group* of responses (say, 16). Score them all. Use the group's mean reward as your baseline.

```python
def grpo_training_step(policy, ref_model, prompt, reward_fn, num_samples=16):
    """
    GRPO: Group Relative Policy Optimization.

    The key insight: generate multiple responses per prompt,
    use the group statistics as your baseline.
    No value function needed.
    """
    # Generate a group of responses
    responses = [policy.generate(prompt) for _ in range(num_samples)]

    # Score each one
    rewards = [reward_fn(r) for r in responses]

    # Compute advantages using GROUP statistics (not a value function)
    mean_reward = sum(rewards) / len(rewards)
    std_reward = std(rewards) + 1e-8  # avoid division by zero
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Now update on each response, weighted by its advantage
    total_loss = 0.0
    for response, advantage in zip(responses, advantages):
        # Standard policy gradient with clipping (like PPO)
        log_prob = compute_log_probability(policy, prompt, response)
        log_prob_old = compute_log_probability(old_policy, prompt, response)

        ratio = exp(log_prob - log_prob_old)
        clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)

        policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

        # KL penalty to stay close to reference
        log_prob_ref = compute_log_probability(ref_model, prompt, response)
        kl_penalty = log_prob - log_prob_ref

        total_loss += policy_loss + beta * kl_penalty

    return total_loss / num_samples
```

Why does this work? Think about it:
- For a hard prompt, most responses fail. Mean reward is low. Even a mediocre success gets positive advantage.
- For an easy prompt, most responses succeed. Mean reward is high. Only exceptional responses get boosted.

The baseline automatically adapts to prompt difficulty; exactly what a value function would do, but without training one.

**GRPO vs PPO comparison:**

|  | PPO | GRPO |
|---|---|---|
| **Baseline** | Value function (learned) | Group mean (computed) |
| **Memory** | Policy + Value + Reference | Policy + Reference |
| **Online/Offline** | Online | Online |
| **Samples per prompt** | 1 (typically) | N (e.g., 16) |
| **Training stability** | Requires careful value function tuning | More stable (no value function to go wrong) |

**The trade-off:** GRPO needs more inference per training step (generating 16 responses instead of 1). But you eliminate an entire model from memory and training. For LLM-scale training where memory is precious, this is often a good trade.

DeepSeek's R1 used GRPO with some aggressive hyperparameters: ε=10.0 for clipping (vs PPO's typical 0.2), 16 samples per prompt, and very long maximum sequence lengths (32K tokens) to allow extended reasoning.

### DPO Variants

After DPO's success, researchers quickly developed variants addressing specific limitations:

**KTO (Kahneman-Tversky Optimization):** DPO requires pairwise comparisons (A is better than B). But sometimes you only have binary feedback: thumbs up or thumbs down on individual responses, without a direct comparison. KTO works with this kind of data. It's named after the psychologists who studied how humans perceive gains vs losses differently.

**IPO (Identity Preference Optimization):** Adds regularization to DPO to prevent overfitting on the preference data.

**ORPO (Odds Ratio Preference Optimization):** Combines the SFT stage and preference stage into one, so you don't need to train twice.

**SimPO (Simple Preference Optimization):** Removes the need for a reference model entirely, making it even simpler than DPO.

You don't need to know the details of all these; the point is that DPO spawned a family of methods, and researchers are still exploring the design space.

### Best-of-N (Rejection Sampling)

This isn't RL at all, but it's surprisingly effective and worth understanding.

The idea is dead simple: instead of training the model to generate better responses, just generate N responses at inference time and pick the best one.

```python
def best_of_n(model, prompt, reward_model, n=16):
    """
    Generate N responses, return the highest-scoring one.

    No training involved; just inference-time filtering.
    """
    responses = [model.generate(prompt) for _ in range(n)]
    scores = [reward_model(prompt + r) for r in responses]
    best_idx = argmax(scores)
    return responses[best_idx]
```

Where does the reward model come from? It's trained exactly as we described in Stage 2 of RLHF: collect human preference pairs (winner vs loser), train a model to predict which response humans would prefer using the Bradley-Terry loss. The reward model takes a prompt and response as input and outputs a scalar score. Here we're just using it to rank responses at inference time rather than to train a policy.

This is often a strong baseline. If your reward model is good, best-of-16 can match or beat models that were trained with RL. DeepSeek uses rejection sampling as part of their R1 training pipeline: generate many responses, keep the correct ones, and use those for further training.

The downside is cost: you're doing N times more inference. But for high-stakes applications where quality matters more than speed, it's a simple way to squeeze more performance out of an existing model.

### RLAIF (RL from AI Feedback)

Collecting human preferences is expensive and slow. What if we used an AI model to provide the feedback instead?

This is RLAIF: instead of humans labeling which response is better, you ask a capable AI model (like GPT-4 or Claude) to judge. The AI provides preferences, you train a reward model on those AI preferences (or use DPO directly), and proceed as usual.

```python
def get_ai_preference(prompt, response_a, response_b, judge_model):
    """
    Ask an AI model which response is better.
    """
    judge_prompt = f"""
    Given this prompt:{prompt}

    Which response is better?

    Response A:{response_a}
    Response B:{response_b}

    Answer with just "A" or "B".
    """
    judgment = judge_model.generate(judge_prompt)
    return "A" if "A" in judgment else "B"
```

Anthropic's Constitutional AI is a version of this, where the AI judges responses based on a set of principles (the "constitution") rather than open-ended preference.

RLAIF is much cheaper and faster than human labeling, though you're ultimately limited by the AI judge's capabilities. If the judge model has blind spots, your trained model will inherit them.

---

## Part 4: Test-Time Compute and Reasoning Models

Everything we've discussed so far is about training-time optimization: making the model better by updating its weights. But starting in late 2024, a different paradigm emerged: **test-time compute scaling**. The idea is that instead of (or in addition to) making the model itself smarter, you can make it *think longer* at inference time.

OpenAI's o1 model, released in September 2024, demonstrated this dramatically. On difficult reasoning tasks, o1 would generate long chains of thought; sometimes thousands of tokens of internal reasoning before producing an answer. This "thinking" wasn't visible to the user, but it allowed the model to work through problems step by step.

Then in January 2025, DeepSeek released their R1 paper, which was a watershed moment for the field. Unlike o1, DeepSeek open-sourced everything: the model, the training methodology, the data recipes. This let us see exactly how test-time reasoning can emerge from RL training.

### The DeepSeek R1 Story

The most surprising finding from DeepSeek was that extended reasoning can emerge purely from RL, without any supervised examples of chain-of-thought reasoning.

Here's the setup. They took a base model that had been pretrained but not instruction-tuned. They gave it math and coding problems where the correct answer is verifiable. The reward was simple: +1 if the final answer matches the ground truth, 0 otherwise. No reward for good reasoning, no reward for style; just binary correctness.

What emerged was remarkable. The model spontaneously started generating long reasoning chains. It would work through problems step by step, backtrack when it hit dead ends, and even express something like surprise when it figured something out (the paper calls these "aha moments"). Nobody taught it to do this; it discovered that thinking step by step helps it get correct answers, which means higher reward.

```python
def r1_zero_reward(response, correct_answer):
    """
    The reward function for DeepSeek R1-Zero.

    Shockingly simple: does the final answer match?
    """
    # Extract the final answer from the response
    # (they used a specific format: answer inside \boxed{})
    extracted_answer = extract_boxed_answer(response)

    # Binary reward
    if extracted_answer == correct_answer:
        return 1.0
    else:
        return 0.0
```

This is one of those results that seems almost too simple to be true. No reward model trained on human preferences. No demonstrations of chain-of-thought reasoning. Just "did you get the right answer?" And yet the model learns to reason.

### GRPO: RL Without a Value Model

The RL algorithm DeepSeek used is called GRPO (Group Relative Policy Optimization). It's a simplification of PPO that's particularly well-suited to language models.

Recall that in PPO, we need advantages (how much better than expected), and these come from a value function that predicts expected reward. Training this value function is tricky; it's another neural network that needs to be kept in sync with the policy, it adds memory overhead, and getting it wrong can destabilize training.

GRPO's insight is that we can estimate the expected reward without a value function. For each prompt, we generate multiple responses (a "group"). We compute the reward for each response. Then we use the mean reward of the group as our baseline.

```python
def grpo_update(policy, ref_model, prompt, num_samples=16, beta=0.01):
    """
    GRPO: Group Relative Policy Optimization.

    Key idea: generate a group of responses for each prompt,
    use the group's mean reward as the baseline.
    """
    # Generate multiple responses for this prompt
    responses = [policy.generate(prompt) for _ in range(num_samples)]

    # Score each response
    rewards = [compute_reward(response) for response in responses]

    # Compute advantages using group statistics
    mean_reward = sum(rewards) / len(rewards)
    std_reward = compute_std(rewards) + 1e-8  # avoid division by zero
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Update policy on each response
    total_loss = 0.0
    for response, advantage in zip(responses, advantages):
        # Compute log probability ratio (like in PPO)
        log_prob_policy = compute_log_probability(policy, prompt, response)
        log_prob_ref = compute_log_probability(ref_model, prompt, response)

        # KL penalty to stay close to reference
        kl_penalty = log_prob_policy - log_prob_ref

        # GRPO uses clipping like PPO, but with different hyperparameters
        ratio = math.exp(log_prob_policy - log_prob_old)
        clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)

        loss = -min(ratio * advantage, clipped_ratio * advantage)
        loss += beta * kl_penalty

        total_loss += loss

    return total_loss / num_samples
```

This has several advantages. Memory usage drops dramatically; you don't need to keep a value model and its optimizer states. Training is more stable because you're not relying on a potentially-incorrect value function. And the baseline naturally adapts to the difficulty of each prompt; for hard prompts where most responses fail, the baseline will be low, so even a barely-correct response gets positive advantage.

DeepSeek's hyperparameters were interesting. They used a much larger clipping range (ε=10.0) than standard PPO (ε=0.2). They generated 16 responses per prompt. They used very long maximum lengths (32,768 tokens) to allow for extended reasoning. The KL coefficient was small (0.001) to allow significant deviation from the reference.

### The Full R1 Pipeline

The pure RL approach (R1-Zero) produces a model that can reason, but it has rough edges. Its responses are sometimes poorly formatted, it might fail to follow instructions unrelated to reasoning, and its behavior on non-reasoning tasks is unchanged from the base model.

DeepSeek's full R1 model uses a multi-stage training process to address this:

```python
def r1_full_training_pipeline(base_model):
    """
    The complete DeepSeek R1 training pipeline.
    """
    # Stage 1: Cold start
    # Train on a small dataset (~1000 examples) of high-quality
    # reasoning chains. This gives the model the right format
    # before we start RL.
    model = supervised_finetune(base_model, cold_start_data)

    # Stage 2: Large-scale RL on reasoning tasks
    # This is where the main reasoning capability is learned.
    # Tens of thousands of math and code problems.
    model = grpo_training(model, reasoning_tasks,
                          reward_fn=correctness_reward)

    # Stage 3: Rejection sampling SFT
    # Generate many responses, keep only the correct ones,
    # fine-tune on this filtered data.
    good_responses = []
    for task in diverse_task_set:
        responses = [model.generate(task) for _ in range(10)]
        good_ones = [r for r in responses if is_correct(r)]
        good_responses.extend(good_ones)

    model = supervised_finetune(model, good_responses)

    # Stage 4: Final RL with broader rewards
    # Add helpfulness, safety, and formatting to the mix.
    model = grpo_training(model, all_tasks,
                          reward_fn=combined_reward)

    return model
```

The rejection sampling stage is worth highlighting. Once you have a model that can sometimes produce good reasoning, you can use it to generate training data for itself. Generate many responses, filter to keep the correct ones, and fine-tune on those. This is a form of self-improvement that amplifies the RL training.

### The Numbers

The R1 results were striking. On AIME 2024 (a challenging high school math competition), the base model achieved 15.6%. R1-Zero, trained with pure RL and no reasoning demonstrations, reached 71.0%. R1, with the full pipeline, achieved competitive results with o1.

Even more interesting was the distillation story. DeepSeek distilled R1's reasoning capabilities into smaller models (1.5B to 70B parameters) by training them on R1's outputs. The 7B distilled model achieved 55.5% on AIME, vastly outperforming what a 7B model could do without this reasoning capability. This suggests that the reasoning patterns learned by R1 can be transferred to much smaller models through standard distillation.

---

## Part 5: Practical RL Training Today

The landscape of RL for LLMs has evolved rapidly. Let me walk through the major approaches available today and when you might use each.

### OpenAI's Reinforcement Fine-Tuning API

In late 2024, OpenAI began rolling out "Reinforcement Fine-Tuning" as an API product. The idea is that you define a scoring function (a "grader"), and OpenAI handles all the RL infrastructure.

```python
def code_grader(model_response, test_cases):
    """
    Example grader for code generation tasks.

    You define this; OpenAI runs RL to maximize it.
    """
    try:
        # Execute the model's code
        exec(model_response)

        # Run test cases
        passed = 0
        for test_input, expected_output in test_cases:
            actual = evaluate(test_input)
            if actual == expected_output:
                passed += 1

        return passed / len(test_cases)
    except:
        return 0.0

def math_grader(model_response, correct_answer):
    """
    Grader for math problems.
    """
    extracted = extract_final_answer(model_response)
    return 1.0 if extracted == correct_answer else 0.0
```

The appeal of RFT is that it handles all the complexity: the RL algorithm, the distributed training, the stability tricks. You just define what "correct" means for your task. OpenAI recommends tasks that are unambiguous (experts would agree on correct answers), have variable difficulty (not all easy or all hard), and are guess-proof (luck shouldn't produce high scores).

Real customers have used this for domain-specific tasks. Stripe used it to train models on their API; the grader checked whether generated code made valid API calls. A scheduling company used it for calendar management; the grader checked whether the model correctly handled edge cases in appointment booking.

### Constitutional AI and Self-Improvement

Anthropic's Constitutional AI takes a different approach. Instead of human feedback, it uses AI feedback based on a set of principles (the "constitution").

```python
def constitutional_revision(model, response, principles):
    """
    Constitutional AI: the model critiques and revises its own outputs.
    """
    # Step 1: Ask the model to critique itself
    critique_prompt = f"""
    Here is a response:{response}

    Please critique this response according to these principles:
{principles}

    What problems do you see?
    """
    critique = model.generate(critique_prompt)

    # Step 2: Ask the model to revise based on the critique
    revision_prompt = f"""
    Original response:{response}
    Critique:{critique}

    Please write an improved response that addresses the critique.
    """
    improved = model.generate(revision_prompt)

    return improved
```

The training process generates many such critique-revision pairs and uses them as training data. The principles might include things like "be helpful and informative," "don't help with dangerous requests," "be honest and don't fabricate information." The model learns to internalize these principles by repeatedly critiquing and improving its outputs according to them.

### Environment-Based Agentic Training

The newest frontier is training models to interact with real environments: code interpreters, web browsers, file systems, APIs. The reward comes from task completion rather than static evaluations.

```python
def code_environment_training(model, problems):
    """
    Train a model to write code by actually running it.
    """
    for problem in problems:
        # Present the problem
        state = {"problem": problem.description, "code": ""}

        done = False
        while not done:
            # Model generates or modifies code
            action = model.generate(state)

            # Actually execute the code
            result = execute_code(action)

            # Check if tests pass
            if run_tests(result, problem.tests):
                reward = 1.0
                done = True
            elif has_syntax_error(result):
                reward = -0.5
                state["error"] = result.error_message
            else:
                reward = -0.1
                state["output"] = result.output

        # Train on this trajectory
        update_policy(model, trajectory, reward)
```

This is conceptually similar to how AlphaGo was trained by playing games, but the "game" is now code execution, web navigation, or computer use. The model learns not just to produce plausible-looking outputs, but to actually accomplish tasks in real environments.

---

## Part 6: Choosing the Right Approach

Having surveyed the landscape, let me offer some practical guidance on which approach to use for different situations.

If you have preference data and want simplicity, use **DPO**. It's stable, easy to implement, and doesn't require juggling multiple models. The main requirement is that your preference data should have clear winners and losers; DPO doesn't handle ties or partial preferences well.

If you have tasks with verifiable answers (math, coding, factual questions), consider **outcome-based RL** like GRPO. The simple binary rewards (correct/incorrect) are often enough to produce significant improvements, as DeepSeek demonstrated. This works especially well when you want to improve reasoning, because extended thinking can be learned from scratch given only outcome feedback.

If you want to use an existing frontier model without running your own training, **OpenAI's RFT** or similar API products let you define custom graders and get RL-trained models without managing infrastructure.

If you need to balance multiple objectives (helpfulness, harmlessness, honesty), **Constitutional AI** or multi-objective RLHF approaches let you specify principles rather than just optimizing a single score.

And if you're building agents that interact with environments, you'll need **agentic RL** setups where the reward comes from task completion in real environments rather than static evaluations.

### The Key Patterns to Remember

Let me distill the key computational patterns that appear across all these methods:

**Policy gradient**: The fundamental idea; weight the gradient of log-probabilities by rewards or advantages.

```python
loss = -advantage * log_prob(response)
```

**Advantage computation**: How much better than expected? This reduces variance.

```python
advantage = reward - baseline
```

**Clipping**: Don't change too much in one update. This adds stability.

```python
clipped_ratio = clip(new_prob / old_prob, 1-epsilon, 1+epsilon)
```

**KL penalty**: Stay close to a reference model. This prevents reward hacking.

```python
adjusted_reward = reward - kl_coef * (log_prob_policy - log_prob_ref)
```

**Direct preference optimization**: Skip the reward model entirely.

```python
loss = -log(sigmoid(beta * (winner_log_ratio - loser_log_ratio)))
```

These patterns combine in different ways across the methods we've discussed, but understanding them individually makes it much easier to understand any specific algorithm you might encounter.

---

## Part 7: The Reasoning Model Landscape (2024-2025)

We've discussed DeepSeek R1 in detail because it's open and well-documented. But what are OpenAI, Anthropic, Google, and other labs doing? Are they doing something similar, or something more sophisticated?

The short answer: they're all converging on similar ideas, but with interesting variations in implementation and philosophy. Let's break it down.

### The Core Insight Everyone Shares

Every major lab has arrived at the same fundamental insight: you can dramatically improve model performance by training models to "think" before answering, and by spending more compute at inference time.

Andrej Karpathy summarized it well:

> By training LLMs against automatically verifiable rewards across a number of environments (e.g. think math/code puzzles), the LLMs spontaneously develop strategies that look like "reasoning" to humans; they learn to break down problem solving into intermediate calculations and they learn a number of problem solving strategies for going back and forth to figure things out.

This is RLVR (Reinforcement Learning from Verifiable Rewards) in a nutshell. The terminology varies; "reasoning models", "inference-scaling", "test-time compute", "thinking models"; but they all refer to the same paradigm.

### OpenAI: o1, o3, and the Private Chain of Thought

OpenAI kicked off the reasoning model era with o1 in September 2024, followed by o3 in late 2024/early 2025.

**Training Approach**: OpenAI uses what they call "large-scale reinforcement learning" to teach models to think productively. The key insight from their public statements:

```python
# Conceptual: OpenAI's approach
# 1. Start with a capable base model (GPT-4 class)
# 2. Train with RL against verifiable rewards
# 3. Model learns to generate "private chain of thought" before answering

def o1_inference(prompt):
    # Generate internal reasoning (hidden from user)
    thinking_tokens = model.generate_thinking(prompt)  # Not shown to user

    # Generate final answer informed by thinking
    answer = model.generate_answer(prompt, thinking_tokens)

    return answer  # Only this is returned
```

**Key characteristics**:
- **Hidden reasoning**: Unlike DeepSeek R1 or Claude, o1/o3's chain of thought is private. Users see a summary, not the raw thinking
- **Emergent strategies**: OpenAI emphasizes these reasoning behaviors emerge spontaneously from RL training, not from explicit programming
- **Test-time scaling**: Performance improves logarithmically with thinking tokens allowed

**Benchmark Results** (to give you a sense of the leap):
- o1 on AIME 2024: 74% (vs GPT-4o at 12%)
- o3 on GPQA Diamond: 87.7%
- o3 on Codeforces: 2727 Elo (International Grandmaster level, top ~200 humans)
- o3 on ARC-AGI: 3x improvement over o1

**The "reasoning effort" dial**: OpenAI lets you control how much thinking the model does:

```python
# API concept (simplified)
response = openai.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": "Solve this math problem..."}],
    reasoning_effort="high"  # low, medium, high
)
```

Higher effort = more thinking tokens = better results on hard problems = higher cost.

**By mid-2025**: OpenAI converged reasoning into their flagship models. GPT-5 family includes reasoning capabilities rather than having separate "reasoning models". The o-series became a dial you can turn rather than a separate product.

### Anthropic: Extended Thinking and User Control

Anthropic took a different philosophical approach with Claude's extended thinking.

**Key difference from OpenAI**: Anthropic makes thinking visible (or at least summarized) and gives users control over the thinking budget.

```python
# Anthropic's approach (conceptual)
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # User controls thinking budget
    },
    messages=[{"role": "user", "content": "Complex reasoning task..."}]
)

# Response includes thinking blocks you can inspect
for block in response.content:
    if block.type == "thinking":
        print(f"Claude's reasoning:{block.thinking[:200]}...")
    elif block.type == "text":
        print(f"Final answer:{block.text}")
```

**Training approach**: Anthropic uses Constitutional AI combined with RL. The model learns when and how to think through a combination of:
- Constitutional principles (what behaviors are good/bad)
- RL on feedback to improve reasoning quality
- Human preference data

**Key characteristics**:
- **Visible thinking**: Users can see Claude's reasoning process (though Claude 4+ models show summarized thinking rather than raw tokens)
- **User-controlled budget**: You specify how many tokens Claude can use for thinking (1,024 to 128K+)
- **Serial test-time compute**: Claude thinks sequentially before answering; more budget = more thorough reasoning
- **Interleaved thinking**: Claude can think between tool calls, not just at the start

```python
# Performance scales with thinking budget
# Anthropic's data shows logarithmic improvement on math problems

thinking_budgets = [1024, 4096, 16384, 65536]
# Rough performance pattern:
# More thinking → Better accuracy, diminishing returns at very high budgets
```

**Claude model evolution**:
- Claude 3.7 Sonnet (Nov 2024): First extended thinking, raw thinking visible
- Claude 4/4.5 family (2025): Summarized thinking, interleaved thinking for tools
- Claude Haiku 4.5: First "small" model with extended thinking capability

**Safety considerations**: Anthropic can "redact" thinking blocks if they contain content flagged by safety systems. The model still uses that reasoning internally, but it's not shown to users.

### Google: Gemini Thinking and Deep Think

Google has been rapidly iterating on reasoning with the Gemini family.

**Evolution**:
- Gemini 2.0 Flash Thinking: First thinking model
- Gemini 2.5 Pro/Flash: Thinking built into all models
- Gemini 3 Pro/Flash: State-of-the-art reasoning
- Gemini 3 Deep Think: Enhanced reasoning mode for hardest problems

**Key characteristics**:
- **Thinking built-in**: Unlike OpenAI's separate o-series, Google builds thinking into all Gemini 2.5+ models
- **Configurable thinking budgets**: Like Anthropic, developers can control thinking depth
- **Deep Think mode**: For the hardest problems (complex math, coding), Deep Think uses iterative rounds of reasoning exploring multiple hypotheses

```python
# Gemini API concept
from google import generativeai as genai

model = genai.GenerativeModel('gemini-3-flash')

# Control thinking with generation config
response = model.generate_content(
    "Solve this olympiad problem...",
    generation_config={
        "thinking_config": {
            "thinking_budget": 8192  # Control thinking tokens
        }
    }
)

# Can access thought summaries
print(response.thought_summary)
```

**Benchmarks** (Gemini 3 Flash):
- GPQA Diamond: 90.4%
- Humanity's Last Exam: 33.7% (no tools)
- SWE-bench Verified: 78%

**Deep Think**: Available to Google AI Ultra subscribers, uses "iterative rounds of reasoning" and "multiple hypotheses" exploration. Takes longer (minutes rather than seconds) but pushes accuracy further on the hardest problems.

### Chinese Labs: QwQ, Qwen3, and Open Weights

Chinese labs, particularly Alibaba's Qwen team, have been remarkably open about their reasoning models.

**QwQ (Qwen with Questions)**: Released November 2024, this was one of the first open-weight reasoning models.

```python
# QwQ inference (real code, works with Hugging Face)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/QwQ-32B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r's are in strawberry?"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# QwQ will "think" before answering
outputs = model.generate(**inputs, max_new_tokens=32768)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
```

**Key characteristics**:
- **Open weights**: You can download and run QwQ locally
- **32B parameters**: Relatively small, runs on consumer hardware with quantization
- **RL-trained**: Like DeepSeek R1, uses reinforcement learning to develop reasoning

**Qwen3 (April 2025)**: The full Qwen3 family supports both thinking and non-thinking modes:

```python
# Qwen3 supports toggling thinking on/off
response = qwen3_model.generate(
    prompt,
    enable_thinking=True  # Toggle reasoning mode
)
```

**Qwen3 highlights**:
- Dense models: 0.6B to 32B parameters
- MoE models: 30B (3B active), 235B (22B active)
- 36 trillion training tokens
- 119 languages supported
- Apache 2.0 license (fully open)

### What's the Same, What's Different?

Let me summarize the key similarities and differences:

**Everyone agrees on**:
1. RL training against verifiable rewards produces reasoning behaviors
2. Test-time compute scaling works; more thinking = better answers
3. These models excel at math, coding, science, and multi-step planning
4. The capability is worth the extra inference cost for hard problems

**Key differences**:

| Aspect | OpenAI | Anthropic | Google | Chinese Labs |
|---|---|---|---|---|
| Thinking visibility | Hidden (private CoT) | Visible/summarized | Summarized | Visible |
| User control | Effort levels (low/med/high) | Token budget | Token budget | Token budget |
| Model separation | Originally separate (o1), now converged | Integrated with toggle | Integrated | Both dedicated (QwQ) and integrated (Qwen3) |
| Open weights | No | No | No | Yes (Qwen, DeepSeek) |

### Process Reward Models vs Outcome Reward Models

One technical distinction worth understanding: how do you provide rewards during training?

**Outcome Reward Model (ORM)**: Only judges the final answer.

```python
def outcome_reward(problem, full_solution, correct_answer):
    """Reward based only on whether final answer is correct."""
    model_answer = extract_final_answer(full_solution)
    return 1.0 if model_answer == correct_answer else 0.0
```

**Process Reward Model (PRM)**: Judges each step of the reasoning.

```python
def process_reward(problem, solution_steps, step_labels):
    """Reward each step of reasoning."""
    rewards = []
    for step, is_correct in zip(solution_steps, step_labels):
        rewards.append(1.0 if is_correct else -1.0)
    return rewards

# PRM can catch errors early:
# Step 1: "Let x = 5" → correct (+1)
# Step 2: "Then 2x = 15" → wrong! (-1) ← PRM catches this
# Step 3: "Therefore..." → doesn't matter, error already flagged
```

**Why this matters**: PRMs provide denser feedback (reward at every step, not just the end), which can make training more efficient. But they require labeled intermediate steps, which is expensive to collect.

OpenAI released PRM800K (800K human-labeled step annotations) to advance this research. The DeepSeek R1 paper uses outcome-based rewards (GRPO with final answer correctness), which is simpler but may be less sample-efficient.

### The Convergence Trend

Perhaps the most important observation: by late 2025, reasoning is no longer a separate "product line" at most labs. Instead:
- OpenAI: Reasoning converged into GPT-5 family
- Anthropic: Extended thinking is a feature of all Claude 4+ models
- Google: Thinking is built into Gemini 2.5+ by default
- Open source: Qwen3 supports both modes in the same model

This suggests reasoning/thinking is becoming a fundamental capability rather than a specialty. The question isn't "should I use a reasoning model?" but "how much thinking budget do I allocate for this task?"

```python
# The emerging pattern: reasoning as a dial, not a switch
def answer_question(question, difficulty):
    if difficulty == "trivial":
        return model.generate(question, thinking_budget=0)
    elif difficulty == "moderate":
        return model.generate(question, thinking_budget=4096)
    elif difficulty == "hard":
        return model.generate(question, thinking_budget=32768)
    else:  # "extreme"
        return model.generate(question, thinking_budget=128000)
```

### What About Safety?

Reasoning models raise interesting safety questions:

**Visible CoT enables monitoring**: If you can see the model's reasoning, you can check for misaligned behaviors. Anthropic and others see this as a safety advantage.

**But is CoT faithful?**: Research shows models sometimes use information without verbalizing it in their chain of thought. The CoT might be a post-hoc rationalization rather than the actual reasoning process.

**Deliberative alignment**: OpenAI published research showing o1-class models can use reasoning to better follow safety guidelines. The model "thinks" about whether a request violates policies before responding.

**Hidden vs visible trade-off**: OpenAI hides reasoning (citing safety and competitive advantage). Anthropic shows it (citing transparency). Both have arguments; hidden CoT prevents users from learning to manipulate reasoning, but visible CoT enables auditing.

### Practical Implications

If you're building with these models, here's what matters:

**When to use reasoning/thinking**:
- Math problems, especially multi-step
- Code that requires planning or debugging
- Scientific reasoning
- Complex analysis or comparison tasks
- Tasks where accuracy matters more than speed

**When NOT to use reasoning**:
- Simple factual questions
- Creative writing (usually)
- Tasks where latency matters more than accuracy
- High-volume, low-stakes queries

**Cost considerations**:
- Thinking tokens cost money (input + output pricing)
- But reasoning models often need fewer retries
- Net cost depends on your accuracy requirements

```python
# Cost-benefit pseudocode
def choose_approach(task, accuracy_requirement, latency_budget):
    if accuracy_requirement > 0.95 and latency_budget > 30:
        return "reasoning_model_high_budget"
    elif accuracy_requirement > 0.8:
        return "reasoning_model_low_budget"
    else:
        return "standard_model"
```

The reasoning model landscape is still evolving rapidly. But the core insight; that training models to think step-by-step via RL dramatically improves their capabilities; is now firmly established. The question isn't whether to use these techniques, but how to use them effectively for your specific needs.

---

## References and Further Reading

The field moves fast, but these resources provide solid foundations:

For RLHF fundamentals, the HuggingFace blog posts (https://huggingface.co/blog/rlhf) are excellent practical introductions. Nathan Lambert's RLHF learning resources (https://www.interconnects.ai/p/rlhf-resources) compile many useful papers and tutorials.

For DPO, the original paper by Rafailov et al. (https://arxiv.org/abs/2305.18290) is clearly written and the math is worth working through.

For test-time compute and reasoning, the DeepSeek R1 paper (https://arxiv.org/abs/2501.12948) is remarkably transparent about methodology and results.

For PPO specifically, the "37 Implementation Details of PPO" blog post (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) documents many practical tricks that aren't in the original paper.

For Constitutional AI, Anthropic's research blog has detailed write-ups of their methodology and the principles they use.
