# Scaling Laws

*How much better does a language model get when you make it bigger, train it longer, or give it more data?*

---

Scaling laws are empirical relationships that predict how language model performance changes as you scale up three key resources: model size (parameters), training data (tokens), and compute (FLOPs). They're not derived from theory—researchers discovered them by training hundreds of models at different scales and fitting curves to the results.

Why do we care? Training a frontier model costs tens of millions of dollars. You don't want to guess how big to make it or how long to train it. Scaling laws let you run small experiments, measure the relationships, and extrapolate to predict what a much larger model would achieve—before committing the compute.

## What Do We Mean by "Compute"?

At first glance, this seems circular. If compute is measured in FLOPs (floating point operations), and FLOPs ≈ 6 × parameters × tokens, then compute is just a function of the other two. Why treat it as a separate axis?

Compute *is* derived from parameters and data. The reason to discuss it separately comes down to **what question each scaling law answers**:

- **L(N)** asks: "If I have unlimited data, what happens as I make the model bigger?" This isolates the effect of model capacity.

- **L(D)** asks: "If I have an infinitely large model, what happens as I train on more data?" This isolates the effect of data.

- **L(C)** asks: "If I have a fixed compute budget and allocate it *optimally* between N and D, what's the best loss I can achieve?"

The compute scaling law is the **envelope**—it tells you the best possible result for a given budget, assuming you don't waste FLOPs on a bad parameter/data split. In practice, this is the question that matters: "I have X dollars worth of GPU time. What's the best model I can train?"

```python
# Same compute budget, different allocations:
budget_flops = 1e21

# Option A: 1B params, lots of data
option_a = {"N": 1e9, "D": budget_flops / (6 * 1e9)}   # D ≈ 167T tokens

# Option B: 10B params, moderate data
option_b = {"N": 10e9, "D": budget_flops / (6 * 10e9)} # D ≈ 16.7T tokens

# Option C: 100B params, little data
option_c = {"N": 100e9, "D": budget_flops / (6 * 100e9)} # D ≈ 1.67T tokens

# L(C) tells you: which option gives the lowest loss?
# Kaplan said option C (favor parameters)
# Chinchilla said option B (balance both)
```

## What Do We Mean by "Loss"?

Loss in scaling laws is cross-entropy loss: for each token position, $-\log p(\text{correct token})$. During training you compute this over a batch and average, but batch size and sequence length don't affect the expected value—only the variance of your estimate. Whether you average over 1,000 tokens or 1 million, you're estimating the same underlying quantity:

$$L = \mathbb{E}_{x \sim p_{\text{data}}}[-\log p_{\text{model}}(x)]$$

Scaling laws report **validation loss**—measured on held-out data—not training loss. Training loss can be artificially low from memorization. The validation loss estimates how well the model generalizes to tokens it hasn't seen, which is what we actually care about.

## A Brief History

**2020: Kaplan et al. at OpenAI** published "Scaling Laws for Neural Language Models," the first systematic study. They trained models ranging from 768 parameters to 1.5 billion and found remarkably clean power-law relationships. The loss followed predictable curves as you scaled parameters, data, or compute. Their key recommendation: **prioritize model size over training duration**. Given a fixed compute budget, you should train a very large model on relatively few tokens rather than a smaller model on more data.

**2022: Hoffmann et al. at DeepMind** (the "Chinchilla" paper) challenged this advice. They trained over 400 models and found that Kaplan's recommendation was wrong—models were being undertrained. Their conclusion: **parameters and tokens should scale equally**. For every doubling of model size, double the training tokens. This led to the "20 tokens per parameter" rule of thumb: a 70B parameter model should see 1.4 trillion tokens.

**2024-2025: Post-Chinchilla refinements** have pushed the ratio even higher. LLaMA 3 trained with over 15,000 tokens per parameter. Researchers found that "overtraining" beyond the compute-optimal point can be worthwhile if you want a smaller, faster model at inference time. Meanwhile, a new scaling axis emerged: **inference-time compute** (thinking tokens, chain-of-thought, search). Models like o1 showed that spending more compute *at generation time* can dramatically improve reasoning, opening a second front in the scaling wars.

## The Core Idea: Power Laws

All scaling laws share the same mathematical form: a **power law**. Before diving into the formulas, let's build intuition for what a power law means.

Imagine you're filling a bucket with water. Linear scaling would mean: 2x the effort → 2x the water. But power laws work differently. They say: 2x the effort → maybe 1.3x the improvement. Or 10x the effort → maybe 2x the improvement. The more you scale, the harder it gets to squeeze out gains.

In Python terms:

```python
def power_law(x, a, alpha):
    """
    a: a constant that sets the overall scale
    alpha: the exponent that controls how fast gains diminish

    If alpha = 1, doubling x doubles the output (linear)
    If alpha = 0.5, doubling x increases output by ~41% (√2 ≈ 1.41)
    If alpha = 0.1, doubling x increases output by only ~7%
    """
    return a * (x ** alpha)
```

Scaling laws for language models typically use *negative* exponents because we're measuring loss (which we want to go *down*). So as you increase parameters/data/compute, the loss decreases—but with diminishing returns governed by the exponent.

### Visualizing Power Laws

The exponent makes a huge difference in how fast you improve. Small exponents (α ≈ 0.05-0.1) mean grinding, incremental gains. Larger exponents (α ≈ 0.5) mean rapid improvement early that tapers off.

![[assets/scaling_exponents_comparison.png]]

**[→ Interactive version with slider](assets/scaling_exponents_interactive.html)** — drag the slider to see how different exponents change the curve shape.

## Kaplan's Scaling Laws (2020)

Kaplan and colleagues at OpenAI discovered that language model loss follows three independent power laws:

### The Three Laws

**1. Scaling with parameters (N):**

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

Breaking down each term:
- $L(N)$ = the loss (cross-entropy) when you have $N$ parameters
- $N$ = the number of non-embedding parameters in your model
- $N_c$ = a fitted constant (~8.8 × 10¹³ in Kaplan's fits). Think of it as a "scale factor" that makes the units work out. It has no deep meaning—it's just whatever value makes the curve fit the data.
- $\alpha_N$ = the scaling exponent (≈ 0.076). This controls how fast loss decreases as you add parameters. A small exponent means slow improvement.

In Python:
```python
def loss_from_params(N, N_c=8.8e13, alpha_N=0.076):
    return (N_c / N) ** alpha_N
```

**2. Scaling with data (D):**

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

Breaking down each term:
- $L(D)$ = the loss when trained on $D$ tokens
- $D$ = the number of training tokens
- $D_c$ = another fitted constant (~5.4 × 10¹³)
- $\alpha_D$ = the data scaling exponent (≈ 0.095). Slightly larger than $\alpha_N$, meaning data scales a bit faster than parameters.

```python
def loss_from_data(D, D_c=5.4e13, alpha_D=0.095):
    return (D_c / D) ** alpha_D
```

**3. Scaling with compute (C):**

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

Breaking down each term:
- $L(C)$ = the loss achievable with compute budget $C$, assuming optimal allocation between N and D
- $C$ = total training compute in FLOPs
- $C_c$ = fitted constant (~3.1 × 10⁸)
- $\alpha_C$ = compute scaling exponent (≈ 0.057). The smallest exponent—compute scaling is the slowest because you're paying for both model size AND training.

```python
def loss_from_compute(C, C_c=3.1e8, alpha_C=0.057):
    return (C_c / C) ** alpha_C
```

### How Did Kaplan Discover These Laws?

**Step 1: Train lots of models at different scales.**

Kaplan's team trained models ranging from 768 parameters to 1.5 billion parameters. For each model size, they trained on varying amounts of data and recorded the final loss. This gave them a big table of (N, D, Loss) tuples.

**Step 2: Isolate the effect of each variable.**

To find L(N)—how loss depends on parameters alone—you need to remove data as a bottleneck. They did this by training each model size "to convergence," meaning they kept training until loss stopped improving. At convergence, the model has seen enough data that adding more wouldn't help. Any remaining loss is due to limited model capacity (N), not limited data (D).

To find L(D)—how loss depends on data alone—you need to remove parameters as a bottleneck. They used their largest models and varied the training data. A sufficiently large model won't be capacity-limited, so any loss differences come from the data.

**Step 3: Plot on log-log axes and look for straight lines.**

If you plot loss vs. parameters on regular axes, you get a curve. But if you plot log(loss) vs. log(parameters), you get a straight line. A straight line on a log-log plot is the signature of a power law.

Why? Take any power law: $L = a \cdot N^{-\alpha}$

Apply log to both sides:
$$\log(L) = \log(a) - \alpha \cdot \log(N)$$

This is just $y = b + mx$—a line with slope $-\alpha$ and intercept $\log(a)$.

Kaplan plotted their data on log-log axes and observed remarkably straight lines. That's how they knew the relationship was a power law, not exponential, not logarithmic, not something else.

```python
import numpy as np

# Simulated experimental data (what Kaplan would have measured)
N_experiments = np.array([1e6, 1e7, 1e8, 1e9])  # model sizes
L_experiments = np.array([4.2, 3.8, 3.5, 3.2])  # measured losses

# Transform to log space
log_N = np.log(N_experiments)
log_L = np.log(L_experiments)

# In log space, this should be a straight line: log(L) = log(a) - α*log(N)
# We can fit this with simple linear regression
# slope = -α, intercept = log(a)
```

**Step 4: Fit the line to get α and a.**

Once you know it's a power law, finding the constants is just linear regression on the log-transformed data:

```python
from scipy import stats

# Linear regression in log space
slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_L)

# Extract the power law parameters
alpha = -slope           # The exponent (negated because loss decreases with N)
a = np.exp(intercept)    # The constant

print(f"Fitted power law: L = {a:.2e} * N^(-{alpha:.3f})")
print(f"R² = {r_value**2:.4f}")  # How well the line fits (should be very close to 1)
```

The R² value tells you how well the power law fits. Kaplan found R² values extremely close to 1, meaning the power law captured almost all the variance in the data.

**Why the ratio form (N_c / N)?**

Kaplan wrote the law as $L = (N_c / N)^{\alpha}$ rather than $L = a \cdot N^{-\alpha}$. These are mathematically equivalent:

$$(N_c / N)^{\alpha} = N_c^{\alpha} \cdot N^{-\alpha} = a \cdot N^{-\alpha}$$

where $a = N_c^{\alpha}$.

The ratio form has a nice interpretation: $N_c$ is the model size where loss would equal 1 (if the power law held that far). It's a characteristic scale. But operationally, it's just a different way of writing the same fitted constant.

**Step 5: Derive the compute law from the other two.**

The compute scaling law L(C) is trickier because compute isn't independent—it's determined by N and D via C = 6ND. To find L(C), Kaplan:

1. For each compute budget C, swept different (N, D) combinations that satisfy C = 6ND
2. Found the (N, D) pair that gave the lowest loss for that budget
3. Plotted the optimal loss against compute budget
4. Observed another power law

This gave them α_C ≈ 0.057, which is smaller than both α_N and α_D because you're paying the cost of both.

Kaplan didn't assume power laws. They trained models, plotted the data, noticed straight lines on log-log plots, and fit the lines. The exponents and constants fell out of basic curve fitting.

### What Do These Exponents Mean?

The exponents are small numbers (0.05-0.1), which tells you something important: **scaling is expensive**. Let's trace through a concrete example with parameters:

```python
# Kaplan's parameter scaling exponent
alpha_N = 0.076

# How much does loss decrease when we double parameters?
# If L ∝ N^(-0.076), then:
# L_new / L_old = (2N / N)^(-0.076) = 2^(-0.076) ≈ 0.95

improvement = 2 ** (-0.076)
print(f"Doubling parameters reduces loss to {improvement:.1%} of original")
# Output: Doubling parameters reduces loss to 94.9% of original
```

So doubling your model size only reduces loss by about 5%. To cut loss in half, you'd need to increase parameters by a factor of:

```python
import math
# We want (factor)^(-0.076) = 0.5
# factor = 0.5^(1/-0.076) = 0.5^(-13.16)
factor = 0.5 ** (1 / -0.076)
print(f"To halve the loss, increase parameters by {factor:.0f}x")
# Output: To halve the loss, increase parameters by 8103x
```

This is the brutal reality of scaling: to halve the loss with parameters alone, you need ~8000x more parameters. The exponents are small, so the curves flatten quickly.

### The Asymmetry Between Parameters and Data

Kaplan noticed something in the exponents: α_N (0.076) < α_D (0.095). This means **scaling parameters is more efficient than scaling data**. A dollar spent on more parameters buys more loss reduction than a dollar spent on more training tokens.

From this, they derived a recommendation: when you have a fixed compute budget, spend most of it on model size. Train a very large model on relatively few tokens, and stop before convergence. Don't waste compute running a smaller model for longer.

This advice shaped GPT-3's training. It's also what Chinchilla would later prove wrong.

### Computing Training FLOPs

How do parameters, data, and compute relate? The commonly-used approximation is:

$$C \approx 6 \times N \times D$$

But where does the 6 come from? And how accurate is this?

**The 6× multiplier explained:**

A "matmul" (matrix multiplication) is the core operation in neural networks. When you multiply a vector by a weight matrix, each output element requires a multiply and an add for every input element. For a weight matrix with W parameters, that's 2W FLOPs per forward pass.

During training, you also need the backward pass. Computing gradients requires roughly twice the FLOPs of the forward pass: once to compute gradients with respect to activations (so you can propagate backward), and once for gradients with respect to weights (so you can update them). That's 4W FLOPs.

Total per token: **2 (forward) + 4 (backward) = 6 FLOPs per parameter per token**.

```python
def simple_flops_estimate(params, tokens):
    """
    The quick approximation used in scaling law papers.

    params: non-embedding parameters
    tokens: training tokens
    """
    return 6 * params * tokens
```

**What the simple formula ignores:**

The 6ND formula counts FLOPs from weight matrices (linear layers), but a transformer also has:

1. **Embedding lookups** — these aren't matmuls, just index lookups, so they contribute zero FLOPs
2. **Attention's query-key dot products** — this is a matmul that scales with sequence length, not parameter count

Karpathy's NanoChat implements the precise formula:

```python
def precise_flops_per_token(n_params, n_embed_params, n_layers, n_heads, head_dim, seq_len):
    """
    From Karpathy's NanoChat, based on the PaLM paper methodology.

    n_params:       total parameters
    n_embed_params: embedding parameters (excluded from 6N term)
    n_layers:       number of transformer layers
    n_heads:        attention heads per layer
    head_dim:       dimension per head (n_embd // n_heads)
    seq_len:        sequence length
    """
    # The 6N term, excluding embeddings (they're lookups, not matmuls)
    matmul_params = n_params - n_embed_params
    matmul_flops = 6 * matmul_params

    # Attention FLOPs: query @ key^T matmul
    # Each layer does: (seq_len, head_dim) @ (head_dim, seq_len)
    # That's 2 * seq_len * head_dim FLOPs per head, times n_heads, times 2 for fwd+bwd
    # Simplified: 12 * n_heads * head_dim * seq_len per layer
    attn_flops_per_layer = 12 * n_heads * head_dim * seq_len
    attn_flops = attn_flops_per_layer * n_layers

    return matmul_flops + attn_flops
```

**How much does attention add? It depends on scale.**

Let's compare both formulas on Karpathy's d12 NanoChat model (768 embedding dim, 12 layers, 6 heads, 2048 seq_len):

```python
# d12 model from NanoChat
n_embd, n_layers, n_heads = 768, 12, 6
head_dim = n_embd // n_heads  # 128
seq_len = 2048
matmul_params = 110_000_000  # ~110M (excluding embeddings)

# Simple 6N formula
simple_flops = 6 * matmul_params
# = 660M FLOPs per token

# Attention term: 12 * heads * head_dim * seq_len per layer
attn_per_layer = 12 * n_heads * head_dim * seq_len  # 18.9M
attn_total = attn_per_layer * n_layers              # 227M

# Karpathy's precise formula
precise_flops = simple_flops + attn_total
# = 887M FLOPs per token
```

| Formula | FLOPs/token | Error |
|---------|-------------|-------|
| Simple 6N | 660M | -26% |
| With attention | 887M | baseline |

At this small scale, **attention adds 34% extra FLOPs**. The simple formula significantly underestimates.

But here's why the approximation works for scaling laws: attention FLOPs scale as `O(n_embd × seq_len × layers)` while matmul FLOPs scale as `O(n_embd² × layers)`. As models get wider, the matmul term (quadratic in width) dominates. At GPT-3 scale with 12,288 embedding dim:

```python
# GPT-3 scale (roughly)
n_embd, n_layers, n_heads = 12288, 96, 96
head_dim = 128
seq_len = 2048
matmul_params = 175_000_000_000  # 175B

simple_flops = 6 * matmul_params           # 1.05e12 per token
attn_total = 12 * n_heads * head_dim * seq_len * n_layers  # 28B per token

# Attention is now only ~2.7% of total
```

The 6ND approximation is bad for small models but good for large ones—exactly where scaling laws matter most.

![[assets/6nd_approximation_accuracy.png]]

```python
# Example: GPT-3 scale with simple approximation
params = 175e9      # 175B parameters
tokens = 300e9      # 300B tokens
flops = 6 * params * tokens
print(f"Training compute: {flops:.2e} FLOPs")
# Output: Training compute: 3.15e+23 FLOPs
```

## Chinchilla's Correction (2022)

Two years after Kaplan, DeepMind's Hoffmann et al. ran a more comprehensive study. They trained over 400 models from 70M to 16B parameters on 5B to 500B tokens, covering a much wider range than Kaplan.

Their finding: **Kaplan got the ratio wrong**. Models weren't just slightly undertrained on data—they were *massively* undertrained. The optimal balance isn't "huge model, small data." It's "scale both equally."

### The Chinchilla Scaling Law

Chinchilla proposed a combined formula that predicts loss from both parameters and data together:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Let's unpack each term:

**E (≈ 1.69)** is the "irreducible loss"—the floor below which you can never go, no matter how much you scale. It represents the inherent entropy of language. Even a perfect model can't predict which of several equally reasonable next words a human will choose.

**A/N^α** is the loss contribution from limited model capacity. Larger N (more parameters) → smaller term → lower loss. The exponent α ≈ 0.34 controls how fast.

**B/D^β** is the loss contribution from limited data. More D (tokens) → smaller term → lower loss. The exponent β ≈ 0.28 controls how fast.

In Python:

```python
def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """
    Predict loss for a model with N parameters trained on D tokens.

    The three terms represent:
    - E: irreducible entropy of language (can't beat this)
    - A/N^α: loss from model being too small
    - B/D^β: loss from not seeing enough data
    """
    model_term = A / (N ** alpha)
    data_term = B / (D ** beta)
    return E + model_term + data_term

# Example: estimate loss for a 7B model on 200B tokens
loss = chinchilla_loss(N=7e9, D=200e9)
print(f"Predicted loss: {loss:.3f}")
```

### Why Does This Formula Make Sense?

Think of it as two separate bottlenecks:

1. **Model capacity bottleneck**: Even with infinite data, a small model can only learn so much. Its limited parameters force it to compress information lossy. Making the model bigger relaxes this constraint.

2. **Data bottleneck**: Even with infinite parameters, a model trained on limited data will overfit or simply not see enough examples to learn rare patterns. More data relaxes this constraint.

The formula says: **your total loss is the irreducible minimum, plus whatever penalty you're paying from each bottleneck**. To minimize loss, you want both bottlenecks relaxed—neither too small a model nor too little data.

### The Equal Scaling Discovery

Here's where it gets practical. If you have a fixed compute budget C, how should you split it between N (parameters) and D (tokens)?

Chinchilla derived that under the constraint $C = 6ND$, the optimal split is:

$$N_{opt} \propto C^{0.5}$$
$$D_{opt} \propto C^{0.5}$$

Both scale as the square root of compute. This means **for every doubling of compute, you should double both the model size AND the training tokens**. Not 10x the model on the same data (Kaplan's advice), but equal growth in both dimensions.

**Deriving the square root scaling.** Start with the loss formula and the compute constraint:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \quad C = 6ND$$

Substitute D = C/(6N) into the loss to eliminate D:

$$L(N) = E + \frac{A}{N^\alpha} + B \cdot \frac{(6N)^\beta}{C^\beta}$$

Take the derivative with respect to N and set it to zero:

$$\frac{dL}{dN} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{\beta B \cdot 6^\beta \cdot N^{\beta-1}}{C^\beta} = 0$$

Solving for N:

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \cdot \frac{C^\beta}{6^\beta}$$

$$N_{opt} = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} \cdot \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}$$

Chinchilla found α ≈ 0.34 and β ≈ 0.28—roughly equal. When α = β, the exponent on C simplifies:

$$\frac{\beta}{\alpha + \beta} = \frac{\beta}{2\beta} = 0.5$$

So $N_{opt} \propto C^{0.5}$. And since $D = C/(6N)$:

$$D_{opt} \propto \frac{C}{C^{0.5}} = C^{0.5}$$

Both scale as the square root of compute because the two exponents in Chinchilla's loss formula are approximately equal.

**Where does "20 tokens per parameter" come from?** The ratio D/N works out to:

$$\frac{D_{opt}}{N_{opt}} = \frac{C / (6 N_{opt})}{N_{opt}} = \frac{C}{6 N_{opt}^2}$$

Since $N_{opt}^2 \propto C$, this ratio is a constant—it doesn't depend on the compute budget. The specific value (≈20) comes from plugging Chinchilla's fitted constants (A, B, α, β) into the optimization. It's not derived from first principles; it's what falls out of their empirical curve fits. Different data or optimizers give different ratios—Karpathy found ~8:1 with Muon on DCLM.

The practical upshot was the "20 tokens per parameter" heuristic:

```python
def chinchilla_optimal_tokens(params):
    """
    Rule of thumb: train on ~20 tokens per parameter.
    """
    return 20 * params

# For a 70B model
tokens = chinchilla_optimal_tokens(70e9)
print(f"Optimal training tokens: {tokens/1e12:.1f}T")
# Output: Optimal training tokens: 1.4T
```

This is exactly how Chinchilla was trained: 70B parameters on 1.4T tokens. Despite being 4x smaller than the 280B-parameter Gopher, it outperformed Gopher on virtually every benchmark—because Gopher was undertrained relative to its size.

### The Proof: Chinchilla vs. Gopher

| Model | Parameters | Training Tokens | Tokens/Param | MMLU Score |
|-------|------------|-----------------|--------------|------------|
| Gopher | 280B | 300B | 1.1 | 60.0% |
| Chinchilla | 70B | 1.4T | 20 | 67.5% |

Same compute budget. Chinchilla is 4x smaller but trained on 4.7x more data. Result: 7.5 percentage points better on MMLU.

This vindicated the equal-scaling hypothesis and triggered a field-wide shift. Labs started training smaller models on much more data.

![[assets/chinchilla_optimal_allocation.png]]

**[→ Interactive Chinchilla loss explorer](assets/chinchilla_loss_interactive.html)** — drag slider to change model size and see how loss varies with training tokens.

**[→ Compute allocation comparison](assets/compute_allocation_interactive.html)** — visualize how Kaplan vs Chinchilla split the compute budget differently.

**[→ Tokens-per-parameter calculator](assets/tokens_calculator_interactive.html)** — see where famous models fall on the tokens/param spectrum.

The curve shows diminishing returns: the first 100B tokens provide large gains, but going from 400B to 500B barely moves the needle.

## The Epoch AI Replication (2024)

In 2024, researchers at Epoch AI [attempted to replicate](https://arxiv.org/abs/2404.10102) Chinchilla's parametric scaling law estimates. What they found raises important caveats.

The original Chinchilla paper used three different methods to estimate optimal scaling. Method 3 (the parametric approach) is what gives us the clean L(N,D) formula. But when Epoch AI re-fit the data, they found substantially different parameter values:

| Parameter | Hoffmann et al. | Epoch AI | Std. Error |
|-----------|-----------------|----------|------------|
| E | 1.69 | 1.82 | ±0.03 |
| A | 406.4 | 482.01 | ±124.58 |
| B | 410.7 | 2085.43 | ±1293.23 |
| α | 0.34 | 0.35 | ±0.02 |
| β | 0.28 | 0.37 | ±0.02 |

The standard errors are huge—B could plausibly range from ~800 to ~3400. And there's a striking inconsistency: Hoffmann's own parameters (when you plug them into the optimization) imply ~70 tokens per parameter, not the 20 they actually used to train Chinchilla. Epoch's re-estimated parameters are the ones that imply ~20 tokens per parameter, matching how Chinchilla was actually trained.

Hoffmann et al. also reported implausibly narrow confidence intervals for their fit. Achieving such precision would require ~600,000 training runs; they ran fewer than 500.

The takeaway isn't that Chinchilla is wrong—the core insight (Kaplan's models were undertrained) holds up. But the precise "20:1" ratio should be treated as a rough guideline, not a law of physics. Different optimizers, data quality, and architectures may shift the optimum.

## Karpathy's Practical Scaling (2025)

Andrej Karpathy's NanoChat project offers a hands-on perspective. Training small models on the DCLM dataset, he confirmed the equal-exponent finding:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

But he found a different optimal ratio: **8 tokens per parameter** rather than Chinchilla's 20. Why the difference?

1. **Optimizer**: NanoChat uses Muon instead of Adam. Different optimizers may favor different parameter/data balances.

2. **Scale**: The experiments are at much smaller scale (millions of parameters vs. billions). Scaling laws extrapolate imperfectly.

3. **Data quality**: DCLM is more aggressively filtered than Chinchilla's training data. Higher-quality data might shift the optimum.

The core insight holds: parameters and data should scale together. The exact ratio depends on your specific setup.

## Overtraining for Inference Efficiency

Chinchilla's "compute-optimal" ratio minimizes training cost for a given loss. But that's not always what you want. In production, you pay for inference too—and inference cost scales with model size, not training tokens.

Consider the math. Suppose you're deploying a chatbot that will handle 1 billion queries. Each query might generate 500 tokens. The inference cost is roughly:

$$\text{Inference cost} \propto N \times \text{tokens generated} \times \text{queries}$$

A 70B model costs 10× more per query than a 7B model. Over a billion queries, that difference dwarfs any training cost savings.

This creates a different optimization: **minimize total cost (training + inference) for a target capability level.** The solution is to train a smaller model on far more data than Chinchilla recommends—"overtraining" relative to the compute-optimal point.

LLaMA 3 is the canonical example:

| Model | Parameters | Training Tokens | Tokens/Parameter | vs. Chinchilla |
|-------|------------|-----------------|------------------|----------------|
| Chinchilla | 70B | 1.4T | 20 | baseline |
| LLaMA 2 7B | 7B | 2T | 286 | 14× more |
| LLaMA 3 8B | 8B | 15T | 1,875 | 94× more |

LLaMA 3 8B trains on 15 trillion tokens—94× more than Chinchilla would recommend for that model size. The training is "wasteful" in compute-optimal terms: those FLOPs could have trained a much larger model to the same loss. But the result is an 8B model that punches far above its weight, cheap to serve at scale.

The tradeoff visualized:

```
                        Chinchilla-optimal         Inference-optimal
                        (minimize training)        (minimize training + inference)

Model size:             Large                      Small
Training tokens:        Moderate                   Very high
Training cost:          Lower                      Higher
Inference cost/query:   Higher                     Lower
Total cost at scale:    Higher                     Lower
```

When does overtraining make sense?

- **High query volume**: If you'll serve billions of requests, inference dominates
- **Latency requirements**: Smaller models are faster
- **Edge deployment**: Can't run 70B on a phone
- **Cost sensitivity**: Inference compute is often more expensive than training

When does Chinchilla-optimal make sense?

- **Research/evaluation**: You just need the best model, serving cost doesn't matter
- **Low query volume**: Training cost dominates
- **One-off tasks**: No repeated inference

The practical upshot: the "right" tokens-per-parameter ratio depends on your deployment scenario. Chinchilla's 20:1 is compute-optimal for training. LLaMA's 1000+:1 is cost-optimal for serving at scale.

## Why Do Scaling Laws Work? (Theory)

Why do neural networks follow power laws at all? Sharma & Kaplan (2020) proposed an explanation connecting scaling exponents to the **intrinsic dimension** of the data manifold. The argument is subtle, so let's build it up step by step.

### The Manifold Hypothesis

Start with a question: how much of "pixel space" contains valid images?

A 224×224 RGB image has 150,528 dimensions (224 × 224 × 3). Each pixel can take values 0-255. The total number of possible images is 256^150528—an incomprehensibly large number. But almost none of these are "real" images. Set pixels randomly and you get static noise, not cats or cars or faces.

Natural images occupy a tiny, thin surface winding through this vast space. That surface is called a **manifold**. The manifold hypothesis says: real-world data doesn't fill its ambient space—it lives on a much lower-dimensional surface embedded within it.

Here's a simple example. Imagine all possible images of a single white dot on a black background:

```
Ambient dimension: 150,528 (all pixels)
Intrinsic dimension: 2 (just x and y position of the dot)
```

You could describe any dot-image with just two numbers. The other 150,526 dimensions are "wasted"—they're determined once you know where the dot is. The data lives on a 2D surface (a plane) embedded in 150,528-dimensional space.

Real images are more complex than a dot, but the principle holds. Researchers estimate natural images have an intrinsic dimension of only 26-43. That's the "true" number of independent factors (pose, lighting, object identity, etc.) needed to specify an image.

### Why Dimension Determines Scaling

Now the key question: if you have N parameters in your neural network, how finely can you approximate a d-dimensional manifold?

Think of it like tiling a floor. With N tiles, you can cover a certain area at a certain resolution.

**1D case (a line):** With N tiles, you can divide a line into N segments. Each segment has length 1/N. Your "resolution" along the line is 1/N.

```
N = 4 tiles on a 1D line:
|████|████|████|████|
  ↑ each segment has length 1/4
```

**2D case (a plane):** With N tiles, you can make a grid of √N × √N. Each tile has side length 1/√N. Your resolution along each axis is 1/√N = 1/N^0.5.

```
N = 16 tiles on a 2D plane:
┌──┬──┬──┬──┐
├──┼──┼──┼──┤     √16 = 4 tiles per side
├──┼──┼──┼──┤     resolution = 1/4 = 1/N^0.5
├──┼──┼──┼──┤
└──┴──┴──┴──┘
```

**3D case (a volume):** With N tiles, you get a cube of N^(1/3) × N^(1/3) × N^(1/3). Resolution along each axis is 1/N^(1/3).

**d-dimensional case:** Resolution along each axis is 1/N^(1/d).

The pattern: **in d dimensions, N parameters give you resolution proportional to N^(1/d) along each dimension.**

### What Exactly Is "Resolution"?

Resolution (ε) is the side length of each cell when you tile the space. If you have N tiles covering a unit hypercube in d dimensions, you arrange them in a grid with N^(1/d) tiles along each axis. Each tile has side length:

$$\varepsilon = \frac{1}{N^{1/d}}$$

Spelled out for each dimension:

| Dimensions | Grid arrangement | Cell size (ε) |
|------------|------------------|---------------|
| d = 1 | N cells in a row | 1/N |
| d = 2 | √N × √N grid | 1/√N = 1/N^0.5 |
| d = 3 | ∛N × ∛N × ∛N grid | 1/∛N = 1/N^0.33 |
| d | N^(1/d) along each axis | 1/N^(1/d) |

### From Resolution to Error

Why does smaller resolution mean lower error? Each tile represents a region where your model outputs roughly the same prediction. Within a tile, the true function varies, but your approximation is constant (or nearly so). The error comes from this mismatch—how much the true function can vary within one cell.

Think of it like pixelating an image. Coarser pixels (larger ε) mean more detail is lost, so the approximation is worse. Finer pixels (smaller ε) capture more variation, reducing error. The resolution ε is literally how finely you can resolve variations in the target function.

If the true function is smooth (doesn't jump around wildly), the variation within a cell of size ε is roughly proportional to ε. So:

$$\text{error} \propto \varepsilon = \frac{1}{N^{1/d}} = N^{-1/d}$$

This is a power law with exponent α = 1/d.

The actual relationship from Sharma & Kaplan is:

$$\alpha \approx \frac{4}{d}$$

The factor of 4 (rather than 1) comes from details of how neural networks approximate smooth functions, related to approximation theory and the smoothness assumptions on the target function. The core insight remains: **higher intrinsic dimension → smaller exponent → slower scaling**.

### Concrete Example: Why Language Scales Slower Than Vision

This explains a real phenomenon. Image classification models scale faster (α ≈ 0.5) than language models (α ≈ 0.07).

Using α ≈ 4/d:
- **Image classification**: α ≈ 0.5 implies d ≈ 8. The manifold of "image → class label" mappings is relatively low-dimensional.
- **Language modeling**: α ≈ 0.07 implies d ≈ 57. The manifold of "context → next token distribution" is much higher-dimensional.

Why the difference? Classification collapses high-dimensional images to a handful of class labels—you're learning a lower-dimensional mapping. Language modeling must predict a full probability distribution over 50,000+ tokens for every position, preserving much more information.

### The Punchline

The exponent α is a property of your **data**, not your architecture. Making your network wider or deeper doesn't change α. The manifold's intrinsic dimension sets a fundamental speed limit on how fast you can improve by scaling.

This is both reassuring and sobering:
- **Reassuring**: Scaling laws aren't arbitrary. They reflect something real about the structure of the problem.
- **Sobering**: You can't escape the exponent by being clever with architecture. If the data manifold has high intrinsic dimension, scaling will be slow no matter what you do.

![[assets/manifold_tiling_visualization.png]]

The plot below shows how this plays out quantitatively. In low dimensions (d=2), resolution improves quickly as you add parameters. In high dimensions (d=50, like language), you need orders of magnitude more parameters for the same improvement.

![[assets/dimension_scaling_comparison.png]]

## Breaking Power Laws: Data Pruning

Power laws imply diminishing returns. To halve error, you need roughly 10x more data. But Sorscher et al. (2022) showed this isn't fundamental—it reflects **redundancy in random sampling**.

Their key finding: with intelligent data pruning, you can achieve **exponential scaling** instead of power-law scaling. The error decays much faster because each retained example teaches the model something new.

The strategy depends on data availability:
- **Data-scarce regime**: Keep easy examples. You need to master common patterns before learning edge cases.
- **Data-abundant regime**: Keep hard examples. Easy cases are already well-represented; examples near the decision boundary provide the most information.

Metrics for identifying "hard" examples include:
- **EL2N**: L2 norm of the error vector (high error = hard)
- **Memorization**: How much does including this example change its prediction?
- **Self-supervised prototypes**: Examples far from cluster centers in embedding space

Practical results: training on 80% of ImageNet (carefully selected) matches full-dataset performance. Some studies find 30% of data is sufficient with proper curation. This has influenced real LLM training—DCLM, FineWeb-Edu, and LLaMA 3 all use aggressive data filtering.

## Broken Scaling Laws

Traditional power laws are smooth curves. But Caballero et al. (2022) showed that real scaling often has **breaks**—points where the exponent changes.

The "Broken Neural Scaling Laws" (BNSL) functional form models this:

$$L = a + b \cdot x^{-c_0} \cdot \prod_i \left(1 + \left(\frac{x}{d_i}\right)^{1/f_i}\right)^{-c_i \cdot f_i}$$

On a log-log plot, this creates multiple approximately-linear segments connected by smooth transitions. Each break represents a change in how the model learns—perhaps a capability threshold being crossed, or a transition between learning regimes.

The canonical example is arithmetic. Models perform at near-random until reaching a critical scale, then suddenly achieve high accuracy. This "emergent" jump corresponds to a break in the scaling law—a transition from near-flat scaling (tiny exponent, barely improving) to rapid improvement (larger exponent). The jump isn't magic; it's baked into the structure of the scaling law itself.

BNSL outperforms standard power laws on 69-75% of tasks. However, there's a fundamental limit: you cannot reliably extrapolate across a break without data points near it. Some capability transitions will remain hard to forecast from smaller-scale experiments.

## Beyond Training: Inference-Time Scaling

Starting in 2024, a new scaling axis emerged. Instead of just making models bigger or training longer, you can spend more compute *at generation time*.

The o1 model demonstrated this dramatically. By generating "thinking tokens"—internal reasoning steps before producing an answer—o1 could trade inference compute for capability. More thinking time → better answers on hard problems. Both axes follow power law relationships—log-scale plots show smooth, linear improvement with more compute on either dimension.

o1 doesn't use explicit search algorithms like Monte Carlo Tree Search. Instead, it's trained via reinforcement learning to perform **implicit search through chain-of-thought**. The model learns when to explore alternatives, how to verify intermediate steps, and when to backtrack. The "thinking" happens in the token stream itself.

Inference strategies fall into two categories:
- **Parallel scaling**: Generate multiple independent completions and pick the best (best-of-N, majority voting)
- **Sequential scaling**: Generate longer reasoning chains with more intermediate steps

A striking finding from Snell et al.: a smaller model with optimized test-time compute can **outperform a 14x larger model** on tasks where the base model has reasonable baseline performance. The compute-optimal strategy varies by problem difficulty—easy problems need simple methods; hard problems need sophisticated search.

| Benchmark | GPT-4o | o1 | o3 |
|-----------|--------|----|----|
| ARC-AGI | 5% | — | 87.5% |
| AIME (math) | ~13% | 89th percentile | higher |

The cost is significant: reasoning models use 10-100x more tokens per query. DeepSeek R1 averages 12,000-23,000 tokens per AIME question. Longer responses don't guarantee accuracy—token usage is a poor proxy for reasoning quality.

Tasks that benefit most from inference scaling: mathematics, competitive programming, logic puzzles—anything with **verifiable intermediate steps**. Tasks that benefit least: open-ended generation, knowledge-intensive QA, NP-hard optimization.

## Current Limitations

Scaling laws have limitations that have become clearer as the field has pushed their boundaries:

**Perplexity ≠ capability**. The laws predict cross-entropy loss, but that doesn't always translate to downstream task performance. Some tasks improve smoothly with scale; others show no improvement or even inverse scaling until a threshold is crossed.

**Emergent abilities and the measurement debate**. Some capabilities appear discontinuously—a model shows near-zero performance until reaching some critical scale, then suddenly works. Wei et al. (2022) catalogued over 137 such tasks.

But Schaeffer et al. (2023) argued this is partly a **measurement artifact**. Over 92% of "emergent" abilities appeared under discontinuous metrics like exact-string-match (you get it exactly right or score zero). Using continuous metrics like token edit distance, the same models show smooth improvement. The "jump" was invisible partial credit becoming visible perfect answers.

The emerging synthesis: both camps are partially right. Some emergence is a metric artifact—researchers should report multiple metrics. But tasks requiring **compositional success** (multi-step reasoning, arithmetic) show genuine sharp transitions even with continuous metrics. If you need 10 steps all correct, and each step improves smoothly with scale, the probability of getting ALL steps right creates a sharp threshold.

Pre-training loss turns out to be more predictive than parameter count. Smaller models trained longer can match larger models' emergent capabilities if they achieve the same training loss.

**Data quality matters**. The laws assume your data quality stays constant as you scale. But high-quality data is finite. As models exhaust curated datasets and turn to synthetic data or lower-quality sources, the relationships may shift.

**Architecture assumptions**. The laws were derived for dense transformer architectures. Mixture-of-experts models, sparse attention, or other architectural innovations may follow different curves.

## Summary

| Paper | Key Finding | Practical Advice |
|-------|-------------|------------------|
| Kaplan 2020 | Loss follows power laws with small exponents | Scale model size preferentially |
| Chinchilla 2022 | Models were undertrained; α ≈ β | Scale parameters and data equally (~20:1 ratio) |
| Sharma & Kaplan 2020 | Exponent α ≈ 4/d (data manifold dimension) | Exponent is a property of the data, not architecture |
| Sorscher 2022 | Data pruning enables exponential scaling | Curate data; keep hard examples when data-abundant |
| Caballero 2022 (BNSL) | Scaling laws can "break" at capability thresholds | Can't extrapolate across breaks without nearby data |
| Epoch 2024 | High uncertainty in exact parameters | Treat ratios as guidelines, not laws |
| NanoChat 2025 | Ratio depends on optimizer, data, scale | Empirically tune for your setup |

The core insight across all this work: **scaling is predictable but expensive**. Power laws with small exponents mean massive investments yield diminishing returns. The practical art is figuring out *where* to invest—model size, training data, data quality, or inference compute—for your specific goals and constraints.

---

## References

**Foundational Papers**
- [Kaplan et al. 2020 - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. 2022 - Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)

**Theory**
- [Sharma & Kaplan 2020 - A Neural Scaling Law from the Dimension of the Data Manifold](https://arxiv.org/abs/2004.10802)
- [Bahri et al. 2021 - Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)

**Refinements and Challenges**
- [Epoch AI 2024 - Chinchilla Scaling: A Replication Attempt](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)
- [Epoch AI - Scaling Laws Literature Review](https://epoch.ai/blog/scaling-laws-literature-review)
- [Sorscher et al. 2022 - Beyond Neural Scaling Laws: Beating Power Laws via Data Pruning](https://arxiv.org/abs/2206.14486)
- [Caballero et al. 2022 - Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891)

**Emergent Abilities**
- [Wei et al. 2022 - Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- [Schaeffer et al. 2023 - Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)

**Inference-Time Scaling**
- [Snell et al. 2024 - Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)

**Practical**
- [Karpathy's NanoChat Scaling Discussion](https://github.com/karpathy/nanochat/discussions/420)
