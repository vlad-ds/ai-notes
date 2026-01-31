"""
Generate scaling law visualizations for the AI Notes.
"""
import matplotlib.pyplot as plt
import numpy as np

# Set up the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

def power_law_loss(x, a, alpha, floor=0):
    """L = floor + a * x^(-alpha)"""
    return floor + a * np.power(x, -alpha)

def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """L(N, D) = E + A/N^α + B/D^β"""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


# ============================================================================
# Chart 1: Power Law Exponents Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.logspace(6, 12, 100)  # 1M to 1T

# Different exponents
exponents = [
    (0.05, 'α = 0.05 (slow, like compute scaling)', '#e74c3c'),
    (0.076, 'α = 0.076 (Kaplan params)', '#3498db'),
    (0.1, 'α = 0.1 (moderate)', '#2ecc71'),
    (0.5, 'α = 0.5 (fast, like image classification)', '#9b59b6'),
]

for alpha, label, color in exponents:
    y = power_law_loss(x, a=10, alpha=alpha, floor=1.5)
    ax.loglog(x, y, label=label, linewidth=2.5, color=color)

ax.set_xlabel('Scale (parameters, tokens, or FLOPs)', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('How Scaling Exponents Affect Improvement Rate', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(1e6, 1e12)
ax.set_ylim(1.5, 12)

plt.tight_layout()
plt.savefig('/Users/vladgheorghe/obsidian/ai-notes/assets/scaling_exponents_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: scaling_exponents_comparison.png")


# ============================================================================
# Chart 2: Chinchilla Optimal Allocation
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Loss surface
N_range = np.logspace(8, 11, 50)  # 100M to 100B params
D_range = np.logspace(9, 13, 50)  # 1B to 10T tokens
N_grid, D_grid = np.meshgrid(N_range, D_range)
L_grid = chinchilla_loss(N_grid, D_grid)

ax = axes[0]
contour = ax.contourf(np.log10(N_grid), np.log10(D_grid), L_grid, levels=20, cmap='viridis_r')
plt.colorbar(contour, ax=ax, label='Loss')

# Mark compute iso-lines (C = 6*N*D)
for C in [1e20, 1e21, 1e22, 1e23, 1e24]:
    D_iso = C / (6 * N_range)
    valid = (D_iso >= 1e9) & (D_iso <= 1e13)
    ax.plot(np.log10(N_range[valid]), np.log10(D_iso[valid]), 'w--', alpha=0.5, linewidth=1)

# Mark optimal path (N ∝ D for Chinchilla)
N_opt = np.logspace(8, 11, 20)
D_opt = 20 * N_opt  # 20 tokens per param
valid = D_opt <= 1e13
ax.plot(np.log10(N_opt[valid]), np.log10(D_opt[valid]), 'r-', linewidth=3, label='Chinchilla optimal (D=20N)')
ax.legend(loc='lower right')

ax.set_xlabel('log₁₀(Parameters)', fontsize=12)
ax.set_ylabel('log₁₀(Tokens)', fontsize=12)
ax.set_title('Loss Landscape: L(N, D)', fontsize=14, fontweight='bold')

# Right: Optimal scaling
ax = axes[1]
compute_budgets = np.logspace(18, 25, 100)

# Kaplan optimal (N ∝ C^0.73)
N_kaplan = 1e6 * np.power(compute_budgets / 1e18, 0.73)
D_kaplan = compute_budgets / (6 * N_kaplan)

# Chinchilla optimal (N ∝ C^0.5, D ∝ C^0.5)
N_chinchilla = 1e7 * np.power(compute_budgets / 1e18, 0.5)
D_chinchilla = compute_budgets / (6 * N_chinchilla)

ax.loglog(compute_budgets, N_kaplan, 'b-', linewidth=2.5, label='Kaplan: N ∝ C^0.73')
ax.loglog(compute_budgets, N_chinchilla, 'r-', linewidth=2.5, label='Chinchilla: N ∝ C^0.5')
ax.loglog(compute_budgets, D_chinchilla, 'r--', linewidth=2.5, label='Chinchilla: D ∝ C^0.5')

# Mark some famous models
models = [
    (3.15e23, 175e9, 'GPT-3'),
    (5e23, 70e9, 'Chinchilla'),
    (4e23, 280e9, 'Gopher'),
]
for c, n, name in models:
    ax.scatter([c], [n], s=100, zorder=5)
    ax.annotate(name, (c, n), textcoords="offset points", xytext=(10, 5), fontsize=10)

ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
ax.set_ylabel('Parameters or Tokens', fontsize=12)
ax.set_title('Optimal Scaling: Kaplan vs Chinchilla', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/vladgheorghe/obsidian/ai-notes/assets/chinchilla_optimal_allocation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: chinchilla_optimal_allocation.png")


# ============================================================================
# Chart 3: The 6ND Approximation at Different Scales
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Model sizes (embedding dim)
dims = np.array([768, 1024, 2048, 4096, 8192, 12288])
labels = ['NanoChat d12\n(768)', '1024', '2048', '4096', '8192', 'GPT-3\n(12288)']

# Approximate calculations
n_layers = dims / 64  # rough scaling
n_heads = dims / 128
head_dim = 128
seq_len = 2048

# Matmul params ≈ 12 * d^2 * layers (very rough)
matmul_params = 12 * dims**2 * n_layers

# Simple 6N
simple_flops = 6 * matmul_params

# Attention term: 12 * heads * head_dim * seq_len * layers per token
attn_flops = 12 * n_heads * head_dim * seq_len * n_layers

# Percentage attention adds
attn_percentage = 100 * attn_flops / simple_flops

ax.bar(range(len(dims)), attn_percentage, color='#3498db', edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Attention FLOPs as % of 6N Term', fontsize=12)
ax.set_xlabel('Model Embedding Dimension', fontsize=12)
ax.set_title('Why 6ND Works Better at Large Scale', fontsize=14, fontweight='bold')

# Add percentage labels on bars
for i, pct in enumerate(attn_percentage):
    ax.annotate(f'{pct:.1f}%', (i, pct + 1), ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(attn_percentage) * 1.15)
ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/vladgheorghe/obsidian/ai-notes/assets/6nd_approximation_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated: 6nd_approximation_accuracy.png")


print("\nAll charts generated successfully!")
