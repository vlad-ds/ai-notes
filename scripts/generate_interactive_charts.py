"""
Generate interactive scaling law visualizations using Plotly.
Opens in browser with working sliders.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = '/Users/vladgheorghe/obsidian/ai-notes/assets'

# ============================================================================
# Chart 1: Interactive Power Law Exponents
# ============================================================================

def create_exponent_explorer():
    """
    Interactive chart to explore how scaling exponents affect improvement rate.
    Slider controls the exponent α.
    """
    x = np.logspace(0, 12, 200)  # 1 to 1T

    fig = go.Figure()

    # Create traces for different alpha values (we'll show/hide with slider)
    alphas = np.arange(0.02, 0.52, 0.02)  # 0.02 to 0.50

    for i, alpha in enumerate(alphas):
        y = 1.5 + 10 * np.power(x, -alpha)
        visible = (abs(alpha - 0.1) < 0.01)  # Start with α=0.1 visible
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'α = {alpha:.2f}',
            line=dict(color='#3498db', width=3),
            visible=visible
        ))

    # Add reference lines (always visible)
    y_kaplan = 1.5 + 10 * np.power(x, -0.076)
    fig.add_trace(go.Scatter(
        x=x, y=y_kaplan,
        mode='lines',
        name='Kaplan α=0.076',
        line=dict(color='#9b59b6', width=2, dash='dash'),
        visible=True
    ))

    y_fast = 1.5 + 10 * np.power(x, -0.5)
    fig.add_trace(go.Scatter(
        x=x, y=y_fast,
        mode='lines',
        name='Fast α=0.5',
        line=dict(color='#2ecc71', width=2, dash='dash'),
        visible=True
    ))

    # Create slider steps
    steps = []
    n_alphas = len(alphas)
    for i, alpha in enumerate(alphas):
        # All alpha traces are hidden except the current one
        # Reference lines (last 2 traces) are always visible
        visibility = [False] * n_alphas + [True, True]
        visibility[i] = True

        step = dict(
            method='update',
            args=[{'visible': visibility}],
            label=f'{alpha:.2f}'
        )
        steps.append(step)

    sliders = [dict(
        active=4,  # Start at α=0.1 (index 4 since we start at 0.02)
        currentvalue={"prefix": "Exponent α = ", "font": {"size": 16}},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=dict(
            text='<b>How Scaling Exponents Affect Improvement Rate</b><br><sup>Drag slider to change α. Dashed lines show Kaplan (slow) and fast scaling.</sup>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Scale (parameters, tokens, or FLOPs)',
            type='log',
            range=[0, 12],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Loss',
            range=[1.5, 12],
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        legend=dict(x=0.7, y=0.95),
        height=600,
        margin=dict(t=100)
    )

    fig.write_html(f'{OUTPUT_DIR}/scaling_exponents_interactive.html')
    print("Generated: scaling_exponents_interactive.html")


# ============================================================================
# Chart 2: Chinchilla Loss Explorer
# ============================================================================

def create_chinchilla_explorer():
    """
    Interactive Chinchilla loss function.
    Two sliders: N (parameters in billions) and see how loss changes with D (tokens).
    """
    fig = go.Figure()

    D = np.logspace(9, 13, 200)  # 1B to 10T tokens

    # Create traces for different N values
    N_values = [1e9, 3e9, 7e9, 10e9, 30e9, 70e9, 100e9, 175e9, 300e9]  # 1B to 300B
    N_labels = ['1B', '3B', '7B', '10B', '30B', '70B', '100B', '175B', '300B']

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#1abc9c', '#3498db', '#9b59b6', '#8e44ad', '#2c3e50']

    for i, (N, label, color) in enumerate(zip(N_values, N_labels, colors)):
        # Chinchilla: L = E + A/N^α + B/D^β
        E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
        L = E + A / np.power(N, alpha) + B / np.power(D, beta)

        visible = (label == '10B')  # Start with 10B visible
        fig.add_trace(go.Scatter(
            x=D, y=L,
            mode='lines',
            name=f'N = {label}',
            line=dict(color=color, width=3),
            visible=visible
        ))

    # Add irreducible loss floor
    fig.add_hline(y=1.69, line_dash="dot", line_color="gray",
                  annotation_text="Irreducible loss (E=1.69)")

    # Create slider steps
    steps = []
    for i, label in enumerate(N_labels):
        visibility = [False] * len(N_values)
        visibility[i] = True

        step = dict(
            method='update',
            args=[{'visible': visibility}],
            label=label
        )
        steps.append(step)

    sliders = [dict(
        active=3,  # Start at 10B
        currentvalue={"prefix": "Model size N = ", "font": {"size": 16}},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=dict(
            text='<b>Chinchilla Loss: L(N, D) = 1.69 + 406/N^0.34 + 411/D^0.28</b><br><sup>Drag slider to change model size (N). X-axis shows training tokens (D).</sup>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Training Tokens (D)',
            type='log',
            gridcolor='lightgray',
            tickformat='.0e'
        ),
        yaxis=dict(
            title='Loss',
            range=[1.5, 6],
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        height=600,
        margin=dict(t=100)
    )

    fig.write_html(f'{OUTPUT_DIR}/chinchilla_loss_interactive.html')
    print("Generated: chinchilla_loss_interactive.html")


# ============================================================================
# Chart 3: Compute Allocation Explorer
# ============================================================================

def create_allocation_explorer():
    """
    Interactive chart showing optimal N vs D allocation for different compute budgets.
    Compare Kaplan vs Chinchilla recommendations.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Kaplan: Favor Parameters (N ∝ C^0.73)', 'Chinchilla: Equal Scaling (N ∝ C^0.5)'),
        horizontal_spacing=0.1
    )

    compute_budgets = np.logspace(18, 25, 100)

    # Kaplan optimal
    N_kaplan = 1e6 * np.power(compute_budgets / 1e18, 0.73)
    D_kaplan = compute_budgets / (6 * N_kaplan)

    # Chinchilla optimal
    N_chinchilla = 1e7 * np.power(compute_budgets / 1e18, 0.5)
    D_chinchilla = compute_budgets / (6 * N_chinchilla)

    # Kaplan plot
    fig.add_trace(go.Scatter(
        x=compute_budgets, y=N_kaplan,
        mode='lines', name='Parameters (N)',
        line=dict(color='#3498db', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=compute_budgets, y=D_kaplan,
        mode='lines', name='Tokens (D)',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ), row=1, col=1)

    # Chinchilla plot
    fig.add_trace(go.Scatter(
        x=compute_budgets, y=N_chinchilla,
        mode='lines', name='Parameters (N)',
        line=dict(color='#3498db', width=2),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=compute_budgets, y=D_chinchilla,
        mode='lines', name='Tokens (D)',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        showlegend=False
    ), row=1, col=2)

    # Add famous models
    models = [
        (3.15e23, 175e9, 300e9, 'GPT-3'),
        (5e23, 70e9, 1.4e12, 'Chinchilla'),
        (4e23, 280e9, 300e9, 'Gopher'),
    ]

    for c, n, d, name in models:
        fig.add_trace(go.Scatter(
            x=[c], y=[n],
            mode='markers+text',
            name=name,
            text=[name],
            textposition='top right',
            marker=dict(size=12, symbol='diamond'),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[c], y=[n],
            mode='markers+text',
            name=name,
            text=[name],
            textposition='top right',
            marker=dict(size=12, symbol='diamond'),
            showlegend=False
        ), row=1, col=2)

    fig.update_xaxes(type='log', title='Compute Budget (FLOPs)', gridcolor='lightgray')
    fig.update_yaxes(type='log', title='Parameters or Tokens', gridcolor='lightgray')

    fig.update_layout(
        title=dict(
            text='<b>Optimal Scaling: How to Split Your Compute Budget</b><br><sup>Kaplan said favor parameters. Chinchilla showed equal scaling is better.</sup>',
            font=dict(size=16)
        ),
        plot_bgcolor='white',
        height=500,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=100)
    )

    fig.write_html(f'{OUTPUT_DIR}/compute_allocation_interactive.html')
    print("Generated: compute_allocation_interactive.html")


# ============================================================================
# Chart 4: Tokens-per-Parameter Calculator
# ============================================================================

def create_tokens_calculator():
    """
    Interactive calculator: given model size, compute Chinchilla-optimal tokens.
    """
    fig = go.Figure()

    # Model sizes from 100M to 1T
    N = np.logspace(8, 12, 100)

    # Different tokens-per-parameter ratios
    ratios = [8, 20, 50, 100, 200]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for ratio, color in zip(ratios, colors):
        D = ratio * N
        fig.add_trace(go.Scatter(
            x=N, y=D,
            mode='lines',
            name=f'{ratio} tokens/param',
            line=dict(color=color, width=2),
            hovertemplate='Model: %{x:.2e} params<br>Tokens: %{y:.2e}<br>Ratio: ' + str(ratio)
        ))

    # Add reference models
    models = [
        (175e9, 300e9, 'GPT-3 (1.7 tok/param)'),
        (70e9, 1.4e12, 'Chinchilla (20 tok/param)'),
        (70e9, 15e12, 'LLaMA 3 (215 tok/param)'),
    ]

    for n, d, name in models:
        fig.add_trace(go.Scatter(
            x=[n], y=[d],
            mode='markers+text',
            text=[name],
            textposition='top left',
            marker=dict(size=15, symbol='star'),
            showlegend=False,
            hovertemplate=name + '<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='<b>Tokens-per-Parameter: How Much Should You Train?</b><br><sup>Chinchilla said 20:1. Modern practice often uses 50-200:1 for inference efficiency.</sup>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Model Parameters (N)',
            type='log',
            gridcolor='lightgray',
            tickformat='.0e'
        ),
        yaxis=dict(
            title='Training Tokens (D)',
            type='log',
            gridcolor='lightgray',
            tickformat='.0e'
        ),
        plot_bgcolor='white',
        height=600,
        legend=dict(x=0.02, y=0.98),
        margin=dict(t=100)
    )

    fig.write_html(f'{OUTPUT_DIR}/tokens_calculator_interactive.html')
    print("Generated: tokens_calculator_interactive.html")


if __name__ == '__main__':
    print("Generating interactive Plotly charts...")
    create_exponent_explorer()
    create_chinchilla_explorer()
    create_allocation_explorer()
    create_tokens_calculator()
    print("\nAll interactive charts generated!")
    print(f"Open the HTML files in {OUTPUT_DIR}/ in your browser.")
