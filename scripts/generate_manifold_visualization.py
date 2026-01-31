"""
Generate visualization showing how tiling resolution scales with dimension.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = '/Users/vladgheorghe/obsidian/ai-notes/assets'

def create_manifold_tiling_figure():
    """
    Create a figure showing how N parameters tile spaces of different dimensions.
    Uses N=64 throughout for consistency: 64 = 64^1 = 8^2 = 4^3
    """
    fig = plt.figure(figsize=(14, 5))

    N = 64  # Number of "parameters" (tiles) - same for all dimensions

    # --- 1D: Line ---
    ax1 = fig.add_subplot(131)
    ax1.set_title(f'1D: N={N} tiles\nResolution = 1/N = 1/{N}', fontsize=11)

    # Draw the line segments (show subset for visibility)
    for i in range(N):
        color = plt.cm.Blues(0.3 + 0.5 * (i % 2))
        ax1.barh(0, 1/N, left=i/N, height=0.3, color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Position along line')
    ax1.set_yticks([])
    ax1.set_aspect('equal')

    # Add resolution annotation (show a few segments worth for visibility)
    ax1.annotate('', xy=(0, -0.25), xytext=(4/N, -0.25),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(2/N, -0.4, f'ε = 1/{N}', ha='center', fontsize=10, color='red')

    # --- 2D: Plane ---
    ax2 = fig.add_subplot(132)
    side_2d = int(np.sqrt(N))  # 8x8 grid
    ax2.set_title(f'2D: N={N} tiles\nResolution = 1/√N = 1/{side_2d}', fontsize=11)

    # Draw the grid
    for i in range(side_2d):
        for j in range(side_2d):
            color = plt.cm.Blues(0.3 + 0.5 * ((i + j) % 2))
            rect = plt.Rectangle((i/side_2d, j/side_2d), 1/side_2d, 1/side_2d,
                                  facecolor=color, edgecolor='black', linewidth=0.5)
            ax2.add_patch(rect)

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.15, 1.05)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')

    # Add resolution annotation
    ax2.annotate('', xy=(0, -0.08), xytext=(1/side_2d, -0.08),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(1/(2*side_2d), -0.13, f'ε = 1/{side_2d}', ha='center', fontsize=10, color='red')

    # --- 3D: Volume ---
    ax3 = fig.add_subplot(133, projection='3d')
    side_3d = int(round(N ** (1/3)))  # 4x4x4 cube
    ax3.set_title(f'3D: N={N} tiles\nResolution = 1/∛N = 1/{side_3d}', fontsize=11)

    # Draw cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for i in range(side_3d):
        for j in range(side_3d):
            for k in range(side_3d):
                # Only draw some cubes (checkerboard pattern) to avoid visual clutter
                if (i + j + k) % 2 == 0:
                    x_base, y_base, z_base = i/side_3d, j/side_3d, k/side_3d
                    s = 1/side_3d * 0.92  # Slightly smaller for gaps

                    # Define cube vertices
                    vertices = [
                        [x_base, y_base, z_base],
                        [x_base + s, y_base, z_base],
                        [x_base + s, y_base + s, z_base],
                        [x_base, y_base + s, z_base],
                        [x_base, y_base, z_base + s],
                        [x_base + s, y_base, z_base + s],
                        [x_base + s, y_base + s, z_base + s],
                        [x_base, y_base + s, z_base + s]
                    ]

                    # Define faces
                    faces = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
                    ]

                    ax3.add_collection3d(Poly3DCollection(
                        faces, alpha=0.7,
                        facecolor=plt.cm.Blues(0.5),
                        edgecolor='black', linewidth=0.5
                    ))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/manifold_tiling_visualization.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved manifold_tiling_visualization.png")


def create_dimension_scaling_figure():
    """
    Create a figure showing how resolution ε scales with N for different dimensions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    N = np.logspace(0, 9, 100)  # 1 to 1 billion parameters

    dimensions = [2, 8, 20, 50]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    # Left plot: Resolution vs N
    ax1.set_title('Resolution (ε) vs Parameters (N)', fontsize=12)
    for d, color in zip(dimensions, colors):
        epsilon = N ** (-1/d)
        ax1.loglog(N, epsilon, label=f'd = {d}', color=color, linewidth=2)

    ax1.set_xlabel('Parameters (N)')
    ax1.set_ylabel('Resolution (ε = 1/N^(1/d))')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.annotate('Higher dimension\n= slower improvement',
                xy=(1e6, 0.25), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right plot: Error vs N (power law)
    ax2.set_title('Error vs Parameters (N)', fontsize=12)
    for d, color in zip(dimensions, colors):
        alpha = 4 / d  # The Sharma-Kaplan relationship
        error = 1.5 + 5 * N ** (-alpha)  # Loss = irreducible + power law term
        ax2.loglog(N, error - 1.5, label=f'd = {d}, α = {alpha:.2f}', color=color, linewidth=2)

    ax2.set_xlabel('Parameters (N)')
    ax2.set_ylabel('Reducible Error (L - E)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotation for real-world reference
    ax2.axhline(y=5 * (1e9) ** (-4/50), color='#e74c3c', linestyle='--', alpha=0.5)
    ax2.annotate('Language models\n(d ≈ 50, α ≈ 0.08)',
                xy=(1e7, 0.5), fontsize=10, color='#e74c3c',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dimension_scaling_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved dimension_scaling_comparison.png")


if __name__ == '__main__':
    create_manifold_tiling_figure()
    create_dimension_scaling_figure()
    print("Done!")
