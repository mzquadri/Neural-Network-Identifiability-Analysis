"""
Visualization utilities for neural network identifiability analysis.
Network architecture diagrams, parameter space plots, and isomorphism visualizations.
"""

from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

try:
    from .identifiability_checks import extract_parameters
except ImportError:  # Supports direct execution: python src/visualization.py
    from identifiability_checks import extract_parameters


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def plot_network_architecture(
    architecture: List[int], title: str = "Network Architecture", save_path: str = None
):
    """Draw a neural network architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))

    layer_positions = np.linspace(0.1, 0.9, len(architecture))
    max_neurons = max(architecture)

    node_positions = {}

    for l, (x_pos, n_neurons) in enumerate(zip(layer_positions, architecture)):
        y_positions = np.linspace(0.1, 0.9, n_neurons) if n_neurons > 1 else [0.5]

        for j, y_pos in enumerate(y_positions):
            node_positions[(l, j)] = (x_pos, y_pos)

            # Color: input=blue, hidden=green, output=red
            if l == 0:
                color = "#3498db"
            elif l == len(architecture) - 1:
                color = "#e74c3c"
            else:
                color = "#2ecc71"

            circle = plt.Circle(
                (x_pos, y_pos), 0.015, color=color, ec="black", zorder=5
            )
            ax.add_patch(circle)

        # Layer label
        ax.text(x_pos, -0.02, f"Layer {l}\n({n_neurons})", ha="center", fontsize=10)

    # Draw connections
    for l in range(len(architecture) - 1):
        for j in range(architecture[l]):
            for k in range(architecture[l + 1]):
                start = node_positions[(l, j)]
                end = node_positions[(l + 1, k)]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    "gray",
                    alpha=0.15,
                    linewidth=0.5,
                )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#3498db", label="Input"),
        mpatches.Patch(color="#2ecc71", label="Hidden"),
        mpatches.Patch(color="#e74c3c", label="Output"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weight_distributions(layers: List[Dict], save_dir: Path = RESULTS_DIR):
    """Plot weight and bias distributions per layer."""
    n_layers = len(layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 8))

    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for l, layer in enumerate(layers):
        W = layer["weight"].numpy().flatten()
        b = layer["bias"].numpy()

        # Weights
        axes[0, l].hist(W, bins=30, color="#3498db", alpha=0.7, edgecolor="black")
        axes[0, l].set_title(f"Layer {l} Weights", fontsize=12)
        axes[0, l].set_xlabel("Value")
        axes[0, l].axvline(0, color="red", linestyle="--", alpha=0.5)
        axes[0, l].grid(True, alpha=0.3)

        # Biases
        axes[1, l].bar(range(len(b)), b, color="#e74c3c", alpha=0.7, edgecolor="black")
        axes[1, l].set_title(f"Layer {l} Biases", fontsize=12)
        axes[1, l].set_xlabel("Neuron Index")
        axes[1, l].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1, l].grid(True, alpha=0.3)

    plt.suptitle("Network Parameter Distributions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Parameter distributions saved to {save_dir}")


def plot_neuron_similarity_matrix(layers: List[Dict], save_dir: Path = RESULTS_DIR):
    """
    Plot cosine similarity heatmaps between neurons in each hidden layer.
    High similarity indicates potential clone pairs.
    """
    hidden_layers = layers[:-1]
    n_hidden = len(hidden_layers)

    fig, axes = plt.subplots(1, n_hidden, figsize=(6 * n_hidden, 5))
    if n_hidden == 1:
        axes = [axes]

    for l, layer in enumerate(hidden_layers):
        W = layer["weight"]
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        similarity = (W_norm @ W_norm.T).numpy()

        im = axes[l].imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[l].set_title(f"Hidden Layer {l}\nNeuron Cosine Similarity", fontsize=12)
        axes[l].set_xlabel("Neuron Index")
        axes[l].set_ylabel("Neuron Index")
        plt.colorbar(im, ax=axes[l], fraction=0.046, pad=0.04)

    plt.suptitle("Neuron Similarity (Clone Detection)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "neuron_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Neuron similarity plots saved to {save_dir}")


def plot_identifiability_summary(
    report: Dict, architecture: List[int], save_dir: Path = RESULTS_DIR
):
    """Create a visual summary of identifiability checks."""
    checks = [
        ("No-Clones", report["no_clones"]["is_clone_free"]),
        ("Non-Degeneracy", report["non_degeneracy"]["is_non_degenerate"]),
        ("Self-Avoiding", report["self_avoiding"]["is_self_avoiding"]),
        ("Activation OK", report["activation_identifiable"]),
        ("Overall", report["is_identifiable"]),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (name, passed) in enumerate(checks):
        color = "#2ecc71" if passed else "#e74c3c"
        symbol = "PASS" if passed else "FAIL"
        ax.barh(i, 1, color=color, alpha=0.7, edgecolor="black")
        ax.text(
            0.5,
            i,
            f"{name}: {symbol}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Identifiability Report: {architecture}", fontsize=15, fontweight="bold"
    )
    ax.invert_yaxis()

    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "identifiability_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary saved to {save_dir}")


if __name__ == "__main__":
    try:
        from .identifiability_checks import build_network, identifiability_report
    except ImportError:
        from identifiability_checks import build_network, identifiability_report

    torch.manual_seed(42)
    arch = [5, 10, 8, 1]
    model = build_network(arch, "tanh")
    layers = extract_parameters(model)

    plot_network_architecture(arch, save_path=str(RESULTS_DIR / "architecture.png"))
    plot_weight_distributions(layers)
    plot_neuron_similarity_matrix(layers)

    report = identifiability_report(model, arch, "tanh")
    plot_identifiability_summary(report, arch)
