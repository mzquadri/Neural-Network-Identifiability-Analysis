"""
Activation function analysis for neural network identifiability.
Compares how different activations affect identifiability guarantees.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    from .identifiability_checks import build_network, identifiability_report
except ImportError:  # Supports direct execution: python src/activation_analysis.py
    from identifiability_checks import build_network, identifiability_report


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ──────────────────────────────────────────────
# Activation function properties
# ──────────────────────────────────────────────
def analyze_activation_properties():
    """Analyze mathematical properties relevant to identifiability."""
    x = torch.linspace(-5, 5, 1000)

    activations = {
        "sigmoid": {
            "fn": torch.sigmoid,
            "properties": {
                "monotonic": True,
                "bounded": True,
                "odd": False,
                "analytic": True,
                "identifiable": True,
                "equivalence": "Up to neuron permutations",
            },
        },
        "tanh": {
            "fn": torch.tanh,
            "properties": {
                "monotonic": True,
                "bounded": True,
                "odd": True,  # tanh(-x) = -tanh(x)
                "analytic": True,
                "identifiable": True,
                "equivalence": "Up to sign flips (+/-) and permutations",
            },
        },
        "relu": {
            "fn": torch.relu,
            "properties": {
                "monotonic": True,
                "bounded": False,
                "odd": False,
                "analytic": False,  # Not differentiable at 0
                "identifiable": False,
                "equivalence": "Fails genericity conditions",
            },
        },
    }

    return activations, x


def plot_activations_and_derivatives(save_dir: Path = RESULTS_DIR):
    """Plot activation functions and their derivatives side by side."""
    activations, x = analyze_activation_properties()
    x_np = x.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {"sigmoid": "#3498db", "tanh": "#e74c3c", "relu": "#2ecc71"}

    for col, (name, info) in enumerate(activations.items()):
        y = info["fn"](x).numpy()
        color = colors[name]

        # Activation function
        axes[0, col].plot(x_np, y, color=color, linewidth=2.5)
        axes[0, col].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0, col].axvline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0, col].set_title(f"{name.upper()}", fontsize=14, fontweight="bold")
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("f(x)")
        axes[0, col].grid(True, alpha=0.3)

        # Identifiability badge
        ident = info["properties"]["identifiable"]
        badge_color = "#2ecc71" if ident else "#e74c3c"
        axes[0, col].text(
            0.02,
            0.98,
            f"Identifiable: {'Yes' if ident else 'No'}",
            transform=axes[0, col].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=badge_color, alpha=0.3),
        )

        # Derivative
        x_grad = x.clone().requires_grad_(True)
        y_grad = info["fn"](x_grad)
        dy = torch.autograd.grad(y_grad.sum(), x_grad)[0]
        axes[1, col].plot(x_np, dy.detach().numpy(), color=color, linewidth=2.5)
        axes[1, col].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1, col].axvline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1, col].set_title(f"{name.upper()} Derivative", fontsize=14)
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("f'(x)")
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle(
        "Activation Functions and Their Derivatives\n"
        "(Identifiability depends on smoothness and odd/even symmetry)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "activation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Activation analysis saved to {save_dir / 'activation_analysis.png'}")


def compare_activations_identifiability(architecture: list = None):
    """
    Run identifiability checks on the same architecture with different activations.
    """
    if architecture is None:
        architecture = [5, 10, 10, 1]

    print("=" * 60)
    print("ACTIVATION FUNCTION COMPARISON FOR IDENTIFIABILITY")
    print("=" * 60)
    print(f"Architecture: {architecture}\n")

    results = {}

    for activation in ["sigmoid", "tanh", "relu"]:
        torch.manual_seed(42)  # Same initialization
        model = build_network(architecture, activation)
        print(f"\n{'─' * 60}")
        report = identifiability_report(model, architecture, activation)
        results[activation] = report

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Activation':<12} {'Identifiable':<15} {'Clone-Free':<12} {'Non-Degenerate':<16} {'Self-Avoiding':<14}"
    )
    print("-" * 70)

    for act, r in results.items():
        print(
            f"{act:<12} "
            f"{'Yes' if r['is_identifiable'] else 'No':<15} "
            f"{'Pass' if r['no_clones']['is_clone_free'] else 'Fail':<12} "
            f"{'Pass' if r['non_degeneracy']['is_non_degenerate'] else 'Fail':<16} "
            f"{'Pass' if r['self_avoiding']['is_self_avoiding'] else 'Fail':<14}"
        )

    return results


# ──────────────────────────────────────────────
# Tanh sign invariance demonstration
# ──────────────────────────────────────────────
def demonstrate_tanh_sign_invariance():
    """
    Demonstrate that tanh(-x) = -tanh(x) leads to sign-flip equivalences.

    For a tanh network, flipping the sign of all incoming weights/bias of a neuron
    and all outgoing weights preserves the input-output map.
    """
    torch.manual_seed(42)
    architecture = [3, 5, 1]

    model = build_network(architecture, "tanh")
    model.eval()

    X = torch.randn(100, 3)

    with torch.no_grad():
        y_original = model(X)

    # Flip signs of neuron 0 in hidden layer
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    with torch.no_grad():
        # Flip incoming weights and bias
        linear_layers[0].weight[0] *= -1
        linear_layers[0].bias[0] *= -1
        # Flip outgoing weights
        linear_layers[1].weight[:, 0] *= -1

        y_flipped = model(X)

    max_diff = (y_original - y_flipped).abs().max().item()
    print(f"\nTanh sign invariance demonstration:")
    print(f"  Max output difference after sign flip: {max_diff:.2e}")
    print(f"  Invariance verified: {max_diff < 1e-6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation function analysis")
    parser.add_argument("--architecture", nargs="+", type=int, default=[5, 10, 10, 1])
    args = parser.parse_args()

    plot_activations_and_derivatives()
    compare_activations_identifiability(args.architecture)
    demonstrate_tanh_sign_invariance()
