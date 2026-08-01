"""
Symmetry-breaking strategies for neural networks.
Proposes and tests architectural constraints that reduce equivalence classes
and improve identifiability.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

try:
    from .identifiability_checks import build_network, extract_parameters, check_no_clones
except ImportError:  # Supports direct execution: python src/symmetry_breaking.py
    from identifiability_checks import build_network, extract_parameters, check_no_clones


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ──────────────────────────────────────────────
# Strategy 1: Ordered initialization
# ──────────────────────────────────────────────
def ordered_initialization(model: nn.Module) -> nn.Module:
    """
    Initialize biases in decreasing order within each layer.
    This breaks permutation symmetry by imposing a canonical ordering.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                # Sort biases in decreasing order
                sorted_indices = torch.argsort(module.bias, descending=True)
                module.bias.data = module.bias.data[sorted_indices]
                module.weight.data = module.weight.data[sorted_indices]
    return model


# ──────────────────────────────────────────────
# Strategy 2: L2 norm ordering regularization
# ──────────────────────────────────────────────
class OrderingRegularizer:
    """
    Regularization loss that encourages neurons to maintain a
    canonical ordering based on their weight norms.

    L_order = sum_{l} sum_{j < j'} max(0, ||w_{j'}||^2 - ||w_j||^2 + margin)
    """

    def __init__(self, margin: float = 0.01, weight: float = 0.1):
        self.margin = margin
        self.weight = weight

    def __call__(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                norms = module.weight.norm(dim=1)  # L2 norm of each neuron
                for j in range(len(norms) - 1):
                    # Encourage norms[j] >= norms[j+1] + margin
                    violation = norms[j + 1] - norms[j] + self.margin
                    loss = loss + torch.relu(violation)
        return self.weight * loss


# ──────────────────────────────────────────────
# Strategy 3: Distinct bias constraint
# ──────────────────────────────────────────────
class DistinctBiasRegularizer:
    """
    Regularization that penalizes neurons with similar bias magnitudes.
    Enforces Fefferman's assumption of distinct bias magnitudes.

    L_bias = sum_{l} sum_{j < j'} exp(-|b_j^2 - b_{j'}^2| / temperature)
    """

    def __init__(self, temperature: float = 0.1, weight: float = 0.05):
        self.temperature = temperature
        self.weight = weight

    def __call__(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                biases = module.bias
                if biases is not None:
                    for j in range(len(biases)):
                        for j_prime in range(j + 1, len(biases)):
                            diff = (biases[j].abs() - biases[j_prime].abs()).abs()
                            loss = loss + torch.exp(-diff / self.temperature)
        return self.weight * loss


# ──────────────────────────────────────────────
# Strategy 4: Anti-clone regularization
# ──────────────────────────────────────────────
class AntiCloneRegularizer:
    """
    Penalizes neurons that are too similar (potential clones).
    Encourages diversity in weight patterns across neurons in the same layer.
    """

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def __call__(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight  # (D_l, D_{l-1})
                # Cosine similarity between all pairs of neurons
                W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
                similarity = W_norm @ W_norm.T

                # Penalize high similarity (exclude diagonal)
                mask = 1.0 - torch.eye(W.shape[0], device=W.device)
                loss = loss + (similarity.abs() * mask).sum()

        return self.weight * loss


# ──────────────────────────────────────────────
# Experiment: Training with symmetry-breaking
# ──────────────────────────────────────────────
def train_with_symmetry_breaking(
    architecture: List[int] = None, n_epochs: int = 200, use_regularizers: bool = True
):
    """
    Train networks with and without symmetry-breaking regularization
    and compare identifiability properties.
    """
    if architecture is None:
        architecture = [2, 8, 8, 1]

    torch.manual_seed(42)

    # Generate synthetic regression data
    X_train = torch.randn(500, architecture[0])
    y_train = (
        torch.sin(X_train[:, 0:1])
        + 0.5 * torch.cos(X_train[:, 1:2])
        + 0.1 * torch.randn(500, 1)
    )

    results = {}

    for setting in ["baseline", "with_regularization"]:
        torch.manual_seed(42)
        model = build_network(architecture, "tanh")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        if setting == "with_regularization" and use_regularizers:
            ordering_reg = OrderingRegularizer(weight=0.05)
            bias_reg = DistinctBiasRegularizer(weight=0.03)
            clone_reg = AntiCloneRegularizer(weight=0.02)

        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            if setting == "with_regularization" and use_regularizers:
                loss = loss + ordering_reg(model) + bias_reg(model) + clone_reg(model)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check identifiability
        layers = extract_parameters(model)
        clones = check_no_clones(layers)

        results[setting] = {
            "final_loss": losses[-1],
            "losses": losses,
            "is_clone_free": clones["is_clone_free"],
            "model": model,
        }

        print(f"\n{setting.upper()}:")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Clone-free: {clones['is_clone_free']}")

    return results


def plot_symmetry_breaking_results(results: dict, save_dir: Path = RESULTS_DIR):
    """Plot training comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        axes[0].plot(r["losses"], label=name, linewidth=2)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss Comparison", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Bias distributions
    for idx, (name, r) in enumerate(results.items()):
        layers = extract_parameters(r["model"])
        all_biases = []
        for l in layers[:-1]:
            all_biases.extend(l["bias"].numpy().tolist())
        axes[1].hist(all_biases, bins=20, alpha=0.5, label=name, edgecolor="black")

    axes[1].set_xlabel("Bias Value", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Hidden Layer Bias Distribution", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Symmetry-Breaking Strategies", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "symmetry_breaking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {save_dir / 'symmetry_breaking.png'}")


if __name__ == "__main__":
    print("=" * 60)
    print("SYMMETRY-BREAKING STRATEGIES EXPERIMENT")
    print("=" * 60)

    results = train_with_symmetry_breaking()
    plot_symmetry_breaking_results(results)
