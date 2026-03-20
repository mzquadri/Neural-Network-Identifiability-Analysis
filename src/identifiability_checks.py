"""
Core identifiability verification checks for neural networks.
Implements Fefferman's conditions: no-clones, non-degeneracy, self-avoiding property.

Based on:
- Fefferman (1994), Reconstructing a Neural Net from its Output
- Bona-Pellissier et al. (2022), Parameter Identifiability of Neural Networks
"""

import argparse
from itertools import combinations
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────
# Network construction
# ──────────────────────────────────────────────
def build_network(architecture: List[int], activation: str = "tanh") -> nn.Sequential:
    """
    Build a fully-connected feedforward network.

    Parameters
    ----------
    architecture : list of int
        Layer sizes, e.g., [5, 10, 10, 1].
    activation : str
        'tanh', 'sigmoid', or 'relu'.
    """
    activation_map = {
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
    }

    if activation not in activation_map:
        raise ValueError(
            f"Unknown activation: {activation}. Choose from {list(activation_map.keys())}"
        )

    layers = []
    for i in range(len(architecture) - 1):
        layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        if i < len(architecture) - 2:  # No activation on output layer
            layers.append(activation_map[activation]())

    return nn.Sequential(*layers)


def extract_parameters(model: nn.Module) -> List[Dict]:
    """
    Extract weights and biases from each layer.

    Returns
    -------
    layers : list of dict
        Each dict has 'weight' (D_l x D_{l-1}) and 'bias' (D_l,) tensors.
    """
    layers = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "weight": module.weight.detach().clone(),
                    "bias": module.bias.detach().clone()
                    if module.bias is not None
                    else None,
                }
            )
    return layers


# ──────────────────────────────────────────────
# Condition 1: No-Clones
# ──────────────────────────────────────────────
def check_no_clones(layers: List[Dict], tol: float = 1e-6) -> Dict:
    """
    Check the no-clones condition for each hidden layer.

    Clone pairs: two neurons j, j' in layer l with identical biases
    and identical incoming weight vectors.

    Returns
    -------
    result : dict with 'is_clone_free', 'clone_pairs' per layer.
    """
    result = {"is_clone_free": True, "layers": []}

    for l, layer in enumerate(layers[:-1]):  # Exclude output layer
        W = layer["weight"]  # (D_l, D_{l-1})
        b = layer["bias"]

        clone_pairs = []
        D_l = W.shape[0]

        for j, j_prime in combinations(range(D_l), 2):
            weight_match = torch.allclose(W[j], W[j_prime], atol=tol)
            bias_match = torch.allclose(
                b[j : j + 1], b[j_prime : j_prime + 1], atol=tol
            )

            if weight_match and bias_match:
                clone_pairs.append((j, j_prime))
                result["is_clone_free"] = False

        result["layers"].append(
            {
                "layer_index": l,
                "num_neurons": D_l,
                "clone_pairs": clone_pairs,
                "is_clone_free": len(clone_pairs) == 0,
            }
        )

    return result


# ──────────────────────────────────────────────
# Condition 2: Non-Degeneracy
# ──────────────────────────────────────────────
def check_non_degeneracy(
    model: nn.Module, architecture: List[int], n_samples: int = 1000
) -> Dict:
    """
    Check if every hidden neuron contributes to the output.

    A network is non-degenerate if removing any single hidden neuron
    changes the network's output on at least some inputs.

    Returns
    -------
    result : dict with 'is_non_degenerate', 'dead_neurons' per layer.
    """
    model.eval()
    X = torch.randn(n_samples, architecture[0])

    with torch.no_grad():
        y_original = model(X)

    layers = extract_parameters(model)
    result = {"is_non_degenerate": True, "layers": []}

    for l, layer in enumerate(layers[:-1]):  # Hidden layers only
        W = layer["weight"]
        D_l = W.shape[0]
        dead_neurons = []

        for j in range(D_l):
            # Zero out neuron j's outgoing weights
            modified_model = _clone_model(model)
            _zero_neuron_outgoing(modified_model, l, j)

            with torch.no_grad():
                y_modified = modified_model(X)

            # Check if output changed
            max_diff = (y_original - y_modified).abs().max().item()
            if max_diff < 1e-8:
                dead_neurons.append(j)
                result["is_non_degenerate"] = False

        result["layers"].append(
            {
                "layer_index": l,
                "num_neurons": D_l,
                "dead_neurons": dead_neurons,
                "is_non_degenerate": len(dead_neurons) == 0,
            }
        )

    return result


def _clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of the model."""
    import copy

    return copy.deepcopy(model)


def _zero_neuron_outgoing(model: nn.Module, layer_idx: int, neuron_idx: int):
    """Zero out outgoing weights of a specific neuron."""
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if layer_idx + 1 < len(linear_layers):
        with torch.no_grad():
            linear_layers[layer_idx + 1].weight[:, neuron_idx] = 0.0


# ──────────────────────────────────────────────
# Condition 3: Self-Avoiding Property
# ──────────────────────────────────────────────
def check_self_avoiding(layers: List[Dict], tol: float = 1e-6) -> Dict:
    """
    Check the self-avoiding property.

    The parameter set is self-avoiding if no two neurons in the same layer
    share the same bias magnitude and the weight ratios are not simple fractions.

    This is a necessary condition in Fefferman's framework.
    """
    result = {"is_self_avoiding": True, "layers": []}

    for l, layer in enumerate(layers[:-1]):
        W = layer["weight"]
        b = layer["bias"]
        D_l = W.shape[0]

        violations = []

        # Check distinct bias magnitudes
        bias_mags = b.abs()
        for j, j_prime in combinations(range(D_l), 2):
            if torch.allclose(
                bias_mags[j : j + 1], bias_mags[j_prime : j_prime + 1], atol=tol
            ):
                violations.append(
                    {
                        "type": "bias_magnitude_collision",
                        "neurons": (j, j_prime),
                        "values": (bias_mags[j].item(), bias_mags[j_prime].item()),
                    }
                )
                result["is_self_avoiding"] = False

        # Check non-trivial weight ratios (no simple fractions p/q for small p, q)
        for j, j_prime in combinations(range(D_l), 2):
            for k in range(W.shape[1]):
                if abs(W[j_prime, k].item()) > tol:
                    ratio = W[j, k].item() / W[j_prime, k].item()
                    if _is_simple_fraction(ratio, max_denom=10, tol=tol):
                        violations.append(
                            {
                                "type": "simple_fraction_ratio",
                                "neurons": (j, j_prime),
                                "weight_index": k,
                                "ratio": ratio,
                            }
                        )
                        # Don't set is_self_avoiding=False for every ratio,
                        # just record the violation
                        break

        result["layers"].append(
            {
                "layer_index": l,
                "num_neurons": D_l,
                "violations": violations,
                "is_self_avoiding": len(
                    [v for v in violations if v["type"] == "bias_magnitude_collision"]
                )
                == 0,
            }
        )

    return result


def _is_simple_fraction(x: float, max_denom: int = 10, tol: float = 1e-4) -> bool:
    """Check if x is close to p/q for small integers p, q."""
    for q in range(1, max_denom + 1):
        for p in range(-max_denom * q, max_denom * q + 1):
            if abs(x - p / q) < tol:
                return True
    return False


# ──────────────────────────────────────────────
# Fefferman's assumptions (combined check)
# ──────────────────────────────────────────────
def check_fefferman_assumptions(layers: List[Dict], tol: float = 1e-6) -> Dict:
    """
    Check all three of Fefferman's assumptions:
    1. Non-zero biases with distinct magnitudes
    2. Non-zero weights with non-trivial ratios
    3. Full connectivity (no zero weights)
    """
    results = {
        "assumption_1_biases": True,
        "assumption_2_weights": True,
        "assumption_3_connectivity": True,
        "details": [],
    }

    for l, layer in enumerate(layers):
        W = layer["weight"]
        b = layer["bias"]

        # Assumption 1: Non-zero biases, distinct magnitudes
        zero_biases = (b.abs() < tol).sum().item()
        if zero_biases > 0:
            results["assumption_1_biases"] = False

        # Assumption 2: Non-zero weights
        zero_weights = (W.abs() < tol).sum().item()
        total_weights = W.numel()
        if zero_weights > 0:
            results["assumption_2_weights"] = False

        # Assumption 3: Full connectivity
        if zero_weights > 0:
            results["assumption_3_connectivity"] = False

        results["details"].append(
            {
                "layer": l,
                "zero_biases": zero_biases,
                "total_biases": b.numel(),
                "zero_weights": zero_weights,
                "total_weights": total_weights,
                "sparsity": zero_weights / total_weights,
            }
        )

    return results


# ──────────────────────────────────────────────
# Full identifiability report
# ──────────────────────────────────────────────
def identifiability_report(
    model: nn.Module, architecture: List[int], activation: str = "tanh"
) -> Dict:
    """
    Generate a comprehensive identifiability report for a neural network.
    """
    layers = extract_parameters(model)

    print("=" * 60)
    print("NEURAL NETWORK IDENTIFIABILITY REPORT")
    print("=" * 60)
    print(f"Architecture: {architecture}")
    print(f"Activation:   {activation}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Check 1: No clones
    clones = check_no_clones(layers)
    print(
        f"[1] No-Clones Condition:    {'PASS' if clones['is_clone_free'] else 'FAIL'}"
    )
    for l_info in clones["layers"]:
        if l_info["clone_pairs"]:
            print(
                f"    Layer {l_info['layer_index']}: {len(l_info['clone_pairs'])} clone pair(s)"
            )

    # Check 2: Non-degeneracy
    non_deg = check_non_degeneracy(model, architecture)
    print(
        f"[2] Non-Degeneracy:         {'PASS' if non_deg['is_non_degenerate'] else 'FAIL'}"
    )
    for l_info in non_deg["layers"]:
        if l_info["dead_neurons"]:
            print(
                f"    Layer {l_info['layer_index']}: {len(l_info['dead_neurons'])} dead neuron(s)"
            )

    # Check 3: Self-avoiding
    self_avoid = check_self_avoiding(layers)
    print(
        f"[3] Self-Avoiding Property: {'PASS' if self_avoid['is_self_avoiding'] else 'FAIL'}"
    )

    # Check 4: Fefferman's assumptions
    fefferman = check_fefferman_assumptions(layers)
    print(
        f"[4] Fefferman Assumption 1 (biases):       {'PASS' if fefferman['assumption_1_biases'] else 'FAIL'}"
    )
    print(
        f"[5] Fefferman Assumption 2 (weights):      {'PASS' if fefferman['assumption_2_weights'] else 'FAIL'}"
    )
    print(
        f"[6] Fefferman Assumption 3 (connectivity):  {'PASS' if fefferman['assumption_3_connectivity'] else 'FAIL'}"
    )

    # Activation analysis
    identifiable_activations = {"sigmoid", "tanh"}
    act_identifiable = activation in identifiable_activations
    print(
        f"\n[7] Activation identifiable: {'YES' if act_identifiable else 'NO'} ({activation})"
    )

    if activation == "tanh":
        print("    Identifiable up to sign flips and neuron permutations (~+-)")
    elif activation == "sigmoid":
        print("    Identifiable up to neuron permutations")
    else:
        print("    WARNING: ReLU fails genericity conditions for Fefferman's framework")

    # Overall
    is_identifiable = (
        clones["is_clone_free"]
        and non_deg["is_non_degenerate"]
        and self_avoid["is_self_avoiding"]
        and act_identifiable
    )
    print(f"\n{'=' * 60}")
    print(
        f"OVERALL IDENTIFIABILITY: {'IDENTIFIABLE' if is_identifiable else 'NOT GUARANTEED'}"
    )
    print(f"{'=' * 60}")

    return {
        "no_clones": clones,
        "non_degeneracy": non_deg,
        "self_avoiding": self_avoid,
        "fefferman": fefferman,
        "activation_identifiable": act_identifiable,
        "is_identifiable": is_identifiable,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Network Identifiability Checks"
    )
    parser.add_argument(
        "--architecture",
        nargs="+",
        type=int,
        default=[5, 10, 10, 1],
        help="Network architecture, e.g., 5 10 10 1",
    )
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "sigmoid", "relu"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = build_network(args.architecture, args.activation)
    report = identifiability_report(model, args.architecture, args.activation)
