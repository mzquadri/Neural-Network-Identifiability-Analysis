"""
Network isomorphism detection.
Implements checks for faithful and extensional isomorphisms between neural networks.

Based on Fefferman's framework: two networks are equivalent if related by
hidden-layer neuron permutations and (for tanh) sign flips.
"""

import argparse
from itertools import permutations
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

from identifiability_checks import build_network, extract_parameters


# ──────────────────────────────────────────────
# Permutation and sign-flip operations
# ──────────────────────────────────────────────
def apply_permutation(
    layers: List[Dict], layer_idx: int, perm: List[int]
) -> List[Dict]:
    """
    Apply a neuron permutation to a specific hidden layer.

    When permuting neurons in layer l:
    - Permute rows of W^l and b^l (incoming connections)
    - Permute columns of W^{l+1} (outgoing connections)
    """
    import copy

    new_layers = copy.deepcopy(layers)
    perm_tensor = torch.tensor(perm, dtype=torch.long)

    # Permute incoming: rows of W^l and b^l
    new_layers[layer_idx]["weight"] = layers[layer_idx]["weight"][perm_tensor]
    new_layers[layer_idx]["bias"] = layers[layer_idx]["bias"][perm_tensor]

    # Permute outgoing: columns of W^{l+1}
    if layer_idx + 1 < len(layers):
        new_layers[layer_idx + 1]["weight"] = layers[layer_idx + 1]["weight"][
            :, perm_tensor
        ]

    return new_layers


def apply_sign_flips(
    layers: List[Dict], layer_idx: int, signs: List[int]
) -> List[Dict]:
    """
    Apply sign flips to neurons in a hidden layer.
    For tanh activation: f(-x) = -f(x), so flipping signs preserves the mapping.

    When flipping sign of neuron j in layer l:
    - Flip sign of W^l[j, :] and b^l[j]
    - Flip sign of W^{l+1}[:, j]
    """
    import copy

    new_layers = copy.deepcopy(layers)
    signs_tensor = torch.tensor(signs, dtype=torch.float32)

    # Flip incoming
    new_layers[layer_idx]["weight"] = layers[layer_idx][
        "weight"
    ] * signs_tensor.unsqueeze(1)
    new_layers[layer_idx]["bias"] = layers[layer_idx]["bias"] * signs_tensor

    # Flip outgoing
    if layer_idx + 1 < len(layers):
        new_layers[layer_idx + 1]["weight"] = layers[layer_idx + 1][
            "weight"
        ] * signs_tensor.unsqueeze(0)

    return new_layers


# ──────────────────────────────────────────────
# Isomorphism checking
# ──────────────────────────────────────────────
def check_extensional_isomorphism(
    model_a: nn.Module,
    model_b: nn.Module,
    architecture: List[int],
    n_test: int = 1000,
    tol: float = 1e-5,
) -> Dict:
    """
    Check if two networks are extensionally isomorphic.

    Two networks are extensionally isomorphic if they produce the same
    output for all inputs (up to numerical tolerance).
    """
    model_a.eval()
    model_b.eval()

    X = torch.randn(n_test, architecture[0])

    with torch.no_grad():
        y_a = model_a(X)
        y_b = model_b(X)

    max_diff = (y_a - y_b).abs().max().item()
    mean_diff = (y_a - y_b).abs().mean().item()
    is_isomorphic = max_diff < tol

    return {
        "is_extensionally_isomorphic": is_isomorphic,
        "max_output_difference": max_diff,
        "mean_output_difference": mean_diff,
        "n_test_points": n_test,
    }


def find_layer_permutation(
    W_a: torch.Tensor,
    b_a: torch.Tensor,
    W_b: torch.Tensor,
    b_b: torch.Tensor,
    allow_sign_flips: bool = True,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Find permutation (and optionally sign flips) that maps layer A to layer B.

    Uses the Hungarian algorithm on a cost matrix of neuron distances.

    Returns
    -------
    permutation : list of int or None
    signs : list of int or None (each +1 or -1)
    """
    D = W_a.shape[0]
    cost_matrix = torch.zeros(D, D)

    best_signs = [1] * D

    for i in range(D):
        for j in range(D):
            # Without sign flip
            diff_pos = (W_a[i] - W_b[j]).norm() + abs(b_a[i] - b_b[j])
            # With sign flip
            diff_neg = (W_a[i] + W_b[j]).norm() + abs(b_a[i] + b_b[j])

            if allow_sign_flips and diff_neg < diff_pos:
                cost_matrix[i, j] = diff_neg
            else:
                cost_matrix[i, j] = diff_pos

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    permutation = col_ind.tolist()

    # Determine signs
    if allow_sign_flips:
        for i, j in zip(row_ind, col_ind):
            diff_pos = (W_a[i] - W_b[j]).norm() + abs(b_a[i] - b_b[j])
            diff_neg = (W_a[i] + W_b[j]).norm() + abs(b_a[i] + b_b[j])
            best_signs[i] = -1 if diff_neg < diff_pos else 1

    total_cost = sum(cost_matrix[i, permutation[i]] for i in range(D))

    return permutation, best_signs, total_cost.item()


def check_faithful_isomorphism(
    layers_a: List[Dict],
    layers_b: List[Dict],
    allow_sign_flips: bool = True,
    tol: float = 1e-4,
) -> Dict:
    """
    Check if two networks are faithfully isomorphic.

    Searches for layer-wise permutations (and sign flips for tanh)
    that transform network A into network B.
    """
    if len(layers_a) != len(layers_b):
        return {
            "is_faithfully_isomorphic": False,
            "reason": "Different number of layers",
        }

    for l in range(len(layers_a)):
        if layers_a[l]["weight"].shape != layers_b[l]["weight"].shape:
            return {
                "is_faithfully_isomorphic": False,
                "reason": f"Different layer {l} sizes",
            }

    permutations_found = []
    signs_found = []
    total_cost = 0.0

    for l in range(len(layers_a) - 1):  # Hidden layers
        perm, signs, cost = find_layer_permutation(
            layers_a[l]["weight"],
            layers_a[l]["bias"],
            layers_b[l]["weight"],
            layers_b[l]["bias"],
            allow_sign_flips=allow_sign_flips,
        )
        permutations_found.append(perm)
        signs_found.append(signs)
        total_cost += cost

    is_isomorphic = total_cost < tol * sum(
        l["weight"].numel() + l["bias"].numel() for l in layers_a
    )

    return {
        "is_faithfully_isomorphic": is_isomorphic,
        "permutations": permutations_found,
        "signs": signs_found,
        "total_alignment_cost": total_cost,
    }


# ──────────────────────────────────────────────
# Create isomorphic network (for testing)
# ──────────────────────────────────────────────
def create_permuted_network(
    model: nn.Module, architecture: List[int], activation: str = "tanh"
) -> nn.Module:
    """
    Create a network that is isomorphic to the given model by
    randomly permuting hidden neurons and (for tanh) flipping signs.
    """
    layers = extract_parameters(model)
    new_model = build_network(architecture, activation)
    new_layers = extract_parameters(new_model)

    # Apply random permutations and sign flips to each hidden layer
    modified_layers = [dict(l) for l in layers]

    for l in range(len(layers) - 1):
        D_l = layers[l]["weight"].shape[0]

        # Random permutation
        perm = torch.randperm(D_l).tolist()
        modified_layers = apply_permutation(modified_layers, l, perm)

        # Random sign flips (only for tanh)
        if activation == "tanh":
            signs = [(-1) ** int(torch.rand(1).item() > 0.5) for _ in range(D_l)]
            modified_layers = apply_sign_flips(modified_layers, l, signs)

    # Load modified parameters into new model
    linear_modules = [m for m in new_model.modules() if isinstance(m, nn.Linear)]
    for i, module in enumerate(linear_modules):
        with torch.no_grad():
            module.weight.copy_(modified_layers[i]["weight"])
            if module.bias is not None:
                module.bias.copy_(modified_layers[i]["bias"])

    return new_model


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────
def demo_isomorphism_detection():
    """Demonstrate isomorphism detection between original and permuted networks."""
    torch.manual_seed(42)
    architecture = [5, 10, 8, 1]
    activation = "tanh"

    print("=" * 60)
    print("ISOMORPHISM DETECTION DEMO")
    print("=" * 60)

    # Create original network
    model_a = build_network(architecture, activation)
    print(f"\nArchitecture: {architecture}, Activation: {activation}")

    # Create isomorphic copy (permuted + sign-flipped)
    model_b = create_permuted_network(model_a, architecture, activation)

    # Create a different network
    model_c = build_network(architecture, activation)

    # Test 1: Extensional isomorphism (A vs B)
    ext_ab = check_extensional_isomorphism(model_a, model_b, architecture)
    print(f"\nA vs B (permuted copy):")
    print(f"  Extensionally isomorphic: {ext_ab['is_extensionally_isomorphic']}")
    print(f"  Max output diff: {ext_ab['max_output_difference']:.2e}")

    # Test 2: Extensional isomorphism (A vs C)
    ext_ac = check_extensional_isomorphism(model_a, model_c, architecture)
    print(f"\nA vs C (different network):")
    print(f"  Extensionally isomorphic: {ext_ac['is_extensionally_isomorphic']}")
    print(f"  Max output diff: {ext_ac['max_output_difference']:.2e}")

    # Test 3: Faithful isomorphism (A vs B)
    layers_a = extract_parameters(model_a)
    layers_b = extract_parameters(model_b)
    faith_ab = check_faithful_isomorphism(layers_a, layers_b)
    print(f"\nFaithful isomorphism (A vs B):")
    print(f"  Is faithfully isomorphic: {faith_ab['is_faithfully_isomorphic']}")
    print(f"  Alignment cost: {faith_ab['total_alignment_cost']:.6f}")

    # Test 4: Faithful isomorphism (A vs C)
    layers_c = extract_parameters(model_c)
    faith_ac = check_faithful_isomorphism(layers_a, layers_c)
    print(f"\nFaithful isomorphism (A vs C):")
    print(f"  Is faithfully isomorphic: {faith_ac['is_faithfully_isomorphic']}")
    print(f"  Alignment cost: {faith_ac['total_alignment_cost']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Isomorphism Detection")
    parser.add_argument("--compare", action="store_true", help="Run isomorphism demo")
    args = parser.parse_args()

    if args.compare:
        demo_isomorphism_detection()
    else:
        demo_isomorphism_detection()
