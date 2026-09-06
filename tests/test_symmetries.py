"""Tests for the parameter symmetries, including the cases that must fail.

A test suite that only checks symmetries that hold would pass on an
implementation that returned "unchanged" for everything. Each positive case here
is paired with the activation where the same transformation must change the
function, so the suite can distinguish a working check from a broken one.

Everything runs in double precision. At single precision a genuine symmetry
leaves a residual near 1e-07, which is not small enough to separate it from a
real difference without a second run at higher precision.
"""

import unittest

import torch
import torch.nn as nn

from src.identifiability_checks import build_network, extract_parameters
from src.network_isomorphisms import (
    apply_permutation,
    apply_positive_scaling,
    apply_sign_flips,
)

ARCHITECTURE = [4, 6, 5, 1]
#: Residual below this at double precision is rounding, not a difference in the
#: represented function. Roughly ten thousand times the double epsilon.
EXACT = 1e-12


def load(layers, activation, dtype=torch.float64) -> nn.Module:
    model = build_network(ARCHITECTURE, activation).to(dtype)
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for module, layer in zip(linear, layers, strict=True):
            module.weight.copy_(layer["weight"].to(dtype))
            module.bias.copy_(layer["bias"].to(dtype))
    return model.eval()


def max_difference(activation, transform, dtype=torch.float64) -> float:
    """Largest output difference between a network and its transformed twin."""
    torch.manual_seed(0)
    base = build_network(ARCHITECTURE, activation).to(dtype)
    layers = [
        {"weight": layer["weight"].to(dtype), "bias": layer["bias"].to(dtype)}
        for layer in extract_parameters(base)
    ]
    modified = transform(layers)
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(1500, ARCHITECTURE[0], generator=generator).to(dtype)
    with torch.no_grad():
        return float((load(layers, activation, dtype)(inputs)
                      - load(modified, activation, dtype)(inputs)).abs().max())


def permute_all(layers):
    out = [dict(layer) for layer in layers]
    for index in range(len(layers) - 1):
        width = layers[index]["weight"].shape[0]
        generator = torch.Generator().manual_seed(index + 5)
        out = apply_permutation(out, index, torch.randperm(width, generator=generator).tolist())
    return out


def negate_all(layers):
    out = [dict(layer) for layer in layers]
    for index in range(len(layers) - 1):
        out = apply_sign_flips(out, index, [-1] * layers[index]["weight"].shape[0])
    return out


def scale_all(layers, factor=1.7):
    out = [dict(layer) for layer in layers]
    for index in range(len(layers) - 1):
        out = apply_positive_scaling(out, index, [factor] * layers[index]["weight"].shape[0])
    return out


class PermutationSymmetry(unittest.TestCase):
    """Relabelling hidden units is a symmetry for every activation."""

    def test_holds_for_every_activation(self):
        for activation in ("tanh", "sigmoid", "relu"):
            with self.subTest(activation=activation):
                self.assertLess(max_difference(activation, permute_all), EXACT)

    def test_the_parameters_really_changed(self):
        """Otherwise the test above would pass on a transformation that does nothing."""
        torch.manual_seed(0)
        layers = extract_parameters(build_network(ARCHITECTURE, "tanh"))
        permuted = permute_all(layers)
        moved = sum(float((a["weight"] - b["weight"]).abs().sum())
                    for a, b in zip(layers, permuted, strict=True))
        self.assertGreater(moved, 1.0)


class SignFlipSymmetry(unittest.TestCase):
    """Negating a hidden unit needs an odd activation."""

    def test_holds_for_tanh(self):
        self.assertLess(max_difference("tanh", negate_all), EXACT)

    def test_fails_for_sigmoid_and_relu(self):
        for activation in ("sigmoid", "relu"):
            with self.subTest(activation=activation):
                # Not merely above the tolerance: a difference of order one.
                self.assertGreater(max_difference(activation, negate_all), 1e-3)

    def test_negation_is_exact_in_binary_floating_point(self):
        """tanh(-z) = -tanh(z) and (-w)(-v) = wv hold with no rounding at all."""
        self.assertEqual(max_difference("tanh", negate_all, torch.float32), 0.0)


class PositiveScalingSymmetry(unittest.TestCase):
    """Rescaling needs positive homogeneity, which among these means ReLU."""

    def test_holds_for_relu(self):
        self.assertLess(max_difference("relu", scale_all), EXACT)

    def test_fails_for_tanh_and_sigmoid(self):
        for activation in ("tanh", "sigmoid"):
            with self.subTest(activation=activation):
                self.assertGreater(max_difference(activation, scale_all), 1e-3)

    def test_a_negative_factor_is_rejected(self):
        """ReLU(cz) = c ReLU(z) is false for c < 0, so the call must not succeed."""
        torch.manual_seed(0)
        layers = extract_parameters(build_network(ARCHITECTURE, "relu"))
        with self.assertRaises(ValueError):
            apply_positive_scaling(layers, 0, [-2.0] * ARCHITECTURE[1])

    def test_scaling_the_weights_without_the_bias_is_not_a_symmetry(self):
        """The bias sets where the kink is, so it has to move with the weights."""
        torch.manual_seed(0)
        base = build_network(ARCHITECTURE, "relu").to(torch.float64)
        layers = [{"weight": layer["weight"].to(torch.float64),
                   "bias": layer["bias"].to(torch.float64)}
                  for layer in extract_parameters(base)]
        broken = [dict(layer) for layer in layers]
        factor = torch.tensor(1.7, dtype=torch.float64)
        broken[0] = dict(broken[0])
        broken[0]["weight"] = layers[0]["weight"] * factor  # bias deliberately left alone
        broken[1] = dict(broken[1])
        broken[1]["weight"] = layers[1]["weight"] / factor

        generator = torch.Generator().manual_seed(1)
        inputs = torch.randn(1500, ARCHITECTURE[0], generator=generator).double()
        with torch.no_grad():
            difference = float((load(layers, "relu")(inputs)
                                - load(broken, "relu")(inputs)).abs().max())
        self.assertGreater(difference, 1e-3)


class DtypeHandling(unittest.TestCase):
    def test_transformations_preserve_double_precision(self):
        """A transformation that downcast to float32 would hide the precision."""
        torch.manual_seed(0)
        base = build_network(ARCHITECTURE, "tanh").to(torch.float64)
        layers = [{"weight": layer["weight"].to(torch.float64),
                   "bias": layer["bias"].to(torch.float64)}
                  for layer in extract_parameters(base)]
        for transform in (permute_all, negate_all, scale_all):
            with self.subTest(transform=transform.__name__):
                for layer in transform(layers):
                    self.assertEqual(layer["weight"].dtype, torch.float64)
                    self.assertEqual(layer["bias"].dtype, torch.float64)

    def test_double_precision_shrinks_a_genuine_residual(self):
        """The evidence that separates a symmetry from a coincidence."""
        single = max_difference("tanh", permute_all, torch.float32)
        double = max_difference("tanh", permute_all, torch.float64)
        self.assertGreater(single, double * 1000)
        self.assertLess(double, EXACT)

    def test_double_precision_does_not_rescue_a_non_symmetry(self):
        single = max_difference("sigmoid", negate_all, torch.float32)
        double = max_difference("sigmoid", negate_all, torch.float64)
        self.assertAlmostEqual(single, double, places=4)


if __name__ == "__main__":
    unittest.main()
