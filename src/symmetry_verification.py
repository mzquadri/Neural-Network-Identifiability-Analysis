"""Check which parameter transformations leave the represented function unchanged.

    python -m src.symmetry_verification

Three transformations are applied to the hidden layers of a small fully connected
network, for three activations, and the resulting function is compared with the
original on sampled inputs.

The point of running each case at two precisions is to separate two things that
look the same in a single number. A transformation that is a genuine symmetry
leaves a residual that shrinks with precision, because the only error is
floating point. A transformation that is not a symmetry leaves a residual of the
same size at any precision, because the functions genuinely differ.

Nothing here proves global functional equivalence. The comparison is over a
finite sample of inputs, which can refute a claimed symmetry but cannot establish
one. What the transformations do establish is the other direction: each verified
case is an explicit pair of different parameter vectors that agree on every input
tested, which is enough to show the parameterisation is not unique.

Writes results/symmetry_verification.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from .identifiability_checks import build_network, extract_parameters
from .network_isomorphisms import (
    apply_permutation,
    apply_positive_scaling,
    apply_sign_flips,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "symmetry_verification.json"

ARCHITECTURE = [5, 10, 8, 1]
ACTIVATIONS = ("tanh", "sigmoid", "relu")
SEED = 42
N_TEST = 4000

#: A scale that is not a power of two. Powers of two are exact in binary floating
#: point, so using 2.0 would report a zero residual that says more about the
#: number chosen than about the symmetry.
SCALE = 1.7

#: Residual below this at double precision is treated as floating point noise
#: rather than a difference in the represented function.
EXACT_TOLERANCE = 1e-12


def rebuild(layers, architecture, activation: str, dtype) -> nn.Module:
    """Load a parameter list into a fresh network at the requested precision."""
    model = build_network(architecture, activation).to(dtype)
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for module, layer in zip(linear, layers, strict=True):
            module.weight.copy_(layer["weight"].to(dtype))
            module.bias.copy_(layer["bias"].to(dtype))
    return model.eval()


def transform(layers, kind: str):
    """Apply one transformation to every hidden layer."""
    out = [dict(layer) for layer in layers]
    for index in range(len(layers) - 1):
        width = layers[index]["weight"].shape[0]
        if kind == "permutation":
            generator = torch.Generator().manual_seed(SEED + index)
            out = apply_permutation(out, index, torch.randperm(width, generator=generator).tolist())
        elif kind == "sign_flip":
            out = apply_sign_flips(out, index, [-1] * width)
        elif kind == "positive_scaling":
            out = apply_positive_scaling(out, index, [SCALE] * width)
        else:
            raise ValueError(f"unknown transformation {kind}")
    return out


def max_output_difference(activation: str, kind: str, dtype) -> tuple[float, float]:
    """Largest and mean absolute output difference over sampled inputs."""
    torch.manual_seed(SEED)
    original = build_network(ARCHITECTURE, activation).to(dtype)
    layers = [
        {"weight": layer["weight"].to(dtype), "bias": layer["bias"].to(dtype)}
        for layer in extract_parameters(original)
    ]
    modified = transform(layers, kind)

    model_a = rebuild(layers, ARCHITECTURE, activation, dtype)
    model_b = rebuild(modified, ARCHITECTURE, activation, dtype)

    generator = torch.Generator().manual_seed(SEED + 1000)
    inputs = torch.randn(N_TEST, ARCHITECTURE[0], generator=generator).to(dtype)
    with torch.no_grad():
        difference = (model_a(inputs) - model_b(inputs)).abs()
    return float(difference.max()), float(difference.mean())


def parameter_distance(activation: str, kind: str) -> float:
    """How far the transformed parameters are from the originals.

    A symmetry that left the parameters alone would be uninteresting. This
    confirms the two networks really are different points in parameter space.
    """
    torch.manual_seed(SEED)
    layers = extract_parameters(build_network(ARCHITECTURE, activation))
    modified = transform(layers, kind)
    total = 0.0
    for before, after in zip(layers, modified, strict=True):
        total += float((before["weight"] - after["weight"]).abs().sum())
        total += float((before["bias"] - after["bias"]).abs().sum())
    return total


def main() -> int:
    print(f"  architecture {ARCHITECTURE}, seed {SEED}, {N_TEST} sampled inputs")
    print(f"  scale factor {SCALE} (not a power of two, so the residual is not "
          f"an artifact of exact binary arithmetic)\n")

    header = (f"  {'activation':<9} {'transformation':<18} {'float32':>11} "
              f"{'float64':>11} {'param dist':>11}   symmetry")
    print(header)
    print("  " + "-" * (len(header) - 2))

    cases = {}
    for activation in ACTIVATIONS:
        for kind in ("permutation", "sign_flip", "positive_scaling"):
            f32_max, f32_mean = max_output_difference(activation, kind, torch.float32)
            f64_max, f64_mean = max_output_difference(activation, kind, torch.float64)
            distance = parameter_distance(activation, kind)
            holds = f64_max < EXACT_TOLERANCE
            cases[f"{activation}|{kind}"] = {
                "activation": activation,
                "transformation": kind,
                "float32_max_output_difference": f32_max,
                "float32_mean_output_difference": f32_mean,
                "float64_max_output_difference": f64_max,
                "float64_mean_output_difference": f64_mean,
                "parameter_l1_distance": distance,
                "function_preserved": bool(holds),
            }
            print(f"  {activation:<9} {kind:<18} {f32_max:11.3e} {f64_max:11.3e} "
                  f"{distance:11.2f}   {'preserved' if holds else 'CHANGED'}")

    preserved = [k for k, v in cases.items() if v["function_preserved"]]
    print(f"\n  {len(preserved)} of {len(cases)} transformations leave the function unchanged:")
    for key in preserved:
        print(f"    {key.replace('|', ', ')}")

    # A negative scale on ReLU, to show the c > 0 restriction is not decorative.
    torch.manual_seed(SEED)
    relu_layers = extract_parameters(build_network(ARCHITECTURE, "relu"))
    try:
        apply_positive_scaling(relu_layers, 0, [-2.0] * ARCHITECTURE[1])
        rejected = False
    except ValueError:
        rejected = True
    print(f"\n  a negative scale factor is rejected by the implementation: {rejected}")

    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch": torch.__version__,
        },
        "setup": {
            "architecture": ARCHITECTURE,
            "activations": list(ACTIVATIONS),
            "seed": SEED,
            "n_test_inputs": N_TEST,
            "input_distribution": "standard normal",
            "scale_factor": SCALE,
            "exact_tolerance_float64": EXACT_TOLERANCE,
        },
        "cases": cases,
        "preserved": preserved,
        "negative_scale_rejected": rejected,
        "note": "A sampled comparison can refute a claimed symmetry but cannot "
                "establish global functional equivalence. Each preserved case is "
                "an explicit pair of distinct parameter vectors agreeing on every "
                "input tested.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n  wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
