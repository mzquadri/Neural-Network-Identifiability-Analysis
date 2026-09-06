"""Figures for the symmetry verification, drawn from the recorded run.

Three figures, each answering one question:

    01  which parameter transformations leave the function unchanged
    02  how to tell a real symmetry from a coincidence
    03  a concrete pair of different networks computing the same function

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import portfolio_style as ps  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "docs" / "figures"
RUN = ROOT / "results" / "symmetry_verification.json"

NICE_T = {
    "permutation": "Relabel hidden units",
    "sign_flip": "Negate a hidden unit",
    "positive_scaling": "Rescale by c > 0",
}
NICE_A = {"tanh": "tanh", "sigmoid": "sigmoid", "relu": "ReLU"}
ORDER_T = ["permutation", "sign_flip", "positive_scaling"]
ORDER_A = ["tanh", "sigmoid", "relu"]


def load() -> dict:
    if not RUN.exists():
        raise SystemExit("results/symmetry_verification.json is missing. "
                         "Run: python -m src.symmetry_verification")
    return json.loads(RUN.read_text(encoding="utf-8"))


def fig_matrix(run):
    """Which transformation is a symmetry, for which activation."""
    cases = run["cases"]
    grid = np.array([[1.0 if cases[f"{a}|{t}"]["function_preserved"] else 0.0
                      for t in ORDER_T] for a in ORDER_A])

    fig = plt.figure(figsize=(13.0, 7.2))
    ax = fig.add_axes([0.235, 0.315, 0.415, 0.375])

    from matplotlib.colors import ListedColormap
    ax.imshow(grid, cmap=ListedColormap(["#FEF2F2", "#ECFDF5"]), vmin=0, vmax=1,
              aspect="auto")
    for i, a in enumerate(ORDER_A):
        for j, t in enumerate(ORDER_T):
            case = cases[f"{a}|{t}"]
            held = case["function_preserved"]
            ax.text(j, i - 0.14, "unchanged" if held else "changed", ha="center",
                    va="center", fontsize=10.4,
                    color=ps.GREEN if held else ps.RED, fontweight="600")
            residual = case["float64_max_output_difference"]
            shown = "0" if residual == 0 else f"{residual:.0e}"
            ax.text(j, i + 0.17, shown, ha="center", va="center", fontsize=9.0,
                    color=ps.MUTED)
    ax.set_xticks(range(len(ORDER_T)))
    ax.set_xticklabels([NICE_T[t].replace(" by ", "\nby ").replace("hidden units", "hidden\nunits")
                        .replace("a hidden unit", "a hidden\nunit") for t in ORDER_T],
                       fontsize=10.2)
    ax.set_yticks(range(len(ORDER_A)))
    ax.set_yticklabels([NICE_A[a] for a in ORDER_A], fontsize=11.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ORDER_T), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER_A), 1), minor=True)
    ax.grid(which="minor", color=ps.PAPER, linewidth=3.0)
    ax.text(0, 1.13, "largest output difference at double precision, over "
                     f"{run['setup']['n_test_inputs']:,} inputs",
            transform=ax.transAxes, fontsize=9.6, color=ps.FAINT, va="bottom")

    # Placed beside the matrix, with lines short enough for the remaining width.
    reasons = [
        ("Relabelling works everywhere",
         ["The hidden units carry no intrinsic",
          "order, so every activation admits it."]),
        ("Negation needs an odd activation",
         ["tanh(-z) = -tanh(z), so the two sign",
          "changes cancel. Sigmoid and ReLU",
          "are not odd, and it fails for them."]),
        ("Rescaling needs positive homogeneity",
         ["ReLU(cz) = c ReLU(z) holds for c > 0.",
          "tanh and sigmoid do not satisfy it,",
          "and c < 0 breaks it even for ReLU."]),
    ]
    y = 0.665
    for head, body in reasons:
        fig.text(0.695, y, head, fontsize=10.2, color=ps.INK, fontweight="600",
                 va="top", transform=fig.transFigure)
        for k, line in enumerate(body):
            fig.text(0.695, y - 0.038 - k * 0.030, line, fontsize=9.2,
                     color=ps.MUTED, va="top", transform=fig.transFigure)
        y -= 0.055 + 0.030 * len(body)

    n_preserved = len(run["preserved"])
    ps.title_block(
        fig, "Different weights, same function",
        "Three ways to change the parameters of a fully connected network, against "
        "three activations. Green means the\nnetwork computes the same function "
        "afterwards.", y=0.955, size=20)
    ps.footnote(fig, [
        f"{n_preserved} of the nine combinations leave the function unchanged, and "
        f"each one is an explicit pair of different parameter vectors that agree on "
        f"every input tested. That is what makes the parameterisation non-unique.",
        "Whether a symmetry holds depends on the activation, not on the network. "
        "The same transformation that is invisible under tanh is plainly visible "
        "under ReLU.",
        "Source: results/symmetry_verification.json."], y=0.115)
    ps.save(fig, OUT, "01_symmetry_matrix")


def fig_precision(run):
    """A real symmetry gets more exact with precision. A coincidence does not."""
    cases = run["cases"]
    keys = [f"{a}|{t}" for a in ORDER_A for t in ORDER_T]
    labels = [f"{NICE_A[cases[k]['activation']]}\n{NICE_T[cases[k]['transformation']]}"
              for k in keys]

    fig = plt.figure(figsize=(13.0, 7.2))
    ax = fig.add_axes([0.075, 0.300, 0.885, 0.415])

    x = np.arange(len(keys))
    floor = 1e-18
    for offset, (field, name, colour) in enumerate((
        ("float32_max_output_difference", "single precision", ps.BLUE_SOFT),
        ("float64_max_output_difference", "double precision", ps.BLUE),
    )):
        values = [max(cases[k][field], floor) for k in keys]
        ax.bar(x + (offset - 0.5) * 0.36, values, width=0.36, color=colour,
               zorder=3, label=name)
    ax.set_yscale("log")
    ax.set_ylim(floor, 1e2)
    ax.axhline(run["setup"]["exact_tolerance_float64"], color=ps.INK, lw=1.3,
               ls="--", zorder=5)
    ax.text(len(keys) - 0.4, run["setup"]["exact_tolerance_float64"] * 2.2,
            "tolerance for calling a residual floating point noise", fontsize=9.2,
            color=ps.INK, ha="right")
    for i, k in enumerate(keys):
        if cases[k]["function_preserved"]:
            ax.text(i, floor * 3, "symmetry", ha="center", fontsize=8.4,
                    color=ps.GREEN, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ps.clean(ax)
    ax.set_ylabel("largest output difference (log scale)", fontsize=10.4)
    ax.legend(fontsize=9.6, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.14))

    ps.title_block(
        fig, "Telling a symmetry from a coincidence",
        "Every case run twice, once at single precision and once at double. What "
        "happens to the residual separates the\ntwo kinds of small number.",
        y=0.955, size=20)
    ps.footnote(fig, [
        "A genuine symmetry is exact in arithmetic, so its residual is only "
        "rounding. It falls by about nine orders of magnitude when the precision "
        "improves.",
        "A transformation that changes the function does not move at all, so "
        "reading a single-precision 1e-07 as a match, without the second run, would "
        "be guesswork.",
        "Negating a tanh unit gives exactly zero at both precisions: negation is "
        "exact in binary floating point, while reordering a sum is not.",
        "Source: results/symmetry_verification.json."], y=0.155)
    ps.save(fig, OUT, "02_precision_evidence")


def fig_counterexample():
    """Two small networks, written out in full, computing the same function."""
    torch.manual_seed(7)
    w1 = torch.tensor([[1.4], [-0.8], [2.1]], dtype=torch.float64)
    b1 = torch.tensor([0.3, -1.1, 0.6], dtype=torch.float64)
    w2 = torch.tensor([[0.9, 1.5, -0.7]], dtype=torch.float64)
    b2 = torch.tensor([0.2], dtype=torch.float64)

    # Relabel units 0 and 2, then negate unit 1. Both are symmetries for tanh.
    perm = [2, 1, 0]
    signs = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    w1b = (w1[perm] * signs.unsqueeze(1))
    b1b = (b1[perm] * signs)
    w2b = (w2[:, perm] * signs.unsqueeze(0))
    b2b = b2.clone()

    def net(x, W1, B1, W2, B2):
        return (torch.tanh(x @ W1.T + B1) @ W2.T + B2).squeeze(-1)

    xs = torch.linspace(-3, 3, 600, dtype=torch.float64).unsqueeze(1)
    ya = net(xs, w1, b1, w2, b2)
    yb = net(xs, w1b, b1b, w2b, b2b)
    residual = (ya - yb).abs().max().item()

    fig = plt.figure(figsize=(13.0, 7.6))
    axL = fig.add_axes([0.075, 0.275, 0.400, 0.440])
    axR = fig.add_axes([0.560, 0.275, 0.400, 0.440])

    axL.plot(xs.squeeze(), ya, color=ps.BLUE, lw=3.0, zorder=3, label="network A")
    axL.plot(xs.squeeze(), yb, color=ps.AMBER, lw=1.4, ls="--", zorder=4,
             label="network B")
    ps.clean(axL)
    axL.set_xlabel("input x", fontsize=10.4)
    axL.set_ylabel("output f(x)", fontsize=10.4)
    axL.legend(fontsize=9.6, ncol=2)
    axL.text(0, 1.06, "the two functions", transform=axL.transAxes, fontsize=10.6,
             color=ps.INK, fontweight="600", va="bottom")

    rows = [
        ("W1", w1.squeeze(-1).tolist(), w1b.squeeze(-1).tolist()),
        ("b1", b1.tolist(), b1b.tolist()),
        ("W2", w2.squeeze(0).tolist(), w2b.squeeze(0).tolist()),
    ]
    axR.axis("off")
    axR.text(0.00, 1.06, "the two parameter sets", transform=axR.transAxes,
             fontsize=10.6, color=ps.INK, fontweight="600", va="bottom")
    axR.text(0.30, 0.95, "network A", transform=axR.transAxes, fontsize=10.0,
             color=ps.BLUE, fontweight="600", ha="center")
    axR.text(0.78, 0.95, "network B", transform=axR.transAxes, fontsize=10.0,
             color=ps.AMBER, fontweight="600", ha="center")
    for i, (name, a_vals, b_vals) in enumerate(rows):
        y = 0.80 - i * 0.20
        axR.text(0.00, y, name, transform=axR.transAxes, fontsize=10.4,
                 color=ps.INK, fontweight="600", va="center")
        axR.text(0.30, y, "[" + ", ".join(f"{v:+.1f}" for v in a_vals) + "]",
                 transform=axR.transAxes, fontsize=10.0, color=ps.BLUE,
                 ha="center", va="center", family="monospace")
        axR.text(0.78, y, "[" + ", ".join(f"{v:+.1f}" for v in b_vals) + "]",
                 transform=axR.transAxes, fontsize=10.0, color=ps.AMBER,
                 ha="center", va="center", family="monospace")
    axR.text(0.00, 0.14, f"largest difference over 600 inputs: {residual:.1e}",
             transform=axR.transAxes, fontsize=10.0, color=ps.GREEN,
             fontweight="600", va="center")
    axR.text(0.00, 0.03, "b2 is +0.2 in both and is not affected.",
             transform=axR.transAxes, fontsize=9.2, color=ps.FAINT, va="center")

    ps.title_block(
        fig, "A worked pair: 1-3-1 with tanh",
        "Network B is network A with hidden units 1 and 3 relabelled and unit 2 "
        "negated. Every weight below differs,\nand the two curves lie on top of "
        "each other.", y=0.958, size=20)
    ps.footnote(fig, [
        "This is the smallest honest illustration of the point. Fitting network A "
        "to data and recovering network B is not a failure of the optimiser: the "
        "two are the same function, and no amount of data separates them.",
        f"The residual of {residual:.0e} at double precision is rounding, not "
        f"disagreement. Written by hand rather than sampled, so the same numbers "
        f"appear on any machine.",
        "Source: scripts/figures/generate_figures.py, worked in double precision."],
        y=0.115)
    ps.save(fig, OUT, "03_worked_counterexample")
    return residual


def main() -> int:
    ps.apply()
    run = load()
    print()
    fig_matrix(run)
    fig_precision(run)
    residual = fig_counterexample()
    print(f"\n  worked example residual: {residual:.3e}")
    print(f"  figures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
