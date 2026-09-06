"""Verify the artifacts, and that the README still matches the recorded run.

The numeric claims in the README come from results/symmetry_verification.json.
This checks that they still do, so a change in the verification cannot leave a
stale number in the text.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "requirements.txt",
    "src/network_isomorphisms.py",
    "src/identifiability_checks.py",
    "src/symmetry_verification.py",
    "results/symmetry_verification.json",
    "docs/figures/01_symmetry_matrix.png",
    "docs/figures/02_precision_evidence.png",
    "docs/figures/03_worked_counterexample.png",
)

LABEL = {"permutation": "relabel", "sign_flip": "negate", "positive_scaling": "rescale"}
NAME = {"tanh": "tanh", "sigmoid": "sigmoid", "relu": "ReLU"}


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    run = json.loads((ROOT / "results" / "symmetry_verification.json").read_text("utf-8"))
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    problems = []

    # The summary table: one row per activation, three verdicts each.
    for activation in ("tanh", "sigmoid", "relu"):
        verdicts = [
            "unchanged" if run["cases"][f"{activation}|{t}"]["function_preserved"]
            else "changed"
            for t in ("permutation", "sign_flip", "positive_scaling")
        ]
        row = (rf"\| {NAME[activation]} \| {verdicts[0]} \| {verdicts[1]} \| "
               rf"{verdicts[2]} \|")
        if not re.search(row, readme):
            problems.append(f"summary row for {activation} does not match the run "
                            f"({', '.join(verdicts)})")

    preserved = len(run["preserved"])
    words = {5: "Five", 4: "Four", 6: "Six", 3: "Three", 7: "Seven"}
    expected = words.get(preserved, str(preserved))
    if f"{expected} of the nine" not in readme:
        problems.append(f"the README does not state that {expected} of nine "
                        f"combinations are preserved")

    # The precision table, checked case by case within a tolerance so that a
    # rebuild on another machine does not fail on the last digit.
    for key, label in (("tanh|permutation", "tanh, relabel"),
                       ("tanh|positive_scaling", "tanh, rescale"),
                       ("relu|positive_scaling", "ReLU, rescale"),
                       ("sigmoid|sign_flip", "sigmoid, negate")):
        case = run["cases"][key]
        row = re.search(rf"\| {re.escape(label)} \| ([\d.e+-]+) \| ([\d.e+-]+) \|", readme)
        if row is None:
            problems.append(f"precision row for {label} is missing")
            continue
        # The single precision column is an order of magnitude claim, not a
        # precise one: a float32 rounding residual depends on the summation order
        # the platform's BLAS chooses, and CI legitimately reports 1.3e-07 where
        # this machine reports 1.0e-07. The double precision column is what the
        # conclusions rest on, so it is held to a tight tolerance.
        pairs = (
            (row.group(1), case["float32_max_output_difference"], "single", 5.0),
            (row.group(2), case["float64_max_output_difference"], "double", 1.25),
        )
        for stated, actual, which, factor in pairs:
            stated_value = float(stated)
            if actual == 0:
                ok = stated_value == 0
            elif stated_value == 0:
                ok = False
            else:
                ratio = max(stated_value / actual, actual / stated_value)
                ok = ratio < factor
            if not ok:
                problems.append(f"precision row {label} ({which}): README says "
                                f"{stated}, the run gives {actual:.2e} "
                                f"(allowed factor {factor})")

    if not run["negative_scale_rejected"]:
        problems.append("a negative scale factor is no longer rejected, but the "
                        "README says it is")

    if problems:
        raise SystemExit("README disagrees with the recorded run:\n" + "\n".join(problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, README "
          f"agrees with results/symmetry_verification.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
