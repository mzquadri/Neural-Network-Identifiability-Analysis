"""Verify that documented source assets are present and importable."""

from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "LICENSE",
    "src/__init__.py",
    "src/identifiability_checks.py",
    "src/network_isomorphisms.py",
    "src/activation_analysis.py",
    "src/symmetry_breaking.py",
    "src/visualization.py",
    "notebooks/01_Identifiability_Conditions.ipynb",
    "notebooks/02_Empirical_Analysis.ipynb",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    for source in (ROOT / "src").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED_FILES)} required artifacts available.")


if __name__ == "__main__":
    main()
