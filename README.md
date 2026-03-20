# Neural Network Identifiability Analysis

Implementation and empirical investigation of identifiability conditions for deep neural networks, based on **Fefferman's framework** and related theoretical results. This project operationalizes mathematical identifiability criteria into implementable checks and invariance tests.

## Background

This project stems from the seminar **"Identification of Neural Networks"** under **Prof. Massimo Fornasier** and **Dr. Alessandro Scagliotti** at the Technical University of Munich (TUM), Department of Mathematics.

### What is Identifiability?

Identifiability ensures that neural networks producing the same input-output mapping are equivalent under specific transformations (permutations and sign flips). If two networks generate the same output for all inputs, they should be considered structurally equivalent.

### Why Does It Matter?

- **Interpreting model behavior**: Uniquely determined parameters enable meaningful interpretation
- **Debugging and diagnosis**: Identifiable networks have well-defined failure modes
- **Training consistency**: Ensures reproducibility across training runs

## Key Concepts Implemented

1. **No-Clones Condition**: Detecting clone pairs (neurons with identical weights/biases in the same layer)
2. **Non-Degeneracy Checks**: Verifying all nodes contribute to the network output
3. **Self-Avoiding Property**: Checking that weight/bias configurations avoid degenerate overlaps
4. **Activation-Dependent Uniqueness**: Analyzing how sigmoid/tanh activations guarantee parameter identifiability vs. ReLU limitations
5. **Network Isomorphisms**: Detecting faithful and extensional isomorphisms between networks
6. **Symmetry-Breaking Strategies**: Proposed constraints to reduce equivalence classes

## Project Structure

```
Neural-Network-Identifiability-Analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── identifiability_checks.py   # Core identifiability verification
│   ├── network_isomorphisms.py     # Isomorphism detection algorithms
│   ├── activation_analysis.py      # Activation function properties
│   ├── symmetry_breaking.py        # Symmetry-breaking strategies
│   └── visualization.py            # Visualization utilities
├── notebooks/
│   ├── 01_Identifiability_Conditions.ipynb
│   └── 02_Empirical_Analysis.ipynb
├── experiments/                    # Experiment configs and logs
└── results/                        # Generated plots and analysis
```

## Quick Start

```bash
git clone https://github.com/mzquadri/Neural-Network-Identifiability-Analysis.git
cd Neural-Network-Identifiability-Analysis

pip install -r requirements.txt

# Run identifiability checks on a sample network
python src/identifiability_checks.py --architecture 5 10 10 1 --activation tanh

# Detect isomorphisms between two networks
python src/network_isomorphisms.py --compare

# Analyze activation function impact
python src/activation_analysis.py
```

## Key Results

| Activation | Identifiable (up to symmetry) | Clone-Free | Self-Avoiding |
|-----------|-------------------------------|------------|---------------|
| Sigmoid   | Yes (up to permutation)       | Checkable  | Verifiable    |
| Tanh      | Yes (up to +/- and permutation)| Checkable  | Verifiable    |
| ReLU      | No (fails genericity)         | N/A        | N/A           |

## Theoretical References

- **Fefferman, C.** (1994). Reconstructing a Neural Net from its Output. *Revista Matematica Iberoamericana*.
- **Bona-Pellissier, Miche, Malgouyres** (2022). Parameter Identifiability of Neural Networks with ReLU, Tanh, and Sigmoid Activations.
- **Petzka, H., Trimmel, M.** (2020). On the Identifiability of Neural Networks.

## Technical Stack

- **Framework**: PyTorch
- **Visualization**: Matplotlib, NetworkX
- **Numerical**: NumPy, SciPy

## Author

**Mohd Zamin Quadri** - M.Sc. Mathematics in Science and Engineering, Technical University of Munich

Seminar supervised by Prof. Massimo Fornasier and Dr. Alessandro Scagliotti (TUM CIT)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohd--zamin-blue)](https://www.linkedin.com/in/mohd-zamin/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
