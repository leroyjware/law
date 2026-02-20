# Ware's Law Verification Suite

Empirical validation suite supporting the paper:

**"Ware's Law: Critical Scaling of Disagreement Variance and the Stability Redline for Distributed AI"**  
(submitted to IEEE Transactions on Artificial Intelligence)

This repository contains Python code to reproduce the numerical experiments in Section VIII (Numerical Validation) and Table I of the manuscript. It simulates stochastic linear consensus dynamics on diverse network topologies, sweeps noise intensity to approach the Ware Redline (χ_w → 0⁺), and fits the log-log scaling of disagreement variance m² vs. Closure Index χ_w.

The simulations confirm Ware's Law (m² ∼ σ² / χ_w) with a grand-average critical exponent of **−0.988 ± 0.044** across eight topology variants (small-world, grids of varying size, hierarchical modular, scale-free, GNN-style, slow edge evolution, heterogeneous γ), closely matching the theoretical prediction of −1.0.

### Key Contribution
This code demonstrates the robustness of the inverse scaling law beyond the mean-field/complete-graph regime, supporting the paper's claim that the Closure Index χ_w = γ λ₂ − c σ² serves as a practical, topology-aware stability metric for distributed AI systems (federated learning, MARL, self-modifying swarms, etc.).

### Requirements
Python 3.10+  
Install dependencies:
```bash
pip install -r requirements.txt
```

### Automated Tests
```bash
pytest tests/ -v
```

Or from any directory:
```bash
python run_tests.py
```

### Google Colab
```python
# Clone, install, run (one cell)
!git clone https://github.com/leroyjware/law.git
%cd law
!pip install -q -r requirements.txt
!python run_tests.py
```