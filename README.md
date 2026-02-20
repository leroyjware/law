# Ware's Law Verification Suite

Empirical validation suite supporting the paper:

**"Ware's Law: Critical Scaling of Disagreement Variance and the Stability Redline for Distributed AI"**  
(submitted to IEEE Transactions on Artificial Intelligence)

This repository contains Python code to reproduce the numerical experiments in Section VIII (Numerical Validation) and Table I of the manuscript. It simulates stochastic linear consensus dynamics on diverse network topologies, sweeps noise intensity to approach the Ware Redline (χ_w → 0⁺), and fits the log-log scaling of disagreement variance m² vs. Closure Index χ_w.

The simulations confirm Ware's Law (m² ∼ σ² / χ_w) with a grand-average critical exponent of **≈ −1.0** across ten topology variants (small-world, grids of varying size, hierarchical modular, scale-free, GNN-style, random regular, Erdős–Rényi, slow edge evolution, heterogeneous γ). Run `--full` or `--paper-quick` to reproduce; committed results in `data/` are from `--paper-quick` (grand mean −1.002 ± 0.66, n=48 fits).

### Key Contribution
This code demonstrates the robustness of the inverse scaling law beyond the mean-field/complete-graph regime, supporting the paper's claim that the Closure Index χ_w = γ λ₂ − c σ² serves as a practical, topology-aware stability metric for distributed AI systems (federated learning, MARL, self-modifying swarms, etc.).

### Requirements
Python 3.10+  
Install dependencies:
```bash
pip install -r requirements.txt
```

### Paper-Ready Empirical Validation
Full suite (8 repeats × 7 seeds × 20 σ points per topology; ~10–25 min):
```bash
PYTHONPATH=. python src/analysis.py --full
```
Output: `data/ware_law_results.csv`, `data/WARE_LAW_REPORT.md`, per-topology log-log CSVs in `data/`, and log-log plots in `data/plots/`.

Fast mode (2 repeats × 3 seeds; ~2 min):
```bash
PYTHONPATH=. python src/analysis.py --fast
# or: FAST_MODE=1 PYTHONPATH=. python src/analysis.py
```

Paper-quick (5 repeats × 5 seeds × 16 points; ~10–15 min):
```bash
PYTHONPATH=. python src/analysis.py --paper-quick --no-plots
```

Skip plots: `--no-plots`  
Custom output dir: `-o my_output`

### Automated Tests
```bash
python run_tests.py
```
Tests 7 topologies: small_world, grid_2d_medium, scale_free, gnn_style, hierarchical_modular, random_regular, erdos_renyi. Use `FAST_MODE=false` for full-mode tests (~2 min per topology).

## Test Expectations
- Slope: roughly -1.0 (topology-dependent; wider tolerance in fast mode)
- R²: >0.50 for most topologies; >0.35 for modular/scale-free

## Key Advantages of the Closure Index (χ_w)

This verification suite demonstrates a simple, topology-aware stability metric derived from linear stochastic consensus dynamics, applicable to distributed AI systems.

- **Predictive redline threshold**: Unlike per-step gradient norm checks or post-hoc ensemble disagreement, χ_w combines coupling strength (via spectral gap λ₂) and noise intensity (σ²) into a single scalar with a theoretical critical point (χ_w → 0⁺) signaling potential divergence. This enables proactive intervention (e.g., reduce learning rate, increase synchronization) before loss spikes or collapse.
- **Topology robustness**: Validated across diverse graphs (small-world, grid, scale-free, modular, etc.), showing consistent inverse scaling (exponent ≈ −1.0) — relevant for real-world decentralized training where communication is sparse or heterogeneous.
- **Low overhead**: Proxies for λ₂ (mixing time, neighbor alignment) and σ² (gradient/message variance) are O(1) to O(n log n) per step — cheap enough for online monitoring in production.

These properties make χ_w complementary to existing tools (gradient clipping, Byzantine aggregation, epistemic uncertainty via ensembles) by providing an early, interpretable warning of systemic instability rather than just local anomalies.

## Immediate Safety Applications in Real Systems

Builders of distributed/federated LLM training or multi-agent systems can apply χ_w today:

- **Distributed pre-training / fine-tuning** — Monitor χ_w per global step (using all-reduce gossip proxies for λ₂); trigger gradient scaling or rollback if χ_w < threshold (e.g., 0.1–0.2).
- **Federated learning** — Clients report local noise proxies; aggregator computes χ_w to detect/reject drifting or poisoned participants.
- **Multi-agent LLM swarms** (e.g., AutoGen, CrewAI) — Treat agent outputs/hidden states as belief vectors; low χ_w flags incoherence → force consensus round or reduce temperature.
- **Inference-time uncertainty** — Multi-sample generations → compute effective χ_w across samples → flag low-confidence outputs for human review.

Code in `src/engine.py` and `analysis.py` is modular — adapt the variance computation and spectral proxy to your setup (e.g., gradient variance instead of state variance, mixing time instead of exact λ₂).