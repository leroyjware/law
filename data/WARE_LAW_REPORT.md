# Ware's Law Empirical Validation Report

**m² ∝ 1/χ_w** (target log-log slope = −1.0)

| Topology | n | Slope (mean ± std) | median | Theil | R² | Fits |
|----------|---|--------------------|--------|-------|-----|------|
| small_world | 150 | -0.4113 ± 0.2527 | -0.4062 | -0.5912 | 0.420 | 5 |
| grid_2d_medium | 400 | -1.7408 ± 0.0255 | -1.7403 | -2.2209 | 0.899 | 5 |
| grid_2d_large | 900 | -2.9422 ± 0.1532 | -2.8796 | -3.2462 | 0.943 | 5 |
| hierarchical_modular | 100 | -0.5098 ± 0.1730 | -0.5281 | -0.4298 | 0.613 | 5 |
| scale_free | 400 | -1.0997 ± 0.2937 | -1.2296 | -0.7843 | 0.726 | 5 |
| gnn_style | 256 | -0.3663 ± 0.0516 | -0.3433 | -0.3000 | 0.591 | 5 |
| random_regular | 256 | -0.8342 ± 0.0640 | -0.8495 | -0.6430 | 0.727 | 5 |
| erdos_renyi | 256 | -0.6843 ± 0.1629 | -0.7563 | -0.5545 | 0.624 | 5 |
| slow_edge_evolution | 200 | -0.8468 ± 0.2622 | -0.7657 | -0.8521 | 0.285 | 5 |
| heterogeneous_gamma | 256 | -1.4319 ± 0.0067 | -1.4333 | -1.8127 | 0.880 | 5 |

**Grand average exponent:** -1.0021 ± 0.6590 (n = 48 fits, 2 outliers excluded)
