import pytest
import numpy as np
from scipy.stats import linregress
from src.engine import run_ware_simulation, get_sigma_sweep

import os
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"
if FAST_MODE:
    TEST_STEPS = 8000
    TEST_BURN_IN = 7000
    TEST_SEEDS = [42, 43, 44]  # 3 seeds for speed
else:
    TEST_STEPS = 25000
    TEST_BURN_IN = 22000
    TEST_SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for robust averaging
TEST_GAMMA = 1.0
TEST_C_COEFF = 0.5

# Per-topology slope ranges. Goal: (-1.5,-0.5) for clean; (-2.5,-0.3) for modular.
# Fast mode: relaxed for quick smoke tests. Full mode: tighter.
SLOPE_RANGES_FULL = {
    "grid_2d_medium": (-1.8, -0.4),
    "small_world": (-1.8, -0.2),
    "scale_free": (-2.5, -0.15),
    "gnn_style": (-1.8, -0.2),
    "hierarchical_modular": (-2.5, -0.15),
    "random_regular": (-1.8, -0.4),
    "erdos_renyi": (-1.8, -0.2),
}
SLOPE_RANGES_FAST = {k: (-2.5, -0.15) for k in SLOPE_RANGES_FULL}  # wider for fast smoke tests

# R² thresholds. Fast mode: relaxed. Full mode: goal >0.70 clean, >0.40 modular.
R2_THRESHOLDS_FULL = {
    "hierarchical_modular": 0.35,
    "scale_free": 0.35,
    "erdos_renyi": 0.35,
    "small_world": 0.40,
}
R2_THRESHOLDS_FAST = {
    "hierarchical_modular": 0.25,
    "scale_free": 0.25,
    "erdos_renyi": 0.15,
    "small_world": 0.35,
    "random_regular": 0.40,
    "gnn_style": 0.40,
    "grid_2d_medium": 0.70,
}
DEFAULT_R2_FULL = 0.50
DEFAULT_R2_FAST = 0.15

TOPOLOGY_CONFIG = [
    ("small_world", 150, {"k": 6, "p": 0.1}),
    ("grid_2d_medium", 400, {}),
    ("scale_free", 400, {"m": 3}),
    ("gnn_style", 256, {"m": 4, "p": 0.2}),
    ("hierarchical_modular", 100, {"s": 25, "v": 10, "p_in": 0.5, "p_out": 0.05}),
    ("random_regular", 256, {"d": 4}),
    ("erdos_renyi", 256, {}),
]

@pytest.mark.parametrize("topology_name,n,topo_kwargs", TOPOLOGY_CONFIG)
def test_scaling_law(topology_name, n, topo_kwargs):
    chis = []
    m2s = []

    sigmas = get_sigma_sweep(topology_name, n, n_points=10, **topo_kwargs)

    for sigma in sigmas:
        m2_repeats = []
        for seed in TEST_SEEDS:
            chi_w, m2, _ = run_ware_simulation(
                n=n,
                topology_name=topology_name,
                gamma=TEST_GAMMA,
                sigma=sigma,
                c_coeff=TEST_C_COEFF,
                seed=seed,
                topology_seed=42,  # same graph for all seeds → consistent chi_w
                steps=TEST_STEPS,
                burn_in_steps=TEST_BURN_IN,
                **topo_kwargs
            )
            if chi_w > 1e-4 and np.isfinite(m2):
                m2_repeats.append(m2)
        if len(m2_repeats) >= len(TEST_SEEDS) // 2 and chi_w > 1e-4:
            chis.append(chi_w)
            m2s.append(np.mean(m2_repeats))

    min_points = 4 if topology_name == "hierarchical_modular" else 5
    assert len(chis) >= min_points, f"Too few valid points for {topology_name} ({len(chis)})"

    chis, m2s = np.array(chis), np.array(m2s)
    mask = chis > 1e-6
    chis, m2s = chis[mask], m2s[mask]  # drop any non-positive
    assert len(chis) >= min_points, f"Too few valid points after filtering for {topology_name}"

    log_chi = np.log(chis)
    log_m2 = np.log(m2s)
    slope, intercept, r_value, _, _ = linregress(log_chi, log_m2)

    slope_ranges = SLOPE_RANGES_FAST if FAST_MODE else SLOPE_RANGES_FULL
    r2_thresholds = R2_THRESHOLDS_FAST if FAST_MODE else R2_THRESHOLDS_FULL
    slope_lo, slope_hi = slope_ranges.get(topology_name, (-1.8, -0.2))
    r2 = r_value**2
    r2_thresh = r2_thresholds.get(topology_name, DEFAULT_R2_FAST if FAST_MODE else DEFAULT_R2_FULL)

    print(f"{topology_name}: slope = {slope:.4f}, R² = {r2:.4f}, points = {len(chis)}")

    # Ware's Law: m² ∝ 1/χ_w → slope ≈ -1; per-topology ranges
    assert slope_lo < slope < slope_hi, \
        f"Exponent {slope:.4f} outside [{slope_lo}, {slope_hi}] for {topology_name}"
    assert r2 > r2_thresh, \
        f"R² {r2:.4f} too low (< {r2_thresh}) for {topology_name}"