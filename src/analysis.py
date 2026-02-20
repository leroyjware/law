import numpy as np
import pandas as pd
from scipy.stats import linregress
from src.engine import run_ware_simulation, get_sigma_sweep
from src.topologies import get_topology  # needed for kwargs

def run_exhaustive_suite(n_repeats=5, seeds_per_sigma=5, base_seed=42, fast_mode=False):
    if fast_mode:
        n_repeats, seeds_per_sigma = 2, 3
        sim_steps, sim_burn = 8000, 7000
    else:
        sim_steps, sim_burn = None, None  # use engine defaults
    # Variants from Table I in the paper
    variants = [
        {"name": "small_world", "n": 150, "kwargs": {"k": 6, "p": 0.1}},
        {"name": "grid_2d_medium", "n": 400},
        {"name": "grid_2d_large", "n": 900},
        {"name": "hierarchical_modular", "n": 100, "kwargs": {"s": 25, "v": 10, "p_in": 0.5, "p_out": 0.01}},
        {"name": "scale_free", "n": 400, "kwargs": {"m": 3}},
        {"name": "gnn_style", "n": 256, "kwargs": {"m": 4, "p": 0.2}},
        {"name": "slow_edge_evolution", "n": 200},
        {"name": "heterogeneous_gamma", "n": 256},
    ]

    results = []
    print(f"{'Variant':<25} | {'n':<6} | {'Exponent':<10} | {'R²':<8} | {'Mean χ_w range'}")
    print("-" * 70)

    for var in variants:
        name = var["name"]
        n = var["n"]
        topo_kwargs = var.get("kwargs", {})

        sigmas = get_sigma_sweep(name, n, n_points=14, **topo_kwargs)
        chis_all, m2s_all = [], []
        for repeat in range(n_repeats):
            chis, m2s = [], []
            for s in sigmas:
                m2_repeats = []
                for k in range(seeds_per_sigma):
                    seed = base_seed + repeat * 100 + k * 7
                    sim_kw = dict(n=n, topology_name=name, gamma=1.0, sigma=s, c_coeff=0.5,
                        seed=seed, topology_seed=base_seed + repeat * 100,
                        slow_edge_decay=(name == "slow_edge_evolution"),
                        heterogeneous_gamma=(name == "heterogeneous_gamma"), **topo_kwargs)
                    if fast_mode:
                        sim_kw["steps"], sim_kw["burn_in_steps"] = sim_steps, sim_burn
                    chi_w, m2, _ = run_ware_simulation(**sim_kw)
                    if np.isfinite(m2) and chi_w > 1e-4:
                        m2_repeats.append(m2)
                if len(m2_repeats) >= seeds_per_sigma // 2:
                    chis.append(chi_w)
                    m2s.append(np.mean(m2_repeats))

            if len(chis) >= 8:  # need enough points for fit
                slope, intercept, r_val, _, _ = linregress(np.log(chis), np.log(m2s))
                results.append(slope)
                chis_all.extend(chis)
                m2s_all.extend(m2s)
            else:
                slope = np.nan
                r_val = np.nan

            print(f"{name:<25} | {n:<6} | {slope:>10.4f} | {r_val**2:>8.4f}")

        # Optional: save per-variant log-log data for figure
        if chis_all:
            pd.DataFrame({"chi_w": chis_all, "m2": m2s_all}).to_csv(f"data/{name}_loglog.csv", index=False)

    grand_mean = np.nanmean(results)
    grand_std = np.nanstd(results)
    print("-" * 70)
    print(f"GRAND AVERAGE EXPONENT: {grand_mean:.3f} ± {grand_std:.3f} (over {len(results)} fits)")

    return results

if __name__ == "__main__":
    import os
    fast = os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")
    run_exhaustive_suite(n_repeats=5, fast_mode=fast)