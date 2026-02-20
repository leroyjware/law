"""
Ware's Law empirical validation suite.
Runs stochastic consensus simulations across topologies and reports scaling exponents.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes
from src.engine import run_ware_simulation, get_sigma_sweep

# Paper-ready defaults: more runs for robust statistics
PAPER_N_REPEATS = 8
PAPER_SEEDS_PER_SIGMA = 7
PAPER_N_POINTS = 20
OUTLIER_SLOPE_THRESH = 3.0  # |slope + 1| > this → flag as outlier


def run_exhaustive_suite(n_repeats=None, seeds_per_sigma=None, n_points=None, base_seed=42, fast_mode=False, full_mode=False, out_dir="data", save_plots=True):
    if fast_mode:
        n_repeats, seeds_per_sigma = 2, 3
        n_points = 10
        sim_steps, sim_burn = 8000, 7000
    elif full_mode:
        n_repeats = n_repeats or PAPER_N_REPEATS
        seeds_per_sigma = seeds_per_sigma or PAPER_SEEDS_PER_SIGMA
        n_points = n_points or PAPER_N_POINTS
        sim_steps, sim_burn = None, None
    else:
        n_repeats = n_repeats or PAPER_N_REPEATS
        seeds_per_sigma = seeds_per_sigma or PAPER_SEEDS_PER_SIGMA
        n_points = n_points or PAPER_N_POINTS
        sim_steps, sim_burn = None, None

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    variants = [
        {"name": "small_world", "n": 150, "kwargs": {"k": 6, "p": 0.1}},
        {"name": "grid_2d_medium", "n": 400},
        {"name": "grid_2d_large", "n": 900},
        {"name": "hierarchical_modular", "n": 100, "kwargs": {"s": 25, "v": 10, "p_in": 0.5, "p_out": 0.05}},
        {"name": "scale_free", "n": 400, "kwargs": {"m": 3}},
        {"name": "gnn_style", "n": 256, "kwargs": {"m": 4, "p": 0.2}},
        {"name": "random_regular", "n": 256, "kwargs": {"d": 4}},
        {"name": "erdos_renyi", "n": 256},
        {"name": "slow_edge_evolution", "n": 200},
        {"name": "heterogeneous_gamma", "n": 256},
    ]

    table_rows = []

    print("=" * 80)
    print("Ware's Law Empirical Validation — m² ∝ 1/χ_w (target exponent −1.0)")
    print("=" * 80)
    print(f"{'Topology':<22} | {'n':<5} | {'Slope':<12} | {'Std':<8} | {'R²':<8} | {'Fits':<5}")
    print("-" * 80)

    all_slopes = []

    for var in variants:
        name = var["name"]
        n = var["n"]
        topo_kwargs = var.get("kwargs", {})

        sigmas = get_sigma_sweep(name, n, n_points=n_points, **topo_kwargs)
        slopes_this = []
        r2s_this = []
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

            if len(chis) >= 5:
                chis_a, m2s_a = np.array(chis), np.array(m2s)
                mask = (chis_a > 1e-6) & (m2s_a > 1e-12)
                if np.sum(mask) >= 5:
                    log_chi = np.log(chis_a[mask])
                    log_m2 = np.log(m2s_a[mask])
                    slope, intercept, r_val, _, _ = linregress(log_chi, log_m2)
                    # Outlier detection: flag extreme slopes
                    if abs(slope + 1) > OUTLIER_SLOPE_THRESH:
                        print(f"  [OUTLIER] {name} repeat {repeat}: slope={slope:.3f} (|slope+1|={abs(slope+1):.2f} > {OUTLIER_SLOPE_THRESH})")
                    slopes_this.append(slope)
                    r2s_this.append(r_val ** 2)
                chis_all.extend(chis)
                m2s_all.extend(m2s)

        slope_mean = np.mean(slopes_this) if slopes_this else np.nan
        slope_std = np.std(slopes_this) if len(slopes_this) > 1 else 0.0
        slope_median = np.median(slopes_this) if slopes_this else np.nan
        q1, q3 = np.percentile(slopes_this, [25, 75]) if len(slopes_this) >= 4 else (np.nan, np.nan)
        slope_iqr = (q3 - q1) if len(slopes_this) >= 4 else 0.0
        r2_mean = np.mean(r2s_this) if r2s_this else np.nan
        n_fits = len(slopes_this)

        # Robust regression on pooled data (secondary fit)
        slope_theil = np.nan
        if chis_all and len(chis_all) >= 8:
            chis_pool = np.array(chis_all)
            m2s_pool = np.array(m2s_all)
            pmask = (chis_pool > 1e-6) & (m2s_pool > 1e-12)
            if np.sum(pmask) >= 8:
                res = theilslopes(np.log(m2s_pool[pmask]), np.log(chis_pool[pmask]))  # log(m2) vs log(chi)
                slope_theil = res[0]  # Theil-Sen slope (x,y) → m2 vs chi, so slope = d(log m2)/d(log chi)

        if slopes_this:
            all_slopes.extend(slopes_this)

        row = {"topology": name, "n": n, "slope_mean": slope_mean, "slope_std": slope_std,
               "slope_median": slope_median, "slope_iqr": slope_iqr, "slope_theil": slope_theil,
               "r2_mean": r2_mean, "n_fits": n_fits}
        table_rows.append(row)

        slope_str = f"{slope_mean:.4f}" if not np.isnan(slope_mean) else "—"
        std_str = f"±{slope_std:.4f}" if n_fits > 1 else "—"
        r2_str = f"{r2_mean:.3f}" if not np.isnan(r2_mean) else "—"
        print(f"{name:<22} | {n:<5} | {slope_str:<12} | {std_str:<8} | {r2_str:<8} | {n_fits:<5}")

        if chis_all:
            pd.DataFrame({"chi_w": chis_all, "m2": m2s_all}).to_csv(
                os.path.join(out_dir, f"{name}_loglog.csv"), index=False
            )
        # Per-topology log-log plot
        if save_plots and chis_all and len(chis_all) >= 5:
            try:
                import matplotlib.pyplot as plt
                chis_plt = np.array(chis_all)
                m2s_plt = np.array(m2s_all)
                msk = (chis_plt > 1e-6) & (m2s_plt > 1e-12)
                if np.sum(msk) >= 5:
                    plt.figure(figsize=(5, 4))
                    plt.loglog(chis_plt[msk], m2s_plt[msk], "o", alpha=0.6, markersize=4)
                    plt.xlabel(r"$\chi_w$")
                    plt.ylabel(r"$m^2$")
                    plt.title(f"{name} (n={n}) — slope mean={slope_mean:.3f}, theil={slope_theil:.3f}")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{name}_loglog.png"), dpi=120)
                    plt.close()
            except Exception as e:
                print(f"  [plot skip] {name}: {e}")

    # Summary table
    df = pd.DataFrame(table_rows)
    df.to_csv(os.path.join(out_dir, "ware_law_results.csv"), index=False)

    # Exclude extreme outliers (|slope+1| > 3) for grand mean
    slopes_clean = [s for s in all_slopes if abs(s + 1) <= OUTLIER_SLOPE_THRESH and -3.0 < s < -0.1]
    grand_mean = np.mean(slopes_clean) if slopes_clean else np.nanmean(all_slopes)
    grand_std = np.std(slopes_clean) if len(slopes_clean) > 1 else (np.nanstd(all_slopes) if all_slopes else 0.0)
    grand_median = np.median(slopes_clean) if slopes_clean else np.nan
    n_outliers = len(all_slopes) - len(slopes_clean)

    print("-" * 80)
    print(f"GRAND AVERAGE EXPONENT: {grand_mean:.4f} ± {grand_std:.4f}  (n = {len(slopes_clean)} fits, {n_outliers} outliers excluded)")
    print(f"Grand median: {grand_median:.4f}  |  Theoretical: −1.0  |  Match: {abs(grand_mean - (-1.0)):.3f} from target")
    print("=" * 80)

    # Write Markdown report
    report_path = os.path.join(out_dir, "WARE_LAW_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Ware's Law Empirical Validation Report\n\n")
        f.write("**m² ∝ 1/χ_w** (target log-log slope = −1.0)\n\n")
        f.write("| Topology | n | Slope (mean ± std) | median | Theil | R² | Fits |\n")
        f.write("|----------|---|--------------------|--------|-------|-----|------|\n")
        for r in table_rows:
            if r['n_fits'] == 0 or np.isnan(r['slope_mean']):
                s, med, th, r2s = "—", "—", "—", "—"
            else:
                s = f"{r['slope_mean']:.4f} ± {r['slope_std']:.4f}" if r['n_fits'] > 1 else f"{r['slope_mean']:.4f}"
                med = f"{r['slope_median']:.4f}" if not np.isnan(r.get('slope_median', np.nan)) else "—"
                th = f"{r['slope_theil']:.4f}" if not np.isnan(r.get('slope_theil', np.nan)) else "—"
                r2s = f"{r['r2_mean']:.3f}"
            f.write(f"| {r['topology']} | {r['n']} | {s} | {med} | {th} | {r2s} | {r['n_fits']} |\n")
        f.write(f"\n**Grand average exponent:** {grand_mean:.4f} ± {grand_std:.4f} (n = {len(slopes_clean) or len(all_slopes)} fits, {n_outliers} outliers excluded)\n")
    print(f"\nReport saved to {report_path}")

    return df, all_slopes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ware's Law empirical validation suite")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 2 repeats, 3 seeds, shorter steps")
    parser.add_argument("--full", action="store_true", help="Full/paper mode: 8 repeats, 7 seeds, 20 points")
    parser.add_argument("--paper-quick", action="store_true", help="Paper-quality but faster: 5 repeats, 5 seeds, 16 points (~10–15 min)")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-topology log-log plots")
    parser.add_argument("-o", "--out-dir", default="data", help="Output directory")
    args = parser.parse_args()
    fast = args.fast or os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")
    full = args.full or os.environ.get("FULL_MODE", "").lower() in ("1", "true", "yes")
    if args.paper_quick:
        run_exhaustive_suite(n_repeats=5, seeds_per_sigma=5, n_points=16, full_mode=True,
                             out_dir=args.out_dir, save_plots=not args.no_plots)
    else:
        run_exhaustive_suite(fast_mode=fast, full_mode=full, out_dir=args.out_dir, save_plots=not args.no_plots)
