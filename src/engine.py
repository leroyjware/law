import numpy as np
import scipy.linalg as la
import networkx as nx

from .topologies import get_topology

# Cap exploding variance to avoid log-log fit dominated by numerical blow-up
M2_CAP = 1e6

# Convergence check: break early when variance plateaus (saves time, ensures steady-state)
CONVERGE_WINDOW = 500   # rolling window for variance std
CONVERGE_THRESH = 0.01  # std < 0.01 * mean → plateau
CONVERGE_STEPS = 200    # need this many consecutive plateau steps to break


def get_sigma_sweep(topology_name, n, gamma=1.0, c_coeff=0.5, n_points=14, chi_min=0.01, chi_max_frac=0.9, chi_critical=0.05, n_critical_min=4, seed=42, **topo_kwargs):
    """
    Return sigma values that span chi_w from chi_min to chi_max.
    Ensures at least n_critical_min points with chi_w < chi_critical (critical regime).
    """
    np.random.seed(seed)
    G = get_topology(topology_name, n, **topo_kwargs)
    L = nx.laplacian_matrix(G).astype(float).toarray()
    evals = np.sort(la.eigvalsh(L))
    lambda_2 = evals[1] if len(evals) > 1 else 0.0
    if lambda_2 < 1e-6:  # disconnected or nearly so
        return np.linspace(0.05, 0.35, n_points)
    chi_upper = gamma * lambda_2 - 1e-6
    chi_max = min(chi_max_frac * gamma * lambda_2, chi_upper)
    chi_max = max(chi_max, chi_min * 2)
    chi_min_actual = min(chi_min, chi_max * 0.5)
    chis = np.linspace(chi_min_actual, chi_max, n_points)
    n_critical = np.sum(chis < chi_critical)
    if n_critical < n_critical_min and chi_critical < chi_max:
        # Ensure enough points in critical regime: densify low-chi region
        chis_lo = np.linspace(chi_min_actual, min(chi_critical, chi_max), n_critical_min)
        chis_hi = chis[chis >= chi_critical]
        chis = np.unique(np.concatenate([chis_lo, chis_hi]))[:n_points + 2]
    sigmas = np.sqrt(np.maximum((gamma * lambda_2 - chis) / c_coeff, 1e-10))
    return sigmas


def run_ware_simulation(
    n=256,
    topology_name="grid_2d_medium",
    gamma=1.0,
    sigma=0.1,
    c_coeff=0.5,
    seed=42,
    topology_seed=None,          # if set, use for graph; else use seed (ensures same graph when averaging)
    steps=15000,
    burn_in_steps=14000,         # track variance only in last 1000
    slow_edge_decay=False,       # for "slow edge evolution" variant
    heterogeneous_gamma=False,   # for "heterogeneous gamma" variant
    **topo_kwargs
):
    graph_seed = topology_seed if topology_seed is not None else seed
    np.random.seed(graph_seed)
    G = get_topology(topology_name, n, **topo_kwargs)
    np.random.seed(seed)  # reseed for simulation noise
    L_sparse = nx.laplacian_matrix(G)
    L_dense = L_sparse.astype(float).toarray()  # for eigvalsh

    # Spectral gap (algebraic connectivity)
    evals = np.sort(la.eigvalsh(L_dense))
    lambda_2 = evals[1] if len(evals) > 1 else 0.0

    # Closure Index χ_w
    chi_w = (gamma * lambda_2) - (c_coeff * sigma**2)

    if chi_w <= 0:
        return chi_w, np.inf, sigma**2

    # Optional: heterogeneous per-agent gamma
    if heterogeneous_gamma:
        gamma_per_agent = gamma * (0.8 + 0.4 * np.random.rand(n))
    else:
        gamma_per_agent = np.full(n, gamma)

    # Optional: slow edge evolution (gradual Laplacian weakening)
    if slow_edge_decay:
        decay_rate = 0.00005  # small decay per step
        current_L = L_dense.copy()
    else:
        decay_rate = 0.0
        current_L = None  # use sparse L directly

    # dt=0.006: slight increase from 0.005 for speed; stability guard remains
    dt = 0.006
    x = np.random.normal(0, 0.5, n)
    variances = []
    var_buffer = []
    converge_count = 0

    for i in range(steps):
        if slow_edge_decay:
            current_L *= (1 - decay_rate)
            current_L = np.maximum(current_L, 0)
            Lx = current_L @ x
        else:
            Lx = L_sparse @ x

        drift = -gamma_per_agent * Lx
        noise = np.random.normal(0, sigma, n)
        x += drift * dt + noise * np.sqrt(dt)

        # Blow-up guard: abort if state explodes
        if np.max(np.abs(x)) > 1e8:
            return chi_w, M2_CAP, sigma**2

        if i >= burn_in_steps:
            mean_x = np.mean(x)
            v = min(np.var(x - mean_x), M2_CAP)
            variances.append(v)
            var_buffer.append(v)
            if len(var_buffer) > CONVERGE_WINDOW:
                var_buffer.pop(0)
            if len(var_buffer) == CONVERGE_WINDOW:
                buf_mean = np.mean(var_buffer)
                buf_std = np.std(var_buffer)
                if buf_std < CONVERGE_THRESH * max(buf_mean, 1e-12):
                    converge_count += 1
                    if converge_count >= CONVERGE_STEPS:
                        break
                else:
                    converge_count = 0

    if not variances:
        m2 = np.nan
    else:
        m2 = min(np.mean(variances), M2_CAP)

    return chi_w, m2, sigma**2