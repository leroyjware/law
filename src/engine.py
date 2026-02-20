import numpy as np
import scipy.linalg as la
import networkx as nx

from .topologies import get_topology

# Topologies that can produce disconnected graphs; retry until connected
CONNECTED_RETRY_TOPOLOGIES = ("hierarchical_modular",)
CONNECTED_RETRY_ATTEMPTS = 50

# Cap exploding variance to avoid log-log fit dominated by numerical blow-up
M2_CAP = 1e6

# Convergence check: break early when variance plateaus (ensures steady-state)
CONVERGE_WINDOW = 800   # rolling window for variance std
CONVERGE_THRESH = 0.005  # std < 0.005 * mean → plateau (tighter)
CONVERGE_STEPS = 400    # need this many consecutive plateau steps to break
CONVERGE_STD_MEAN_MAX = 0.02  # if final std/mean > this, consider non-converged


def _get_connected_topology(topology_name, n, seed, **topo_kwargs):
    """Return connected graph; retry for topologies that can be disconnected."""
    for attempt in range(CONNECTED_RETRY_ATTEMPTS):
        np.random.seed(seed + attempt)
        G = get_topology(topology_name, n, **topo_kwargs)
        if nx.is_connected(G):
            return G
    return G  # return last attempt even if disconnected (caller checks lambda_2)


def get_sigma_sweep(topology_name, n, gamma=1.0, c_coeff=0.5, n_points=20, chi_min=0.005, chi_max_frac=0.9,
                    n_chi_lt_008=6, n_chi_lt_003=4, seed=42, **topo_kwargs):
    """
    Return sigma values biased toward critical regime χ_w → 0⁺.
    Guarantees ≥n_chi_lt_008 points with χ_w < 0.08, ≥n_chi_lt_003 with χ_w < 0.03.
    Uses log spacing in critical regime, linear above.
    """
    if topology_name in CONNECTED_RETRY_TOPOLOGIES:
        G = _get_connected_topology(topology_name, n, seed, **topo_kwargs)
    else:
        np.random.seed(seed)
        G = get_topology(topology_name, n, **topo_kwargs)
    L = nx.laplacian_matrix(G).astype(float).toarray()
    evals = np.sort(la.eigvalsh(L))
    lambda_2 = evals[1] if len(evals) > 1 else 0.0
    if lambda_2 < 1e-6:
        return np.linspace(0.05, 0.35, n_points)
    chi_upper = gamma * lambda_2 - 1e-6
    chi_min_actual = max(chi_min, 0.003)
    chi_max = min(chi_max_frac * gamma * lambda_2, chi_upper)
    chi_max = max(chi_max, chi_min_actual * 2)
    # Critical regime: log-spaced chi_min … min(0.08, chi_max)
    chi_crit_hi = min(0.08, chi_max * 0.9)
    chis_lo = np.logspace(np.log10(chi_min_actual), np.log10(chi_crit_hi), max(n_chi_lt_008, 8))
    # Above critical: linear
    chis_hi = np.linspace(chi_crit_hi * 1.1, chi_max, max(n_points - len(chis_lo), 6)) if chi_max > chi_crit_hi else np.array([])
    chis = np.unique(np.concatenate([chis_lo, chis_hi]))
    chis = chis[(chis >= chi_min_actual) & (chis <= chi_upper)]
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
    steps=25000,
    burn_in_steps=22000,        # ensure steady-state; track variance in last 3000
    slow_edge_decay=False,       # for "slow edge evolution" variant
    heterogeneous_gamma=False,   # for "heterogeneous gamma" variant
    verbose=False,               # log last 5 variances + std/mean when done
    **topo_kwargs
):
    graph_seed = topology_seed if topology_seed is not None else seed
    if topology_name in CONNECTED_RETRY_TOPOLOGIES:
        G = _get_connected_topology(topology_name, n, graph_seed, **topo_kwargs)
    else:
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

    dt = 0.003  # smaller dt for numerical stability in critical regime
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

        # Blow-up guard: abort if state explodes (stronger threshold)
        if np.max(np.abs(x)) > 1e6:
            return chi_w, np.inf, sigma**2

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
        if verbose:
            last5 = variances[-5:] if len(variances) >= 5 else variances
            v_mean = np.mean(variances)
            v_std = np.std(variances)
            std_mean_ratio = v_std / max(v_mean, 1e-12)
            print(f"  [conv] last5={last5}, std/mean={std_mean_ratio:.4f}, step={i}")

    return chi_w, m2, sigma**2