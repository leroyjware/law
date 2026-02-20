import networkx as nx
import numpy as np


def get_topology(name, n, **kwargs):
    """
    Generate various network topologies used in the paper.
    """
    if name == "small_world":
        # Watts-Strogatz (small-world)
        k = kwargs.get("k", 6)
        p = kwargs.get("p", 0.1)
        return nx.watts_strogatz_graph(n, k, p)

    elif name == "grid_2d_medium":
        side = int(np.sqrt(n))  # e.g. n≈400 → 20x20
        return nx.grid_2d_graph(side, side)

    elif name == "grid_2d_large":
        side = int(np.sqrt(n))  # e.g. n≈900 → 30x30
        return nx.grid_2d_graph(side, side)

    elif name == "hierarchical_modular":
        # 4 communities with sparse inter-links; p_out=0.05 improves connectivity
        return nx.gaussian_random_partition_graph(
            n, s=kwargs.get("s", 25), v=kwargs.get("v", 10),
            p_in=kwargs.get("p_in", 0.5), p_out=kwargs.get("p_out", 0.05)
        )

    elif name == "scale_free":
        # Barabási–Albert (scale-free)
        m = kwargs.get("m", 3)
        return nx.barabasi_albert_graph(n, m)

    elif name == "gnn_style":
        # Power-law cluster graph (mimics message-passing / GNN layers)
        m = kwargs.get("m", 4)
        p = kwargs.get("p", 0.2)
        return nx.powerlaw_cluster_graph(n, m, p)

    elif name == "slow_edge_evolution":
        # Small-world with decaying edges over time (placeholder; dynamics added in engine)
        G = nx.watts_strogatz_graph(n, k=6, p=0.1)
        return G  # actual decay handled in simulation loop if needed

    elif name == "heterogeneous_gamma":
        # Grid with per-node gamma variation (handled in engine)
        side = int(np.sqrt(n))
        return nx.grid_2d_graph(side, side)

    elif name == "random_regular":
        # d-regular random graph (same degree, good connectivity)
        d = kwargs.get("d", 4)
        d = min(d, n - 1)
        return nx.random_regular_graph(d, n)

    elif name == "erdos_renyi":
        # Erdős–Rényi G(n,p); p tuned for connectivity
        p = kwargs.get("p", 2.5 * np.log(n) / n)  # above connectivity threshold
        return nx.erdos_renyi_graph(n, min(p, 0.99))

    else:
        # Default: complete graph (for baseline comparison)
        return nx.complete_graph(n)