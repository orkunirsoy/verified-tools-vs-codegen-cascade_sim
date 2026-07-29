"""Grid data loading: IEEE/MATPOWER test cases -> NetworkX graphs.

This module is the single place where "what is the graph, exactly?" is decided.
Both agent arms and the oracle load topology through here, so that any difference
in their answers comes from the agent, never from a different view of the data.

Modeling decisions (deliberate, and reported in the writeup):

1. Undirected. We study topological cascades (Motter-Lai), not directed power flow.
2. Simple graph, not a multigraph. Parallel branches between the same bus pair
   collapse to one edge. On case118 this takes 186 branches -> 179 edges. A
   multigraph would change betweenness, so this choice is load-bearing and is
   asserted in the graph metadata below.
3. Lines and transformers are both edges. A transformer is a real electrical
   connection; dropping it would disconnect voltage levels that are physically joined.
4. Out-of-service elements are excluded. The bundled cases have none (verified),
   but a real case would, and silently including them would be wrong.
5. Bus IDs are pandapower's integer indices, kept as-is so answers can be traced
   back to the source case.

Everything here is deterministic: same case name -> same graph, every run.
"""

from __future__ import annotations

import warnings
from typing import Callable

import networkx as nx

# pandapower emits deprecation chatter on import; it is noise for our purposes.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandapower.networks as pn

#: Test cases we support. case14 is the development case (small enough to check
#: answers by hand); case118 is the main evaluation case (big enough for a
#: non-trivial cascade, small enough that a betweenness sweep stays fast).
CASES: dict[str, Callable] = {
    "case14": pn.case14,
    "case30": pn.case30,
    "case57": pn.case57,
    "case118": pn.case118,
}

DEV_CASE = "case14"
MAIN_CASE = "case118"


def load_grid(case_name: str = MAIN_CASE) -> nx.Graph:
    """Load an IEEE test case as an undirected simple NetworkX graph.

    Args:
        case_name: one of CASES, e.g. "case118".

    Returns:
        A connected undirected graph. Nodes are integer bus IDs. Edges carry
        `element` ("line" or "trafo") and `n_parallel` (how many branches
        collapsed into this edge). Graph-level metadata records the case name and
        the branch/edge counts, so a run log can prove which topology it used.

    Raises:
        KeyError: unknown case name.
        ValueError: the loaded topology is disconnected, which would silently
            invalidate every downstream metric.
    """
    if case_name not in CASES:
        raise KeyError(f"unknown case {case_name!r}; available: {sorted(CASES)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = CASES[case_name]()

    G = nx.Graph()
    G.add_nodes_from(int(b) for b in net.bus.index[net.bus.in_service])

    n_branches = 0
    for frame, u_col, v_col, kind in (
        (net.line, "from_bus", "to_bus", "line"),
        (net.trafo, "hv_bus", "lv_bus", "trafo"),
    ):
        for _, row in frame[frame.in_service].iterrows():
            u, v = int(row[u_col]), int(row[v_col])
            if u == v:
                continue  # self-loop: electrically meaningless here, and it would
                          # distort degree without adding connectivity
            n_branches += 1
            if G.has_edge(u, v):
                # Parallel branch: collapse (decision 2 above), but keep the count
                # so the collapse is visible rather than silent.
                G.edges[u, v]["n_parallel"] += 1
            else:
                G.add_edge(u, v, element=kind, n_parallel=1)

    G.graph.update(
        case=case_name,
        n_branches=n_branches,
        n_edges=G.number_of_edges(),
        n_collapsed=n_branches - G.number_of_edges(),
    )

    if not nx.is_connected(G):
        raise ValueError(
            f"{case_name} loaded disconnected "
            f"({nx.number_connected_components(G)} components); every metric "
            f"downstream assumes a connected base network"
        )
    return G


def grid_summary(G: nx.Graph) -> str:
    """One-line human-readable description of a loaded grid, for logs and traces."""
    return (
        f"{G.graph['case']}: {G.number_of_nodes()} buses, "
        f"{G.number_of_edges()} edges ({G.graph['n_branches']} branches, "
        f"{G.graph['n_collapsed']} parallel collapsed), "
        f"avg degree {2 * G.number_of_edges() / G.number_of_nodes():.2f}"
    )
