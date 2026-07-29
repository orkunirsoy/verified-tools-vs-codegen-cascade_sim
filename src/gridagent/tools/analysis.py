"""Arm A's verified analysis functions.

INDEPENDENT FROM THE ORACLE, ON PURPOSE
---------------------------------------
`gridagent.oracle.reference` is the answer key; this module is what Arm A executes.
They are written separately and take different code paths -- the oracle implements
Brandes from scratch, this calls NetworkX -- so when they agree that is evidence, not
a tautology. `scripts/verify_tools.py` asserts they agree on every gold question.

WHAT "VERIFIED" BUYS ARM A
--------------------------
This is the AAG position made concrete: the cascade loop here is written once, tested,
and reused. Arm A cannot get the redistribution logic wrong, because it never writes
it. Arm A can still fail by choosing the wrong tool, passing the wrong arguments, or
misreporting what a tool returned -- and those failures are the interesting ones.

KB SCOPE IS A DESIGN DECISION WITH CONSEQUENCES
-----------------------------------------------
These functions cover the analyses a competent engineer would pre-write for cascade
work, including topology reinforcement (`added_line`), because "should we build this
line" is a standard question in the domain. A narrower KB would have handed the
comparison to Arm B by construction; a KB with one function per graded question would
have handed it to Arm A. The line drawn here is "primitives that compose", so multi-step
questions still require the model to plan.
"""

from __future__ import annotations

import networkx as nx

DEFAULT_ALPHA = 0.2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _loads(G: nx.Graph) -> dict[int, float]:
    """Load = unnormalized betweenness. NetworkX handles disconnected graphs by
    counting only within-component pairs, which is what the cascade needs."""
    return nx.betweenness_centrality(G, normalized=False)


def _prepare(G: nx.Graph, added_line: tuple[int, int] | None) -> nx.Graph:
    """The INITIAL network for this scenario. If a line is added, the upgraded grid
    is the initial network and capacities are baselined on it."""
    if added_line is None:
        return G
    # Say what is wrong with the argument rather than letting it fall through to an
    # unpack error. With validation off there is no repair loop, so this message is the
    # only signal the log carries about why the step died.
    if not (isinstance(added_line, (list, tuple)) and len(added_line) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in added_line)):
        raise ValueError(
            f"added_line must be a pair of bus ids like [31, 81], got {added_line!r}")
    u, v = added_line
    if G.has_edge(u, v):
        raise ValueError(f"line {u}-{v} already exists")
    H = G.copy()
    H.add_edge(u, v)
    return H


def _components(G: nx.Graph) -> list[int]:
    return sorted((len(c) for c in nx.connected_components(G)), reverse=True)


# --------------------------------------------------------------------------
# the KB functions
# --------------------------------------------------------------------------


def grid_info(G: nx.Graph) -> dict:
    """Size and shape of the grid."""
    deg = dict(G.degree())
    return {
        "n_buses": G.number_of_nodes(),
        "n_lines": G.number_of_edges(),
        "mean_degree": round(2 * G.number_of_edges() / G.number_of_nodes(), 3),
        "is_connected": nx.is_connected(G),
    }


def degree_ranking(G: nx.Graph, k: int = 5) -> dict:
    """Buses ranked by number of directly connected lines, highest first."""
    deg = sorted(G.degree(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "ranking": [{"bus": b, "degree": d} for b, d in deg[:k]],
        # Flat list of ids so a later step can reference it directly. Without this,
        # a multi-bus trigger cannot be built from a ranking at all.
        "top_buses": [b for b, _ in deg[:k]],
        "top_bus": deg[0][0],
        "n_tied_at_top": sum(1 for _, d in deg if d == deg[0][1]),
    }


def load_ranking(G: nx.Graph, k: int = 5) -> dict:
    """Buses ranked by LOAD (betweenness), highest first."""
    lo = sorted(_loads(G).items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "ranking": [{"bus": b, "load": round(v, 2)} for b, v in lo[:k]],
        # Flat list of ids so a later step can reference it directly, e.g. to trigger a
        # cascade on the top 3 buses. The dict-of-dicts `ranking` cannot be passed as a
        # trigger_buses argument.
        "top_buses": [b for b, _ in lo[:k]],
        "busiest_bus": lo[0][0],
        "next_load_outside_top_k": round(lo[k][1], 2) if len(lo) > k else None,
    }


def simulate_cascade(
    G: nx.Graph,
    trigger_buses: list[int] | None = None,
    trigger_line: tuple[int, int] | None = None,
    alpha: float = DEFAULT_ALPHA,
    added_line: tuple[int, int] | None = None,
) -> dict:
    """Run one Motter-Lai overload cascade and report the outcome.

    Exactly one trigger must be given. `added_line` upgrades the grid FIRST, and the
    upgraded grid becomes the initial network for load and capacity purposes.

    A bus has FAILED if it is not in the largest connected component at the end, which
    covers both overloaded buses and buses cut off without overloading.
    """
    if (trigger_buses is None) == (trigger_line is None):
        raise ValueError("give exactly one of trigger_buses or trigger_line")

    G0 = _prepare(G, added_line)
    n0 = G0.number_of_nodes()
    capacity = {b: (1.0 + alpha) * load for b, load in _loads(G0).items()}

    H = G0.copy()
    if trigger_buses is not None:
        missing = [b for b in trigger_buses if b not in H]
        if missing:
            raise KeyError(f"trigger buses not in grid: {missing}")
        H.remove_nodes_from(trigger_buses)
        trigger_desc = {"buses": sorted(trigger_buses)}
    else:
        u, v = trigger_line
        if not H.has_edge(u, v):
            raise KeyError(f"no such line: {u}-{v}")
        H.remove_edge(u, v)
        trigger_desc = {"line": [u, v]}

    overloaded: list[int] = []
    rounds = 0
    while H.number_of_nodes():
        rounds += 1
        current = _loads(H)
        blown = [b for b, load in current.items() if load > capacity[b]]
        if not blown:
            break
        H.remove_nodes_from(blown)
        overloaded.extend(blown)

    comps = _components(H)
    lcc = comps[0] if comps else 0
    n_trigger = len(trigger_buses) if trigger_buses is not None else 0
    return {
        "trigger": trigger_desc,
        "alpha": alpha,
        "added_line": list(added_line) if added_line else None,
        "rounds": rounds,
        "lcc_size": lcc,
        "lcc_fraction": lcc / n0,
        "n_islands": len(comps),
        "n_failed_total": n0 - lcc,
        "n_failed_excluding_trigger": n0 - lcc - n_trigger,
        "n_overloaded": len(overloaded),
        "overloaded_buses": sorted(overloaded),
    }


def list_buses(G: nx.Graph) -> dict:
    """Every bus id in the grid. Use this to know what a sweep would range over."""
    return {"buses": sorted(G), "n_buses": G.number_of_nodes()}


def list_lines(G: nx.Graph) -> dict:
    """Every line in the grid as [busA, busB] pairs."""
    lines = [list(e) for e in sorted(tuple(sorted(x)) for x in G.edges())]
    return {"lines": lines, "n_lines": len(lines)}


# --------------------------------------------------------------------------
# generic data operations
#
# These know nothing about grids or cascades. They are the numpy/pandas-level
# operations AAG's own prototype used alongside graph algorithms ("NumPy-based
# aggregation for transaction summarization"). Keeping selection generic means the
# knowledge base contains no function shaped like an answer to a graded question:
# deciding WHICH column to optimize and in WHICH direction stays a planning decision.
# --------------------------------------------------------------------------


def column(records: list[dict], field: str) -> dict:
    """Pull one field out of a list of records, in order. Like `df[field]`."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    missing = [i for i, r in enumerate(records) if not isinstance(r, dict) or field not in r]
    if missing:
        have = sorted(records[0]) if isinstance(records[0], dict) else type(records[0]).__name__
        raise KeyError(f"field {field!r} missing from {len(missing)} records; available: {have}")
    return {"field": field, "values": [r[field] for r in records], "n": len(records)}


def argmin(values: list) -> dict:
    """Index of the smallest value. Like `np.argmin`. Ties are reported, not hidden."""
    return _arg_extreme(values, "min")


def argmax(values: list) -> dict:
    """Index of the largest value. Like `np.argmax`. Ties are reported, not hidden."""
    return _arg_extreme(values, "max")


def _arg_extreme(values: list, mode: str) -> dict:
    if not isinstance(values, list) or not values:
        raise ValueError("values must be a non-empty list")
    best = (min if mode == "min" else max)(values)
    idxs = [i for i, v in enumerate(values) if v == best]
    return {
        "index": idxs[0],
        "value": best,
        "n_tied": len(idxs),
        "tied_indices": idxs if len(idxs) > 1 else [],
    }


def at(records: list, index: int) -> dict:
    """The element at a position. Like indexing a list or `df.loc[i]`."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    if not isinstance(index, int) or not (0 <= index < len(records)):
        raise IndexError(f"index {index!r} out of range for {len(records)} records")
    return {"index": index, "record": records[index]}
