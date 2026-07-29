"""The correctness oracle: independent reference implementations.

WHY THIS MODULE EXISTS SEPARATELY FROM `gridagent.tools`
--------------------------------------------------------
Arm A (the verified-tool agent) executes hand-written analysis functions, so
whenever its plan is right, its numbers are right *by construction*. Grading Arm A
against Arm A's own tools would therefore measure tool selection, not correctness,
and putting that number in the same column as Arm B's correctness would be
circular. So the oracle is written here, separately, and both arms are graded
against it.

HOW INDEPENDENT IS "INDEPENDENT"?
---------------------------------
Honestly: partially, and the writeup says so rather than overclaiming.

  * Betweenness centrality is the load variable in Motter-Lai, so it is the single
    most load-bearing quantity in the study. Here it is a from-scratch Brandes
    implementation (`_brandes_betweenness`), NOT a NetworkX call. It is
    cross-checked against `nx.betweenness_centrality` to 1e-9 by
    `scripts/verify_oracle.py`. Two implementations agreeing is real evidence;
    one implementation agreeing with itself is not.
  * The cascade loop is independently written from the Arm A tool version.
  * Bookkeeping metrics (connected components, articulation points, diameter) call
    NetworkX in both places. Reimplementing Tarjan to grade a question like "how
    many articulation points are there" would buy accuracy we can already get by
    checking case14 by hand, and case14 is small enough to check by hand.

Every function is pure and deterministic: same graph in, same numbers out, so gold
answers can be frozen and re-derived.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict

import networkx as nx

# --------------------------------------------------------------------------
# Betweenness: independent Brandes implementation
# --------------------------------------------------------------------------


def _brandes_betweenness(G: nx.Graph) -> dict[int, float]:
    """Unnormalized betweenness centrality via Brandes' algorithm.

    Written from the algorithm rather than called from NetworkX, so that the
    oracle's load variable is not the same code path the tools use. Handles
    disconnected graphs: unreachable nodes are simply never visited, which is the
    behavior the cascade needs once the network starts to fragment.

    Unnormalized (raw pair counts) and halved for undirected graphs, matching the
    Motter-Lai convention where load is the number of shortest paths through a node.
    """
    betweenness = dict.fromkeys(G, 0.0)

    for s in G:
        # --- single-source shortest paths (BFS), counting path multiplicity ---
        stack: list[int] = []
        preds: dict[int, list[int]] = {w: [] for w in G}
        sigma = dict.fromkeys(G, 0.0)
        sigma[s] = 1.0
        dist: dict[int, int] = {s: 0}
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            dv = dist[v]
            for w in G[v]:
                if w not in dist:
                    dist[w] = dv + 1
                    queue.append(w)
                if dist[w] == dv + 1:  # w is one hop further: v is a predecessor
                    sigma[w] += sigma[v]
                    preds[w].append(v)

        # --- accumulate dependencies back-to-front ---
        delta = dict.fromkeys(G, 0.0)
        while stack:
            w = stack.pop()
            coeff = (1.0 + delta[w]) / sigma[w]
            for v in preds[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w]

    # Undirected: every pair (s,t) is traversed from both ends.
    return {v: b / 2.0 for v, b in betweenness.items()}


def betweenness(G: nx.Graph) -> dict[int, float]:
    """Public alias for the oracle's betweenness."""
    return _brandes_betweenness(G)


def top_k_betweenness(G: nx.Graph, k: int = 5) -> list[tuple[int, float]]:
    """The k buses carrying the most shortest paths, highest first.

    Ties are broken by bus ID so the ordering is deterministic. Where a tie
    straddles the k-th position the gold answer records it (see build_gold), because
    grading a ranked list against an arbitrary tiebreak would punish a correct agent.
    """
    b = _brandes_betweenness(G)
    return sorted(b.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# --------------------------------------------------------------------------
# Structure and connectivity
# --------------------------------------------------------------------------


def degree_stats(G: nx.Graph) -> dict:
    """Degree distribution summary, plus the single highest-degree bus."""
    deg = dict(G.degree())
    hi = max(sorted(deg), key=lambda n: deg[n])
    return {
        "max_degree": deg[hi],
        "max_degree_bus": hi,
        "mean_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "n_buses": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }


def articulation_points(G: nx.Graph) -> list[int]:
    """Buses whose removal disconnects the network: topological single points of failure."""
    return sorted(nx.articulation_points(G))


def bridges(G: nx.Graph) -> list[tuple[int, int]]:
    """Edges whose removal disconnects the network: critical branches."""
    return sorted(tuple(sorted(e)) for e in nx.bridges(G))


def distance_stats(G: nx.Graph) -> dict:
    """Diameter and mean shortest-path length. Assumes a connected graph."""
    return {
        "diameter": nx.diameter(G),
        "avg_shortest_path": nx.average_shortest_path_length(G),
    }


def islanding(G: nx.Graph, removed: list[int]) -> dict:
    """What the network looks like after buses are removed outright.

    This is the static "does it split" question, with no load redistribution: the
    contrast case for the cascade questions below.
    """
    H = G.copy()
    H.remove_nodes_from(removed)
    comps = sorted((sorted(c) for c in nx.connected_components(H)), key=len, reverse=True)
    return {
        "n_islands": len(comps),
        "island_sizes": [len(c) for c in comps],
        "lcc_size": len(comps[0]) if comps else 0,
        "lcc_fraction": (len(comps[0]) / G.number_of_nodes()) if comps else 0.0,
        "splits": len(comps) > 1,
    }


def worst_single_line_removal(G: nx.Graph) -> dict:
    """Sweep every line; find the one whose loss strands the most buses.

    Deliberately sweeps ALL edges rather than only the bridges. A non-bridge leaves
    the network connected and strands nobody, so the answer is identical either way,
    but not routing through `bridges()` keeps this derivation independent of the
    concept the question is testing.

    "Cut off" means "not in the largest surviving component", so a line whose loss
    splits the grid 116/2 strands 2 buses.
    """
    n = G.number_of_nodes()
    best_line, best_cut = None, 0
    for e in sorted(tuple(sorted(edge)) for edge in G.edges()):
        H = G.copy()
        H.remove_edge(*e)
        comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
        cut = n - comps[0]
        if cut > best_cut:
            best_line, best_cut = e, cut
    return {"line": best_line, "n_cut_off": best_cut}


def worst_single_removal(G: nx.Graph) -> dict:
    """Sweep every bus; find the one whose removal shrinks the giant component most.

    O(n) connectivity checks -- cheap, and it is the honest answer to "which single
    bus matters most", as opposed to a centrality proxy for it.
    """
    best_bus, best_frac = None, 1.1
    for n in sorted(G):
        frac = islanding(G, [n])["lcc_fraction"]
        if frac < best_frac:
            best_bus, best_frac = n, frac
    return {"bus": best_bus, "lcc_fraction": best_frac}


# --------------------------------------------------------------------------
# Motter-Lai cascade
# --------------------------------------------------------------------------


@dataclass
class CascadeResult:
    """Outcome of one cascade run. Frozen into the gold answers verbatim."""

    trigger: list[int]
    alpha: float
    rounds: int
    failed: list[int] = field(default_factory=list)  # OVERLOADED only, excludes trigger
    n_failed: int = 0  # overloaded count -- diagnostic, not what the questions ask
    n_dead: int = 0  # buses not in the largest component at the end, INCLUDING trigger
    lcc_size: int = 0
    lcc_fraction: float = 0.0
    n_islands: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def cascade(G: nx.Graph, trigger: list[int], alpha: float = 0.2) -> CascadeResult:
    """Motter-Lai cascading failure by load redistribution.

    The model (Motter & Lai 2002), stated exactly as implemented:

      1. Load L_i is bus i's betweenness on the *initial* network.
      2. Capacity C_i = (1 + alpha) * L_i(G_0), fixed for all time. alpha is the
         tolerance: how much spare headroom the network was built with.
      3. Remove the trigger bus(es).
      4. Recompute betweenness on the *surviving* network. Every bus with
         L_i > C_i is overloaded and fails. Remove them all at once.
      5. Repeat step 4 until a round produces no new failures.
      6. Report the surviving giant component as a fraction of the original N.

    The subtlety that makes this a good test of a code-generating agent: load is
    recomputed each round but capacity is *not*. An agent that recomputes capacity
    from the current graph produces a cascade that can never propagate, and the
    result looks entirely plausible. That is precisely the plausible-but-wrong
    failure this project is built to detect.

    Strict inequality (L > C, not >=) matters at alpha=0: a bus exactly at capacity
    survives.

    Degenerate corner, verified rather than assumed: a bus with zero initial load
    has C_i = 0 at *every* alpha, so it fails as soon as any shortest path routes
    through it. Raising alpha therefore does not guarantee "no cascade" -- in the
    limit it guarantees only that every bus with positive initial load is immune
    (`verify_oracle.py` asserts exactly this). case118 has 19 zero-load buses, but a
    sweep of all 118 triggers at alpha=0.2 shows none of them ever fail, so this
    corner does not affect any gold answer on the main case. It is documented
    because it would otherwise look like a bug in someone's reimplementation --
    including an agent's.
    """
    if not trigger:
        raise ValueError("cascade needs at least one trigger bus")
    missing = [t for t in trigger if t not in G]
    if missing:
        raise KeyError(f"trigger buses not in grid: {missing}")

    H = G.copy()
    H.remove_nodes_from(trigger)
    return _run_cascade(G, H, alpha, trigger=sorted(trigger))


def _run_cascade(G0: nx.Graph, H: nx.Graph, alpha: float, trigger: list[int]) -> CascadeResult:
    """Shared overload loop for every trigger type.

    `G0` is the INITIAL network: capacities are baselined on it, and fractions are
    reported against its node count. `H` is that network with the trigger already
    applied -- buses removed for a bus trigger, a line removed for a line trigger.
    Splitting it this way keeps one implementation of the Motter-Lai rounds, so a
    bus-triggered and a line-triggered cascade cannot drift apart.
    """
    n0 = G0.number_of_nodes()
    capacity = {v: (1.0 + alpha) * load for v, load in _brandes_betweenness(G0).items()}

    failed: list[int] = []
    rounds = 0
    while H.number_of_nodes():
        rounds += 1
        loads = _brandes_betweenness(H)
        overloaded = [v for v, load in loads.items() if load > capacity[v]]
        if not overloaded:
            break
        H.remove_nodes_from(overloaded)
        failed.extend(overloaded)

    comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    return CascadeResult(
        trigger=trigger,
        alpha=alpha,
        rounds=rounds,
        failed=sorted(failed),
        n_failed=len(failed),
        n_dead=n0 - (comps[0] if comps else 0),
        lcc_size=comps[0] if comps else 0,
        lcc_fraction=(comps[0] / n0) if comps else 0.0,
        n_islands=len(comps),
    )


def cascade_from_line(G: nx.Graph, line: tuple[int, int], alpha: float = 0.2) -> CascadeResult:
    """Cascade triggered by a LINE tripping rather than a bus failing.

    Motter-Lai as published removes nodes. A line trip is the natural operational
    analogue: the line opens, shortest paths reroute, and any bus whose load now
    exceeds the capacity it was built with fails. Capacity is still baselined on the
    intact network, with the line present.

    `trigger` on the result is the empty list, since no bus was removed to start it;
    the line is recorded by the caller.
    """
    u, v = line
    if not G.has_edge(u, v):
        raise KeyError(f"no such line: {line}")
    H = G.copy()
    H.remove_edge(u, v)
    return _run_cascade(G, H, alpha, trigger=[])


def worst_line_trigger(G: nx.Graph, alpha: float = 0.2) -> dict:
    """Which single line, if it trips, leaves the smallest surviving network."""
    best_line, best_frac, best = None, 1.1, None
    for e in sorted(tuple(sorted(edge)) for edge in G.edges()):
        r = cascade_from_line(G, e, alpha)
        if r.lcc_fraction < best_frac:
            best_line, best_frac, best = e, r.lcc_fraction, r
    return {"line": best_line, "lcc_fraction": best_frac, "result": best.to_dict()}


def worst_rounds_trigger(G: nx.Graph, alpha: float = 0.2) -> dict:
    """Which single bus, if it fails, produces the cascade that takes the most rounds.

    Deliberately a different objective from `worst_cascade_trigger`, which minimizes
    the surviving component. Once a failed bus is defined as "not in the largest
    component", damage-count and smallest-component are the same quantity, so duration
    is what still separates them: the hub does the most damage in only 2 rounds, while
    bus 76 grinds on for 8. Duration cannot be read off any centrality measure -- the
    cascade has to be run to completion -- which makes this the sharper test of whether
    an agent computed or pattern-matched.
    """
    best_bus, best_rounds, best = None, -1, None
    for n in sorted(G):
        r = cascade(G, [n], alpha)
        if r.rounds > best_rounds:
            best_bus, best_rounds, best = n, r.rounds, r
    return {"bus": best_bus, "rounds": best_rounds, "result": best.to_dict()}


def with_line(G: nx.Graph, line: tuple[int, int]) -> nx.Graph:
    """The grid with one new line added. The result is a new INITIAL network.

    Capacities must be rebaselined on it -- a bus's headroom reflects the grid it
    operates in, not a grid that no longer exists. An agent that adds the line but
    keeps the original capacities produces a plausible wrong answer.
    """
    u, v = line
    if G.has_edge(u, v):
        raise ValueError(f"line already present: {line}")
    H = G.copy()
    H.add_edge(u, v)
    return H


def no_cascade_value(G: nx.Graph, trigger: list[int]) -> float:
    """The surviving fraction if the trigger is removed and NOTHING cascades.

    Not a question. This is grader metadata: an agent that recomputes capacity from
    the current graph each round makes overload impossible (C_i = (1+alpha)L_i > L_i
    always), so its cascade halts immediately and it reports exactly this number.
    Matching it is the signature of that specific bug, and lets the grader name the
    failure instead of just scoring it wrong.
    """
    return islanding(G, trigger)["lcc_fraction"]


def worst_cascade_trigger(G: nx.Graph, alpha: float = 0.2) -> dict:
    """Which single bus, if it fails, does the most cascade damage?

    Full sweep over every bus. On case118 this is 118 cascades, each recomputing
    betweenness per round -- slow but exact, and it runs once when gold is built,
    not per agent run.
    """
    best_bus, best_frac, best = None, 1.1, None
    for n in sorted(G):
        r = cascade(G, [n], alpha)
        if r.lcc_fraction < best_frac:
            best_bus, best_frac, best = n, r.lcc_fraction, r
    return {"bus": best_bus, "lcc_fraction": best_frac, "result": best.to_dict()}
