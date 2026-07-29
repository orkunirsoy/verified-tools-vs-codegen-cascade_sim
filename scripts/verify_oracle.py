"""Cross-check the oracle against independent implementations.

The oracle is the answer key for the whole study, so it does not get to assert its
own correctness -- it has to earn it. This script is the evidence, and it is cited
in the report.

Checks:
  1. The from-scratch Brandes betweenness matches NetworkX to 1e-9 on every case.
     Two independent implementations agreeing is real evidence.
  2. Betweenness still matches after the graph is fragmented, since that is the
     state the cascade actually runs in and the easiest place for a hand-rolled
     implementation to be subtly wrong.
  3. Cascade invariants hold (monotone shrinkage, termination, alpha ordering).
  4. Motter-Lai's headline qualitative result reproduces: low tolerance cascades,
     high tolerance does not.

Run: python scripts/verify_oracle.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import networkx as nx

from gridagent.data import CASES, load_grid, grid_summary
from gridagent.oracle import reference as ref

TOL = 1e-9
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def check_betweenness(G: nx.Graph, label: str) -> None:
    """Our Brandes vs NetworkX's, unnormalized, on the same graph."""
    ours = ref.betweenness(G)
    theirs = nx.betweenness_centrality(G, normalized=False)
    worst = max((abs(ours[v] - theirs[v]) for v in G), default=0.0)
    check(f"betweenness matches NetworkX [{label}]", worst < TOL, f"max abs diff {worst:.2e}")


def main() -> int:
    for case in ["case14", "case30", "case57", "case118"]:
        G = load_grid(case)
        print(f"\n=== {grid_summary(G)} ===")

        # 1. connected base graph
        check_betweenness(G, "intact")

        # 2. fragmented graph -- the state the cascade runs in
        H = G.copy()
        H.remove_nodes_from([n for n, _ in ref.top_k_betweenness(G, 3)])
        check_betweenness(H, f"fragmented, {nx.number_connected_components(H)} components")

        # 3. cascade invariants
        hub = ref.top_k_betweenness(G, 1)[0][0]
        alphas = (0.0, 0.2, 0.5, 1.0, 10.0)
        results = {a: ref.cascade(G, [hub], alpha=a) for a in alphas}

        check(
            "cascade terminates and never grows the network",
            all(r.lcc_fraction <= 1.0 and r.rounds >= 1 for r in results.values()),
        )
        check(
            "trigger is never counted among cascade failures",
            all(hub not in r.failed for r in results.values()),
        )
        fracs = [results[a].lcc_fraction for a in alphas]
        check(
            "more tolerance never means more damage (monotone in alpha)",
            all(x <= y + TOL for x, y in zip(fracs, fracs[1:])),
            " <= ".join(f"{f:.3f}" for f in fracs),
        )
        check(
            "low tolerance does cascade (Motter-Lai reproduces)",
            results[0.0].n_failed > 0,
            f"alpha=0 -> {results[0.0].n_failed} failed, "
            f"LCC {results[0.0].lcc_fraction:.3f}",
        )
        check(
            "tolerance strictly helps: alpha=10 beats alpha=0",
            results[10.0].lcc_fraction > results[0.0].lcc_fraction,
            f"{results[0.0].lcc_fraction:.3f} -> {results[10.0].lcc_fraction:.3f}",
        )

        # The degenerate corner of Motter-Lai, asserted as a theorem rather than
        # assumed away. C_i = (1+alpha)*L_i(G_0), so as alpha -> inf any bus with
        # positive initial load becomes uncapacitated and can never fail. But a bus
        # with L_i(G_0) = 0 has C_i = 0 at EVERY alpha, so it fails the instant any
        # shortest path routes through it. Unbounded tolerance therefore does not
        # mean "no cascade" -- it means "only zero-initial-load buses can fail".
        b0 = ref.betweenness(G)
        zero_load = {v for v, load in b0.items() if load == 0.0}
        r_inf = ref.cascade(G, [hub], alpha=1e9)
        check(
            "under unbounded tolerance, only zero-initial-load buses can fail",
            all(f in zero_load for f in r_inf.failed),
            f"{len(zero_load)} zero-load buses; alpha=1e9 -> {r_inf.n_failed} failed",
        )

        # Why a naive "alpha=10 means no cascade" expectation is wrong: a bus can
        # gain load far out of proportion to what it started with, when the hub's
        # removal turns a peripheral bus into a bottleneck.
        H = G.copy()
        H.remove_nodes_from([hub])
        bH = ref.betweenness(H)
        growth = max(
            ((bH[v] / b0[v], v) for v in H if b0[v] > 0),
            default=(0.0, None),
        )
        print(f"  [info] hub bus {hub}; worst load growth after trigger: "
              f"bus {growth[1]} x{growth[0]:.1f}")

        print(f"  [info] hub bus {hub}; alpha=0.2 -> {results[0.2].n_failed} cascade failures, "
              f"LCC {results[0.2].lcc_fraction:.3f}, {results[0.2].rounds} rounds")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All oracle checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
