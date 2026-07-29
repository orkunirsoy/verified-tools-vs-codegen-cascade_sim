"""Regenerate the report's accuracy-versus-cost figure from the run logs.

Pure function over `results*/`: reads logs, grades them, plots. No API calls, no
network. Safe to run as often as you like.

    python scripts/plot_accuracy_vs_cost.py                    # -> Report/fig_accuracy_vs_cost.pdf
    python scripts/plot_accuracy_vs_cost.py --out /tmp/f.png   # any matplotlib format
    python scripts/plot_accuracy_vs_cost.py --linear           # linear x axis

Grading, condition order and per-model pricing are imported from `report.py` rather
than restated here, so the figure cannot drift from the tables it sits next to. Cost
includes retries and failed attempts, for the same reason `report.py` does: excluding
them would flatter whichever arm retries more.

Every point is one (model, condition) cell over all 45 runs -- the same denominator as
the "Totals" table -- so the figure and Table 3 are two views of one number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gridagent import conditions as C
from gridagent.evaluation import grader as GR
from gridagent.evaluation.runlog import load_all_runs
from report import _ordered_models, price_for

#: Model -> colour. The two arm colours match `armA`/`armB` in report.tex so the figure
#: reads against the architecture diagram; the third is the amber that separates the
#: reasoning model from both without colliding with either.
MODEL_COLOR = {
    "gpt-4o-mini": "#14607A",
    "gpt-4o": "#8B4513",
    "gpt-5-nano": "#E4A03C",
}
DEFAULT_COLOR = "#666666"

#: Condition -> marker. Arm A takes the filled round/angular family, Arm B the two
#: triangles, so arm membership survives greyscale printing.
COND_MARKER = {
    "A-full": "o",
    "A-noval": "s",
    "A-zeroshot": "D",
    "B-hints": "^",
    "B-bare": "v",
}
DEFAULT_MARKER = "P"


def cells(vs):
    """Yield (model, condition, accuracy_pct, mean_cost_usd, n) per non-empty cell."""
    models = _ordered_models({v.model for v in vs})
    conds = [c for c in C.DEFAULT_ORDER if any(v.condition == c for v in vs)]
    for m in models:
        p_in, p_out = price_for(m)
        for c in conds:
            g = [v for v in vs if v.model == m and v.condition == c]
            if not g:
                continue
            cost = sum(v.prompt_tokens * p_in + v.completion_tokens * p_out
                       for v in g) / 1e6
            ncorr = sum(v.bucket == GR.CORRECT for v in g)
            yield m, c, 100 * ncorr / len(g), cost / len(g), len(g)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", type=Path, default=None, metavar="DIR",
                    help="results directories to grade (default: every results*/ found)")
    ap.add_argument("--out", type=Path,
                    default=ROOT.parent / "Report" / "fig_accuracy_vs_cost.pdf",
                    help="output path; extension picks the format")
    ap.add_argument("--linear", action="store_true",
                    help="linear cost axis (default: log, since costs span ~15x)")
    args = ap.parse_args()

    dirs = args.results
    if dirs is None:
        dirs = sorted(p for p in ROOT.glob("results*")
                      if p.is_dir() and not p.name.startswith("results_archive"))
    runs = [r for d in dirs for r in load_all_runs(d)]
    if not runs:
        print(f"no run logs in {[str(d) for d in dirs]}; run scripts/run_matrix.py first")
        return 1
    vs = GR.grade_all(runs, GR.load_gold(ROOT / "gold" / "gold_answers.json"))

    rows = list(cells(vs))
    if not rows:
        print("nothing to plot")
        return 1

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ns = set()
    for model, cond, acc, cost, n in rows:
        ns.add(n)
        ax.scatter(cost, acc,
                   color=MODEL_COLOR.get(model, DEFAULT_COLOR),
                   marker=COND_MARKER.get(cond, DEFAULT_MARKER),
                   s=95, edgecolors="white", linewidths=0.8, zorder=3)
        print(f"  {model:>12s} {cond:>11s}  {acc:5.1f}%  ${cost:.5f}/run  (n={n})")

    if not args.linear:
        ax.set_xscale("log")
    ax.set_xlabel("Cost per run (USD)")
    # The denominator is stated on the axis so the figure is readable detached from the
    # caption; it collapses to "all runs" if the cells ever stop being equal-sized.
    ax.set_ylabel(f"Accuracy, all {ns.pop()} runs (%)" if len(ns) == 1
                  else "Accuracy, all runs (%)")
    ax.set_title("Accuracy vs cost by model and condition")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    models = _ordered_models({m for m, *_ in rows})
    conds = [c for c in C.DEFAULT_ORDER if any(c == r[1] for r in rows)]
    cond_legend = ax.legend(
        handles=[Line2D([], [], linestyle="none", color="#333333",
                        marker=COND_MARKER.get(c, DEFAULT_MARKER), markersize=8, label=c)
                 for c in conds],
        title="Condition", loc="upper left", frameon=True, fontsize=9)
    ax.add_artist(cond_legend)
    ax.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=8,
                              color=MODEL_COLOR.get(m, DEFAULT_COLOR), label=m)
                       for m in models],
              title="Model", loc="lower right", frameon=True, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
