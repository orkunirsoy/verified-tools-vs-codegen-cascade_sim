"""Grade every run log and print the results tables.

Pure function over `results/`: reads logs, applies the grader, prints. No API calls, no
network, nothing written unless `--csv` is given. Safe to run as often as you like, which
is the point of deriving labels rather than storing them.

    python scripts/report.py                # the tables
    python scripts/report.py --flags        # + diagnostic flag counts
    python scripts/report.py --failures     # + one line per non-correct run
    python scripts/report.py --csv out.csv  # one row per run, for plotting

Costs are computed at gpt-4o-mini list price and include retries and failed attempts,
because excluding them would flatter whichever arm retries more.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from gridagent import conditions as C
from gridagent.evaluation import grader as GR
from gridagent.evaluation.runlog import load_all_runs
from gridagent.questions import load_questions

#: USD per 1M tokens, OpenAI list price. Keyed by model id, because a single hardcoded
#: pair silently misreports every dollar figure the moment a second model is run --
#: and cost per correct answer is a headline metric here.
#:
#: Reasoning models (the gpt-5 family) bill hidden thinking tokens as OUTPUT, and the
#: API already counts them inside `completion_tokens`. Cost below is therefore correct
#: as-is. What is NOT comparable across families is token COUNT, so the tables report
#: reasoning tokens separately -- see `visible_completion_tokens` in runlog.py.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-3.5-turbo": (0.50, 1.50),
}
DEFAULT_PRICE = (0.15, 0.60)


def price_for(model: str) -> tuple[float, float]:
    """List price for a model id, longest-prefix matched so dated snapshots resolve."""
    if model in PRICES:
        return PRICES[model]
    hits = [k for k in PRICES if model.startswith(k)]
    if hits:
        return PRICES[max(hits, key=len)]
    print(f"  ! no price for model {model!r}; using gpt-4o-mini rates. Costs for this "
          f"model are WRONG until you add it to PRICES.", file=sys.stderr)
    return DEFAULT_PRICE


# Back-compat for any caller still importing the old names.
PRICE_IN, PRICE_OUT = DEFAULT_PRICE
MARK = {GR.CORRECT: "C", GR.WRONG_ANSWER: "W", GR.FAILED: "F"}

#: Model order for cross-model tables: the axis under test, weakest/oldest first, so a
#: trend reads left to right rather than in whatever order the directories globbed.
MODEL_ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o",
               "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5.4-mini", "gpt-5.4"]


def _ordered_models(models) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _cross_model(vs, qs, split: str) -> None:
    """Correct-rate, silent-failure rate and cost, per condition per model.

    Reasoning tokens are reported as their own column rather than folded into `tokens`.
    A reasoning model's billed output is dominated by hidden thinking -- on the smoke
    test, 1408 of 1506 output tokens -- so a single token column would compare model
    families rather than the arms, which is the comparison this project exists to make.
    """
    models = _ordered_models({v.model for v in vs})
    conds = [c for c in C.DEFAULT_ORDER if any(v.condition == c for v in vs)]

    for title, key in (("CORRECT RATE", "correct"), ("SILENT-FAILURE RATE", "silent")):
        print(f"{title}")
        print(f"  {'condition':>11s} " + "".join(f"{m:>16s}" for m in models))
        print("  " + "-" * (11 + 16 * len(models)))
        for c in conds:
            row = f"  {c:>11s} "
            for m in models:
                g = [v for v in vs if v.condition == c and v.model == m]
                if not g:
                    row += f"{'-':>16s}"; continue
                n = sum(v.bucket == (GR.CORRECT if key == "correct" else GR.WRONG_ANSWER)
                        for v in g)
                row += f"{f'{n}/{len(g)} ({100*n/len(g):.0f}%)':>16s}"
            print(row)
        print()

    print("COST AND TOKENS")
    hdr = f"  {'model':>14s} {'correct':>12s} {'tokens':>9s} {'reasoning':>10s} {'$/run':>9s} {'$/correct':>10s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for m in models:
        g = [v for v in vs if v.model == m]
        if not g:
            continue
        p_in, p_out = price_for(m)
        cost = sum(v.prompt_tokens * p_in + v.completion_tokens * p_out for v in g) / 1e6
        ncorr = sum(v.bucket == GR.CORRECT for v in g)
        reas = sum(v.reasoning_tokens for v in g)
        print(f"  {m:>14s} {f'{ncorr}/{len(g)} ({100*ncorr/len(g):.0f}%)':>12s} "
              f"{sum(v.total_tokens for v in g)//len(g):>9d} "
              f"{(reas//len(g)) if reas else 0:>10d} "
              f"${cost/len(g):>8.4f} {f'${cost/ncorr:.4f}' if ncorr else 'n/a':>10s}")
    total = sum(v.prompt_tokens * price_for(v.model)[0]
                + v.completion_tokens * price_for(v.model)[1] for v in vs) / 1e6
    print(f"\n  total spend across all models shown: ${total:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", action="store_true")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--csv", type=Path)
    # One directory per model (see run_matrix --results-dir). Default reads whatever
    # results*/ directories exist, so adding a model needs no flag change here.
    ap.add_argument("--results", nargs="+", type=Path, default=None, metavar="DIR",
                    help="results directories to grade (default: every results*/ found)")
    ap.add_argument("--split", choices=["development", "test", "all", "validation"], default="all",
                    help="grade only one question split (default: all)")
    args = ap.parse_args()

    dirs = args.results
    if dirs is None:
        dirs = sorted(p for p in ROOT.glob("results*")
                      if p.is_dir() and not p.name.startswith("results_archive"))
    runs = [r for d in dirs for r in load_all_runs(d)]
    if not runs:
        print(f"no run logs in {[str(d) for d in dirs]}; run scripts/run_matrix.py first")
        return 1
    gold = GR.load_gold(ROOT / "gold" / "gold_answers.json")
    qs = load_questions()
    if args.split == "validation":
        args.split = "development"  # legacy alias
    if args.split != "all":
        keep = set(qs.split_ids(args.split))
        runs = [r for r in runs if r["question_id"] in keep]
    vs = GR.grade_all(runs, gold)

    models = sorted({v.model for v in vs})
    if len(models) > 1:
        # Never average across models: a mixed table hides the comparison being made.
        print(f"grading {len(vs)} runs across {len(models)} models: {', '.join(models)}")
        print(f"split={args.split}\n")
        _cross_model(vs, qs, args.split)
        return 0
    if models:
        print(f"model: {models[0]}   split: {args.split}")

    conds = [c for c in C.DEFAULT_ORDER if any(v.condition == c for v in vs)]
    qids = [q for q in qs.ids if any(v.question_id == q for v in vs)]

    def cell(cond: str, qid: str) -> list:
        return sorted([v for v in vs if v.condition == cond and v.question_id == qid],
                      key=lambda v: v.run_index)

    print(f"{len(vs)} runs   C=correct  W=wrong answer  F=no answer produced")
    print()
    print(f"{'q':>4s} {'category':>17s} " + "".join(f"{c:>12s}" for c in conds))
    print("-" * (23 + 12 * len(conds)))
    for qid in qids:
        cat = next((v.category for v in vs if v.question_id == qid), "")
        row = "".join(f"{''.join(MARK[v.bucket] for v in cell(c, qid)):>12s}" for c in conds)
        print(f"{qid:>4s} {cat:>17s} " + row)

    print()
    hdr = f"{'condition':>11s} {'n':>4s} {'correct':>13s} {'wrong':>7s} {'failed':>7s}"
    hdr += f" {'tokens':>8s} {'reason':>8s} {'$/run':>8s} {'$/correct':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for c in conds:
        g = [v for v in vs if v.condition == c]
        b = Counter(v.bucket for v in g)
        # Priced per run's OWN model, so a mixed-model results directory is still costed
        # correctly rather than at whatever single rate was hardcoded.
        cost = sum((lambda p: v.prompt_tokens * p[0] + v.completion_tokens * p[1])(
            price_for(v.model)) for v in g) / 1e6
        ncorr = b[GR.CORRECT]
        per_correct = f"${cost / ncorr:.4f}" if ncorr else "n/a"
        reas = sum(v.reasoning_tokens for v in g)
        reas_col = f"{reas // len(g):>8d}" if reas else f"{'-':>8s}"
        print(f"{c:>11s} {len(g):>4d} {ncorr:>4d} ({100 * ncorr / len(g):>4.0f}%) "
              f"{b[GR.WRONG_ANSWER]:>7d} {b[GR.FAILED]:>7d} "
              f"{sum(v.total_tokens for v in g) // len(g):>8d} {reas_col} "
              f"${cost / len(g):>7.4f} {per_correct:>10s}")

    print()
    print("by question category:")
    cats = sorted({v.category for v in vs})
    print(f"{'condition':>11s} " + "".join(f"{c:>19s}" for c in cats))
    for c in conds:
        cells = []
        for cat in cats:
            g = [v for v in vs if v.condition == c and v.category == cat]
            n = sum(1 for v in g if v.bucket == GR.CORRECT)
            cells.append(f"{n}/{len(g)}" if g else "-")
        print(f"{c:>11s} " + "".join(f"{x:>19s}" for x in cells))

    if args.flags:
        print("\ndiagnostic flags (derived, never stored):")
        for c in conds:
            fl = Counter(f for v in vs if v.condition == c for f in v.flags)
            if fl:
                print(f"  {c}:")
                for name, n in fl.most_common():
                    print(f"      {n:>3d}  {name}")

    if args.failures:
        print("\nevery non-correct run:")
        for v in sorted([v for v in vs if v.bucket != GR.CORRECT],
                        key=lambda v: (v.question_id, v.condition, v.run_index)):
            print(f"  {v.question_id} {v.condition:11s} r{v.run_index} {v.bucket:12s} "
                  f"submitted={str(v.submitted)[:26]:>26s}  gold={str(v.gold)[:18]}")
            print(f"       {v.reason[:96]}")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(vs[0].to_dict()))
            w.writeheader()
            for v in vs:
                w.writerow({k: (";".join(str(i) for i in x) if isinstance(x, list) else x)
                            for k, x in v.to_dict().items()})
        print(f"\nwrote {args.csv} ({len(vs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
