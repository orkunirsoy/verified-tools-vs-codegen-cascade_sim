"""Run the evaluation matrix: every question, every condition, n runs each.

This is the single command that produced `results/`. It exists so the experiment can be
re-run, resumed, or extended without reconstructing a shell loop, and so a reader can see
exactly what was executed rather than inferring it from 165 log files.

    python scripts/run_matrix.py --dry-run          # what would run, and the cost estimate
    python scripts/run_matrix.py                    # the full matrix, skipping finished runs
    python scripts/run_matrix.py -q q04 q08         # only these questions
    python scripts/run_matrix.py -c A-full B-hints  # only these conditions
    python scripts/run_matrix.py --force            # redo runs that already have a log

RESUMABLE BY DEFAULT. A run whose log already exists is skipped, so an interrupted batch
continues where it stopped and a re-run costs nothing. `--force` overrides that, and is the
only way to spend money on work already done.

EVERY RUN IS ONE PAID API CALL SEQUENCE. The dry run prints the count and an estimate
first; nothing is sent until you confirm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from gridagent import conditions as C
from gridagent.client import OpenAIClient
from gridagent.data import load_grid
from gridagent.questions import load_questions

DEFAULT_RESULTS = ROOT / "results"

#: Rough per-run token cost, from the observed mean across the 165-run matrix. The estimate
#: is deliberately an order-of-magnitude check, not a billing figure.
EST_TOKENS_PER_RUN = 4000

#: List price per 1M tokens, for the pre-flight estimate only. The authoritative table
#: lives in scripts/report.py; this one exists so the estimate printed before spending
#: money is not silently a gpt-4o-mini figure while a different model is being run.
EST_PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
    "gpt-5-nano": (0.05, 0.40), "gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00),
    "gpt-5.4": (2.50, 15.00), "gpt-5.4-mini": (0.75, 4.50), "gpt-3.5-turbo": (0.50, 1.50),
}


def est_price(model: str) -> tuple[float, float]:
    hits = [k for k in EST_PRICES if model.startswith(k)]
    return EST_PRICES[max(hits, key=len)] if hits else (0.15, 0.60)


def log_path(results: Path, condition: str, qid: str, run_index: int) -> Path:
    return results / condition / f"{qid}_run{run_index}.json"


def main() -> int:
    qs = load_questions()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--questions", nargs="+", default=qs.ids, metavar="QID")
    ap.add_argument("-c", "--conditions", nargs="+", default=C.DEFAULT_ORDER, metavar="COND")
    ap.add_argument("-n", "--runs", type=int, default=3, help="runs per cell (default 3)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 by default: n>1 only measures variance if the model samples")
    ap.add_argument("--force", action="store_true", help="redo cells that already have a log")
    ap.add_argument("--dry-run", action="store_true", help="list the work, send nothing")
    # Log paths carry no model id, so a second model would overwrite the first model's
    # runs in place. Results are expensive (an API call each) and labels are free, so the
    # irreversible thing must be namespaced. One directory per model.
    ap.add_argument("--results-dir", type=Path, default=None, metavar="PATH",
                    help="where to write logs (default: results/ for gpt-4o-mini, "
                         "results-<model>/ for anything else)")
    args = ap.parse_args()

    results = args.results_dir
    if results is None:
        results = (DEFAULT_RESULTS if args.model.startswith("gpt-4o-mini")
                   else ROOT / f"results-{args.model}")
    results = Path(results)
    manifest = results / "manifest.json"

    for qid in args.questions:
        qs.by_id(qid)  # raises on an unknown id before anything is spent
    for cond in args.conditions:
        C.get(cond)

    todo = [(c, q, i)
            for c in args.conditions
            for q in args.questions
            for i in range(1, args.runs + 1)
            if args.force or not log_path(results, c, q, i).exists()]
    done = len(args.conditions) * len(args.questions) * args.runs - len(todo)

    print(f"matrix: {len(args.conditions)} conditions x {len(args.questions)} questions "
          f"x {args.runs} runs = {len(args.conditions) * len(args.questions) * args.runs}")
    print(f"already logged: {done}    to run: {len(todo)}")
    print(f"model={args.model}  temperature={args.temperature}")
    print(f"writing to: {results.relative_to(ROOT) if results.is_relative_to(ROOT) else results}")
    p_in, p_out = est_price(args.model)
    est = len(todo) * EST_TOKENS_PER_RUN * (p_in + p_out) / 2 / 1e6
    print(f"rough cost estimate: ${est:.2f}  (~{EST_TOKENS_PER_RUN} tokens/run)")

    if args.dry_run:
        for c, q, i in todo:
            print(f"  would run {c:11s} {q} run{i}")
        return 0
    if not todo:
        print("nothing to do; use --force to redo completed cells")
        return 0

    G = load_grid(qs.case)
    client = OpenAIClient(model=args.model)
    started = time.time()
    buckets: dict[str, int] = {}

    for n, (cond, qid, idx) in enumerate(todo, 1):
        t0 = time.time()
        log = C.get(cond).run(
            G=G, question=qs.by_id(qid), qs=qs, client=client,
            run_index=idx, temperature=args.temperature,
        )
        log.write(results)
        buckets[log.status] = buckets.get(log.status, 0) + 1
        print(f"[{n:3d}/{len(todo)}] {cond:11s} {qid} run{idx}  "
              f"{log.status:12s} {log.total_tokens:6d} tok  {time.time() - t0:5.1f}s")

    elapsed = time.time() - started
    print(f"\ndone in {elapsed / 60:.1f} min; terminal statuses: {buckets}")
    print("grade with:  python scripts/report.py   (or verify_grader.py to see every run)")

    # Provenance: what produced this results directory, appended not overwritten.
    manifest.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(manifest.read_text()) if manifest.exists() else []
    history.append({
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model, "temperature": args.temperature, "runs_per_cell": args.runs,
        "conditions": args.conditions, "questions": args.questions,
        "n_executed": len(todo), "n_skipped": done, "elapsed_min": round(elapsed / 60, 1),
        "statuses": buckets,
    })
    manifest.write_text(json.dumps(history, indent=2) + "\n")
    print(f"appended to {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
