"""Does Arm B's retry loop recover from its own errors?

Pure function over `results*/`: reads logs, applies the grader, prints. No API calls,
no network, nothing written. Safe to run as often as you like.

    python scripts/retry_analysis.py
    python scripts/retry_analysis.py --results results results-gpt-4o

Arm B regenerates code after reading stdout or a traceback, up to an attempt budget.
This script asks how often that loop actually fires, and whether it fires on the failure
mode that matters. An attempt counts as errored when it wrote to stderr, exited non-zero,
or timed out; every flagged attempt on gpt-4o-mini was checked by hand and is a genuine
Python traceback.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gridagent.evaluation import grader as GR  # noqa: E402


def errored(attempt: dict) -> bool:
    """An attempt that did not run clean."""
    return (bool(attempt.get("stderr"))
            or attempt.get("exit_code", 0) != 0
            or attempt.get("timed_out", False))


def load_arm_b(results_dir: Path, gold: dict) -> list[tuple[dict, object]]:
    out = []
    for f in sorted(results_dir.glob("*/*.json")):
        log = json.loads(f.read_text())
        if not log["condition"].startswith("B"):
            continue
        out.append((log, GR.grade(log, gold)))
    return out


def report(results_dir: Path, gold: dict) -> None:
    runs = load_arm_b(results_dir, gold)
    if not runs:
        print(f"\n{results_dir}: no Arm B runs found")
        return

    model = runs[0][0]["model"]
    total_attempts = sum(len(log["code_attempts"]) for log, _ in runs)
    print(f"\n{'=' * 72}")
    print(f"{results_dir}  model={model}  {len(runs)} Arm B runs  "
          f"{total_attempts} code attempts")
    print("=" * 72)

    # How many attempts each run used, against how the run ended.
    by_n = collections.defaultdict(collections.Counter)
    for log, v in runs:
        by_n[len(log["code_attempts"])][v.bucket] += 1

    print("\nattempts used -> outcome")
    print(f"{'attempts':>9} {'runs':>5} {'CORRECT':>8} {'WRONG':>7} {'FAILED':>7}")
    for n in sorted(by_n):
        c = by_n[n]
        print(f"{n:>9} {sum(c.values()):>5} {c[GR.CORRECT]:>8} "
              f"{c[GR.WRONG_ANSWER]:>7} {c[GR.FAILED]:>7}")

    # Did the first attempt break, and did the loop rescue the run?
    first = collections.defaultdict(collections.Counter)
    for log, v in runs:
        attempts = log["code_attempts"]
        if not attempts:
            first["no code produced"][v.bucket] += 1
        elif errored(attempts[0]):
            first["attempt 1 errored"][v.bucket] += 1
        else:
            first["attempt 1 ran clean"][v.bucket] += 1

    print("\nfirst attempt -> final outcome")
    print(f"{'':>20} {'runs':>5} {'CORRECT':>8} {'WRONG':>7} {'FAILED':>7}")
    for key in ("attempt 1 ran clean", "attempt 1 errored", "no code produced"):
        c = first.get(key)
        if not c:
            continue
        print(f"{key:>20} {sum(c.values()):>5} {c[GR.CORRECT]:>8} "
              f"{c[GR.WRONG_ANSWER]:>7} {c[GR.FAILED]:>7}")

    bad = sum(sum(1 for a in log["code_attempts"] if errored(a)) for log, _ in runs)
    revised_clean = sum(1 for log, _ in runs
                        if len(log["code_attempts"]) > 1
                        and not errored(log["code_attempts"][0]))
    print(f"\nerrored attempts: {bad} of {total_attempts} "
          f"({100 * bad / max(total_attempts, 1):.1f}%)")
    print(f"runs that revised after a CLEAN first attempt: {revised_clean} "
          f"(incremental work, not recovery)")

    # The load-bearing number: wrong answers the loop could never have seen.
    silent_clean = silent_err = 0
    for log, v in runs:
        if v.bucket != GR.WRONG_ANSWER:
            continue
        if any(errored(a) for a in log["code_attempts"]):
            silent_err += 1
        else:
            silent_clean += 1
    print(f"WRONG_ANSWER runs: {silent_clean + silent_err}   "
          f"never saw an error: {silent_clean}   saw at least one: {silent_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, nargs="+", default=None,
                    help="results directories to read (default: every results*/ found)")
    args = ap.parse_args()

    # Mirror report.py: never sweep archived pre-fix logs into a live number.
    dirs = args.results or sorted(d for d in ROOT.glob("results*")
                                  if d.is_dir()
                                  and not d.name.startswith("results_archive"))
    gold = GR.load_gold(ROOT / "gold" / "gold_answers.json")
    for d in dirs:
        report(d if d.is_absolute() else ROOT / d, gold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
