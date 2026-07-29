"""Exercise the grader against synthetic runs plus any real logs on disk.

The grader decides every number in the report, so its edge cases matter more than its
happy path. Each case below is a claim about what SHOULD be counted as correct, and the
generous ones are deliberate: an agent that got the physics right and the formatting
wrong was right, and grading it down would measure formatting compliance.

Fully offline. No API calls.

Run:  python scripts/verify_grader.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from gridagent.evaluation import grader as GR
from gridagent.evaluation.runlog import load_all_runs

ROOT = Path(__file__).resolve().parents[1]
GOLD = GR.load_gold(ROOT / "gold" / "gold_answers.json")

failures: list[str] = []


def mkrun(qid: str, answer, status: str = "completed", **kw) -> dict:
    """Minimal log shaped like the real thing."""
    log = {
        "condition": kw.pop("condition", "T"), "question_id": qid,
        "run_index": 1, "status": status,
        "parsed_answer": None if answer is _NONE else {"answer": answer},
        "raw_final_response": "", "parse_error": kw.pop("parse_error", None),
        "tool_calls": kw.pop("tool_calls", []), "code_attempts": kw.pop("code_attempts", []),
        "validation": kw.pop("validation", []), "wall_time_s": 1.0,
        "totals": {"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20,
                   "n_llm_calls": 2, "n_tool_calls": 0, "n_code_attempts": 0},
        "error": kw.pop("error", None),
    }
    log.update(kw)
    return log


_NONE = object()


def case(label: str, qid: str, answer, expect_bucket: str, expect_flags=(), **kw) -> None:
    v = GR.grade(mkrun(qid, answer, **kw), GOLD)
    ok = v.bucket == expect_bucket and all(
        any(f.startswith(e) for f in v.flags) for e in expect_flags)
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:52s} -> {v.bucket:12s} {v.flags}")
    if not ok:
        failures.append(f"{label}: got {v.bucket} {v.flags}, wanted {expect_bucket} {list(expect_flags)}")


def main() -> int:
    a = GOLD["answers"]
    print("GRADER VERIFICATION (offline)")
    print("=" * 92)

    print("\n-- correct answers, awkward formats (must all be CORRECT) --")
    case("fraction exact", "q03", 0.1864406779661017, GR.CORRECT)
    case("fraction rounded within tolerance", "q03", 0.186, GR.CORRECT)
    case("fraction as percent string", "q03", "18.6%", GR.CORRECT)
    case("fraction as bare percent number", "q03", 18.64, GR.CORRECT)
    case("fraction as prose", "q03", "about 0.186 of the grid", GR.CORRECT)
    case("count as int", "q04", 95, GR.CORRECT)
    case("count as string", "q04", "95 buses", GR.CORRECT)
    case("bus as int", "q05", 68, GR.CORRECT)
    case("bus as 'bus 68'", "q05", "bus 68", GR.CORRECT)
    case("bus as single-element list", "q05", [68], GR.CORRECT)
    case("bus list exact", "q02", [68, 76, 64, 79, 37], GR.CORRECT)
    case("bus list as string", "q02", "68, 76, 64, 79, 37", GR.CORRECT)
    case("line as pair", "q09", [79, 80], GR.CORRECT)
    case("line REVERSED (unordered)", "q09", [80, 79], GR.CORRECT)
    case("line as '79-80'", "q09", "79-80", GR.CORRECT)
    case("line as prose", "q09", "the line between buses 79 and 80", GR.CORRECT)

    print("\n-- wrong answers that ran clean (must be WRONG_ANSWER) --")
    case("plain wrong fraction", "q03", 0.5, GR.WRONG_ANSWER)
    case("THE CAPACITY-RECOMPUTE BUG", "q03", 0.9915, GR.WRONG_ANSWER,
         ["matched_no_cascade_value"])
    case("same bug reported as percent", "q03", "99.15%", GR.WRONG_ANSWER,
         ["matched_no_cascade_value"])
    case("counted overloads not failures", "q04", 29, GR.WRONG_ANSWER)
    case("hub shortcut on q06", "q06", 68, GR.WRONG_ANSWER)
    case("bus list right set wrong order", "q02", [76, 68, 64, 79, 37], GR.WRONG_ANSWER)
    case("line that is not the worst", "q09", [7, 8], GR.WRONG_ANSWER)
    case("fraction just outside tolerance", "q03", 0.1930, GR.WRONG_ANSWER)

    print("\n-- no answer produced (must be FAILED) --")
    case("crashed", "q03", _NONE, GR.FAILED, status="crashed", error="tool blew up")
    case("sandbox timeout", "q03", _NONE, GR.FAILED, status="timeout")
    case("hit attempt budget", "q03", _NONE, GR.FAILED, status="max_attempts")
    case("api error", "q03", _NONE, GR.FAILED, status="api_error")
    case("finished but unparseable", "q03", _NONE, GR.FAILED, parse_error="no JSON")
    case("answer field present but empty", "q03", None, GR.FAILED)
    case("prose with no number", "q03", "most of the grid survived", GR.WRONG_ANSWER,
         ["wrong_answer_type"])
    case("boolean is not a count", "q04", True, GR.WRONG_ANSWER, ["wrong_answer_type"])

    print("\n-- answered, but not a usable value: WRONG_ANSWER, not FAILED --")
    case("fraction returned for a 'which line' question", "q09", 0.983, GR.WRONG_ANSWER,
         ["wrong_answer_type"])
    case("score returned instead of the line", "q11", 53, GR.WRONG_ANSWER,
         ["wrong_answer_type"])

    print("\n-- Arm A: tool was right, model misreported (the interesting one) --")
    case("model garbled a correct tool result", "q03", 0.75, GR.WRONG_ANSWER,
         ["tool_had_correct_value"],
         tool_calls=[{"name": "simulate_cascade",
                      "result": {"lcc_fraction": 0.1864406779661017, "rounds": 2}}])
    case("tool itself was wrong", "q03", 0.75, GR.WRONG_ANSWER,
         tool_calls=[{"name": "simulate_cascade", "result": {"lcc_fraction": 0.75}}])

    print("\n-- diagnostic flags --")
    case("plan rejections counted", "q03", 0.1864406779661017, GR.CORRECT,
         ["plan_rejected_x"], validation=[{"problems": ["bad tool"]}, {"problems": []}])
    case("code retries counted", "q03", 0.1864406779661017, GR.CORRECT, ["code_retries_x"],
         code_attempts=[{"exit_code": 1, "timed_out": False},
                        {"exit_code": 0, "timed_out": False}])
    case("sandbox timeout flagged", "q03", 0.1864406779661017, GR.CORRECT,
         ["sandbox_timeout"], code_attempts=[{"exit_code": None, "timed_out": True}])

    print("\n-- real logs on disk --")
    runs = load_all_runs(ROOT / "results")
    if not runs:
        print("  (none yet)")
    else:
        for v in GR.grade_all(runs, GOLD):
            print(f"  {v.condition:10s} {v.question_id}  {v.bucket:12s} "
                  f"submitted={v.submitted!r} gold={v.gold!r} flags={v.flags}")

    print("\n" + "=" * 92)
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All grader cases behave as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
