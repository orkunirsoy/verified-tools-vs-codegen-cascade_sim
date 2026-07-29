"""Run one question through one condition and print the trace.

This is the debugging surface and the demo surface. Day 1 is spent staring at traces,
and a screenshot of this output is the fallback exhibit if the HTML trace viewer slips.

    python ask.py q03 --condition A-full
    python ask.py q03 --condition B-hints --stub    # offline, no API key
    python ask.py q10 --condition A-full --save

`--stub` replays canned responses so the whole path can be exercised with no key and no
network. It verifies prompt assembly, both arms, the sandbox, and the log schema. It
cannot verify auth, real token accounting, the true response shape, or rate limits --
that is what the one real smoke call is for, and why it stays a gate before any batch.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from gridagent import conditions as C
from gridagent.client import OpenAIClient, StubClient
from gridagent.data import load_grid
from gridagent.questions import load_questions
from gridagent.evaluation.runlog import RunLog

GOLD = json.loads((ROOT / "gold" / "gold_answers.json").read_text())

RULE = "=" * 78


def _trim(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more chars]"


def print_trace(log: RunLog, gold, verbose: int) -> None:
    print(RULE)
    print(f"{log.condition}  |  {log.question_id}  |  {log.model}  |  {log.case}")
    print(RULE)
    print("QUESTION (as the arm received it)")
    print(_trim(log.question_text, 0 if verbose >= 2 else 600))
    print(f"\ncontext blocks: {log.context_blocks or '(none)'}")

    if log.plan is not None:
        print(f"\n{'-' * 78}\nPLAN")
        print(json.dumps(log.plan, indent=2))
    for v in (log.validation or []):
        verdict = "PASS" if not v["problems"] else f"REJECTED: {v['problems']}"
        print(f"validation (enabled={v['enabled']}, attempt {v['attempt']}): {verdict}")

    for t in log.tool_calls:
        print(f"\n{'-' * 78}\nTOOL {t.name}({json.dumps(t.args, default=str)})  "
              f"[{t.duration_s:.2f}s]")
        if t.error:
            print(f"  ERROR: {t.error}")
        else:
            shown = json.dumps(t.result, indent=2, default=str)
            print(_trim(shown, 0 if verbose >= 1 else 700))

    for c in log.code_attempts:
        print(f"\n{'-' * 78}\nCODE attempt {c.attempt}  [{c.duration_s:.2f}s, "
              f"exit={c.exit_code}{', TIMED OUT' if c.timed_out else ''}]")
        print(_trim(c.code, 0 if verbose >= 1 else 900))
        if c.stdout:
            print(f"  stdout: {_trim(c.stdout.strip(), 400)}")
        if c.stderr:
            print(f"  stderr: {_trim(c.stderr.strip(), 400)}")

    print(f"\n{RULE}\nFINAL RESPONSE")
    print(_trim(log.raw_final_response, 0 if verbose >= 1 else 800) or "(none)")
    if log.parse_error:
        print(f"PARSE ERROR: {log.parse_error}")

    answer = (log.parsed_answer or {}).get("answer") if isinstance(log.parsed_answer, dict) else None
    print(f"\nanswer : {answer!r}")
    print(f"gold   : {gold['value']!r}")
    print(f"status : {log.status}" + (f"  ({log.error})" if log.error else ""))
    print(f"tokens : {log.total_tokens} ({log.prompt_tokens} in / "
          f"{log.completion_tokens} out) over {log.n_llm_calls} calls, "
          f"retries included")
    print(f"time   : {log.wall_time_s:.1f}s")
    print(RULE)


def stub_responses(arm: str, question_id: str) -> list[str]:
    """Canned responses that exercise the full path offline.

    Arm B's script deliberately fails first: attempt 1 raises, so the observe-and-revise
    loop, the traceback capture, and the retry token accounting all get exercised rather
    than assumed.
    """
    gold_value = GOLD["answers"][question_id]["value"]
    answer = json.dumps({"answer": gold_value, "units": "stub", "reasoning": "stub run"})
    if arm == "A":
        return [
            '{"steps": [{"id": "s1", "tool": "load_ranking", "args": {"k": 1}}, '
            '{"id": "s2", "tool": "simulate_cascade", '
            '"args": {"trigger_buses": ["$s1.busiest_bus"], "alpha": 0.2}}]}',
            answer,
        ]
    return [
        "```python\nprint(undefined_name_to_force_a_traceback)\n```",
        "```python\nprint('buses:', G.number_of_nodes())\n```",
        answer,
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question_id", help="e.g. q03")
    ap.add_argument("--condition", default="A-full", choices=C.DEFAULT_ORDER)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--run-index", type=int, default=1)
    ap.add_argument("--stub", action="store_true", help="offline canned responses")
    ap.add_argument("--save", action="store_true", help="write the log to results/")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v full tool output and code, -vv also the full question")
    args = ap.parse_args()

    qs = load_questions()
    question = qs.by_id(args.question_id)
    G = load_grid(qs.case)
    cond = C.get(args.condition)

    if args.stub:
        client = StubClient(stub_responses(cond.arm, question.id))
    else:
        client = OpenAIClient(model=args.model)

    log = cond.run(G=G, question=question, qs=qs, client=client,
                   run_index=args.run_index, temperature=args.temperature)

    print_trace(log, GOLD["answers"][question.id], args.verbose)

    if args.save:
        path = log.write(ROOT / "results")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
