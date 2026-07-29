"""Grade a run log against the frozen gold key.

A PURE FUNCTION OVER LOGS
-------------------------
Nothing here calls a model or touches the network. Given a run log and the gold key it
returns a verdict, so it can be rerun for free every time the failure taxonomy changes.
That is what made it safe to defer classification until after seeing the data: labels are
cheap to recompute, raw traces cost an API call to regenerate.

THREE OUTCOME BUCKETS, NOT TWO
------------------------------
    CORRECT       finished and matched gold
    FAILED        never produced an answer: crashed, timed out, hit the attempt
                  budget, an API error, or emitted something unparseable
    WRONG_ANSWER  finished cleanly, produced a well-formed answer, and it is wrong

The third bucket is the entire point of the project. A single "incorrect" category cannot
distinguish an agent that fell over from one that returned a plausible number nobody would
question, and the second is the failure that motivates verified tools.

DIAGNOSTIC FLAGS
----------------
Buckets say what happened; flags say WHY, and are derived, never stored. The important one
is `no_cascade_value`: an agent that recomputes capacity each round makes overload
impossible, so its cascade halts immediately and it reports the structural-removal number.
Matching that value is the signature of a specific, known bug, so the grader can name the
failure rather than merely scoring it wrong.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

CORRECT = "CORRECT"
FAILED = "FAILED"
WRONG_ANSWER = "WRONG_ANSWER"

#: Terminal statuses that mean no answer was ever produced.
NO_ANSWER_STATUSES = {"crashed", "timeout", "max_attempts", "api_error", "incomplete"}


@dataclass
class Verdict:
    condition: str
    question_id: str
    run_index: int
    bucket: str
    gold: Any = None
    submitted: Any = None
    normalized: Any = None
    answer_type: str = ""
    category: str = ""
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    #: accounting carried forward so the results tables need only the verdicts
    model: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Hidden reasoning tokens (gpt-5 family). Already inside completion_tokens, which is
    #: what the API bills; carried separately because token COUNT stops being comparable
    #: across model families once a reasoning model is in the table.
    reasoning_tokens: int = 0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_code_attempts: int = 0
    n_plan_rejections: int = 0
    wall_time_s: float = 0.0
    status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# answer normalization
# --------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

#: Unsigned, for bus ids. Bus ids are never negative, and a signed pattern misreads the
#: most natural way to write a line: "79-80" parses as [79, -80], which then fails to
#: match gold [79, 80]. That would have silently marked correct q09/q11 answers wrong.
_UINT = re.compile(r"\d+")


def _first_number(text: str) -> float | None:
    m = _NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


def _bus_ids(text: str) -> list[int]:
    """Every bus id in a string, unsigned so hyphens read as separators."""
    return [int(n) for n in _UINT.findall(text.replace(",", " "))]


def _validation_entries(log: dict) -> list[dict]:
    """Plan-validation records, tolerant of logs written by older code.

    An earlier version stored a single dict here rather than a list of attempts.
    Iterating that yields key strings and crashes the grader, so anything that is not a
    list of dicts is treated as absent rather than guessed at.
    """
    v = log.get("validation")
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return [v] if isinstance(v, dict) else []


def normalize(value: Any, answer_type: str) -> tuple[Any, str | None]:
    """Coerce a submitted answer into a comparable form.

    Deliberately generous about FORMAT and strict about VALUE. An agent that understood
    the physics but wrote "18.6%" instead of 0.186, or "bus 68" instead of 68, was right;
    penalising that would measure formatting compliance rather than correctness. What is
    never repaired is a wrong number.
    """
    if value is None:
        return None, "no answer field"

    if answer_type == "fraction":
        if isinstance(value, str):
            pct = value.strip().endswith("%")
            n = _first_number(value)
            if n is None:
                return None, f"no number in {value!r}"
            return (n / 100.0 if pct else n), None
        if isinstance(value, (int, float)):
            # A bare number above 1 for a fraction is a percentage in disguise.
            return (float(value) / 100.0 if value > 1.0 else float(value)), None
        return None, f"cannot read a fraction from {type(value).__name__}"

    if answer_type in {"count", "number"}:
        if isinstance(value, bool):
            return None, "boolean is not a number"
        if isinstance(value, (int, float)):
            return float(value), None
        if isinstance(value, str):
            n = _first_number(value)
            return (n, None) if n is not None else (None, f"no number in {value!r}")
        return None, f"cannot read a number from {type(value).__name__}"

    if answer_type == "bus":
        if isinstance(value, bool):
            return None, "boolean is not a bus id"
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        if isinstance(value, str):
            ids = _bus_ids(value)
            return (ids[0], None) if ids else (None, f"no bus id in {value!r}")
        if isinstance(value, list) and len(value) == 1:
            return normalize(value[0], "bus")
        return None, f"cannot read a bus id from {type(value).__name__}"

    if answer_type == "bus_list_ordered":
        if isinstance(value, str):
            ids = _bus_ids(value)
            if not ids:
                return None, f"no bus ids in {value!r}"
            return ids, None
        if isinstance(value, list):
            out = []
            for v in value:
                one, err = normalize(v, "bus")
                if err:
                    return None, f"bad element in list: {err}"
                out.append(one)
            return out, None
        return None, f"cannot read a bus list from {type(value).__name__}"

    if answer_type == "line":
        # A line is an UNORDERED pair: "79-80", [80, 79] and "the line between buses
        # 79 and 80" are the same answer.
        if isinstance(value, str):
            ids = _bus_ids(value)
            if len(ids) < 2:
                return None, f"need two bus ids, found {len(ids)} in {value!r}"
            return sorted(ids[:2]), None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                return None, f"a line needs exactly 2 bus ids, got {len(value)}"
            out = []
            for v in value:
                one, err = normalize(v, "bus")
                if err:
                    return None, f"bad bus id in line: {err}"
                out.append(one)
            return sorted(out), None
        return None, f"cannot read a line from {type(value).__name__}"

    return value, None


def matches(submitted: Any, gold: Any, answer_type: str, tolerance: float | None) -> bool:
    """Is the normalized answer right?"""
    if submitted is None:
        return False
    if answer_type in {"fraction", "number"}:
        tol = tolerance if tolerance is not None else 1e-9
        return isinstance(submitted, float) and math.isclose(
            submitted, float(gold), abs_tol=tol)
    if answer_type == "count":
        return float(submitted) == float(gold)
    if answer_type == "bus":
        return int(submitted) == int(gold)
    if answer_type in {"bus_list_ordered"}:
        return list(submitted) == list(gold)
    if answer_type == "line":
        return list(submitted) == sorted(int(x) for x in gold)
    return submitted == gold


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _fraction_flags(entry: dict, normalized: float, n_buses: int) -> list[str]:
    """Name the specific arithmetic mistakes behind a wrong fraction.

    Two of these were found on q10 and neither was caught by bucket alone:

      * dividing by the SURVIVORS instead of the pre-cascade component. 14/117 rather
        than 14/118 is a genuine misreading of the spec, but it lands 0.001 from gold and
        the 0.005 tolerance absorbs it, so the run scores CORRECT. The flag records that
        it happened even when the verdict does not.
      * reporting 1.0 for a bus-triggered cascade, which is impossible: removing the
        trigger caps the answer at (N-1)/N. It means the trigger was never removed.
    """
    flags: list[str] = []
    gold_value = entry.get("value")
    if not isinstance(gold_value, (int, float)) or not n_buses:
        return flags

    ceiling = (n_buses - 1) / n_buses
    if normalized > ceiling + 1e-9:
        flags.append("impossible_full_survival")

    lcc = round(float(gold_value) * n_buses)
    for removed in (1, 2, 3):
        denom = n_buses - removed
        if denom > 0 and math.isclose(normalized, lcc / denom, abs_tol=1e-4):
            flags.append(f"wrong_denominator_minus{removed}")
            break
    return flags


def diagnose(log: dict, entry: dict, normalized: Any, answer_type: str,
             n_buses: int = 0) -> list[str]:
    """Derived flags explaining a run. Never stored in the log, always recomputed."""
    flags: list[str] = []
    prov = entry.get("provenance", {}) or {}

    # The signature bug: capacity recomputed each round -> nothing ever overloads ->
    # the cascade halts immediately and the answer collapses to structural removal.
    if answer_type == "fraction" and isinstance(normalized, float):
        flags.extend(_fraction_flags(entry, normalized, n_buses))

    ncv = prov.get("no_cascade_value")
    if ncv is not None and answer_type == "fraction" and isinstance(normalized, float):
        if math.isclose(normalized, float(ncv), abs_tol=0.005):
            flags.append("matched_no_cascade_value")

    # Arm A only: did the tools produce the right answer and the model still misreport it?
    # Deliberately independent of whether the SUBMITTED value parsed. An earlier version
    # only ran this when `normalized` existed, so it stayed silent on the clearest case of
    # all: the tools returned [31, 81] and the model answered 53, which is a wrong type and
    # therefore had no normalized value to check.
    tool_calls = log.get("tool_calls") or []
    if tool_calls:
        if _tools_contained(tool_calls, entry.get("value"), answer_type):
            flags.append("tool_had_correct_value")

    # Arm B answering with no execution behind it: the number was asserted, not computed.
    # Distinct from "computed and wrong", and it must not be counted as evidence either
    # way about code generation's correctness.
    if (log.get("notes") or {}).get("arm") == "B" and not (log.get("code_attempts") or []):
        flags.append("answered_without_executing")
    if any(c.get("session_lost") for c in (log.get("code_attempts") or [])):
        flags.append("session_restarted")
    # Neutral: changing G is allowed. Recorded because "was the graph edited before this
    # measurement" is worth knowing when reading a wrong answer, not because it is a fault.
    if any(c.get("grid_mutated") for c in (log.get("code_attempts") or [])):
        flags.append("modified_grid")

    if log.get("parse_error"):
        flags.append("parse_error")
    if any(c.get("timed_out") for c in (log.get("code_attempts") or [])):
        flags.append("sandbox_timeout")
    if any(c.get("exit_code") not in (0, None) for c in (log.get("code_attempts") or [])):
        flags.append("code_raised")
    n_rej = sum(1 for v in _validation_entries(log) if v.get("problems"))
    if n_rej:
        flags.append(f"plan_rejected_x{n_rej}")
    if len(log.get("code_attempts") or []) > 1:
        flags.append(f"code_retries_x{len(log['code_attempts']) - 1}")
    return flags


def _tools_contained(tool_calls: list[dict], gold: Any, answer_type: str) -> bool:
    """Did any tool return the gold value anywhere in its result?

    Crude by design: a deep scan for the value. It exists to separate 'the verified tool
    was wrong' from 'the tool was right and the model reported something else', which is
    the failure mode most worth reporting for Arm A.
    """
    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            if answer_type in {"line", "bus_list_ordered"}:
                try:
                    if answer_type == "line" and len(node) == 2:
                        if sorted(int(x) for x in node) == sorted(int(x) for x in gold):
                            return True
                    if answer_type == "bus_list_ordered" and list(node) == list(gold):
                        return True
                except (TypeError, ValueError):
                    pass
            return any(walk(v) for v in node)
        if answer_type in {"fraction", "number"} and isinstance(node, (int, float)):
            return math.isclose(float(node), float(gold), abs_tol=1e-9)
        if answer_type in {"count", "bus"} and isinstance(node, (int, float)):
            return not isinstance(node, bool) and float(node) == float(gold)
        return False

    return any(walk(tc.get("result")) for tc in tool_calls)


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


def grade(log: dict, gold: dict) -> Verdict:
    """Grade one run log. `gold` is the full gold_answers.json payload."""
    qid = log["question_id"]
    entry = gold["answers"][qid]
    answer_type = entry["answer_type"]
    n_buses = gold.get("n_buses", 0)
    totals = log.get("totals", {}) or {}

    v = Verdict(
        condition=log["condition"], question_id=qid, run_index=log["run_index"],
        bucket=FAILED, gold=entry["value"], answer_type=answer_type,
        category=entry.get("category", ""), status=log.get("status", ""),
        model=log.get("model", ""),
        total_tokens=totals.get("total_tokens", 0),
        prompt_tokens=totals.get("prompt_tokens", 0),
        completion_tokens=totals.get("completion_tokens", 0),
        reasoning_tokens=totals.get("reasoning_tokens", 0),
        n_llm_calls=totals.get("n_llm_calls", 0),
        n_tool_calls=totals.get("n_tool_calls", 0),
        n_code_attempts=totals.get("n_code_attempts", 0),
        n_plan_rejections=sum(1 for x in _validation_entries(log) if x.get("problems")),
        wall_time_s=log.get("wall_time_s", 0.0),
    )

    if log.get("status") in NO_ANSWER_STATUSES:
        v.reason = log.get("error") or f"run ended as {log.get('status')}"
        v.flags = diagnose(log, entry, None, answer_type, n_buses)
        return v

    parsed = log.get("parsed_answer")
    if not isinstance(parsed, dict) or "answer" not in parsed:
        v.reason = "response did not contain a JSON object with an 'answer' field"
        v.flags = diagnose(log, entry, None, answer_type, n_buses)
        return v

    v.submitted = parsed.get("answer")
    if v.submitted is None:
        # An explicit null is no answer at all, unlike a present-but-unusable value.
        v.reason = "the answer field was null"
        v.flags = diagnose(log, entry, None, answer_type, n_buses)
        return v

    normalized, err = normalize(v.submitted, answer_type)
    v.normalized = normalized
    if err:
        # The run finished and DID answer; the value is just not a usable <answer_type>,
        # e.g. a surviving fraction returned for a "which line" question. That is a wrong
        # answer, not a missing one. Bucketing it FAILED would hide it in the same column
        # as crashes and understate exactly the silent-failure rate this project measures.
        v.bucket = WRONG_ANSWER
        v.reason = f"answered {v.submitted!r}, which is not a usable {answer_type}: {err}"
        v.flags = diagnose(log, entry, None, answer_type, n_buses) + ["wrong_answer_type"]
        return v

    v.flags = diagnose(log, entry, normalized, answer_type, n_buses)
    if matches(normalized, entry["value"], answer_type, entry.get("tolerance")):
        v.bucket = CORRECT
        v.reason = "matched gold"
    else:
        v.bucket = WRONG_ANSWER
        v.reason = f"ran clean but answered {normalized!r}, gold {entry['value']!r}"
    return v


def grade_all(logs: list[dict], gold: dict) -> list[Verdict]:
    return [grade(log, gold) for log in logs]


def summarize(verdicts: list[Verdict]) -> dict:
    """Counts by condition, and by condition x category. Pure aggregation."""
    out: dict[str, Any] = {"by_condition": {}, "by_condition_category": {}, "flags": {}}
    for v in verdicts:
        c = out["by_condition"].setdefault(
            v.condition, {CORRECT: 0, WRONG_ANSWER: 0, FAILED: 0, "n": 0, "tokens": 0})
        c[v.bucket] += 1
        c["n"] += 1
        c["tokens"] += v.total_tokens

        k = (v.condition, v.category)
        cc = out["by_condition_category"].setdefault(
            f"{v.condition}|{v.category}", {CORRECT: 0, WRONG_ANSWER: 0, FAILED: 0, "n": 0})
        cc[v.bucket] += 1
        cc["n"] += 1

        for f in v.flags:
            out["flags"][f] = out["flags"].get(f, 0) + 1
    return out


def load_gold(path: Path) -> dict:
    return json.loads(Path(path).read_text())
