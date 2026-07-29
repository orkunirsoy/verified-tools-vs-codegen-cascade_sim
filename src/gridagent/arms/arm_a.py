"""Arm A: the verified-tool agent (the AAG position).

The model never computes. It plans a small dependency graph of KB tool calls, a
validator checks the plan before anything runs, our code executes the tools, and the
model writes the final answer from the summarized results.

Three conditions share this implementation:

  A-full      few-shot planning + validation      (the AAG position as stated)
  A-noval     few-shot planning, no validation    (how much reliability is validation?)
  A-zeroshot  zero-shot planning + validation     (what do the examples buy?)

The residual failure modes are the interesting ones, and each is logged distinctly:
picking the wrong tool, passing wrong arguments, and -- the one worth reporting --
misreporting a number the tool returned correctly.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import networkx as nx

from ..client import LLMClient
from ..evaluation.runlog import LLMCall, RunLog, ToolCall
from ..questions import ANSWER_SCHEMA, Question, QuestionSet, assemble_question, grid_description
from ..tools import kb as KBMOD

MAX_PLAN_REPAIRS = 2

PLANNER_SYSTEM = """You are a planning component in a power-grid analysis system.

You do NOT perform any analysis yourself. You choose which verified analysis functions \
to call, in what order, and with what arguments. Another component executes them.

Available functions:
{catalogue}

Produce a plan as a single JSON object and nothing else:

{{"steps": [{{"id": "s1", "tool": "<name>", "args": {{...}}}}, ...]}}

REFERENCES. A later step may use an earlier step's output by passing the string \
"$<step_id>.<field>" as an argument value. "$s1.busiest_bus" is the busiest_bus field \
returned by step s1. Dotted paths reach nested fields. If a reference already points at a \
list, pass it on its own -- do NOT wrap it in another list. Wrap only a single value that \
is going into an argument expecting a list.

REPEATING A STEP. Add "foreach" to a step to run it once per element of a list. The list \
is either a reference to an earlier step's output ("$s1.buses") or, when the question \
names the candidates itself, a literal list written straight into the plan (for example \
[[31, 81], [48, 67]]). Inside that step's args, "$item" is the current element. The step's \
result is {{"results": [...]}}, one record per element, and each record also carries the \
element under the key "item". For example, to run an analysis over every bus:

  {{"id": "s1", "tool": "list_buses", "args": {{}}}},
  {{"id": "s2", "tool": "simulate_cascade", "foreach": "$s1.buses",
    "args": {{"trigger_buses": ["$item"], "alpha": 0.2}}}}

There is no function that sweeps or ranks for you. To find a best or worst case you must \
do four things: repeat the analysis over the candidates, pull out the column you care \
about, take its argmin or argmax, and then look up WHICH candidate that position refers \
to. argmin and argmax return a position, not a candidate, so without that last lookup you \
only know how good the winner was, not which one it is. Choosing which column and which \
direction is your decision.

WHERE THE ANSWER COMES FROM. Every step's full output is handed to the reporter, so you never need a step just to read a value out of a result. If the analysis step already produces the number asked for, the plan ENDS there. column/argmin/argmax/at exist only for picking a winner out of a repeated step's `results` list, never for reading a field off a single result.

Keep the plan minimal. Do not add steps whose results you would not use."""

#: Worked examples of the PLAN LANGUAGE, not of the question set.
#:
#: Both examples are held to one rule: an example must not be a graded question, and must
#: not become one by substituting constants. An earlier version violated this outright --
#: Example 1 was q03 verbatim together with its correct plan, which also handed q04, q07
#: and q10 their shared stem. Arm B receives no worked examples in either condition, so a
#: solved instance here is not a prompt-design choice, it is contamination.
#:
#: What the examples legitimately teach is mechanics the planner cannot infer from the
#: catalogue alone: how $step.field references work, how `foreach` binds "$item", and when
#: a plan should STOP. Example 1 therefore triggers an explicitly named bus at a
#: non-default alpha, and Example 2 sweeps for a quantity (fewest overloads) that no
#: graded question asks for. Choosing the column and the direction stays the planner's job.
FEWSHOT = """Example 1.
QUESTION: Take bus 12 as the one that fails, with the tolerance parameter set to 0.5 \
rather than 0.2. How big is the component left standing, relative to the whole?
PLAN: {"steps": [
  {"id": "s1", "tool": "simulate_cascade", "args": {"trigger_buses": [12], "alpha": 0.5}}
]}

The plan ENDS at s1. lcc_fraction is already part of s1's result, and the reporter is \
given every step's full output, so no further step is needed to read it out.

Example 2.
QUESTION: Taking the buses one by one as the thing that breaks, find the one whose \
failure pushes the fewest others past capacity.
PLAN: {"steps": [
  {"id": "s1", "tool": "list_buses", "args": {}},
  {"id": "s2", "tool": "simulate_cascade", "foreach": "$s1.buses", "args": {"trigger_buses": ["$item"], "alpha": 0.2}},
  {"id": "s3", "tool": "column", "args": {"records": "$s2.results", "field": "n_overloaded"}},
  {"id": "s4", "tool": "argmin", "args": {"values": "$s3.values"}},
  {"id": "s5", "tool": "at", "args": {"records": "$s2.results", "index": "$s4.index"}}
]}

Note in Example 2 that finding a best or worst case takes several steps: build the \
candidate list, repeat the analysis over it, pull out the column that matters, take its \
argmin or argmax, then look up which candidate that was. No single function does this \
for you. Which column and which direction depend on the question and are your decision."""

REPORTER_SYSTEM = """You are the reporting component of a power-grid analysis system.

The analysis has already been performed by verified functions. Their results are given \
to you below. Your job is to read the answer off those results and report it.

Do NOT recompute anything. Do NOT second-guess the numbers. If the results contain the \
answer, report exactly that value.

{schema}"""


# --------------------------------------------------------------------------
# plan parsing and validation
# --------------------------------------------------------------------------


def parse_json_object(text: str) -> tuple[Any, str | None]:
    """Pull one JSON object out of a model response, fences or not."""
    if not text or not text.strip():
        return None, "empty response"
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1]), None
                except json.JSONDecodeError:
                    start = None
    return None, "no parseable JSON object in response"


#: A reference to an earlier step's output. Dotted paths are allowed so a nested field
#: can be reached, e.g. "$s1.smallest_surviving_network.bus". Single-level-only matching
#: would reject that as malformed and force the planner into shapes the KB cannot serve.
_REF = re.compile(r"^\$(\w+)((?:\.\w+)+)$")

#: Inside a `foreach` step, this stands for the current element. It deliberately has no
#: dot, so it can never collide with a $step.field reference.
ITEM = "$item"

#: Reserved key under which the executor stashes the current element while resolving a
#: repeated step's arguments. Not a step id, so a plan cannot reference it directly.
_ITEM_KEY = "__item__"


def validate_plan(plan: Any) -> list[str]:
    """Structural check of a plan. Returns a list of problems; empty means valid.

    Catches what an executor would otherwise hit as a runtime crash: unknown tools,
    unknown or missing arguments, references to steps that do not exist or have not run
    yet. This is the component `A-noval` switches off.
    """
    problems: list[str] = []
    if not isinstance(plan, dict) or "steps" not in plan:
        return ["plan is not an object with a 'steps' key"]
    steps = plan["steps"]
    if not isinstance(steps, list) or not steps:
        return ["'steps' must be a non-empty list"]

    # step id -> the fields that step's output actually exposes. The KB declares every
    # tool's return schema, so a reference to a field that does not exist is catchable
    # before execution instead of surfacing as a KeyError mid-plan.
    seen: set[str] = set()
    fields_of: dict[str, set[str]] = {}
    lists_of: dict[str, set[str]] = {}
    for i, step in enumerate(steps):
        where = f"step {i + 1}"
        if not isinstance(step, dict):
            problems.append(f"{where}: not an object")
            continue
        sid = step.get("id")
        if not sid:
            problems.append(f"{where}: missing 'id'")
        elif sid in seen:
            problems.append(f"{where}: duplicate id {sid!r}")
        name = step.get("tool")
        if name not in KBMOD.KB:
            problems.append(f"{where}: unknown tool {name!r}; available: {sorted(KBMOD.KB)}")
            if sid:
                seen.add(sid)
            continue
        tool = KBMOD.KB[name]
        args = step.get("args", {})
        if not isinstance(args, dict):
            problems.append(f"{where}: 'args' must be an object")
            args = {}
        # A reference to a list field wrapped in another list produces [[...]], which
        # only surfaces deep inside the tool as a confusing KeyError.
        for arg_name, arg_val in args.items():
            if (isinstance(arg_val, list) and len(arg_val) == 1
                    and isinstance(arg_val[0], str) and _REF.match(arg_val[0])):
                m2 = _REF.match(arg_val[0])
                src2, f2 = m2.group(1), m2.group(2).lstrip(".").split(".")[0]
                if f2 in lists_of.get(src2, set()):
                    problems.append(
                        f"{where} ({name}): {arg_name} is {arg_val!r}, but "
                        f"{arg_val[0]} is already a list. Pass it directly as "
                        f"{arg_val[0]!r}, not wrapped in another list.")

        for missing in [a for a in tool.required if a not in args]:
            problems.append(f"{where} ({name}): missing required argument {missing!r}")
        for unknown in [a for a in args if a not in tool.args]:
            problems.append(
                f"{where} ({name}): unknown argument {unknown!r}; accepts {sorted(tool.args)}")
        fe = step.get("foreach")
        if fe is not None:
            literal = coerce_foreach(fe)
            if literal is not None:
                # A literal candidate list is legitimate: several questions name the
                # candidates outright, and no step produces them.
                if not literal:
                    problems.append(f"{where}: 'foreach' list is empty")
                # The literal twin of the reference double-wrap check above. A candidate
                # list wrapped in one more list iterates ONCE with the whole list bound to
                # "$item", which then reaches the tool as a single argument and dies as an
                # unpack error from deep inside it. The earlier double-wrap fix only
                # covered references, so this shape validated and crashed at execution.
                # One element that is itself a list OF LISTS is the signature: a genuine
                # single candidate like [[31, 81]] holds a list of ints, not of lists.
                elif (len(literal) == 1 and isinstance(literal[0], list) and literal[0]
                        and all(isinstance(x, list) for x in literal[0])):
                    problems.append(
                        f"{where}: 'foreach' is {json.dumps(literal)}, which is one "
                        f"candidate wrapped in an extra list, so the step would run once "
                        f"with every candidate at once. Pass the candidates directly as "
                        f"{json.dumps(literal[0])}.")
            elif isinstance(fe, str) and _REF.match(fe):
                if _REF.match(fe).group(1) not in seen:
                    problems.append(
                        f"{where}: foreach {fe!r} points to a step that has not run")
            else:
                problems.append(f"{where}: 'foreach' must be a reference like $s1.buses "
                                f"or a literal list of candidates, got {fe!r}")
            if not any(v == ITEM for v in _iter_refs(args)):
                problems.append(f"{where}: has 'foreach' but never uses \"$item\" in args, "
                                f"so every repetition would be identical")
        for val in _iter_refs(args):
            if val == ITEM:
                if fe is None:
                    problems.append(f"{where}: uses \"$item\" without a 'foreach'")
                continue
            m = _REF.match(val)
            if not m:
                problems.append(f"{where}: malformed reference {val!r}, expected $step.field")
            elif m.group(1) not in seen:
                problems.append(f"{where}: reference {val!r} points to a step that has not run")
            else:
                src, first = m.group(1), m.group(2).lstrip(".").split(".")[0]
                known = fields_of.get(src)
                if known and first not in known:
                    problems.append(
                        f"{where}: {val!r} does not exist. Step {src} produces a single "
                        f"result, not a list. If you are only trying to report a value "
                        f"that step {src} already computed, DELETE this step: the "
                        f"reporter is given every step's full output.")
                elif known and first in known:
                    # A list-typed argument fed a scalar field. Without this the plan
                    # validates and dies mid-execution, and the repair message above
                    # reads as an invitation to substitute another scalar.
                    arg_name = _arg_of(args, val)
                    if arg_name in tool.list_args and first not in lists_of.get(src, set()):
                        problems.append(
                            f"{where} ({name}): argument {arg_name!r} needs a list, but "
                            f"{val!r} is a single value. Only a repeated (foreach) step "
                            f"produces a list. If you just want to report that value, "
                            f"DELETE this step.")
        if sid:
            seen.add(sid)
            # A repeated step returns the collected rows, not the tool's own fields.
            if step.get("foreach") is not None:
                fields_of[sid] = {"results", "n_results"}
                lists_of[sid] = {"results"}
            else:
                fields_of[sid] = set(tool.returns)
                lists_of[sid] = set(tool.list_returns)

    if name := _trigger_conflict(steps):
        problems.append(name)
    return problems


def coerce_foreach(fe: Any) -> list | None:
    """The candidate list for a repeated step, or None if this is not one.

    Accepts a real JSON array and an array serialised as a string. Models routinely emit
    "[[31,81],[48,67]]" with quotes, and the intent is unambiguous; rejecting it made a
    correct plan unrunnable and the error message never mentioned the quoting.
    """
    if isinstance(fe, list):
        return fe
    if isinstance(fe, str) and fe.strip().startswith("["):
        try:
            parsed = json.loads(fe)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _arg_of(args: dict, target: str) -> str | None:
    """Which top-level argument does this reference sit under?"""
    for k, v in args.items():
        if target in list(_iter_refs(v)):
            return k
    return None


def _iter_refs(value: Any):
    if isinstance(value, str) and value.startswith("$"):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from _iter_refs(v)


def _trigger_conflict(steps: list) -> str | None:
    for i, s in enumerate(steps):
        if isinstance(s, dict) and s.get("tool") == "simulate_cascade":
            args = s.get("args", {}) or {}
            has_bus, has_line = "trigger_buses" in args, "trigger_line" in args
            if has_bus == has_line:  # foreach binds $item into one of them, not both
                return (f"step {i + 1} (simulate_cascade): give exactly one of "
                        f"trigger_buses or trigger_line")
    return None


def resolve(value: Any, outputs: dict[str, Any]) -> Any:
    """Substitute $step.field references with values from earlier steps."""
    if isinstance(value, str):
        if value == ITEM:
            if _ITEM_KEY not in outputs:
                raise KeyError('"$item" used outside a foreach step')
            return outputs[_ITEM_KEY]
        m = _REF.match(value)
        if m:
            sid, path = m.groups()
            if sid not in outputs:
                raise KeyError(f"reference to unknown step {sid!r}")
            current: Any = outputs[sid]
            walked = f"${sid}"
            for field in path.lstrip(".").split("."):
                if not isinstance(current, dict) or field not in current:
                    available = (sorted(current) if isinstance(current, dict)
                                 else type(current).__name__)
                    raise KeyError(f"{walked} has no field {field!r}; has {available}")
                current = current[field]
                walked += f".{field}"
            return current
        return value
    if isinstance(value, dict):
        return {k: resolve(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, outputs) for v in value]
    return value


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------


def run(
    G: nx.Graph,
    question: Question,
    qs: QuestionSet,
    client: LLMClient,
    condition: str,
    run_index: int = 1,
    temperature: float = 0.0,
    few_shot: bool = True,
    validate: bool = True,
) -> RunLog:
    """Plan, validate, execute, report. Everything is logged, including rejections."""
    text = assemble_question(question, qs)
    log = RunLog(
        condition=condition, question_id=question.id, run_index=run_index,
        model=client.model, temperature=temperature, case=G.graph.get("case", "?"),
        question_text=text, context_blocks=list(question.context),
    )
    log.notes["arm"] = "A"
    log.notes["few_shot"] = few_shot
    log.notes["validation"] = validate

    system = PLANNER_SYSTEM.format(catalogue=KBMOD.catalogue())
    user = f"{grid_description(G)}\n\n{text}"
    if few_shot:
        user = f"{FEWSHOT}\n\nNow the real question.\n\n{user}"

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    plan, problems = None, ["no plan produced"]
    for attempt in range(1, MAX_PLAN_REPAIRS + 2):
        comp = client.complete(messages, temperature=temperature)
        log.add_llm_call(LLMCall(
            step="plan", attempt=attempt, messages=list(messages),
            response_text=comp.text, prompt_tokens=comp.prompt_tokens,
            completion_tokens=comp.completion_tokens,
            reasoning_tokens=comp.reasoning_tokens, latency_s=comp.latency_s,
            error=comp.error,
        ))
        if comp.error:
            log.finish("api_error", comp.error)
            return log

        candidate, parse_err = parse_json_object(comp.text)
        problems = [parse_err] if parse_err else (validate_plan(candidate) if validate else [])
        # Append, never overwrite: the count and content of rejections IS the measurement
        # that separates A-full from A-noval.
        log.validation = (log.validation or []) + [
            {"attempt": attempt, "problems": problems, "enabled": validate,
             "plan": candidate}
        ]
        if not problems:
            plan = candidate
            break
        # Validation off: a parse failure is still fatal, there is nothing to execute.
        if not validate and parse_err is None:
            plan = candidate
            break
        messages = messages + [
            {"role": "assistant", "content": comp.text},
            {"role": "user", "content": "That plan has problems:\n- " + "\n- ".join(problems)
                                        + "\nReturn a corrected plan as JSON only."},
        ]

    log.plan = plan
    if plan is None:
        log.finish("max_attempts", f"plan never validated: {problems}")
        return log

    # ---- execute ----
    outputs: dict[str, Any] = {}
    summaries: list[dict] = []
    for step in plan.get("steps", []):
        name = step.get("tool")
        tool = KBMOD.KB.get(name)
        t0 = time.time()
        if tool is None:
            log.add_tool_call(ToolCall(name=str(name), args=step.get("args", {}),
                                       error=f"unknown tool {name!r}",
                                       duration_s=time.time() - t0))
            log.finish("crashed", f"unknown tool {name!r}")
            return log
        try:
            raw_args = step.get("args", {}) or {}
            fe = step.get("foreach")
            if fe is None:
                args = resolve(raw_args, outputs)
                result = tool.fn(G, **args) if tool.needs_graph else tool.fn(**args)
            else:
                literal = coerce_foreach(fe)
                items = ([resolve(x, outputs) for x in literal] if literal is not None
                         else resolve(fe, outputs))
                if not isinstance(items, list) or not items:
                    raise ValueError(f"foreach {fe!r} did not resolve to a non-empty list")
                rows = []
                for item in items:
                    a = resolve(raw_args, {**outputs, _ITEM_KEY: item})
                    r = tool.fn(G, **a) if tool.needs_graph else tool.fn(**a)
                    # Tag each row with the element it came from, so a later argmin can be
                    # mapped back to the candidate that produced it.
                    rows.append({"item": item, **r} if isinstance(r, dict) else
                                {"item": item, "value": r})
                args = {"foreach": fe, "n_items": len(items), "args": raw_args}
                result = {"n_results": len(rows), "results": rows}
        except Exception as exc:
            log.add_tool_call(ToolCall(name=name, args=step.get("args", {}),
                                       error=f"{type(exc).__name__}: {exc}",
                                       duration_s=time.time() - t0))
            log.finish("crashed", f"tool {name} failed: {type(exc).__name__}: {exc}")
            return log

        # Full result goes in the log; the model only ever sees the summary. That gap
        # is what makes "the tool was right, the model misreported it" detectable.
        log.add_tool_call(ToolCall(name=name, args=args, result=result,
                                   duration_s=time.time() - t0))
        outputs[step.get("id", name)] = result
        summaries.append({"step": step.get("id"), "tool": name,
                          "args": KBMOD.compact(args),
                          "result": KBMOD.summarize(result)})

    # ---- report ----
    report_user = (
        f"{text}\n\nVerified analysis results:\n"
        f"{json.dumps(summaries, indent=2, default=str)}"
    )
    report_messages = [
        {"role": "system", "content": REPORTER_SYSTEM.format(schema=ANSWER_SCHEMA)},
        {"role": "user", "content": report_user},
    ]
    comp = client.complete(report_messages, temperature=temperature)
    log.add_llm_call(LLMCall(
        step="report", attempt=1, messages=report_messages, response_text=comp.text,
        prompt_tokens=comp.prompt_tokens, completion_tokens=comp.completion_tokens,
            reasoning_tokens=comp.reasoning_tokens,
        latency_s=comp.latency_s, error=comp.error,
    ))
    if comp.error:
        log.finish("api_error", comp.error)
        return log

    log.raw_final_response = comp.text
    parsed, err = parse_json_object(comp.text)
    log.parsed_answer, log.parse_error = parsed, err
    log.finish("completed")
    return log
