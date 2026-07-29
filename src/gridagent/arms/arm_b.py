"""Arm B: the code-generation agent (the CodeAct position).

The model writes Python, we run it, it sees stdout or the traceback, and it revises.
There is no menu of functions: if the question needs a cascade, the model writes the
cascade. That is the whole claim under test, since writing the redistribution logic is
exactly where a plausible-but-wrong answer comes from.

Two conditions share this implementation:

  B-hints  a NetworkX API cheat-sheet in the prompt
  B-bare   nothing

That pair is load-bearing. Without it, any Arm A win could be attributed to the
hand-written knowledge base rather than to the action format, and the headline claim
would not survive a reviewer.

The cheat-sheet is deliberately API syntax only. It never states the cascade model --
that lives in the shared question context both arms receive -- or the ablation would be
measuring two things at once.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import networkx as nx

from ..client import LLMClient
from ..evaluation.runlog import CodeAttempt, LLMCall, RunLog
from ..questions import ANSWER_SCHEMA, Question, QuestionSet, assemble_question, grid_description
from . import sandbox
from .arm_a import parse_json_object

MAX_CODE_ATTEMPTS = 4

SYSTEM = """You are a data analyst who answers questions about a power grid by writing \
and running Python.

You send one snippet at a time. It is executed and you see its stdout and stderr. If it \
fails, or the output is not what you needed, you send another.

THE SESSION IS PERSISTENT. All your snippets run in one interpreter, so variables, \
imports and functions you define in one snippet are still there in the next. Build up \
your analysis across turns instead of repeating yourself.

The environment already provides:
  * `networkx` imported as `nx`
  * `G`, the grid as an undirected networkx Graph with integer bus ids

Rules:
  * Reply with ONE Python code block and nothing else, until you are ready to answer.
  * `print()` whatever you need to see. You cannot inspect variables any other way.
  * `G` persists across snippets. If you change it, it stays changed; copy it first
    (`H = G.copy()`) if you want the original back.
  * Each snippet is killed after {timeout}s, so keep an eye on cost. A full sweep over
    every bus or every line does fit, but recomputing more than you need may not.
  * You have at most {max_attempts} snippets before the run ends, so plan accordingly.
  * There is no network access.
  * Write the analysis yourself. There are no pre-built cascade helpers.

When you have the final answer, reply with the JSON object described below INSTEAD of \
code, and nothing else. Do not send code and an answer in the same reply: a reply \
containing code is treated as code, and its answer will be ignored.

{schema}"""

HINTS = """NetworkX API reference (syntax only):

  nx.betweenness_centrality(G, normalized=False) -> dict {node: betweenness}
      Counts shortest paths through each node. On a disconnected graph it counts only
      within-component pairs.
  nx.connected_components(G) -> iterator of sets of nodes
  max(nx.connected_components(G), key=len) -> the largest component
  G.copy(), G.remove_node(n), G.remove_nodes_from(iterable)
  G.remove_edge(u, v), G.add_edge(u, v), G.has_edge(u, v)
  G.degree(n) -> int, dict(G.degree()) -> {node: degree}
  G.nodes(), G.edges(), G.number_of_nodes(), G.number_of_edges()
  nx.articulation_points(G), nx.bridges(G) -> iterators
"""

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _is_bare_json_object(text: str) -> bool:
    """Is the response nothing but a JSON object?"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        ast.literal_eval(t)
        return True
    except (ValueError, SyntaxError):
        return False


def extract_code(text: str) -> str | None:
    """Pull Python out of a response. None means the model really wrote no code.

    Fenced blocks are the happy path, but models frequently emit bare Python with no
    fences. An earlier fenced-only version silently treated those turns as final answers,
    so Arm B "answered" questions whose code was never executed and reported numbers it
    had hallucinated. Anything that parses as Python and is not just a JSON object is
    therefore treated as code.
    """
    m = CODE_BLOCK.search(text or "")
    if m:
        return m.group(1).strip()

    body = (text or "").strip()
    if not body or _is_bare_json_object(body):
        return None
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    # A lone dict/string/number expression is prose or JSON, not a program.
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        if isinstance(tree.body[0].value, (ast.Dict, ast.Constant)):
            return None
    return body


def looks_like_answer(text: str) -> bool:
    """A response is an ANSWER only if it contains no code at all.

    Strict on purpose: a reply carrying both a snippet and a JSON object is a code turn
    whose author got ahead of itself. Accepting it as an answer is how a run ends up
    reporting the output of code that never ran.
    """
    if extract_code(text) is not None:
        return False
    parsed, _ = parse_json_object(text)
    return isinstance(parsed, dict) and "answer" in parsed


def run(
    G: nx.Graph,
    question: Question,
    qs: QuestionSet,
    client: LLMClient,
    condition: str,
    run_index: int = 1,
    temperature: float = 0.0,
    hints: bool = True,
    max_attempts: int = MAX_CODE_ATTEMPTS,
    timeout_s: int = sandbox.DEFAULT_TIMEOUT_S,
) -> RunLog:
    """Generate, execute, observe, revise, until an answer or the attempt budget."""
    text = assemble_question(question, qs)
    log = RunLog(
        condition=condition, question_id=question.id, run_index=run_index,
        model=client.model, temperature=temperature, case=G.graph.get("case", "?"),
        question_text=text, context_blocks=list(question.context),
    )
    log.notes["arm"] = "B"
    log.notes["hints"] = hints
    log.notes["max_attempts"] = max_attempts

    system = SYSTEM.format(schema=ANSWER_SCHEMA, timeout=timeout_s,
                           max_attempts=max_attempts)
    user = f"{grid_description(G)}\n\n{text}"
    if hints:
        user = f"{HINTS}\n{user}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    nudged_to_execute = False

    # One session for the whole run: variables, imports and helper functions persist
    # across turns the way they do in CodeAct's interpreter.
    with sandbox.Session(G, timeout_s=timeout_s) as session:
        for attempt in range(1, max_attempts + 1):
            comp = client.complete(messages, temperature=temperature)
            log.add_llm_call(LLMCall(
                step="codegen", attempt=attempt, messages=list(messages),
                response_text=comp.text, prompt_tokens=comp.prompt_tokens,
                completion_tokens=comp.completion_tokens,
            reasoning_tokens=comp.reasoning_tokens, latency_s=comp.latency_s,
                error=comp.error,
            ))
            if comp.error:
                log.finish("api_error", comp.error)
                return log

            if looks_like_answer(comp.text):
                # An answer before anything has run is a claim about output that does not
                # exist. Push back once; if it insists, accept it but mark the run so the
                # grader can separate "computed and wrong" from "never computed".
                if not log.code_attempts and not nudged_to_execute:
                    nudged_to_execute = True
                    messages = messages + [
                        {"role": "assistant", "content": comp.text},
                        {"role": "user", "content":
                            "You have not run any code yet, so that answer is not based on "
                            "any executed result. Send one Python snippet that computes it "
                            "and prints what you need to see."},
                    ]
                    continue
                log.raw_final_response = comp.text
                parsed, err = parse_json_object(comp.text)
                log.parsed_answer, log.parse_error = parsed, err
                log.notes["answered_without_executing"] = not log.code_attempts
                log.finish("completed")
                return log

            code = extract_code(comp.text)
            if code is None:
                # Neither code nor an answer. Nudge rather than burning the budget.
                messages = messages + [
                    {"role": "assistant", "content": comp.text},
                    {"role": "user", "content": "Reply with either one Python code block or "
                                                "the final JSON answer object."},
                ]
                continue

            result = session.run(code)
            log.add_code_attempt(CodeAttempt(
                attempt=attempt, code=code, stdout=result.stdout, stderr=result.stderr,
                exit_code=result.exit_code, timed_out=result.timed_out,
                duration_s=result.duration_s, session_lost=result.session_lost,
                grid_mutated=result.grid_mutated,
            ))

            observation = f"stdout:\n{result.stdout or '(empty)'}"
            if result.stderr:
                observation += f"\n\nstderr:\n{result.stderr}"
            if result.timed_out:
                observation += (f"\n\nThe snippet was killed after {timeout_s}s. Make it "
                                f"faster before trying again.")
            if result.session_lost:
                observation += ("\n\nNOTE: the interpreter was restarted, so variables and "
                                "functions defined earlier are gone. Redefine anything you "
                                "still need.")
            messages = messages + [
                {"role": "assistant", "content": comp.text},
                {"role": "user", "content": observation + "\n\nContinue, or give the final "
                                                          "JSON answer if you have it."},
            ]

    log.notes["answered_without_executing"] = not log.code_attempts
    log.finish("max_attempts", f"no answer after {max_attempts} attempts")
    return log
