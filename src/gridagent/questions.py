"""Question loading and prompt assembly.

The `context` list on each question names shared definition blocks that must be
prepended to it. Both arms get byte-identical text, so no definition can bias the
comparison. This module is the only place that assembly happens.

It is also the thing the smoke test must check first: if a context block is silently
dropped, the agent is graded on guessing a convention it was never given, and 165 runs
produce garbage that looks fine. `assemble_question` therefore fails loudly on an
unknown context name rather than skipping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT / "questions" / "questions.yaml"
GOLD_PATH = ROOT / "gold" / "gold_answers.json"

#: Every arm emits this, so the grader never has to parse prose. Sharing it between
#: arms keeps the comparison fair: neither is advantaged by output formatting.
ANSWER_SCHEMA = """Respond with a single JSON object and nothing else:

{
  "answer": <the answer, as a bare number, a bus id, a list of bus ids, or a
             two-element list [busA, busB] for a power line>,
  "units": "<fraction | count | bus | bus_list | line>",
  "reasoning": "<two or three sentences on how you got it>"
}

Rules for `answer`:
  * A fraction is a decimal between 0 and 1, not a percentage.
  * A bus is a bare integer, e.g. 68 (not "bus 68").
  * An ordered list of buses is a JSON list, most important first.
  * A power line is a two-element list of bus ids, e.g. [79, 80].
Do not wrap the JSON in markdown fences. Do not add commentary outside the JSON."""


@dataclass
class Question:
    id: str
    category: str
    context: list[str]
    text: str  # raw text from the YAML, before assembly
    answer_type: str
    tolerance: float | None = None
    notes: str = ""
    #: "development" (q01-q11, used to build and debug the harness) or "test" (t01-t04, held out).
    #: Reporting the two together would overstate Arm A, whose validator guards were each
    #: written against a specific dev-question failure. See the header of questions.yaml.
    split: str = "development"


@dataclass
class QuestionSet:
    case: str
    definitions: dict[str, str]
    questions: list[Question]

    def by_id(self, qid: str) -> Question:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(f"no question {qid!r}; have {[q.id for q in self.questions]}")

    @property
    def ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def split(self, name: str) -> list[Question]:
        """Questions in one split. Use this rather than slicing ids by prefix."""
        return [q for q in self.questions if q.split == name]

    def split_ids(self, name: str) -> list[str]:
        return [q.id for q in self.split(name)]


def load_questions(path: Path = QUESTIONS_PATH) -> QuestionSet:
    spec = yaml.safe_load(Path(path).read_text())
    definitions = spec.get("definitions", {})
    questions = []
    for q in spec["questions"]:
        unknown = [c for c in q.get("context", []) if c not in definitions]
        if unknown:
            raise ValueError(
                f"{q['id']}: unknown context {unknown}; defined: {sorted(definitions)}"
            )
        split = q.get("split", "development")
        if split == "validation":
            split = "development"  # legacy alias
        if split not in {"development", "test"}:
            raise ValueError(
                f"{q['id']}: split must be 'development' or 'test', got {split!r}")
        questions.append(Question(
            id=q["id"],
            category=q["category"],
            context=list(q.get("context", [])),
            text=" ".join(q["question"].split()),
            answer_type=q["answer_type"],
            tolerance=q.get("tolerance"),
            notes=q.get("notes", ""),
            split=split,
        ))
    return QuestionSet(case=spec["case"], definitions=definitions, questions=questions)


def assemble_question(q: Question, qs: QuestionSet) -> str:
    """The full question text an arm receives: definition blocks, then the question.

    Order is fixed (the order the blocks are declared in the YAML, not the order they
    appear in `context`) so that two runs of the same question are byte-identical.
    """
    ordered = [name for name in qs.definitions if name in q.context]
    missing = set(q.context) - set(ordered)
    if missing:
        raise ValueError(f"{q.id}: context {sorted(missing)} not in definitions")

    parts = [qs.definitions[name].rstrip() for name in ordered]
    parts.append(f"QUESTION: {q.text}")
    return "\n\n".join(parts)


def grid_description(G) -> str:
    """What every arm is told about the grid itself. Identical for both."""
    return (
        f"The grid is an undirected graph with {G.number_of_nodes()} buses (nodes) and "
        f"{G.number_of_edges()} power lines (edges). Bus ids are integers. Parallel "
        f"branches between the same pair of buses are collapsed into a single edge."
    )
