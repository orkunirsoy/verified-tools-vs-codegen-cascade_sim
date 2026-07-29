"""The five experimental conditions.

Three are Arm A and two are Arm B, each single-arm, which avoids a cross-product while
still separating the action format from the scaffolding around it.

If a run has to be cut for time, the drop order is A-zeroshot first (cheapest finding).
B-hints and B-bare are protected: without that pair, an Arm A win cannot be attributed
to the action format rather than to the hand-written knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .arms import arm_a, arm_b


@dataclass(frozen=True)
class Condition:
    name: str
    arm: str
    tests: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self, **call_kwargs):
        fn = arm_a.run if self.arm == "A" else arm_b.run
        return fn(condition=self.name, **{**self.kwargs, **call_kwargs})


CONDITIONS: dict[str, Condition] = {
    "A-full": Condition(
        "A-full", "A", "few-shot planning + validation (the AAG position)",
        {"few_shot": True, "validate": True}),
    "A-noval": Condition(
        "A-noval", "A", "validation off: how much reliability comes from validation "
                        "rather than from the tool format",
        {"few_shot": True, "validate": False}),
    "A-zeroshot": Condition(
        "A-zeroshot", "A", "zero-shot planning, validation on: what the examples buy",
        {"few_shot": False, "validate": True}),
    "B-hints": Condition(
        "B-hints", "B", "CodeAct plus a NetworkX API cheat-sheet",
        {"hints": True}),
    "B-bare": Condition(
        "B-bare", "B", "CodeAct with nothing",
        {"hints": False}),
}

DEFAULT_ORDER = ["A-full", "A-noval", "A-zeroshot", "B-hints", "B-bare"]


def get(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; available: {DEFAULT_ORDER}")
    return CONDITIONS[name]
