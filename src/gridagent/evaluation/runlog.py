"""Run logging: capture everything raw, derive labels later.

DESIGN RULE: LOG RAW, DERIVE LABELS
-----------------------------------
The grader is a pure function over these logs, so it can be rerun for free as many
times as we change our mind about how to classify a failure. Bucket assignments and
any later taxonomy are DERIVED from a log, never stored in it as ground truth.

That inverts what is expensive. Labels cost a function call to recompute; a raw trace
costs an API call to regenerate, times 165 runs. So the irreversible decision is what
the log captures, not how we classify it afterwards.

Two fields carry more weight than they look:

  * Tool return values (Arm A). "Arm A answered 0.99" is two opposite findings
    depending on whether the tool returned 0.99 and the model reported it faithfully,
    or the tool returned 0.186 and the model garbled it. Only the logged return value
    separates them, and that distinction is a contribution of the project.

  * Raw response text alongside the parsed answer. If the JSON parser misfires on a
    run that answered correctly in prose and we kept only the parse result, that run
    is permanently misfiled as a failure. Keeping both makes parsing re-runnable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class LLMCall:
    """One request/response pair, including the ones that failed or were retried."""

    step: str  # "plan", "report", "codegen", ...
    attempt: int
    messages: list[dict[str, str]]  # exactly what was sent
    response_text: str  # exactly what came back
    prompt_tokens: int = 0
    completion_tokens: int = 0  # INCLUDES reasoning_tokens; that is how the API bills it
    reasoning_tokens: int = 0  # hidden thinking tokens; 0 for non-reasoning models
    latency_s: float = 0.0
    error: str | None = None  # transport/API error, if the call itself failed


@dataclass
class ToolCall:
    """One Arm A tool invocation and what it returned."""

    name: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class CodeAttempt:
    """One Arm B generate-execute cycle."""

    attempt: int
    code: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    session_lost: bool = False
    grid_mutated: bool = False


@dataclass
class RunLog:
    """Everything one (condition, question, run) produced.

    `status` records how the run ENDED, not whether it was right. Correctness is the
    grader's job and is derived later from `parsed_answer` plus the gold key.
    """

    condition: str
    question_id: str
    run_index: int
    model: str
    temperature: float
    case: str
    question_text: str  # the fully assembled text, context blocks included
    context_blocks: list[str] = field(default_factory=list)

    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    code_attempts: list[CodeAttempt] = field(default_factory=list)
    plan: Any = None  # Arm A: the plan as the model produced it
    validation: Any = None  # Arm A: validator verdict, including rejections

    raw_final_response: str = ""
    parsed_answer: Any = None
    parse_error: str | None = None

    status: str = "incomplete"  # completed | crashed | timeout | max_attempts | api_error
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    wall_time_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    # ---------- accumulation ----------

    def add_llm_call(self, call: LLMCall) -> LLMCall:
        self.llm_calls.append(call)
        return call

    def add_tool_call(self, call: ToolCall) -> ToolCall:
        self.tool_calls.append(call)
        return call

    def add_code_attempt(self, attempt: CodeAttempt) -> CodeAttempt:
        self.code_attempts.append(attempt)
        return attempt

    # ---------- derived accounting ----------

    @property
    def prompt_tokens(self) -> int:
        """Includes retries and failed attempts. Excluding them would flatter Arm B."""
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        """Billed output tokens. For a reasoning model this INCLUDES hidden thinking."""
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def reasoning_tokens(self) -> int:
        """Hidden thinking tokens. 0 for non-reasoning models."""
        return sum(c.reasoning_tokens for c in self.llm_calls)

    @property
    def visible_completion_tokens(self) -> int:
        """Output minus hidden reasoning.

        Use THIS when comparing verbosity across model families. `completion_tokens` is
        the right number for cost, but a reasoning model's total is dominated by tokens
        the agent never saw, so comparing it to a non-reasoning model's output measures
        the model family rather than the arm.
        """
        return max(0, self.completion_tokens - self.reasoning_tokens)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def n_llm_calls(self) -> int:
        return len(self.llm_calls)

    def cost_usd(self, price_per_1m_in: float, price_per_1m_out: float) -> float:
        """Dollars, the number a practitioner actually decides on."""
        return (self.prompt_tokens * price_per_1m_in
                + self.completion_tokens * price_per_1m_out) / 1_000_000

    def finish(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.wall_time_s = time.time() - self.started_at

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["totals"] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "visible_completion_tokens": self.visible_completion_tokens,
            "total_tokens": self.total_tokens,
            "n_llm_calls": self.n_llm_calls,
            "n_tool_calls": len(self.tool_calls),
            "n_code_attempts": len(self.code_attempts),
        }
        return d

    def write(self, results_dir: Path) -> Path:
        """One file per run: results/<condition>/<question_id>_run<N>.json."""
        out = Path(results_dir) / self.condition
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.question_id}_run{self.run_index}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return path


def load_run(path: Path) -> dict:
    """Read a run log back. The grader consumes these, never live objects."""
    return json.loads(Path(path).read_text())


def load_all_runs(results_dir: Path) -> list[dict]:
    """Every run log under results/, sorted for deterministic reporting."""
    return [load_run(p) for p in sorted(Path(results_dir).rglob("*_run*.json"))]
