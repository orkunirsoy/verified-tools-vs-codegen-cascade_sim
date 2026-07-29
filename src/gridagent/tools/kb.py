"""The tool knowledge base Arm A plans over.

Each entry is a name, a description the planner sees, an argument schema the validator
enforces, and the callable the executor runs. The descriptions are the retrieval
surface: this is a hand-written KB, not a scraped one, which is a deliberate difference
from the AAG paper and is reported as such.

Tool outputs get SUMMARIZED before reaching the reporter. A sweep returns 118 rows;
feeding those to the model would blow the context and would also let the model redo the
analysis in prose rather than trust the tool. Summarizing is part of the AAG design, not
a workaround: the tool decides the answer, the model reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import analysis as A

# The KB holds ANALYSIS FUNCTIONS and GENERIC DATA OPERATIONS, nothing shaped like an
# answer to a graded question. AAG builds its knowledge base from library documentation,
# so the granularity here matches a library: `simulate_cascade` is an algorithm the way
# PageRank is, and column/argmin/argmax/at are the numpy-level operations their own
# prototype used alongside graph algorithms. Sweeping and ranking are NOT functions --
# repetition is a `foreach` in the plan, and choosing what to optimize is the planner's
# decision. An earlier draft had `sweep_bus_triggers` and `select_extreme`, which
# answered whole questions in one call and made the planning ablations meaningless.
#
# DESCRIPTIONS SAY WHAT A FUNCTION DOES, NEVER WHEN TO REACH FOR IT.
# An audit found the descriptions had been written with the question set open: they
# carried routing hints phrased in the questions' own words -- "use for 'most direct
# connections'" (q01 verbatim), "use to find the 'busiest' bus" (q03/q04), "the upgraded
# grid" (q11), "repeated over all buses" (q05/q06). Arm B's API reference is pure syntax
# and carries no equivalents, so those hints were an uncontrolled advantage on tool
# selection sitting inside the arm whose selection reliability is the thing being measured.
# Library documentation describes behaviour; it does not anticipate the caller's question.
# When adding a tool, write its description without the question set in front of you.

#: Keys dropped when a tool result is shown to the reporter. Each is a bulk listing
#: whose summary statistics are already present in the same result.
BULK_KEYS = {"overloaded_buses", "results", "buses", "lines", "values"}


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str]  # arg name -> human-readable type/meaning
    required: list[str]
    fn: Callable[..., Any]
    returns: dict[str, str] = None  # output field -> meaning; SHOWN to the planner
    needs_graph: bool = True  # False for pure data operations
    #: Which of `returns` are lists, and which args must be given a list. The validator
    #: needs both to catch a scalar being passed where a column is expected -- otherwise
    #: that only surfaces as a ValueError halfway through executing the plan.
    list_returns: frozenset = frozenset()
    list_args: frozenset = frozenset()

    def __post_init__(self):
        if self.returns is None:
            self.returns = {}


KB: dict[str, Tool] = {
    "grid_info": Tool(
        name="grid_info",
        description="Size and shape of the grid: number of buses, number of lines, "
                    "mean degree, whether it is connected.",
        args={},
        required=[],
        fn=A.grid_info,
        returns={"n_buses": "int", "n_lines": "int", "mean_degree": "float",
                 "is_connected": "bool"},
    ),
    "degree_ranking": Tool(
        name="degree_ranking",
        description="Rank buses by DEGREE, the number of lines directly attached to a "
                    "bus. Degree is not the same quantity as load.",
        args={"k": "int, how many to return (default 5)"},
        required=[],
        fn=A.degree_ranking,
        list_returns=frozenset({"ranking", "top_buses"}),
        returns={"top_bus": "int, the single highest-degree bus",
                 "top_buses": "list[int], the k highest-degree bus ids, highest first",
                 "ranking": "list of {bus, degree}",
                 "n_tied_at_top": "int, how many buses share the top degree"},
    ),
    "load_ranking": Tool(
        name="load_ranking",
        description="Rank buses by LOAD (betweenness centrality), highest first. "
                    "Returns the single top-ranked bus and the top k of them.",
        args={"k": "int, how many to return (default 5)"},
        required=[],
        fn=A.load_ranking,
        list_returns=frozenset({"ranking", "top_buses"}),
        returns={"busiest_bus": "int, the single highest-load bus",
                 "top_buses": "list[int], the k highest-load bus ids, highest first",
                 "ranking": "list of {bus, load}",
                 "next_load_outside_top_k": "float"},
    ),
    "simulate_cascade": Tool(
        name="simulate_cascade",
        description="Run ONE Motter-Lai overload cascade and report the outcome: "
                    "surviving fraction, buses failed, buses overloaded, rounds. "
                    "The trigger is either a list of buses that fail or a single line "
                    "that trips. An optional new line may be added before the run, in "
                    "which case the modified network is the initial one.",
        args={
            "trigger_buses": "list[int], buses that fail (give this OR trigger_line)",
            "trigger_line": "[int, int], a line that trips (give this OR trigger_buses)",
            "alpha": "float, tolerance parameter (default 0.2)",
            "added_line": "[int, int], a new line built before the cascade (optional)",
        },
        required=[],
        fn=A.simulate_cascade,
        returns={"lcc_fraction": "float, surviving fraction of the grid",
                 "lcc_size": "int, buses left in the largest component",
                 "n_failed_total": "int, buses NOT in the largest component (includes "
                                   "the trigger buses)",
                 "n_failed_excluding_trigger": "int, same but minus the trigger buses",
                 "n_overloaded": "int, buses that exceeded capacity",
                 "rounds": "int, cascade rounds until it stopped",
                 "n_islands": "int, connected components at the end"},
    ),
    "list_buses": Tool(
        name="list_buses",
        description="Every bus id in the grid, as a list.",
        args={},
        required=[],
        fn=A.list_buses,
        list_returns=frozenset({"buses"}),
        returns={"buses": "list[int], every bus id", "n_buses": "int"},
    ),
    "list_lines": Tool(
        name="list_lines",
        description="Every line in the grid, as a list of [busA, busB] pairs.",
        args={},
        required=[],
        fn=A.list_lines,
        list_returns=frozenset({"lines"}),
        returns={"lines": "list of [busA, busB]", "n_lines": "int"},
    ),
    "column": Tool(
        name="column",
        description="Pull one field out of a LIST of records, in order, like selecting a "
                    "column from a table. Only for the `results` of a repeated step. A "
                    "single analysis result is not a list and does not need this.",
        args={"records": "a list of records, e.g. the `results` of a repeated step",
              "field": "which field to extract"},
        required=["records", "field"],
        fn=A.column,
        list_returns=frozenset({"values"}),
        list_args=frozenset({"records"}),
        needs_graph=False,
        returns={"values": "list of that field's values, in record order", "n": "int",
                 "field": "str"},
    ),
    "argmin": Tool(
        name="argmin",
        description="Position of the smallest value in a list of numbers.",
        args={"values": "list of numbers"},
        required=["values"],
        fn=A.argmin,
        list_returns=frozenset({"tied_indices"}),
        list_args=frozenset({"values"}),
        needs_graph=False,
        returns={"index": "int, position of the smallest value",
                 "value": "the smallest value",
                 "n_tied": "int, how many positions share it; >1 means no unique answer",
                 "tied_indices": "the tied positions when there is no unique winner"},
    ),
    "argmax": Tool(
        name="argmax",
        description="Position of the largest value in a list of numbers.",
        args={"values": "list of numbers"},
        required=["values"],
        fn=A.argmax,
        list_returns=frozenset({"tied_indices"}),
        list_args=frozenset({"values"}),
        needs_graph=False,
        returns={"index": "int, position of the largest value",
                 "value": "the largest value",
                 "n_tied": "int, how many positions share it; >1 means no unique answer",
                 "tied_indices": "the tied positions when there is no unique winner"},
    ),
    "at": Tool(
        name="at",
        description="The record at a given position in a list. Use it to look up which "
                    "candidate an argmin or argmax position refers to.",
        args={"records": "a list of records", "index": "int, the position"},
        required=["records", "index"],
        fn=A.at,
        list_args=frozenset({"records"}),
        needs_graph=False,
        returns={"record": "the record at that position", "index": "int"},
    ),
}


def catalogue() -> str:
    """The KB rendered for the planner prompt."""
    lines = []
    for t in KB.values():
        lines.append(f"- {t.name}: {t.description}")
        if t.args:
            lines.append("    arguments:")
            for arg, meaning in t.args.items():
                req = " (required)" if arg in t.required else ""
                lines.append(f"      {arg}: {meaning}{req}")
        else:
            lines.append("    arguments: none")
        if t.returns:
            lines.append("    returns:")
            for field_, meaning in t.returns.items():
                lines.append(f"      {field_}: {meaning}")
    return "\n".join(lines)


def summarize(result: Any) -> Any:
    """Drop bulk listings before a tool result is shown to the model, at ANY depth.

    A shallow version missed nested ones: `at` returns {"index", "record": {...}} and the
    record's 34-element `overloaded_buses` went straight into the reporter prompt, so the
    prompt ended on a wall of bus numbers with the actual answer buried at the top.
    """
    if isinstance(result, list):
        return [summarize(v) for v in result]
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        if k in BULK_KEYS and isinstance(v, list):
            out[f"{k}_omitted"] = f"{len(v)} entries omitted for brevity"
        else:
            out[k] = summarize(v)
    return out


#: Longest list shown to the model in a step's arguments. Anything longer is elided.
MAX_ARG_LIST = 8


def compact(value: Any, max_list: int = MAX_ARG_LIST) -> Any:
    """Shrink long lists anywhere inside a step's ARGUMENTS.

    Without this, a plan step like select_extreme(results=$s2.results) pastes an entire
    118-row sweep into the reporter prompt, because the executor resolves the reference
    before logging. That inflated Arm A's token count roughly ninefold on sweep
    questions and would have corrupted the cost comparison, which is a headline metric.

    The full arguments still go to the log. Only the model's view is shrunk.
    """
    if isinstance(value, list):
        if len(value) > max_list:
            head = [compact(v, max_list) for v in value[:2]]
            return head + [f"... {len(value) - 2} more entries elided"]
        return [compact(v, max_list) for v in value]
    if isinstance(value, dict):
        return {k: compact(v, max_list) for k, v in value.items()}
    return value
