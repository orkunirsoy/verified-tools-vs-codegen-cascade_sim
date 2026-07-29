# Grid Agent — Verified Tools vs Code Generation for Power-Grid Analysis

**Course:** 17-630 Prompt Engineering (CMU)  
**Author:** Orkun Irsoy (`oirsoy@andrew.cmu.edu`)

Same power-grid analysis assistant, two action formats. **Arm A (verified tools)**
plans over a hand-written Knowledge Base of verified graph tools (AAG-style).
**Arm B (code generation)** writes and executes Python in a sandbox (CodeAct-style).
Both arms see the same IEEE 118-bus Motter–Lai cascade questions, the same models,
and the same JSON answer schema. A frozen oracle grades the final answer.

This README uses the same vocabulary as `presentation/main.tex` and `Report/report.pdf`.

## Layout

```
ask.py                 CLI demo / single-run debugger
src/gridagent/         Package
  arms/                Arm A (planner→validator→executor→reporter)
                       and Arm B (code prompt→sandbox→observe/revise)
  tools/               Knowledge Base + verified analysis functions
  oracle/              Independent Motter–Lai reference (not Arm A's tools)
  evaluation/          Grader (CORRECT / FAILED / WRONG_ANSWER) + run logs
  conditions.py        The five ablation conditions
questions/             15 questions (11 development + 4 held-out test)
gold/                  Frozen gold answers (oracle-derived)
scripts/               Matrix runner, report, oracle/tool checks, trace viewer builder
viewer/                Static HTML trace viewer (open in a browser)
results/               gpt-4o-mini logs (225 runs)
results-gpt-4o/        gpt-4o logs (225 runs)
results-gpt-5-nano/    gpt-5-nano logs (225 runs)
```

## Vocabulary (matches the talk and report)

| Term | Meaning |
|------|---------|
| **Arm A / verified tools** | planner → local validator → Knowledge-Base executor → reporter |
| **Arm B / code generation** | code prompt (± NetworkX cheat-sheet) → sandbox → observe/revise |
| **planner** | LLM prompt that emits a JSON tool plan (`arm_a.py`) |
| **local validator** | Deterministic `validate_plan` (ablated in `A-noval`) |
| **reporter** | LLM prompt that maps tool results into the JSON answer schema |
| **code prompt** | Arm B system prompt; writes Python, not tool calls |
| **NetworkX cheat-sheet** | API syntax only; present in `B-hints`, absent in `B-bare` |
| **CORRECT** | Final answer matched the frozen oracle |
| **FAILED** | No usable answer (crash, timeout, attempt budget, bad JSON, null field) |
| **WRONG_ANSWER / silent failure** | Finished clean with a well-formed wrong answer |
| **development split** | q01–q11 (11 questions; used to build/debug the harness) |
| **held-out test split** | t01–t04 (4 harder composition questions) |

## The five conditions

| Condition     | Arm            | What it tests                                      |
|---------------|----------------|----------------------------------------------------|
| `A-full`      | verified tools | Few-shot planning + plan validation (AAG position) |
| `A-noval`     | verified tools | Validation off                                     |
| `A-zeroshot`  | verified tools | Zero-shot planning, validation on                  |
| `B-hints`     | code generation| CodeAct + NetworkX cheat-sheet                     |
| `B-bare`      | code generation| CodeAct with no API hints                          |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...   # only needed for live runs; never commit a key
```

## Reproduce the report numbers (no API calls)

```bash
python scripts/report.py --results results            --split all
python scripts/report.py --results results-gpt-4o     --split all
python scripts/report.py --results results-gpt-5-nano --split all
python scripts/retry_analysis.py --results results results-gpt-4o
```

These scripts are pure functions over the logged traces. Bucket labels are derived,
never stored as ground truth.

## Rebuild the report figure (no API calls)

```bash
python scripts/plot_accuracy_vs_cost.py    # -> ../Report/fig_accuracy_vs_cost.pdf
```

Grading, condition order and per-model pricing are imported from `report.py`, so the
figure cannot drift from the tables it sits next to.

## Offline smoke test (no API key)

```bash
PYTHONPATH=src python ask.py q03 --condition A-full --stub
PYTHONPATH=src python ask.py q03 --condition B-hints --stub
```

## Live single run (costs money — ask before batching)

```bash
PYTHONPATH=src python ask.py q03 --condition A-full
PYTHONPATH=src python scripts/run_matrix.py --help
```

## Trace viewer

Open `viewer/trace_viewer.html` in a browser. Rebuild from logs with:

```bash
python scripts/build_viewer.py
```

## Where the prompts live

| Prompt | File |
|--------|------|
| Arm A planner + few-shot examples + reporter | `src/gridagent/arms/arm_a.py` |
| Arm A local validator | `validate_plan()` in the same file |
| Arm B code prompt + NetworkX cheat-sheet | `src/gridagent/arms/arm_b.py` |
| Shared JSON answer schema | `src/gridagent/questions.py` (`ANSWER_SCHEMA`) |
| KB catalogue the planner sees | `src/gridagent/tools/kb.py` |

## Design notes for graders

- Correctness is graded on the **final answer**, not intermediate tool values. An Arm A
  run that computes the right number and reports the wrong field counts as WRONG_ANSWER.
- `B-hints` vs `B-bare` separates the action format from the hand-written KB; without
  that ablation an Arm A win would be confounded with documentation quality.
- Token and dollar costs include retries and (for reasoning models) reasoning tokens.
- The oracle's betweenness is a from-scratch Brandes implementation, separate from
  Arm A's tools, so gold and tools cannot silently share a bug.
