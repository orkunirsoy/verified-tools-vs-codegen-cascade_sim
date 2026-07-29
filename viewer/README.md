# Trace viewer

`trace_viewer.html` is a single self-contained page showing 11 questions with **Arm A and
Arm B side by side on the same question**. Open it in any browser; there is no server, no
build step and no external asset.

## Regenerate

```
python scripts/build_viewer.py                  # -> viewer/trace_viewer.html
python scripts/build_viewer.py --out docs/x.html
```

Pure function over `results*/`, like `report.py` and `retry_analysis.py`: it reads logs,
recomputes verdicts with `evaluation/grader.py` at build time, and writes one file. No API
calls. Verdicts are never stored, so the page cannot drift from the grader.

## What is on it

22 runs, chosen to cover both arms and all three outcome buckets: 13 correct, 7 silent
failures, 2 crashes. The selection is the `CURATED` list at the top of
`scripts/build_viewer.py`, so the page is reproducible rather than hand-assembled.

Each panel shows the full trace. For Arm A: the plan, the validator's verdict on each
attempt, every tool call with its return value, then the submitted answer. For Arm B: every
code attempt with its stdout and stderr, then the submitted answer. Both carry a verdict
badge, the grader's reason, any diagnostic flags, token count and wall time.

The cases worth opening first:

| Case | Why |
|---|---|
| `q04`, `q08` | Arm B runs clean and returns a confidently wrong number |
| `q09` (B-bare) | The one repeated crash: four attempts, three tracebacks, no answer |
| `q11`, `t04` | Arm A's residual failure: right computation, wrong field in the answer slot |
| `t02`, `t04` on gpt-4o | The same questions both arms failed on gpt-4o-mini, answered correctly |

## Editing the selection

Add or change entries in `CURATED`. Each is
`(results_dir, question_id, run_index, caption, arm_a_condition, arm_b_condition)`; the two
condition fields may be `None`, meaning "first condition holding that run". Pin one when the
behaviour you want lives in a specific condition, as `q09` does with `B-bare`.

**Check the caption against the run it actually renders.** Captions written from notes go
stale when the selection changes, and a caption that describes a different run than the one
on screen is worse than no caption. `build_viewer.py` prints what it selected.
