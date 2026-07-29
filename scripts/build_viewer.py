"""Render a curated set of run traces to one self-contained HTML page.

Pure function over `results*/`: reads logs, applies the grader, writes one file. No API
calls, no network, no dependencies beyond the standard library.

    python scripts/build_viewer.py                 # -> viewer/trace_viewer.html
    python scripts/build_viewer.py --out docs/traces.html

The point of the viewer is the pairing: for a handful of questions it puts Arm A and
Arm B side by side on the SAME question, so the difference between "planned over verified
tools" and "wrote and ran Python" is visible rather than asserted. Runs are picked to
cover both arms, both outcomes, and the silent-failure cases the whole project is about.

Selection is deterministic (see CURATED below), so the page is reproducible. Everything
shown is read straight from the logs; nothing is summarised by hand.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gridagent.evaluation import grader as GR  # noqa: E402

# Curated cases. Each renders Arm A and Arm B on the SAME question and run, side by side.
#
#   (results_dir, question_id, run_index, caption, arm_a_condition|None, arm_b_condition|None)
#
# A None condition means "first condition that has this run on disk", which is the usual
# case. Pin a condition when the interesting behaviour lives in a specific one; every
# caption below was checked against the run it actually renders.
CURATED: list[tuple[str, str, int, str, str | None, str | None]] = [
    ("results", "q01", 1,
     "Control question. Both arms answer 48 correctly; the contrast is how they get there.",
     None, None),
    ("results", "q03", 1,
     "A plain cascade. Arm A composes verified steps; Arm B writes the physics itself. "
     "Both reach 0.186.", None, None),
    ("results", "q04", 1,
     "Arm B answers 29 where gold is 95. It counted the buses the cascade removed rather "
     "than the buses that ended up outside the surviving network, and never errored.",
     None, None),
    ("results", "q05", 1,
     "A 118-way sweep. Arm A's plan uses foreach over every bus; Arm B writes the loop.",
     None, None),
    ("results", "q08", 1,
     "Arm B answers 42, the overloaded count, where gold is 83. Three clean attempts, "
     "wrong quantity.", None, None),
    ("results", "q09", 2,
     "The one repeated crash in the whole Arm B set: four attempts, three tracebacks, "
     "no answer produced. This is what a visible failure looks like.", None, "B-bare"),
    ("results", "q11", 1,
     "Arm A's residual failure. The tool returns the right record and the prose names the "
     "right line, but the answer field carries 53, the lcc_size from that record.",
     None, None),
    ("results", "t02", 1,
     "Held-out. Arm B defines a trigger parameter and never uses it, so no cascade runs and "
     "the answer is 0. Arm A never gets a plan past the validator.", None, None),
    ("results", "t04", 1,
     "Held-out, both arms wrong in different ways. Arm B's overload test is L > 1.2*L, "
     "false for every bus. Arm A computes the right line and submits 42, the bus count.",
     None, None),
    ("results-gpt-4o", "t02", 1,
     "The same question on gpt-4o: both arms answer 6. Capability, not architecture.",
     None, None),
    ("results-gpt-4o", "t04", 1,
     "The same question on gpt-4o: both arms answer 23-76 correctly.", None, None),
]

ARM_A_CONDITIONS = ["A-full", "A-noval", "A-zeroshot"]
ARM_B_CONDITIONS = ["B-hints", "B-bare"]

BUCKET_CLASS = {GR.CORRECT: "ok", GR.WRONG_ANSWER: "silent", GR.FAILED: "crash"}
BUCKET_LABEL = {GR.CORRECT: "correct", GR.WRONG_ANSWER: "silent failure", GR.FAILED: "crashed"}

CSS = """
:root{--ink:#1f1e1c;--paper:#fbfaf7;--muted:#79756e;--hair:#d8d3ca;
--armA:#14607a;--armB:#a8432e;--ok:#2f6b45;--accent:#be7f23;--code:#f3f1ec}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:34px 40px 22px;border-bottom:1px solid var(--hair)}
header .rule{width:60px;height:3px;background:var(--accent);margin-bottom:14px}
h1{margin:0 0 6px;font-size:26px;letter-spacing:-.01em}
header p{margin:0;color:var(--muted);max-width:70ch}
main{padding:0 40px 60px}
.case{margin:34px 0 0;padding-top:26px;border-top:1px solid var(--hair)}
.case:first-child{border-top:none}
.case>h2{margin:0 0 4px;font-size:19px}
.why{margin:0 0 6px;color:var(--muted);max-width:90ch}
.q{margin:0 0 16px;padding:10px 14px;background:#fff;border-left:3px solid var(--accent);
max-width:90ch}
.q .gold{color:var(--ok);font-weight:600}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}
.arm{background:#fff;border:1px solid var(--hair);border-radius:3px;overflow:hidden}
.arm>h3{margin:0;padding:10px 14px;font-size:14px;letter-spacing:.02em;
border-bottom:1px solid var(--hair)}
.armA>h3{color:var(--armA);border-top:3px solid var(--armA)}
.armB>h3{color:var(--armB);border-top:3px solid var(--armB)}
.body{padding:12px 14px}
.badge{float:right;font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;
letter-spacing:.02em}
.badge.ok{background:#e6f2ea;color:var(--ok)}
.badge.silent{background:#fbe9e5;color:var(--armB)}
.badge.crash{background:#fdf2dd;color:#8a6012}
.step{margin:0 0 10px;padding-left:11px;border-left:2px solid var(--hair)}
.step .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin-bottom:3px}
pre{margin:0;padding:9px 11px;background:var(--code);border-radius:2px;overflow-x:auto;
font:12px/1.5 "IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:pre-wrap;word-break:break-word}
pre.err{background:#fbe9e5;color:#7d2d1e}
.meta{font-size:12px;color:var(--muted);margin-top:10px;padding-top:8px;
border-top:1px solid var(--hair)}
.answer{font-weight:600}
.answer.wrong{color:var(--armB)}
.answer.right{color:var(--ok)}
details>summary{cursor:pointer;color:var(--muted);font-size:12px;margin-bottom:5px}
footer{padding:26px 40px;border-top:1px solid var(--hair);color:var(--muted);font-size:13px}
"""


def esc(x) -> str:
    return html.escape(str(x))


def clip(text: str, n: int = 1400) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"\n... [{len(text) - n} more characters]"


def load(results_dir: Path, condition: str, qid: str, run: int) -> dict | None:
    f = results_dir / condition / f"{qid}_run{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def pick(results_dir: Path, conditions: list[str], qid: str, run: int) -> dict | None:
    """First condition in the list that has this run on disk."""
    for c in conditions:
        log = load(results_dir, c, qid, run)
        if log:
            return log
    return None


def render_arm_a(log: dict) -> str:
    out = []
    plan = log.get("plan")
    if plan:
        out.append('<div class="step"><div class="lbl">plan</div>'
                   f'<pre>{esc(json.dumps(plan, indent=2))}</pre></div>')
    for v in (log.get("validation") or []):
        problems = v.get("problems") or []
        verdict = "accepted" if not problems else "rejected: " + "; ".join(map(str, problems))
        cls = " class=err" if problems else ""
        out.append(f'<div class="step"><div class="lbl">validation, attempt {v.get("attempt")}'
                   f'</div><pre{cls}>{esc(verdict)}</pre></div>')
    for t in (log.get("tool_calls") or []):
        head = f'{t.get("name")}({json.dumps(t.get("args"))})'
        res = t.get("error") or json.dumps(t.get("result"))
        cls = " class=err" if t.get("error") else ""
        out.append('<div class="step"><div class="lbl">tool call</div>'
                   f'<pre>{esc(clip(head, 400))}</pre>'
                   f'<pre{cls}>{esc(clip(str(res), 700))}</pre></div>')
    return "".join(out)


def render_arm_b(log: dict) -> str:
    out = []
    for a in (log.get("code_attempts") or []):
        out.append(f'<div class="step"><div class="lbl">attempt {a.get("attempt")}</div>'
                   f'<pre>{esc(clip(a.get("code", ""), 1100))}</pre>')
        if a.get("stdout"):
            out.append(f'<pre>stdout: {esc(clip(a["stdout"], 400))}</pre>')
        if a.get("stderr"):
            out.append(f'<pre class="err">{esc(clip(a["stderr"], 500))}</pre>')
        out.append("</div>")
    return "".join(out)


def render_run(log: dict, verdict, arm: str) -> str:
    cls = BUCKET_CLASS.get(verdict.bucket, "crash")
    steps = render_arm_a(log) if arm == "A" else render_arm_b(log)
    parsed = log.get("parsed_answer") or {}
    submitted = parsed.get("answer", "(none)")
    right = "right" if verdict.bucket == GR.CORRECT else "wrong"
    reasoning = parsed.get("reasoning") or ""
    flags = ", ".join(verdict.flags) if verdict.flags else "none"
    t = log.get("totals") or {}
    return f"""
    <div class="arm arm{arm}">
      <h3>Arm {arm} &middot; <code>{esc(log['condition'])}</code>
        <span class="badge {cls}">{BUCKET_LABEL.get(verdict.bucket, verdict.bucket)}</span></h3>
      <div class="body">
        {steps}
        <div class="step"><div class="lbl">submitted answer</div>
          <pre class="answer {right}">{esc(submitted)}   (gold: {esc(verdict.gold)})</pre></div>
        {f'<details><summary>model reasoning</summary><pre>{esc(clip(reasoning, 900))}</pre></details>' if reasoning else ''}
        <div class="meta">
          status <b>{esc(log.get('status'))}</b> &middot;
          {t.get('total_tokens', 0):,} tokens &middot;
          {log.get('wall_time_s', 0):.1f}s &middot;
          grader: {esc(verdict.reason)} &middot; flags: {esc(flags)}
        </div>
      </div>
    </div>"""


def build(out_path: Path) -> int:
    gold = GR.load_gold(ROOT / "gold" / "gold_answers.json")
    cases, shown = [], 0

    for dirname, qid, run, why, a_cond, b_cond in CURATED:
        rdir = ROOT / dirname
        a = pick(rdir, [a_cond] if a_cond else ARM_A_CONDITIONS, qid, run)
        b = pick(rdir, [b_cond] if b_cond else ARM_B_CONDITIONS, qid, run)
        if not a and not b:
            print(f"  skip {dirname}/{qid} run{run}: no logs on disk")
            continue

        model = (a or b)["model"]
        qtext = (a or b).get("question_text", "")
        gold_val = gold["answers"].get(qid, {}).get("value", "?")

        panels = []
        for log, arm in ((a, "A"), (b, "B")):
            if log:
                panels.append(render_run(log, GR.grade(log, gold), arm))
                shown += 1

        cases.append(f"""
        <section class="case">
          <h2>{esc(qid)} &middot; run {run} &middot; <code>{esc(model)}</code></h2>
          <p class="why">{esc(why)}</p>
          <div class="q">{esc(qtext)}<br><span class="gold">gold: {esc(gold_val)}</span></div>
          <div class="cols">{''.join(panels)}</div>
        </section>""")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>grid-agent &middot; run traces</title>
<style>{CSS}</style></head>
<body>
<header>
  <div class="rule"></div>
  <h1>Verified tools vs. code generation: run traces</h1>
  <p>{len(cases)} questions, {shown} runs, rendered directly from the run logs by
  <code>scripts/build_viewer.py</code>. Each row shows both arms answering the same
  question. Arm A plans over verified functions; Arm B writes and executes Python.
  A <b>silent failure</b> is a run that finished, returned a well-formed answer, and was
  wrong.</p>
</header>
<main>{''.join(cases)}</main>
<footer>
  Generated from <code>results/</code> and <code>results-gpt-4o/</code>. Verdicts are
  recomputed by <code>evaluation/grader.py</code> at build time, never stored. Long code,
  outputs and reasoning are truncated for readability; the full text is in the JSON logs.
</footer>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(cases)} questions, {shown} runs, "
          f"{len(page) / 1024:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "viewer" / "trace_viewer.html",
                    help="output HTML file (default: viewer/trace_viewer.html)")
    args = ap.parse_args()
    return build(args.out if args.out.is_absolute() else ROOT / args.out)


if __name__ == "__main__":
    raise SystemExit(main())
