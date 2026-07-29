"""Check Arm A's tools against the oracle on every gold answer.

Two implementations agreeing is evidence; one agreeing with itself is not. The oracle
implements Brandes from scratch, the tools call NetworkX, and the cascade loops are
written separately. If they agree on all 11 gold answers, both are probably right.

This does NOT make the comparison circular. Arm A executes these tools, so a correct
plan yields correct numbers by construction -- that is exactly the AAG claim under
test. Arm A is still graded on its final natural-language answer against the frozen
oracle key, so it can fail by choosing the wrong tool or misreporting what it returned.

Run:  python scripts/verify_tools.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from gridagent.data import load_grid
from gridagent.tools import analysis as T

ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "gold" / "gold_answers.json").read_text())
ALPHA = GOLD["alpha"]

failures: list[str] = []


def check(qid: str, got, want, note: str = "") -> None:
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-9
    status = "ok " if ok else "FAIL"
    print(f"  [{status}] {qid}: tools={got!r} gold={want!r} {note}")
    if not ok:
        failures.append(qid)


def sweep(G, candidates, kind, alpha, trigger_buses=None):
    """What a `foreach` step does: one simulate_cascade per candidate, tagged by item."""
    rows = []
    for item in candidates:
        if kind == "trigger_buses":
            r = T.simulate_cascade(G, trigger_buses=[item], alpha=alpha)
        elif kind == "trigger_line":
            r = T.simulate_cascade(G, trigger_line=tuple(item), alpha=alpha)
        else:
            r = T.simulate_cascade(G, trigger_buses=list(trigger_buses), alpha=alpha,
                                   added_line=tuple(item))
        rows.append({"item": item, **r})
    return rows


def pick(rows, field, mode):
    """What column -> argmin/argmax -> at does: three generic ops, no domain knowledge."""
    vals = T.column(rows, field)["values"]
    idx = (T.argmin if mode == "min" else T.argmax)(vals)
    return T.at(rows, idx["index"])["record"], idx["n_tied"]


def main() -> int:
    G = load_grid(GOLD["case"])
    a = GOLD["answers"]
    print(f"Arm A tools vs oracle gold -- {GOLD['case']}, alpha={ALPHA}")
    print("Answers are reached ONLY by composing library-shaped primitives:")
    print("  analysis: grid_info degree_ranking load_ranking simulate_cascade "
          "list_buses list_lines")
    print("  data ops: column argmin argmax at        repetition: foreach (plan-level)")
    print("=" * 78)

    check("q01", T.degree_ranking(G, k=5)["top_bus"], a["q01"]["value"], "(1 step)")
    check("q02", T.load_ranking(G, k=5)["top_buses"], a["q02"]["value"], "(1 step)")

    hub = T.load_ranking(G, k=1)["busiest_bus"]
    c = T.simulate_cascade(G, trigger_buses=[hub], alpha=ALPHA)
    check("q03", c["lcc_fraction"], a["q03"]["value"], "(2 steps)")
    check("q04", c["n_failed_excluding_trigger"], a["q04"]["value"], "(2 steps)")

    # q05/q06: identical foreach, DIFFERENT column and direction. That choice is the
    # whole planning decision, and it is no longer hidden inside a sweep function.
    buses = T.list_buses(G)["buses"]
    rows_b = sweep(G, buses, "trigger_buses", ALPHA)
    r5, t5 = pick(rows_b, "lcc_fraction", "min")
    check("q05", r5["item"], a["q05"]["value"],
          f"(5 steps: list->foreach->column(lcc_fraction)->argmin->at, ties={t5})")
    r6, t6 = pick(rows_b, "rounds", "max")
    check("q06", r6["item"], a["q06"]["value"],
          f"(same foreach, column(rounds)->argmax, ties={t6})")

    top3 = T.load_ranking(G, k=3)["top_buses"]
    check("q07", T.simulate_cascade(G, trigger_buses=top3, alpha=ALPHA)["lcc_fraction"],
          a["q07"]["value"], "(2 steps)")

    check("q08", T.simulate_cascade(G, trigger_line=(48, 68), alpha=ALPHA)["n_failed_total"],
          a["q08"]["value"], "(1 step)")

    lines = T.list_lines(G)["lines"]
    rows_l = sweep(G, lines, "trigger_line", ALPHA)
    r9, t9 = pick(rows_l, "lcc_fraction", "min")
    check("q09", r9["item"], a["q09"]["value"], f"(5 steps over 179 lines, ties={t9})")

    check("q10", T.simulate_cascade(G, trigger_buses=[hub], alpha=ALPHA,
                                    added_line=(23, 81))["lcc_fraction"],
          a["q10"]["value"], "(2 steps)")

    cand = [[31, 81], [48, 67], [23, 76], [48, 82], [23, 81]]
    rows_c = sweep(G, cand, "added_line", ALPHA, trigger_buses=[hub])
    r11, t11 = pick(rows_c, "lcc_fraction", "max")
    check("q11", r11["item"], a["q11"]["value"],
          f"(5 named candidates: no foreach needed, 5 explicit steps work too, ties={t11})")

    print("=" * 78)
    if failures:
        print(f"MISMATCH on {failures}. One of the two implementations is wrong.")
        return 1
    print("Tools agree with the oracle on all 11 gold answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
