"""Derive the gold answer key from the oracle and freeze it to JSON.

Every gold value is computed here from `gridagent.oracle.reference` -- none is typed
in by hand. That means the answer key cannot silently drift from the oracle, and
regenerating it after an oracle change is one command.

Each entry records not just the value but *how it was derived* (`derivation`) and
supporting context (`provenance`), so the human review pass can check the reasoning
rather than squinting at a bare number. That review is the step that makes the whole
comparison credible: the oracle is only an answer key if a domain expert has signed
off on it.

Run:  python scripts/build_gold.py            # write gold/gold_answers.json
      python scripts/build_gold.py --review   # human-readable review sheet
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import yaml

from gridagent.data import load_grid, grid_summary
from gridagent.oracle import reference as ref

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "questions" / "questions.yaml"
GOLD = ROOT / "gold" / "gold_answers.json"

ALPHA = 0.2  # must match cascade_spec in questions.yaml


def compute_gold(G) -> dict[str, dict]:
    """Compute every gold answer. Keys must match question ids in questions.yaml."""
    bet = ref.betweenness(G)
    ranked = sorted(bet.items(), key=lambda kv: (-kv[1], kv[0]))
    hub = ranked[0][0]
    top3 = [b for b, _ in ranked[:3]]

    gold: dict[str, dict] = {}

    # ---- controls ----
    ds = ref.degree_stats(G)
    gold["q01"] = {
        "value": ds["max_degree_bus"],
        "derivation": "degree_stats(G).max_degree_bus",
        "provenance": {"max_degree": ds["max_degree"],
                       "n_tied_at_max": sum(1 for _, d in G.degree() if d == ds["max_degree"])},
    }

    gold["q02"] = {
        "value": [b for b, _ in ranked[:5]],
        "derivation": "top 5 by unnormalized betweenness, descending",
        "provenance": {"top5_loads": [round(v, 2) for _, v in ranked[:5]],
                       "next_load": round(ranked[5][1], 2),
                       "boundary_gap": round(ranked[4][1] - ranked[5][1], 2)},
    }

    # ---- cascade ----
    casc = ref.cascade(G, [hub], alpha=ALPHA)
    no_casc = ref.no_cascade_value(G, [hub])
    gold["q03"] = {
        "value": casc.lcc_fraction,
        "derivation": f"cascade(G, trigger=[{hub}], alpha={ALPHA}).lcc_fraction",
        "provenance": {"trigger": hub, "lcc_size": casc.lcc_size, "rounds": casc.rounds,
                       "n_overloaded": casc.n_failed,
                       "no_cascade_value": no_casc,
                       "note": "GRADER METADATA: an answer matching no_cascade_value is the "
                               "signature of recomputing capacity each round, which makes "
                               "overload impossible and halts the cascade immediately."},
    }
    gold["q04"] = {
        "value": casc.n_dead - len(casc.trigger),
        "derivation": (f"cascade(G, trigger=[{hub}], alpha={ALPHA}).n_dead minus the trigger "
                       "-- failed means not in the largest component"),
        "provenance": {"trigger": hub, "n_dead_including_trigger": casc.n_dead,
                       "lcc_size": casc.lcc_size, "rounds": casc.rounds,
                       "of_which_overloaded": casc.n_failed,
                       "of_which_cut_off_without_overloading":
                           casc.n_dead - len(casc.trigger) - casc.n_failed,
                       "overloaded_buses": casc.failed},
    }

    worst = ref.worst_cascade_trigger(G, alpha=ALPHA)
    gold["q05"] = {
        "value": worst["bus"],
        "derivation": f"argmin over all buses of cascade(G,[b],alpha={ALPHA}).lcc_fraction",
        "provenance": {"lcc_fraction": worst["lcc_fraction"], "busiest_bus": hub,
                       "note": "answer IS the busiest bus; shortcut accidentally right here, "
                               "wrong on q06"},
    }

    worst_rd = ref.worst_rounds_trigger(G, alpha=ALPHA)
    gold["q06"] = {
        "value": worst_rd["bus"],
        "derivation": f"argmax over all buses of cascade(G,[b],alpha={ALPHA}).rounds",
        "provenance": {"rounds": worst_rd["rounds"],
                       "lcc_fraction": worst_rd["result"]["lcc_fraction"],
                       "busiest_bus": hub, "busiest_bus_rounds": casc.rounds,
                       "note": "answer is NOT the busiest bus. 68 does the most damage in "
                               "only 2 rounds; this is the shortcut detector, and duration "
                               "cannot be read off any centrality measure"},
    }

    c3 = ref.cascade(G, top3, alpha=ALPHA)
    gold["q07"] = {
        "value": c3.lcc_fraction,
        "derivation": f"cascade(G, trigger={top3}, alpha={ALPHA}).lcc_fraction",
        "provenance": {"trigger": top3, "n_overloaded": c3.n_failed, "rounds": c3.rounds,
                       "single_hub_lcc_for_contrast": casc.lcc_fraction,
                       "note": "LESS damaging than the hub alone -- deliberately "
                               "counterintuitive"},
    }

    # ---- extended: line triggers and topology changes ----
    Q8_LINE = (48, 68)
    c8 = ref.cascade_from_line(G, Q8_LINE, alpha=ALPHA)
    gold["q08"] = {
        "value": c8.n_dead,
        "derivation": f"cascade_from_line(G, line={Q8_LINE}, alpha={ALPHA}).n_dead",
        "provenance": {"line": list(Q8_LINE), "lcc_fraction": c8.lcc_fraction,
                       "lcc_size": c8.lcc_size, "rounds": c8.rounds,
                       "of_which_overloaded": c8.n_failed,
                       "note": "no trigger bus to exclude: a line tripped, both end buses "
                               "stay in service unless they overload"},
    }

    wl = ref.worst_line_trigger(G, alpha=ALPHA)
    n_zero = sum(1 for e in G.edges()
                 if ref.cascade_from_line(G, tuple(sorted(e)), alpha=ALPHA).n_failed == 0)
    gold["q09"] = {
        "value": list(wl["line"]),
        "derivation": f"argmin over all lines of cascade_from_line(G,e,alpha={ALPHA}).lcc_fraction",
        "provenance": {"lcc_fraction": wl["lcc_fraction"],
                       "n_overloaded": wl["result"]["n_failed"],
                       "rounds": wl["result"]["rounds"],
                       "n_lines": G.number_of_edges(),
                       "n_lines_causing_zero_overloads": n_zero},
    }

    Q10_LINE = (23, 81)
    c10 = ref.cascade(ref.with_line(G, Q10_LINE), [hub], alpha=ALPHA)
    gold["q10"] = {
        "value": c10.lcc_fraction,
        "derivation": f"cascade(with_line(G,{Q10_LINE}), [{hub}], alpha={ALPHA}).lcc_fraction",
        "provenance": {"added_line": list(Q10_LINE), "n_overloaded": c10.n_failed,
                       "baseline_lcc_without_line": casc.lcc_fraction,
                       "note": "adding a line makes the cascade WORSE (Braess-like)"},
    }

    Q11_CANDIDATES = [(31, 81), (48, 67), (23, 76), (48, 82), (23, 81)]
    scored = [(e, ref.cascade(ref.with_line(G, e), [hub], alpha=ALPHA).lcc_fraction)
              for e in Q11_CANDIDATES]
    best = max(scored, key=lambda t: t[1])
    gold["q11"] = {
        "value": list(best[0]),
        "derivation": ("argmax over the 5 named candidate lines of "
                       f"cascade(with_line(G,e), [{hub}], alpha={ALPHA}).lcc_fraction"),
        "provenance": {"candidates": {f"{u}-{v}": round(f, 4) for (u, v), f in scored},
                       "baseline_lcc_without_line": round(casc.lcc_fraction, 4),
                       "note": "2 of the 5 candidates are WORSE than building nothing"},
    }

    # ================= HELD-OUT TEST SPLIT =================
    #
    # Same rule as every entry above: derived from the oracle, never typed in by hand.
    # Uses only oracle functions that already existed for the dev split -- no new
    # machinery, which is what keeps the test split from becoming a second dev set.
    #
    # Every one of these is an argmin/argmax, so UNIQUENESS IS PART OF VALIDITY. Each
    # entry records `n_tied`, and `main()` refuses to write the file if any is > 1. A
    # question with a tied extremum has no single correct answer and must be cut, not
    # resolved by picking one. The original t02 (longest cascade over line triggers) was
    # cut for exactly this: a seven-way tie at 6 rounds.

    def extreme(scored, direction):
        """(winner, value, n_tied) over [(candidate, value), ...]."""
        best = (max if direction == "max" else min)(v for _, v in scored)
        tied = [c for c, v in scored if v == best]
        return tied[0], best, len(tied)

    T01_LINE = (31, 81)
    H01 = ref.with_line(G, T01_LINE)
    s01 = [(b, ref.cascade(H01, [b], alpha=ALPHA).lcc_fraction) for b in sorted(H01)]
    w01, v01, n01 = extreme(s01, "min")
    gold["t01"] = {
        "value": w01,
        "derivation": (f"argmin over every bus b of "
                       f"cascade(with_line(G,{T01_LINE}), [b], alpha={ALPHA}).lcc_fraction"),
        "provenance": {"lcc_fraction": round(v01, 4), "n_tied": n01,
                       "added_line": list(T01_LINE),
                       "q05_answer_on_original_grid": worst["bus"],
                       # The margin to the runner-up is ONE bus (21 vs 22, buses 41/43/81).
                       # That is deliberate and was checked, not overlooked. A thin margin
                       # is only a validity problem when the deciding convention is left to
                       # the agent -- cf. q06, whose thin round-count margin needed a
                       # uniformity check because round counting is underspecified. Here
                       # the only variant that moves the answer is >= instead of >, and the
                       # `overload` block says "strictly greater than" in words. An agent
                       # using >= has contradicted text it was given, which is an error, not
                       # a defensible reading. Verified stable under 1e-9 and 1e-6 float
                       # tolerance, so this is not a floating-point knife-edge either.
                       "margin_to_runner_up_buses": 1,
                       "runners_up": [41, 43, 81],
                       "answer_if_overload_test_were_non_strict": 31,
                       "note": "transpose of q11: sweeps TRIGGERS with the upgrade fixed. "
                               "Answer differs from q05's, so the upgrade genuinely moves "
                               "the worst trigger and the dev answer cannot be reused. "
                               "DIAGNOSTIC: an agent answering 31 used >= rather than > in "
                               "the overload test -- the same class of named signature as "
                               "t02's -4 and q03's no_cascade_value"},
    }

    c68 = ref.cascade(G, [68], alpha=ALPHA)
    c76 = ref.cascade(G, [76], alpha=ALPHA)
    gold["t02"] = {
        "value": c68.n_dead - c76.n_dead,
        "derivation": (f"cascade(G,[68],alpha={ALPHA}).n_dead "
                       f"- cascade(G,[76],alpha={ALPHA}).n_dead"),
        "provenance": {"n_dead_bus68": c68.n_dead, "n_dead_bus76": c76.n_dead,
                       "n_tied": 1,
                       # The wording "how many ADDITIONAL buses fail" admits a
                       # trigger-inclusive and a trigger-exclusive reading. Both give the
                       # same answer here because the -1 cancels on each side (96-90 and
                       # 95-89), so the question is safe as written.
                       "same_answer_excluding_triggers": (c68.n_dead - 1) - (c76.n_dead - 1),
                       # The one reading that diverges counts OVERLOADS instead of
                       # not-in-LCC. It contradicts the `overload` block, which defines
                       # failed explicitly, so it is a genuine error rather than a guess
                       # about our convention -- and it is SIGN-FLIPPED, which makes it an
                       # unusually clean diagnostic signature.
                       "overloaded_only_reading": c68.n_failed - c76.n_failed,
                       "note": "two scenarios plus arithmetic. The KB has no subtraction "
                               "tool, so Arm A must run two cascades and let the REPORTER "
                               "subtract -- against a reporter prompt that says 'Do NOT "
                               "recompute anything'. Riskiest question in the split. An "
                               "agent counting overloads rather than not-in-LCC answers "
                               "-4, i.e. the wrong sign"},
    }

    top5 = [b for b, _ in ref.top_k_betweenness(G, 5)]
    s03 = [(b, ref.cascade(G, [b], alpha=ALPHA).lcc_fraction) for b in top5]
    w03, _v03, n03 = extreme(s03, "max")
    gold["t03"] = {
        "value": w03,
        "derivation": (f"argmax over the top-5 betweenness buses of "
                       f"cascade(G,[b],alpha={ALPHA}).lcc_fraction"),
        "provenance": {"top5_by_load": top5, "n_tied": n03,
                       "lcc_by_candidate": {b: round(v, 4) for b, v in s03},
                       "note": "foreach over a DERIVED candidate list ($s1.top_buses), a "
                               "shape no dev question uses. Not monotonic in load rank "
                               "(rank 3 beats rank 4), so it is not simply 'pick the "
                               "least central of the five' by construction"},
    }

    T04_CANDIDATES = [(31, 81), (23, 76), (48, 82)]
    T04_TRIGGER = (48, 68)
    s04 = [(e, ref.cascade_from_line(ref.with_line(G, e), T04_TRIGGER,
                                     alpha=ALPHA).lcc_fraction)
           for e in T04_CANDIDATES]
    w04, _v04, n04 = extreme(s04, "max")
    gold["t04"] = {
        "value": list(w04),
        "derivation": (f"argmax over the 3 named candidate lines of "
                       f"cascade_from_line(with_line(G,e), {T04_TRIGGER}, "
                       f"alpha={ALPHA}).lcc_fraction"),
        "provenance": {"candidates": {f"{u}-{v}": round(f, 4) for (u, v), f in s04},
                       "n_tied": n04, "trigger_line": list(T04_TRIGGER),
                       "note": "added_line combined with a LINE trigger, the one pairing "
                               "the dev split never uses. 3 candidates so a blind guess "
                               "scores 1/3"},
    }
    return gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true", help="print a human review sheet")
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS.read_text())
    G = load_grid(spec["case"])
    gold = compute_gold(G)

    q_ids = [q["id"] for q in spec["questions"]]
    if set(q_ids) != set(gold):
        raise SystemExit(f"question/gold mismatch: questions={q_ids} gold={sorted(gold)}")

    # A tied extremum means the question has no single correct answer, and grading it
    # would score agents on a coin flip. Refuse to write rather than freeze one.
    tied = {qid: e["provenance"]["n_tied"] for qid, e in gold.items()
            if isinstance(e.get("provenance"), dict) and e["provenance"].get("n_tied", 1) > 1}
    if tied:
        raise SystemExit(
            f"REFUSING TO WRITE: tied extremum in {tied}. A tied question has no unique "
            f"answer. Cut it or change its objective; do not pick a winner.")

    # Every `context` name must resolve to a real definition block. A typo here would
    # silently strip a definition from the prompt -- the agent would be graded on
    # guessing a convention it was never given, which is the one thing the question
    # set exists to prevent.
    known = set(spec.get("definitions", {}))
    for q in spec["questions"]:
        if "context" not in q:
            raise SystemExit(f"{q['id']}: missing `context` (use [] if none needed)")
        unknown = [c for c in q["context"] if c not in known]
        if unknown:
            raise SystemExit(f"{q['id']}: unknown context {unknown}; known: {sorted(known)}")

    # Any cascade question must carry the cascade model, or the agent is left guessing
    # whether overloads propagate and how capacity is baselined.
    for q in spec["questions"]:
        if q["category"].startswith("cascade") and "overload" not in q["context"]:
            raise SystemExit(f"{q['id']}: cascade question missing `overload` context")

    payload = {
        "case": spec["case"],
        "grid": grid_summary(G),
        "n_buses": G.number_of_nodes(),
        "alpha": ALPHA,
        "oracle_verified_by": "scripts/verify_oracle.py",
        "answers": {
            q["id"]: {
                "question": " ".join(q["question"].split()),
                "category": q["category"],
                "answer_type": q["answer_type"],
                "tolerance": q.get("tolerance"),
                **gold[q["id"]],
            }
            for q in spec["questions"]
        },
    }

    if args.review:
        print(f"GOLD ANSWER REVIEW SHEET -- {payload['grid']}\n" + "=" * 78)
        for q in spec["questions"]:
            a = payload["answers"][q["id"]]
            print(f"\n[{q['id']}] {a['category'].upper()}   answer_type={a['answer_type']}"
                  + (f"  tol={a['tolerance']}" if a["tolerance"] is not None else ""))
            print(f"  Q: {a['question']}")
            print(f"  GOLD ANSWER  ->  {a['value']}")
            print(f"  derived by:  {a['derivation']}")
            for k, v in a["provenance"].items():
                print(f"    - {k}: {v}")
        print("\n" + "=" * 78)
        print("Check: is each GOLD ANSWER the answer you would give a colleague?")
        return 0

    GOLD.parent.mkdir(exist_ok=True)
    GOLD.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {GOLD.relative_to(ROOT)} ({len(gold)} answers, case {spec['case']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
