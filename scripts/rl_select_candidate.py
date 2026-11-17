#!/usr/bin/env python
"""Apply the declared validation-only selection rule to complete candidate suites.

Rules (all over the 12 main-decoder runs: EEGNet + CBraMod, participants s11/s13, validation seeds; ties 1e-9):
  --metric success (default; protocol v2, reports/rl_v2_experiment_plan.md Section 4): highest mean final-step
      success; ties: lower mean final distance, then the smaller EEG weight.
  --metric auc (protocol v3, FINAL_V3_HANDOFF.md Section 6.3): highest mean normalized success AUC; ties: higher
      mean final success, then the smaller EEG weight.
  --metric preferred_pass (route preference, FINAL_V3_HANDOFF.md Section 7.3 and
      reports/final_v3/preference_plan.md): highest mean final preferred-pass rate; ties: higher mean task
      (final) success, then the smaller EEG weight.
Every candidate must be a complete suite (rl.summarize).

Usage: python scripts/rl_select_candidate.py <suite> [<suite> ...] --out <json> --markdown <md> [--metric M]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.summarize import summarize  # noqa: E402

MAIN = ("eegnet", "cbramod")
TIE = 1e-9


def score(suite):
    s = summarize(suite)
    if s["split"] != "validation":
        raise ValueError(f"Not a validation suite: {suite}")
    main = [r for r in s["runs"] if r["condition"] in MAIN]
    expected = len(MAIN) * 2 * len(s["config"]["validation_seeds"])
    if len(main) != expected:
        raise ValueError(f"Expected {expected} main-decoder runs, found {len(main)}")
    per_condition = {}
    for cond in ("sparse", "dense", "ideal", "eegnet", "cbramod", "independent_eegnet", "independent_cbramod"):
        rs = [r for r in s["runs"] if r["condition"] == cond]
        per_condition[cond] = {m: statistics.mean(r[m] for r in rs) for m in
                               ("success_rate", "mean_final_distance", "normalized_auc_success")}
        per_condition[cond]["n_runs"] = len(rs)
    pref = "preferred_pass_rate" in main[0]
    if pref:
        for cond in per_condition:
            rs = [r for r in s["runs"] if r["condition"] == cond]
            per_condition[cond].update({m: statistics.mean(r[m] for r in rs) for m in
                                        ("preferred_pass_rate", "violation_free_rate", "normalized_auc_preferred_pass")})
    return {"suite": str(suite), "eeg_weight": s["config"]["eeg_weight"],
            "selection_preferred_pass": statistics.mean(r["preferred_pass_rate"] for r in main) if pref else None,
            "dense_weight": s["config"]["dense_weight"], "n_runs": len(s["runs"]),
            "selection_success": statistics.mean(r["success_rate"] for r in main),
            "selection_final_distance": statistics.mean(r["mean_final_distance"] for r in main),
            "selection_auc": statistics.mean(r["normalized_auc_success"] for r in main),
            "n_selection_runs": len(main), "per_condition": per_condition,
            "training_source_sha256": s["training_source_sha256"], "training_git_shas": s["training_git_shas"]}


RULES = {
    # metric: (primary key, higher is better), (tie-break key, higher is better), description
    "auc": ("selection_auc", "selection_success",
            "highest mean normalized success AUC", "higher mean final success"),
    "preferred_pass": ("selection_preferred_pass", "selection_success",
                       "highest mean final preferred-pass rate", "higher mean task success"),
}


def choose_by(scores, metric):
    primary, second, d1, d2 = RULES[metric]
    best = max(c[primary] for c in scores)
    tied = [c for c in scores if best - c[primary] <= TIE]
    reason = d1
    if len(tied) > 1:
        best2 = max(c[second] for c in tied)
        tied = [c for c in tied if best2 - c[second] <= TIE]
        reason = f"tie on the primary metric; {d2}"
        if len(tied) > 1:
            tied = [min(tied, key=lambda c: c["eeg_weight"])]
            reason = f"tie on the primary metric and {d2.split(' ', 1)[1]}; smaller EEG weight"
    return tied[0], reason


def choose(scores):
    best_success = max(c["selection_success"] for c in scores)
    tied = [c for c in scores if best_success - c["selection_success"] <= TIE]
    reason = "highest mean final success"
    if len(tied) > 1:
        best_dist = min(c["selection_final_distance"] for c in tied)
        tied = [c for c in tied if c["selection_final_distance"] - best_dist <= TIE]
        reason = "tie on success; lowest mean final distance"
        if len(tied) > 1:
            tied = [min(tied, key=lambda c: c["eeg_weight"])]
            reason = "tie on success and distance; smaller EEG weight"
    return tied[0], reason


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--markdown", required=True)
    p.add_argument("--metric", choices=["success", "auc", "preferred_pass"], default="success")
    a = p.parse_args()
    scores = [score(s) for s in a.suites]
    if len({json.dumps(c["training_source_sha256"], sort_keys=True) for c in scores}) != 1:
        raise ValueError("Candidates used different training sources")
    if a.metric != "success":
        return main_generic(a, scores)
    winner, reason = choose(scores)
    lines = ["| candidate | EEG weight | main-decoder success (12 runs) | final distance (m) | normalized AUC |",
             "|---|---|---|---|---|"]
    for c in sorted(scores, key=lambda c: c["eeg_weight"]):
        lines.append(f"| `{Path(c['suite']).name}` | {c['eeg_weight']} | {c['selection_success']:.4f} | "
                     f"{c['selection_final_distance']:.4f} | {c['selection_auc']:.4f} |")
    rationale = (f"Validation-only selection by the rule declared in reports/rl_v2_experiment_plan.md before any "
                 f"validation run: {reason}. Candidates (EEG weight: mean final success / final distance m / AUC over "
                 f"the 12 EEGNet+CBraMod validation runs): " +
                 "; ".join(f"{Path(c['suite']).name} (w={c['eeg_weight']}): {c['selection_success']:.4f} / "
                           f"{c['selection_final_distance']:.4f} / {c['selection_auc']:.4f}"
                           for c in sorted(scores, key=lambda c: c['eeg_weight'])) +
                 f". Selected {Path(winner['suite']).name} (EEG weight {winner['eeg_weight']}). All three 33-run "
                 "suites are complete and preserved; no test-participant feedback was used.")
    result = {"rule": __doc__.split("Usage")[0].strip(), "candidates": scores, "selected": winner["suite"],
              "selected_eeg_weight": winner["eeg_weight"], "reason": reason, "rationale": rationale}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    Path(a.markdown).write_text("\n".join(lines) + "\n\n" + rationale + "\n")
    print(rationale)


def main_generic(a, scores):
    if a.metric == "preferred_pass" and any(c["selection_preferred_pass"] is None for c in scores):
        raise ValueError("preferred_pass selection needs preference suites")
    winner, reason = choose_by(scores, a.metric)
    primary = RULES[a.metric][0]
    source = {"auc": "FINAL_V3_HANDOFF.md Section 6.3 (protocol v3)",
              "preferred_pass": "reports/final_v3/preference_plan.md (route preference)"}[a.metric]
    lines = ["| candidate | EEG weight | selection metric (12 runs) | final success | final distance (m) | "
             "normalized AUC | preferred pass |", "|---|---|---|---|---|---|---|"]
    for c in sorted(scores, key=lambda c: c["eeg_weight"]):
        pp = "n/a" if c["selection_preferred_pass"] is None else f"{c['selection_preferred_pass']:.4f}"
        lines.append(f"| `{Path(c['suite']).name}` | {c['eeg_weight']} | {c[primary]:.4f} | {c['selection_success']:.4f} | "
                     f"{c['selection_final_distance']:.4f} | {c['selection_auc']:.4f} | {pp} |")
    rationale = (f"Validation-only selection by the rule declared in {source} before any validation run of this "
                 f"suite family: {reason}. Candidates (EEG weight: {primary} / final success over the 12 EEGNet+CBraMod "
                 "validation runs): " +
                 "; ".join(f"{Path(c['suite']).name} (w={c['eeg_weight']}): {c[primary]:.4f} / {c['selection_success']:.4f}"
                           for c in sorted(scores, key=lambda c: c["eeg_weight"])) +
                 f". Selected {Path(winner['suite']).name} (EEG weight {winner['eeg_weight']}). All candidate suites are "
                 "complete and preserved; no test-participant feedback was used.")
    result = {"rule": __doc__.split("Usage")[0].strip(), "metric": a.metric, "candidates": scores,
              "selected": winner["suite"], "selected_eeg_weight": winner["eeg_weight"], "reason": reason,
              "rationale": rationale}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    Path(a.markdown).write_text("\n".join(lines) + "\n\n" + rationale + "\n")
    print(rationale)


if __name__ == "__main__":
    main()
