#!/usr/bin/env python
"""Apply the declared validation-only selection rule to complete candidate suites.

Rule (reports/rl_v2_experiment_plan.md, Section 4): highest mean final-step success over the 12 main-decoder
runs (EEGNet + CBraMod, participants s11/s13, validation seeds); ties (1e-9): lower mean final distance over
the same runs, then the smaller EEG weight. Every candidate must be a complete suite (rl.summarize).

Usage: python scripts/rl_select_candidate.py <suite> [<suite> ...] --out <json> --markdown <md>
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
    return {"suite": str(suite), "eeg_weight": s["config"]["eeg_weight"],
            "dense_weight": s["config"]["dense_weight"], "n_runs": len(s["runs"]),
            "selection_success": statistics.mean(r["success_rate"] for r in main),
            "selection_final_distance": statistics.mean(r["mean_final_distance"] for r in main),
            "selection_auc": statistics.mean(r["normalized_auc_success"] for r in main),
            "n_selection_runs": len(main), "per_condition": per_condition,
            "training_source_sha256": s["training_source_sha256"], "training_git_shas": s["training_git_shas"]}


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
    a = p.parse_args()
    scores = [score(s) for s in a.suites]
    if len({json.dumps(c["training_source_sha256"], sort_keys=True) for c in scores}) != 1:
        raise ValueError("Candidates used different training sources")
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


if __name__ == "__main__":
    main()
