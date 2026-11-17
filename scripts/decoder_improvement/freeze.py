#!/usr/bin/env python
"""Pre-test freeze after the five-seed confirmation (Stage F). No reselection happens here.

Usage: python scripts/decoder_improvement/freeze.py --plan-f PLANS/stage_F.json --decisions-dir DEC --runs-root RUNS --out DEC/freeze.json

For every recipe in Stage F (selected candidate and historical-recipe reference per architecture):
  - 15 CV fits (3 folds x seeds 42-46); final epoch count = ceil(median one-based best epoch), clamped to [1, max];
  - five-seed CV score (participant-macro of seed-mean AUROC) and the screening (42-44) vs new-seed (45-46) scores
    (selection optimism);
  - paired selected-minus-reference differences per participant and seed;
  - hashes of every CV prediction file and the training sources of the fits.
The calibration procedure and reporting conventions are restated from study.json so that the freeze is complete.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from errp.dimp.spec import load_study, selection_score, short_hash  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-f", required=True)
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    study = load_study()
    plan = json.loads(Path(a.plan_f).read_text())
    runs = Path(a.runs_root)
    recipes, by_arch = [], {}
    for c in plan["candidates"]:
        sts = [json.loads((runs / f / "status.json").read_text()) for f in c["fit_ids"]]
        assert all(s["state"] == "completed" for s in sts) and len(sts) == 15
        epochs = [s["best_epoch_one_based"] for s in sts]
        final_epochs = int(min(max(math.ceil(statistics.median(epochs)), 1), c["recipe"]["max_epochs"]))
        rows = [{"seed": s["seed"], "per_participant_auroc": s["per_participant_auroc"]} for s in sts]
        five = selection_score(rows)
        scr = selection_score([r for r in rows if r["seed"] in study["seeds"]["screening"]])
        new = selection_score([r for r in rows if r["seed"] not in study["seeds"]["screening"]])
        item = {"arch": c["arch"], "role": c["role"], "candidate_id": c["candidate_id"], "recipe_id": c["recipe_id"],
                "recipe": c["recipe"], "recipe_hash": short_hash(c["recipe"], 16),
                "identical_to_reference": c.get("identical_to_reference", False),
                "cv_best_epochs_one_based": epochs, "final_epochs": final_epochs,
                "final_epoch_rule": "ceil(median of 15 one-based CV best epochs), clamped to [1, max_epochs]",
                "scheduler_horizon_epochs": c["recipe"].get("scheduler_horizon_epochs"),
                "cv_score_5seed": five["score"], "cv_score_screening_seeds": scr["score"],
                "cv_score_new_seeds_45_46": new["score"],
                "per_participant_seed_mean_5seed": five["per_participant_seed_mean"],
                "per_fit": [{"fit_id": s["fit_id"], "fold": s["fold"], "seed": s["seed"],
                             "best_epoch_one_based": s["best_epoch_one_based"], "macro_auroc": s["macro_auroc"],
                             "per_participant_auroc": s["per_participant_auroc"],
                             "inner_val_predictions_sha256": sha(runs / s["fit_id"] / "inner_val_predictions.csv")}
                            for s in sts],
                "training_source_sha256": sts[0]["provenance"]["training_source_sha256"],
                "initialization_sha256": sts[0]["provenance"]["initialization_sha256"]}
        recipes.append(item)
        by_arch.setdefault(c["arch"], {})[c["role"]] = item
    paired = {}
    for arch, d in by_arch.items():
        if "reference" not in d:
            paired[arch] = {"note": "selected recipe identical to the reference; no distinct candidate"}
            continue
        sel = {(f["seed"], p): v for f in d["selected"]["per_fit"] for p, v in f["per_participant_auroc"].items()}
        ref = {(f["seed"], p): v for f in d["reference"]["per_fit"] for p, v in f["per_participant_auroc"].items()}
        diffs = [{"seed": k[0], "participant": k[1], "selected_minus_reference": sel[k] - ref[k]} for k in sorted(sel)]
        paired[arch] = {"pairs": diffs, "mean": statistics.mean(d_["selected_minus_reference"] for d_ in diffs),
                        "n_positive": sum(d_["selected_minus_reference"] > 0 for d_ in diffs), "n": len(diffs),
                        "five_seed_score_difference": d["selected"]["cv_score_5seed"] - d["reference"]["cv_score_5seed"],
                        "selection_optimism_selected": d["selected"]["cv_score_screening_seeds"] - d["selected"]["cv_score_new_seeds_45_46"]}
    freeze = {"schema": "csci642-dimp-freeze/v1", "study_sha": short_hash(study, 16), "plan_f_sha256": sha(a.plan_f),
              "decisions": {s: sha(Path(a.decisions_dir) / f"stage_{s}.json") for s in "ABCDE"
                            if (Path(a.decisions_dir) / f"stage_{s}.json").exists()},
              "recipes": recipes, "confirmation_paired_differences": paired,
              "final_seeds": study["seeds"]["confirmation_and_final"], "feedback_export_seed": study["seeds"]["feedback_export"],
              "final_training": "all seven outer-training participants, fixed epoch count, no early stopping or checkpoint "
                                "selection; the fixed-epoch final snapshot is the evaluation model",
              "calibration": study["calibration"],
              "reporting": {"controlled_comparison": "identity and temperature (per final model, fitted on s11/s13)",
                            "deployment": "selected candidates: method chosen by leave-one-validation-participant-out "
                                          "cross-fitting; references: temperature",
                            "summaries": "seed mean +/- sample SD; participant tables; paired differences on the same people/seeds",
                            "test_status": study["outer_split"]["test_status"]},
              "test_accessed_before_freeze": False}
    Path(a.out).write_text(json.dumps(freeze, indent=1) + "\n")
    for r in recipes:
        print(f"{r['arch']:8s} {r['role']:9s} {r['candidate_id']:40s} final_epochs={r['final_epochs']} "
              f"cv5={r['cv_score_5seed']:.4f} screen={r['cv_score_screening_seeds']:.4f} new={r['cv_score_new_seeds_45_46']:.4f}")
    print(json.dumps({k: {kk: v[kk] for kk in v if kk != "pairs"} for k, v in paired.items()}, indent=1))


if __name__ == "__main__":
    main()
