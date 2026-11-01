#!/usr/bin/env python
"""Select the best candidate of a config's grid by validation AUROC only, then calibrate
(validation), evaluate and export per-trial probabilities for the selected run.

  python scripts/select_and_evaluate.py --config configs/models/eegnet.yaml --runs-root artifacts/runs \
      --prep-dir data/derived/prep_v1 --selection-dir artifacts/selection
  # evaluate one specific run (extra seeds / supplementary models), no selection:
  python scripts/select_and_evaluate.py --config ... --run-id eegnet_lr1e-3_s43 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from errp.data import load_splits  # noqa: E402
from errp.evaluate import evaluate_run  # noqa: E402
from errp.tuning import candidates, load_yaml  # noqa: E402
from errp.utils import write_json  # noqa: E402


def model_tag(cfg: dict) -> str:
    n = cfg["model"]["name"]
    return n[3:] if n.startswith("bd_") else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--selection-dir", default=None)
    ap.add_argument("--run-id", default=None, help="evaluate this run only (no selection)")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    splits = load_splits(a.splits)
    prep_id = json.loads((Path(a.prep_dir) / "preprocessing_config.json").read_text())["preprocessing_id"]
    runs = Path(a.runs_root)
    if a.run_id:
        chosen = a.run_id
        seed = json.loads((runs / chosen / "run_config.json").read_text())["config"]["seed"]
    else:
        rows = []
        for c in candidates(a.config):
            s = json.loads((runs / c["run_id"] / "train_summary.json").read_text())
            rows.append({"run_id": c["run_id"], "overrides": c["overrides"], "best_val_auroc": s["best_metric"],
                         "best_epoch": s["best_epoch"], "epochs_completed": s["epochs_completed"],
                         "stopped_early": s["stopped_early"]})
        best = max(range(len(rows)), key=lambda i: (rows[i]["best_val_auroc"], -i))  # ties -> first listed
        chosen, seed = rows[best]["run_id"], cfg["seed"]
        sel = {"config": a.config, "model": model_tag(cfg), "criterion": "highest best-epoch validation AUROC; "
               "ties -> first listed candidate", "candidates": rows, "selected_run_id": chosen,
               "test_data_used_for_selection": False}
        if a.selection_dir:
            write_json(sel, Path(a.selection_dir) / f"{model_tag(cfg)}.json")
        print("[select]", json.dumps(sel))
    m = evaluate_run(runs / chosen, a.prep_dir, splits, chosen, model_tag(cfg), seed, prep_id)
    t = m["test_pooled"]["calibrated@0.5"]
    print("[eval]", chosen, json.dumps({"test_auroc": t["auroc"], "test_bacc": t["balanced_accuracy"],
                                        "test_brier": t["brier"], "T": m["calibration"]["temperature"]}))


if __name__ == "__main__":
    main()
