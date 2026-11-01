#!/usr/bin/env python
"""Assemble the RL-facing prediction tables, run the acceptance checks, and write
artifacts/manifest.json + artifacts/metrics_summary.json.

Main feedback library (README Section 9): the selected seed-42 runs of EEGNet and CBraMod ->
  artifacts/predictions/test_predictions.csv, artifacts/predictions/validation_predictions.csv
Additional decoder seeds -> artifacts/predictions/additional_seeds/*.csv
Supplementary baselines  -> artifacts/predictions/supplementary/*.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from errp.evaluate import PRED_COLUMNS  # noqa: E402
from errp.utils import env_info, git_sha, sha256_file, write_json  # noqa: E402


def run_files(run_dir: Path, root: Path) -> dict:
    out = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def check_table(df: pd.DataFrame, split: str, manifest: pd.DataFrame, models: list[str]) -> dict:
    assert list(df.columns) == PRED_COLUMNS, df.columns
    assert (df.split == split).all()
    for c in ["p_error_raw", "p_error"]:
        assert np.isfinite(df[c]).all() and df[c].between(0, 1).all(), c
    kept = manifest[manifest.retained == 1].set_index("trial_id")
    j = df.join(kept[["subject_id", "label", "split"]], on="trial_id", rsuffix="_man")
    assert j.subject_id_man.notna().all(), "trial_id not in manifest"
    assert (j.subject_id == j.subject_id_man).all() and (j.label == j.label_man.astype(int)).all()
    assert (j.split_man == split).all()
    ids = {}
    for (m, s), g in df.groupby(["model", "decoder_seed"]):
        assert g.trial_id.is_unique
        ids[(m, s)] = g.set_index("trial_id").label.sort_index()
    ref = next(iter(ids.values()))
    for k, v in ids.items():
        assert v.index.equals(ref.index) and (v.values == ref.values).all(), f"trial/label mismatch for {k}"
    expected = set(kept[kept.split == split].index)
    assert set(ref.index) == expected, "table does not cover all retained trials of the split"
    return {"rows": int(len(df)), "trials_per_model": int(len(ref)), "models": sorted({m for m, _ in ids}),
            "n_pos": int(ref.sum()), "n_neg": int((ref == 0).sum()), "checks": "passed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="artifacts root (contains runs/, selection/)")
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--extra-seed-runs", nargs="*", default=[])
    ap.add_argument("--supplementary-runs", nargs="*", default=[])
    a = ap.parse_args()
    art = Path(a.artifacts)
    runs = art / "runs"
    sel = {m: json.loads((art / "selection" / f"{m}.json").read_text()) for m in ["eegnet", "cbramod"]}
    main_runs = [sel["eegnet"]["selected_run_id"], sel["cbramod"]["selected_run_id"]]
    manifest = pd.read_csv(Path(a.prep_dir) / "trial_manifest.csv", dtype={"subject_id": str, "trial_id": str})
    prep_cfg = json.loads((Path(a.prep_dir) / "preprocessing_config.json").read_text())

    groups = {"main": main_runs, "additional_seeds": a.extra_seed_runs, "supplementary": a.supplementary_runs}
    checks = {}
    for g, rids in groups.items():
        if not rids:
            continue
        outdir = art / "predictions" if g == "main" else art / "predictions" / g
        outdir.mkdir(parents=True, exist_ok=True)
        for split in ["validation", "test"]:
            df = pd.concat([pd.read_csv(runs / r / f"{split}_predictions.csv", dtype={"subject_id": str, "trial_id": str})
                            for r in rids], ignore_index=True)
            checks[f"{g}/{split}"] = check_table(df, split, manifest, [])
            df.to_csv(outdir / f"{split}_predictions.csv", index=False, float_format="%.10g")

    # manifest.json ------------------------------------------------------------------
    entries, calibs, metrics_summary = {}, {}, {}
    for g, rids in groups.items():
        for r in rids:
            rd = runs / r
            rc = json.loads((rd / "run_config.json").read_text())
            met = json.loads((rd / "metrics.json").read_text())
            cal = json.loads((rd / "calibration.json").read_text())
            ck = rd / "checkpoints"
            ckpt_id = met.get("best_ckpt_sha256") and f"{r}:best" or f"{r}:model"
            entries[ckpt_id] = {
                "run_id": r, "group": g, "model": met["model"], "decoder_seed": met["decoder_seed"],
                "best_ckpt": str((ck / "best.ckpt").relative_to(art)) if (ck / "best.ckpt").exists() else None,
                "last_ckpt": str((ck / "last.ckpt").relative_to(art)) if (ck / "last.ckpt").exists() else None,
                "periodic_ckpts": sorted(str(p.relative_to(art)) for p in ck.glob("epoch_*.ckpt")),
                "other_model_files": sorted(str(p.relative_to(art)) for p in ck.glob("*.joblib")),
                "config": rc["config"], "source_sha": rc.get("source_sha"), "env": rc.get("env"),
                "best_epoch": met.get("best_epoch"), "best_val_auroc": met.get("best_val_metric"),
                "files": run_files(rd, art),
            }
            calibs[cal["calibration_id"]] = cal
            metrics_summary[r] = {
                "group": g, "model": met["model"], "decoder_seed": met["decoder_seed"],
                "test_pooled": met["test_pooled"], "test_subject_macro_calibrated@0.5": met["test_subject_macro_calibrated@0.5"],
                "test_subject_macro_raw@0.5": met["test_subject_macro_raw@0.5"],
                "test_per_subject_calibrated@0.5": met["test_per_subject_calibrated@0.5"],
                "validation": met["validation"], "temperature": cal["temperature"],
                "val_youden_threshold": cal["val_selected_threshold_youden_on_calibrated"],
            }
    prep_files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in sorted(Path(a.prep_dir).iterdir()) if p.is_file()}
    man = {
        "schema": "csci642-eeg-stage1-manifest/v1",
        "source_sha": git_sha(),
        "label_mapping": {"0": "non-error (S4)", "1": "error (S6 machine error)"},
        "probability_semantics": "p_error_raw = sigmoid(error-logit margin); p_error = temperature-calibrated P(error)",
        "splits": {"file": "configs/splits.json", "sha256": sha256_file(a.splits), **json.loads(Path(a.splits).read_text())},
        "preprocessing": {prep_cfg["preprocessing_id"]: {"config": prep_cfg, "files": prep_files,
                                                         "note": "derived EEG arrays are not uploaded (raw data license undeclared); "
                                                                 "rebuild with scripts/prepare_data.py"}},
        "checkpoints": entries,
        "calibrations": calibs,
        "selection": sel,
        "prediction_tables": {k: v for k, v in checks.items()},
        "export_env": env_info(),
    }
    write_json(man, art / "manifest.json")
    write_json(metrics_summary, art / "metrics_summary.json")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
