#!/usr/bin/env python
"""Dataset table of ErrP-Bench from the prepared directories: python bench/dataset_stats.py --B $B --out stats.json"""
import argparse
import json
from pathlib import Path

import pandas as pd

DS = ["errphri_robot", "errphri_cursor", "bnci2015_013", "erpcore_ern", "interhri"]
ap = argparse.ArgumentParser()
ap.add_argument("--B", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []
for d in DS:
    p = Path(a.B) / "data/prep" / d
    c = json.loads((p / "preprocessing_config.json").read_text())
    m = pd.read_csv(p / "trial_manifest.csv", dtype={"subject_id": str})
    m = m[m.retained == 1]
    ch = c.get("channels") or c["config"]["channels"]["order"]
    win = c.get("window_s") or [c["config"]["epoch"]["tmin_s"], c["config"]["epoch"]["tmax_s"]]
    per = m.groupby("subject_id").label.agg(["count", "sum"])
    rows.append({"dataset": d, "n_subjects": int(m.subject_id.nunique()), "n_epochs": int(len(m)),
                 "n_error": int(m.label.sum()), "error_rate": float(m.label.mean()), "n_channels": len(ch),
                 "window_s": win, "min_errors_per_subject": int(per["sum"].min()),
                 "max_errors_per_subject": int(per["sum"].max()),
                 "reference": c.get("reference", c.get("config", {}).get("reference")),
                 "license": (c.get("dataset") or {}).get("license") if isinstance(c.get("dataset"), dict) else None})
Path(a.out).write_text(json.dumps(rows, indent=1, default=str) + "\n")
for r in rows:
    print(r["dataset"], r["n_subjects"], r["n_epochs"], r["n_error"], round(r["error_rate"], 3), r["n_channels"], r["window_s"],
          r["min_errors_per_subject"], str(r["reference"])[:60], r["license"])
