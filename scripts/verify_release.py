#!/usr/bin/env python
"""Verify downloaded Release assets.

1. sizes + SHA-256 of every downloaded asset against SHA256SUMS (and the expected list);
2. extract archives into a fresh directory;
3. reload each main-run best.ckpt from the download, recompute validation/test predictions on
   the locally rebuilt data, and compare with the released prediction tables
   (declared tolerance: max |dp| <= 1e-6 on p_error_raw and p_error);
4. the training-resumption check runs separately (scripts/check_resume.py --mode continue)
   on a copy of each downloaded last.ckpt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOL = 1e-6


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--expected-sizes", default=None, help="json {asset: bytes} from the GitHub API")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dl, ex = Path(a.download_dir), Path(a.extract_dir)
    res = {"tolerance_abs_p": TOL, "assets": {}, "reproduction": {}}
    sums = {}
    for line in (dl / "SHA256SUMS").read_text().splitlines():
        h, n = line.split()
        sums[n] = h
    sizes = json.loads(Path(a.expected_sizes).read_text()) if a.expected_sizes else {}
    ok = True
    for n, h in sums.items():
        p = dl / n
        r = {"present": p.exists()}
        if p.exists():
            r.update({"bytes": p.stat().st_size, "sha256": sha256(p)})
            r["sha256_ok"] = r["sha256"] == h
            if n in sizes:
                r["size_matches_api"] = sizes[n] == r["bytes"]
        ok &= r.get("sha256_ok", False) and r.get("size_matches_api", True)
        res["assets"][n] = r
    res["all_assets_ok"] = bool(ok)
    ex.mkdir(parents=True, exist_ok=False)  # must be fresh
    for p in sorted(dl.glob("*.tar.gz")):
        with tarfile.open(p) as tf:
            tf.extractall(ex, filter="data")
    import torch  # noqa: E402
    from errp.data import load_splits  # noqa: E402
    from errp.evaluate import margins_from_ckpt  # noqa: E402
    from errp.calibration import apply_calibration  # noqa: E402
    splits = load_splits(a.splits)
    manifest = json.loads((dl / "manifest.json").read_text())
    for ckpt_id, e in manifest["checkpoints"].items():
        if e["group"] != "main" or not e["best_ckpt"]:
            continue
        ck = ex / e["best_ckpt"]
        assert ck.exists(), ck
        rec = {"best_ckpt": str(ck), "best_ckpt_sha256": sha256(ck),
               "sha256_matches_manifest": sha256(ck) == e["files"][e["best_ckpt"]]["sha256"]}
        out, st = margins_from_ckpt(ck, a.prep_dir, splits, a.device)
        calib_id = f"{e['run_id']}:temperature"
        cal = manifest["calibrations"][calib_id]
        for split, fname in [("validation", "validation_predictions.csv"), ("test", "test_predictions.csv")]:
            d, z = out[split]
            tab = pd.read_csv(dl / fname, dtype={"trial_id": str})
            tab = tab[tab.checkpoint_id == ckpt_id].set_index("trial_id").loc[d.trial_ids]
            p_raw = 1 / (1 + np.exp(-z))
            p_cal = apply_calibration(z, cal)
            rec[split] = {"n": int(len(z)),
                          "max_abs_diff_p_error_raw": float(np.abs(p_raw - tab.p_error_raw.values).max()),
                          "max_abs_diff_p_error": float(np.abs(p_cal - tab.p_error.values).max()),
                          "labels_match": bool((d.y == tab.label.values).all())}
            rec[split]["within_tolerance"] = (rec[split]["max_abs_diff_p_error_raw"] <= TOL
                                              and rec[split]["max_abs_diff_p_error"] <= TOL)
        rec["ok"] = rec["sha256_matches_manifest"] and all(rec[s]["within_tolerance"] and rec[s]["labels_match"]
                                                          for s in ["validation", "test"])
        res["reproduction"][ckpt_id] = rec
    res["all_reproduction_ok"] = all(r["ok"] for r in res["reproduction"].values()) and bool(res["reproduction"])
    res["torch"] = torch.__version__
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"all_assets_ok": res["all_assets_ok"], "all_reproduction_ok": res["all_reproduction_ok"],
                      **{k: {s: v[s] for s in ["validation", "test"]} for k, v in res["reproduction"].items()}}, indent=1))


if __name__ == "__main__":
    main()
