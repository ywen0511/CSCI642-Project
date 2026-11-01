#!/usr/bin/env python
"""Package Release assets: one .tar.gz per run (all checkpoints, logs, predictions, metrics,
calibration), one shared-metadata archive, the RL-facing prediction tables and manifest as
individual assets, and SHA256SUMS. Archives above --max-part-bytes are split into parts
(reassemble with `cat <name>.part-* > <name>`); full-file checksums are listed as well.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tar_dir(src: Path, arcname: str, out: Path) -> None:
    # deterministic member order; gzip level 6
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        for p in sorted(src.rglob("*")):
            tf.add(p, arcname=f"{arcname}/{p.relative_to(src)}", recursive=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--repo", required=True, help="repository checkout (configs, reports)")
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-part-bytes", type=int, default=1_900_000_000)
    a = ap.parse_args()
    art, repo, out = Path(a.artifacts), Path(a.repo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = []
    for rd in sorted((art / "runs").iterdir()):
        # every run directory: evaluated runs and non-selected tuning candidates (best + last ckpts)
        if rd.is_dir() and ((rd / "metrics.json").exists() or (rd / "train_summary.json").exists()):
            dst = out / f"run_{rd.name}.tar.gz"
            tar_dir(rd, f"runs/{rd.name}", dst)
            assets.append(dst)
    # shared metadata
    stage = out / "_stage_shared"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data").mkdir(parents=True)
    for f in ["manifest.json", "metrics_summary.json"]:
        shutil.copy2(art / f, stage / f)
    shutil.copytree(art / "predictions", stage / "predictions")
    shutil.copytree(art / "selection", stage / "selection")
    for f in ["trial_manifest.csv", "preprocessing_config.json", "audit.json", "raw_files.csv"]:
        shutil.copy2(Path(a.prep_dir) / f, stage / "data" / f)
    shutil.copytree(repo / "configs", stage / "configs")
    for f in ["requirements.txt", "requirements-lock.txt", "SOURCE_SHA"]:
        if (repo / f).exists():
            shutil.copy2(repo / f, stage / f)
    dst = out / "shared_metadata.tar.gz"
    tar_dir(stage, "shared_metadata", dst)
    shutil.rmtree(stage)
    assets.append(dst)
    # RL-facing tables + manifest as plain assets
    for src, name in [(art / "predictions" / "test_predictions.csv", "test_predictions.csv"),
                      (art / "predictions" / "validation_predictions.csv", "validation_predictions.csv"),
                      (art / "manifest.json", "manifest.json"),
                      (art / "metrics_summary.json", "metrics_summary.json")]:
        shutil.copy2(src, out / name)
        assets.append(out / name)
    # split oversize archives
    final, full_sums = [], []
    for p in assets:
        if p.stat().st_size > a.max_part_bytes:
            full_sums.append(f"{sha256(p)}  {p.name}")
            subprocess.run(["split", "-b", str(a.max_part_bytes), "-d", "-a", "2", str(p), str(p) + ".part-"], check=True)
            final += sorted(out.glob(p.name + ".part-*"))
            p.unlink()
        else:
            final.append(p)
    lines = [f"{sha256(p)}  {p.name}" for p in final]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    if full_sums:
        (out / "SHA256SUMS.full-files").write_text("\n".join(full_sums) + "\n")
    for p in final:
        print(f"{p.stat().st_size:>12}  {p.name}")


if __name__ == "__main__":
    sys.exit(main())
