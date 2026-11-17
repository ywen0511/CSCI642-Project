#!/usr/bin/env python
"""Acquire the pinned external inputs of the decoder study and fail loudly on any version/checksum mismatch.

Usage: python scripts/decoder_improvement/acquire_inputs.py --out DIR [--only dataset|weights]
Creates DIR/data/raw/dataset-ErrP-HRI (raw ErrP-HRI files at commit 4bc3e524...) and
DIR/weights/CBraMod/pretrained_weights.pth (Hugging Face weighting666/CBraMod at revision 500543c7...), checks every
file against configs/decoder_improvement/external_inputs.json (sizes and SHA-256 from the original acquisition;
reports/data/raw_files.csv), and writes DIR/acquisition_report.json. Exit status 2 = blocked (network/upstream
unavailable), 1 = mismatch, 0 = verified. Raw EEG and weights are NOT redistributed by this project.
"""
import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs" / "decoder_improvement" / "external_inputs.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "csci642-acquire"}),
                                        timeout=600) as r, open(dest, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
            return
        except Exception as exc:
            last = exc
            time.sleep(10 * (i + 1))
    raise ConnectionError(f"{url}: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["dataset", "weights"])
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"spec_sha256": sha256(SPEC), "items": {}, "status": "verified"}
    try:
        if a.only in (None, "dataset"):
            ds = spec["dataset"]
            raw = out / ds["target_dir"]
            arc = out / "_dataset.tar.gz"
            if not raw.exists():
                fetch(ds["archive_url"], arc)
                tmp = out / "_dataset_extract"
                with tarfile.open(arc) as tf:
                    for m in tf.getmembers():
                        if m.name.startswith("/") or ".." in Path(m.name).parts or m.issym() or m.islnk():
                            raise ValueError(f"unsafe archive member {m.name}")
                    tf.extractall(tmp, filter="data")
                top = next(tmp.iterdir())
                raw.parent.mkdir(parents=True, exist_ok=True)
                top.rename(raw)
                tmp.rmdir()
                arc.unlink()
            bad = []
            for f in ds["files"]:
                p = raw / f["path"]
                if not p.is_file() or p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
                    bad.append(f["path"])
            rep["items"]["dataset"] = {"url": ds["archive_url"], "revision": ds["revision"], "n_files_checked": len(ds["files"]),
                                       "mismatched": bad}
            if bad:
                rep["status"] = "mismatch"
        if a.only in (None, "weights"):
            w = spec["weights"]
            dest = out / w["target"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                fetch(w["url"], dest)
            ok = dest.stat().st_size == w["bytes"] and sha256(dest) == w["sha256"]
            rep["items"]["weights"] = {"url": w["url"], "revision": w["revision"], "bytes": dest.stat().st_size,
                                       "sha256": sha256(dest), "ok": ok}
            if not ok:
                rep["status"] = "mismatch"
    except (ConnectionError, OSError) as exc:
        rep["status"] = "blocked"
        rep["error"] = f"{type(exc).__name__}: {exc}"
    (out / "acquisition_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1)[:2000])
    sys.exit({"verified": 0, "mismatch": 1, "blocked": 2}[rep["status"]])


if __name__ == "__main__":
    main()
