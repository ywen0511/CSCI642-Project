#!/usr/bin/env python
"""Restore a packaged RL suite from downloaded Release assets (FINAL_V3_HANDOFF.md, Section 5 step 1).

Reads the Release's SHA256SUMS (asset hashes), reassembles split archives and checks every full-archive hash
against manifest.json, extracts every run archive and shared_evidence.tar.gz into a NEW suite directory, and
checks every extracted file against the manifest's per-run inventory (files not in the archive, i.e. periodic
snapshots of final-snapshots-only packages, are reported as absent). rl.summarize must then pass on the result.

Usage: python scripts/final_v3_restore_suite.py --download DIR --scope validation_w010 --out SUITE_DIR --report JSON
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", required=True)
    p.add_argument("--scope", required=True, help="asset name prefix before '__' (e.g. validation_w010, final)")
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    dl, out = Path(a.download), Path(a.out)
    sums = {}
    for line in (dl / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    bad = [n for n, d in sums.items() if not (dl / n).is_file() or sha256(dl / n) != d]
    if bad:
        raise SystemExit(f"asset hash mismatch/missing: {bad}")
    pre = a.scope + "__"
    manifest = json.loads((dl / (pre + "manifest.json")).read_text())
    work = out.parent / (out.name + ".assembling")
    work.mkdir(parents=True, exist_ok=False)
    archives = {}
    for item in manifest["assets"]:
        name = item["name"]
        parts = manifest["split_parts"].get(name)
        target = work / name
        if parts:
            with target.open("wb") as f:
                for part in parts:
                    f.write((dl / (pre + part)).read_bytes())
        else:
            target.symlink_to((dl / (pre + name)).resolve())
        if sha256(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise SystemExit(f"full-archive hash mismatch: {name}")
        archives[name] = target
    out.mkdir(parents=True, exist_ok=False)
    for name, path in sorted(archives.items()):
        with tarfile.open(path) as tar:
            tar.extractall(out, filter="data")
    report = {"download": str(dl), "scope": a.scope, "suite": str(out), "manifest_scope": manifest["scope"],
              "source_dir": manifest["source_dir"], "assets_checked": len(sums),
              "archives_checked": len(archives), "runs": len(manifest["runs"]), "files_checked": 0,
              "files_absent_by_package_scope": 0, "mismatches": []}
    for run in manifest["runs"]:
        for f in run["files"]:
            path = out / f["path"]
            if not f.get("in_archive", True):
                report["files_absent_by_package_scope"] += 1
                if path.exists():
                    report["mismatches"].append(f"unexpected file present: {f['path']}")
                continue
            if not path.is_file() or path.stat().st_size != f["bytes"] or sha256(path) != f["sha256"]:
                report["mismatches"].append(f["path"])
            report["files_checked"] += 1
    for path in archives.values():
        path.unlink()
    work.rmdir()
    report["ok"] = not report["mismatches"]
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    if not report["ok"]:
        raise SystemExit(f"{len(report['mismatches'])} file mismatches")


if __name__ == "__main__":
    main()
