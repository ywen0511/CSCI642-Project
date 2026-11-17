#!/usr/bin/env python
"""Audit every expected historical EEG / RL v2 run against local files and the published Release manifests.

For each run: read status/run metadata, actual steps/epochs and snapshot inventory from disk; compare every file
(SHA-256) with the uploaded package manifest(s); classify every file as
  published     - inside an uploaded archive listed in the Release manifest (hash matches),
  recoverable   - not in any uploaded archive but present locally with the hash recorded in the manifest,
  local_only    - present locally, not listed in any manifest (e.g. created after packaging),
  missing       - listed in a manifest but absent / hash-mismatched locally and not published.
Run-level category: independently_published (all files published), recoverable_upload (some recoverable, none
missing), incomplete_execution (status not completed), missing_local_artifacts (any missing).

Usage (cluster, filer access):
  python scripts/decoder_improvement/audit_historical.py --inventory remote_inventory.json \
     --eeg-root <EEG project> --rl-root <RL project repo> --out historical_delivery_audit.json
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

RL_SCOPES = {  # tag -> (local suite dir under artifacts/, asset prefix in the Release)
    "rl-v2-smoke-20261110-cfe0305": [("rl_v2_smoke_r4", "smoke_r4__"), ("rl_v2_smoke", "superseded_rl_v2_smoke__"),
                                     ("rl_v2_smoke_r2", "superseded_rl_v2_smoke_r2__"),
                                     ("rl_v2_smoke_r3", "superseded_rl_v2_smoke_r3__")],
    "rl-v2-pilots-20261110-50888a9": [("rl_v2_pilots", "pilots__"),
                                      ("rl_v2_pilots_attempt1_failed_sb3_info_episode", "pilots_attempt1_failed__")],
    "rl-v2-validation-w010-20261110-cfe0305": [("rl_v2_validation/candidate_w010", "validation_w010__")],
    "rl-v2-validation-w050-20261110-cfe0305": [("rl_v2_validation/candidate_w050", "validation_w050__")],
    "rl-v2-validation-w100-20261110-cfe0305": [("rl_v2_validation/candidate_w100", "validation_w100__")],
    "rl-v2-final-20261110-25bb0f4": [("rl_v2_final", "final__")],
}
EXPECTED_RUNS = {"rl_v2_pilots": 15, "rl_v2_validation/candidate_w010": 33, "rl_v2_validation/candidate_w050": 33,
                 "rl_v2_validation/candidate_w100": 33, "rl_v2_final": 75}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "csci642-audit"}), timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def audit_rl(inv, rl_root):
    rl_root = Path(rl_root)
    rel = {r["tag"]: r for r in inv["releases"]}
    runs, gaps = [], []
    for tag, scopes in RL_SCOPES.items():
        assets = {a["name"]: a for a in rel[tag]["assets"]} if tag in rel else {}
        for suite, prefix in scopes:
            man_asset = assets.get(prefix + "manifest.json")
            manifest = fetch_json(man_asset["url"]) if man_asset else None
            published_runs = {r["run"]: r for r in (manifest or {}).get("runs", [])}
            uploaded = {n for n, a in assets.items() if a["state"] == "uploaded"}
            suite_dir = rl_root / "artifacts" / suite
            local_runs = sorted(p.parent.name for p in suite_dir.glob("*/run.json"))
            if suite in EXPECTED_RUNS:
                completed = [r for r in local_runs if json.loads((suite_dir / r / "status.json").read_text()).get("completed")]
                if len(completed) != EXPECTED_RUNS[suite]:
                    gaps.append(f"{suite}: {len(completed)} completed runs on disk, expected {EXPECTED_RUNS[suite]}")
            for name in sorted(set(local_runs) | set(published_runs)):
                run_dir = suite_dir / name
                rec = {"release": tag, "suite": suite, "run": name}
                status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
                rec.update(completed=bool(status.get("completed")), final_steps=status.get("final_steps"),
                           error=status.get("error"))
                snaps = sorted(p.parent.name for p in run_dir.glob("checkpoints/*/metadata.json"))
                rec["local_snapshots"] = snaps
                pub = published_runs.get(name)
                archive = prefix + (pub["archive"] if pub else name + ".tar.gz")
                archive_uploaded = archive in uploaded or any(n.startswith(archive + ".part-") for n in uploaded)
                counts = Counter()
                missing, recoverable = [], []
                listed = set()
                for f in (pub or {}).get("files", []):
                    rel_path = f["path"].split("/", 1)[1]
                    listed.add(rel_path)
                    local = run_dir / rel_path
                    in_archive = f.get("in_archive", True) and archive_uploaded
                    ok_local = local.is_file() and local.stat().st_size == f["bytes"] and sha256(local) == f["sha256"]
                    if in_archive:
                        counts["published"] += 1
                    elif ok_local:
                        counts["recoverable"] += 1
                        recoverable.append(rel_path)
                    else:
                        counts["missing"] += 1
                        missing.append(rel_path)
                for p in run_dir.rglob("*") if run_dir.exists() else []:
                    if p.is_file() and str(p.relative_to(run_dir)) not in listed:
                        counts["local_only"] += 1
                rec.update(published_in_manifest=pub is not None, archive=archive, archive_uploaded=archive_uploaded,
                           file_counts=dict(counts), recoverable_files=recoverable, missing_files=missing,
                           recoverable_periodic_snapshots=sorted({r.split("/")[1] for r in recoverable
                                                                  if r.startswith("checkpoints/step_")}))
                if not rec["completed"]:
                    cat = "incomplete_execution"
                elif counts["missing"]:
                    cat = "missing_local_artifacts"
                elif counts["recoverable"] or counts["local_only"] and not pub:
                    cat = "recoverable_upload"
                else:
                    cat = "independently_published"
                rec["category"] = cat
                runs.append(rec)
    return runs, gaps


def audit_eeg(inv, eeg_root):
    eeg_root = Path(eeg_root)
    rel = {r["tag"]: r for r in inv["releases"]}
    tag = "eeg-stage1-20261101-bccbb8a"
    assets = {a["name"]: a for a in rel[tag]["assets"]}
    local_rel = eeg_root / "release" / tag
    out = []
    sums = {}
    if (local_rel / "SHA256SUMS").exists():
        sums = dict(reversed(l.split()) for l in (local_rel / "SHA256SUMS").read_text().splitlines() if l.strip())
    for run_dir in sorted((eeg_root / "artifacts" / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        name = f"run_{run_dir.name}.tar.gz"
        ck = sorted(p.name for p in (run_dir / "checkpoints").glob("*.ckpt")) if (run_dir / "checkpoints").exists() else []
        summ = json.loads((run_dir / "train_summary.json").read_text()) if (run_dir / "train_summary.json").exists() else {}
        local_pkg = local_rel / name
        rec = {"release": tag, "run": run_dir.name, "checkpoints_local": ck, "epochs_completed": summ.get("epochs_completed"),
               "best_epoch": summ.get("best_epoch"), "asset_uploaded": assets.get(name, {}).get("state") == "uploaded",
               "asset_bytes": assets.get(name, {}).get("size"),
               "local_package_matches_sums": bool(local_pkg.exists() and sums.get(name) == sha256(local_pkg))}
        rec["category"] = "independently_published" if rec["asset_uploaded"] else (
            "recoverable_upload" if ck else "missing_local_artifacts")
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inventory", required=True)
    p.add_argument("--eeg-root", required=True)
    p.add_argument("--rl-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    inv = json.loads(Path(a.inventory).read_text())
    rl_runs, gaps = audit_rl(inv, a.rl_root)
    eeg_runs = audit_eeg(inv, a.eeg_root)
    by = defaultdict(Counter)
    for r in rl_runs:
        by[r["release"]][r["category"]] += 1
    for r in eeg_runs:
        by[r["release"]][r["category"]] += 1
    result = {"inventory_generated_utc": inv["generated_utc"],
              "releases": [{k: r[k] for k in ("tag", "n_assets", "n_uploaded", "draft")} for r in inv["releases"]],
              "category_counts": {k: dict(v) for k, v in by.items()}, "suite_gaps": gaps,
              "recoverable_runs": sorted(f"{r['suite']}/{r['run']}" for r in rl_runs if r["category"] == "recoverable_upload"),
              "missing_runs": sorted(f"{r['suite']}/{r['run']}" for r in rl_runs if r["category"] == "missing_local_artifacts"),
              "eeg_runs": eeg_runs, "rl_runs": rl_runs}
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("category_counts", "suite_gaps")}, indent=1))
    print("recoverable runs:", len(result["recoverable_runs"]), "missing runs:", len(result["missing_runs"]))


if __name__ == "__main__":
    main()
