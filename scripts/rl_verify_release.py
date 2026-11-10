#!/usr/bin/env python
"""Fresh-download verification of an RL v2 Release.

1. List the Release's assets through the public GitHub API (paginated) and download every asset into a NEW
   directory (no token needed for this public repository).
2. Check the inventory (names and byte sizes against the API) and every SHA-256 in SHA256SUMS.
3. Reassemble split archives (<name>.part-NN) and check each full-archive SHA-256 against the package
   manifest(s) (and SHA256SUMS.full-files when present).
4. Extract every run archive into a NEW directory (never over the originals) and validate the SHA-256 of every
   file of every snapshot (checkpoint.validate_snapshot).
5. Run rl.verify_checkpoint (reload, replay restore, fixed inference probes on CPU) on every final snapshot and
   on --periodic-per-condition periodic snapshots per condition.

Usage: python scripts/rl_verify_release.py --tag TAG --download-dir NEW --extract-dir NEW --out report.json
"""
import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.checkpoint import validate_snapshot  # noqa: E402
from rl.verify_checkpoint import verify  # noqa: E402

REPO = "ywen0511/CSCI642-Project"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def api(url):
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "csci642-verify"})
    for attempt in range(5):
        try:
            with urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def list_assets(tag):
    rel = api(f"https://api.github.com/repos/{REPO}/releases/tags/{tag}")
    assets, page = [], 1
    while True:
        batch = api(f"https://api.github.com/repos/{REPO}/releases/{rel['id']}/assets?per_page=100&page={page}")
        assets += batch
        if len(batch) < 100:
            break
        page += 1
    return rel, assets


def download(url, dest, size):
    for attempt in range(6):
        try:
            req = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "csci642-verify"})
            with urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                for block in iter(lambda: r.read(1 << 20), b""):
                    f.write(block)
            if dest.stat().st_size == size:
                return attempt
        except Exception:
            pass
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Download failed: {dest.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--download-dir", required=True)
    p.add_argument("--extract-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--periodic-per-condition", type=int, default=1)
    a = p.parse_args()
    dl, ex = Path(a.download_dir), Path(a.extract_dir)
    dl.mkdir(parents=True, exist_ok=False)
    ex.mkdir(parents=True, exist_ok=False)
    rel, listed_assets = list_assets(a.tag)
    # assets whose upload never completed (state != "uploaded") are not downloadable; record, do not verify
    assets = [x for x in listed_assets if x.get("state") == "uploaded"]
    report = {"not_uploaded_assets": [x["name"] for x in listed_assets if x.get("state") != "uploaded"],
              "tag": a.tag, "release_url": rel["html_url"], "target_commitish": rel["target_commitish"],
              "n_assets_api": len(assets), "bytes_api": sum(x["size"] for x in assets), "ok": False}
    retries = 0
    for asset in sorted(assets, key=lambda x: x["name"]):
        retries += download(asset["browser_download_url"], dl / asset["name"], asset["size"])
    report["download_retries"] = retries
    on_disk = {f.name: f.stat().st_size for f in dl.iterdir()}
    report["inventory_ok"] = on_disk == {x["name"]: x["size"] for x in assets}
    sums = dict(reversed(line.split()) for line in (dl / "SHA256SUMS").read_text().splitlines() if line.strip())
    listed = set(sums) | {"SHA256SUMS"}
    report["sha256sums_cover_all_assets"] = listed == set(on_disk)
    bad = [n for n, h in sums.items() if sha256(dl / n) != h]
    report["sha256_mismatches"] = bad
    # reassemble split archives and check full-archive hashes from the package manifests
    full_expected = {}
    for manifest in dl.glob("*manifest.json"):
        prefix = manifest.name[: -len("manifest.json")]
        for item in json.loads(manifest.read_text())["assets"]:
            full_expected[prefix + item["name"]] = item["sha256"]
    if (dl / "SHA256SUMS.full-files").exists():
        for line in (dl / "SHA256SUMS.full-files").read_text().splitlines():
            h, n = line.split()
            full_expected[n] = h
    groups = defaultdict(list)
    for f in sorted(dl.iterdir()):
        m = re.match(r"(.+)\.part-(\d+)$", f.name)
        if m:
            groups[m.group(1)].append(f)
    assembled = Path(str(ex) + "_assembled")
    assembled.mkdir(exist_ok=False)
    archives = {}
    for name, parts in groups.items():
        out = assembled / name
        with open(out, "wb") as w:
            for part in parts:
                w.write(part.read_bytes())
        archives[name] = out
    for f in dl.glob("*.tar.gz"):
        archives[f.name] = f
    full_bad = [n for n, path in archives.items() if n in full_expected and sha256(path) != full_expected[n]]
    report.update(n_split_archives=len(groups), n_archives=len(archives),
                  full_archive_hash_checked=sum(n in full_expected for n in archives), full_archive_mismatches=full_bad)
    # extract into a fresh directory
    for name, path in sorted(archives.items()):
        dest = ex / name[: -len(".tar.gz")]
        dest.mkdir(parents=True, exist_ok=False)
        with tarfile.open(path) as tar:
            tar.extractall(dest, filter="data")
    # reassembled copies are reconstructible from the verified parts; remove them to bound disk use
    for path in assembled.iterdir():
        path.unlink()
    assembled.rmdir()
    snapshots = sorted(m.parent for m in ex.glob("*/*/checkpoints/*/metadata.json"))
    snap_bad = []
    for s in snapshots:
        try:
            validate_snapshot(s)
        except Exception as exc:  # noqa: BLE001
            snap_bad.append(f"{s}: {exc}")
    report.update(n_snapshots_hash_validated=len(snapshots), snapshot_hash_failures=snap_bad)
    # inference/reload verification
    by_condition = defaultdict(list)
    for s in snapshots:
        if s.name.startswith("step_"):
            run = json.loads((s.parent.parent / "run.json").read_text())
            by_condition[run["condition"]].append(s)
    targets = [s for s in snapshots if s.name.startswith("final_")]
    for cond, snaps in sorted(by_condition.items()):
        snaps = sorted(snaps)
        targets += snaps[len(snaps) // 2: len(snaps) // 2 + a.periodic_per_condition]
    results, max_diff = {}, 0.0
    for s in targets:
        try:
            r = verify(s)
            max_diff = max(max_diff, r["max_action_difference"])
            results[str(s.relative_to(ex))] = {"ok": True, "max_action_difference": r["max_action_difference"],
                                               "steps": r["steps"], "updates": r["updates"], "replay_size": r["replay_size"]}
        except Exception as exc:  # noqa: BLE001
            results[str(s.relative_to(ex))] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report.update(n_checkpoints_inference_verified=len(results),
                  n_final_snapshots_verified=sum(1 for k in results if "/final_" in k),
                  n_periodic_snapshots_verified=sum(1 for k in results if "/step_" in k),
                  inference_failures=[k for k, v in results.items() if not v["ok"]],
                  max_action_difference=max_diff, tolerance=1e-6, verification=results)
    report["ok"] = bool(not report["not_uploaded_assets"] and report["inventory_ok"] and report["sha256sums_cover_all_assets"] and not bad and not full_bad
                        and not snap_bad and not report["inference_failures"] and targets)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "verification"}, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
