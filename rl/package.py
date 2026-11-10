"""Package RL runs, including every retained checkpoint and replay buffer.

Scopes:
  suite (default)  a complete validation/final suite; summarize() must pass
  --smoke          the integration smoke directory; smoke_report.json must be ok
  --pilot          individual pilot / interrupted runs; NOT a passed suite. Every run is
                   archived; incomplete runs are included and flagged in the manifest.

Archives above --max-part-bytes are split losslessly into numbered parts
(<name>.part-00, ...); reassemble with `cat <name>.part-* > <name>`. SHA256SUMS lists every
uploaded file; manifest.json also lists full-archive hashes and every file inside each run.
"""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from .checkpoint import validate_snapshot
from .feedback import sha256
from .summarize import summarize


def file_inventory(run):
    return [{"path": str(p.relative_to(run.parent)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(run.rglob("*")) if p.is_file()]


def package(root, destination, smoke=False, pilot=False, scope_note=None, max_part_bytes=1_900_000_000):
    root, destination = Path(root), Path(destination)
    if smoke and pilot:
        raise ValueError("Choose one scope")
    if smoke:
        report = json.loads((root / "smoke_report.json").read_text())
        if not report.get("ok"):
            raise ValueError("Cannot package a failed smoke test as complete")
        scope = "integration-smoke-only; not a formal performance release"
    elif pilot:
        scope = "pilot/diagnostic runs; NOT a complete validation or final suite"
    else:
        report = summarize(root)
        scope = report["split"] + " suite"
    if scope_note:
        scope += f" ({scope_note})"
    runs = sorted(p.parent for p in root.glob("*/run.json"))
    if not runs:
        raise ValueError("No runs found")
    destination.mkdir(parents=True, exist_ok=False)
    inventory, run_records = [], []
    for run in runs:
        status_path = run / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"completed": False}
        completed = bool(status.get("completed"))
        if not completed and not pilot:
            raise ValueError(f"Incomplete run: {run}")
        checkpoints = sorted((run / "checkpoints").glob("*/metadata.json"))
        if completed and not checkpoints:
            raise ValueError(f"No checkpoints: {run}")
        for metadata in checkpoints:
            validate_snapshot(metadata.parent)
        archive = destination / (run.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(run, arcname=run.name)
            log = root / (run.name + ".log")
            if log.is_file():
                tar.add(log, arcname=run.name + ".log")
        run_records.append({"run": run.name, "completed": completed, "error": status.get("error"),
                            "final_steps": status.get("final_steps"), "snapshots": [m.parent.name for m in checkpoints],
                            "archive": archive.name, "files": file_inventory(run)})
        inventory.append({"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive)})
    metadata_archive = destination / "shared_evidence.tar.gz"
    with tarfile.open(metadata_archive, "w:gz") as tar:
        for path in sorted(root.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
    inventory.append({"name": metadata_archive.name, "bytes": metadata_archive.stat().st_size,
                      "sha256": sha256(metadata_archive)})
    parts = {}
    for item in inventory:
        path = destination / item["name"]
        if item["bytes"] > max_part_bytes:
            subprocess.run(["split", "-b", str(max_part_bytes), "-d", "-a", "2", str(path), str(path) + ".part-"],
                           check=True)
            parts[item["name"]] = [p.name for p in sorted(destination.glob(item["name"] + ".part-*"))]
            path.unlink()
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({
        "scope": scope, "source_dir": str(root), "assets": inventory, "split_parts": parts,
        "reassemble": "cat <name>.part-* > <name>; then check the full-archive sha256 in 'assets'",
        "runs": run_records,
        "required": "Upload ALL assets and SHA256SUMS to a new Release. Re-download, verify hashes, "
                    "run rl.verify_checkpoint and a separate continuation check."}, indent=2) + "\n")
    files = sorted(p for p in destination.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (destination / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    return inventory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--smoke", action="store_true")
    scope.add_argument("--pilot", action="store_true")
    p.add_argument("--scope-note")
    p.add_argument("--max-part-bytes", type=int, default=1_900_000_000)
    a = p.parse_args()
    print(json.dumps(package(a.runs, a.out, a.smoke, a.pilot, a.scope_note, a.max_part_bytes), indent=2))
