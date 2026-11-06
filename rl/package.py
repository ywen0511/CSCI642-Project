"""Package completed RL runs, including every retained checkpoint and replay buffer."""
import argparse
import json
import tarfile
from pathlib import Path

from .checkpoint import validate_snapshot
from .feedback import sha256
from .summarize import summarize


def package(root, destination, smoke=False):
    root, destination = Path(root), Path(destination)
    if smoke:
        report = json.loads((root / "smoke_report.json").read_text())
        if not report.get("ok"):
            raise ValueError("Cannot package a failed smoke test as complete")
        scope = "integration-smoke-only; not a formal performance release"
    else:
        report = summarize(root)
        scope = report["split"] + " suite"
    runs = sorted(p.parent for p in root.glob("*/status.json"))
    if not runs:
        raise ValueError("No runs found")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = []
    for run in runs:
        status = json.loads((run / "status.json").read_text())
        if not status.get("completed"):
            raise ValueError(f"Incomplete run: {run}")
        checkpoints = sorted((run / "checkpoints").glob("*/metadata.json"))
        if not checkpoints:
            raise ValueError(f"No checkpoints: {run}")
        for metadata in checkpoints:
            validate_snapshot(metadata.parent)
        archive = destination / (run.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(run, arcname=run.name)
        inventory.append({"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive)})
    metadata_archive = destination / "shared_evidence.tar.gz"
    with tarfile.open(metadata_archive, "w:gz") as tar:
        for path in sorted(root.iterdir()):
            if path.is_file():
                tar.add(path, arcname=path.name)
    inventory.append({"name": metadata_archive.name, "bytes": metadata_archive.stat().st_size,
                      "sha256": sha256(metadata_archive)})
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"scope": scope, "assets": inventory,
                                    "required": "Upload ALL assets and SHA256SUMS to a new Release. Re-download, verify hashes, run rl.verify_checkpoint and a separate continuation check."}, indent=2) + "\n")
    files = sorted(p for p in destination.iterdir() if p.is_file())
    (destination / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    return inventory


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(json.dumps(package(a.runs, a.out, a.smoke), indent=2))
