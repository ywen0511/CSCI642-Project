#!/usr/bin/env python
"""Short continuation from a freshly downloaded v2 snapshot, with its exact config and identity.

Run from a clean checkout of the snapshot's recorded training commit (the source-hash gate in
rl.checkpoint.restore_snapshot rejects other training sources). Writes the snapshot's own config to a file,
continues training into a NEW directory, then checks: steps and gradient updates increase, the replay buffer
grows and keeps the restored transitions unchanged, and the new final snapshot verifies (hashes, reload,
inference probes). Not part of any performance summary.

Usage: python scripts/rl_continuation_check.py --snapshot DIR --feedback-dir DIR --steps 2000 --work NEW --out report.json
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.checkpoint import validate_snapshot  # noqa: E402
from rl.verify_checkpoint import verify  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--feedback-dir", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkout", default=str(Path(__file__).resolve().parents[1]),
                   help="clean checkout of the snapshot's training commit; rl.train runs from here")
    a = p.parse_args()
    checkout = Path(a.checkout).resolve()
    snap, work = Path(a.snapshot).resolve(), Path(a.work)
    work.mkdir(parents=True, exist_ok=False)
    meta = validate_snapshot(snap)
    spec = meta["run_spec"]
    config = work / "snapshot_config.json"
    config.write_text(json.dumps(spec["config"], indent=2) + "\n")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout, text=True).strip()
    if dirty:
        raise ValueError("Checkout is not clean")
    cmd = [sys.executable, "-m", "rl.train", "--config", str(config), "--phase", spec["phase"],
           "--condition", spec["condition"], "--seed", str(spec["seed"]), "--feedback-dir", a.feedback_dir,
           "--resume", str(snap), "--steps", str(a.steps), "--out", str(work / "continued")]
    if spec["subject"]:
        cmd += ["--subject", spec["subject"]]
    with (work / "continue.log").open("w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=checkout).returncode
    report = {"snapshot": str(snap), "snapshot_steps": meta["num_timesteps"], "snapshot_updates": meta["n_updates"],
              "snapshot_replay_size": meta["replay_size"], "checkout_git_sha": git, "command": cmd, "returncode": rc,
              "ok": False}
    if rc == 0:
        status = json.loads((work / "continued" / "status.json").read_text())
        new = Path(status["final_checkpoint"])
        v = verify(new)
        with open(snap / "replay_buffer.pkl", "rb") as f:
            old_buf = pickle.load(f)
        with open(new / "replay_buffer.pkl", "rb") as f:
            new_buf = pickle.load(f)
        n_old = old_buf.size()
        kept = all(np.array_equal(old_buf.observations[k][:n_old], new_buf.observations[k][:n_old])
                   for k in old_buf.observations) and np.array_equal(old_buf.actions[:n_old], new_buf.actions[:n_old])
        report.update(status=status, verification=v, restored_transitions_unchanged=bool(kept),
                      steps_increased=status["final_steps"] == meta["num_timesteps"] + a.steps,
                      updates_increased=status["updates"] > meta["n_updates"],
                      replay_grew=status["replay_size"] > meta["replay_size"])
        report["ok"] = bool(v["ok"] and kept and report["steps_increased"] and report["updates_increased"]
                            and report["replay_grew"] and status["resume_from"] == str(snap))
    Path(a.out).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("verification", "command")}, indent=2, default=str))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
