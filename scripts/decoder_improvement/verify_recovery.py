#!/usr/bin/env python
"""Checks on a downloaded RL v2 recovery Release (after release_tools.py fetch has verified every member hash).

Usage (RL environment python, repository root): python scripts/decoder_improvement/verify_recovery.py \
    --extract <fetch dir>/extract --feedback-dir <historical feedback tables> --work NEW_DIR --out report.json \
    [--continuation-steps 1000] [--continuation-all-conditions]

- Re-validates the SHA-256 of every file of every snapshot (rl.checkpoint.validate_snapshot).
- rl.verify_checkpoint (reload, replay restore, stored inference probe on CPU) on EVERY final snapshot and on the middle
  periodic snapshot of every condition.
- With --continuation-all-conditions: a short isolated continuation (scripts/rl_continuation_check.py) from a
  recovered periodic snapshot of one run per condition, in a new directory; requires the training sources of this
  checkout to equal the snapshot's (checked by the RL code).
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.checkpoint import validate_snapshot  # noqa: E402
from rl.verify_checkpoint import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", required=True)
    ap.add_argument("--feedback-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--continuation-steps", type=int, default=1000)
    ap.add_argument("--continuation-all-conditions", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=False)
    snaps = sorted(m.parent for m in Path(a.extract).glob("*/*/checkpoints/*/metadata.json"))
    rep = {"n_snapshots": len(snaps), "hash_failures": [], "verified": {}, "continuation": {}, "problems": []}
    by_cond = defaultdict(list)
    runs = {}
    for s in snaps:
        try:
            meta = validate_snapshot(s)
        except Exception as exc:  # noqa: BLE001
            rep["hash_failures"].append(f"{s}: {exc}")
            continue
        cond = meta["run_spec"]["condition"]
        if s.name.startswith("step_"):
            by_cond[cond].append(s)
        runs.setdefault(cond, set()).add(s.parent.parent)
    targets = [s for s in snaps if s.name.startswith("final_")]
    for cond, ss in sorted(by_cond.items()):
        ss = sorted(ss, key=lambda p: (p.parent.parent.name, p.name))
        targets.append(ss[len(ss) // 2])
    worst = 0.0
    for s in targets:
        try:
            r = verify(s)
            worst = max(worst, r["max_action_difference"])
            rep["verified"][str(s.relative_to(a.extract))] = {"ok": True, "max_action_difference": r["max_action_difference"],
                                                             "steps": r["steps"], "replay_size": r["replay_size"]}
        except Exception as exc:  # noqa: BLE001
            rep["verified"][str(s.relative_to(a.extract))] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            rep["problems"].append(f"verify {s}")
    rep["n_final_verified"] = sum(1 for k in rep["verified"] if "/final_" in k)
    rep["n_periodic_verified"] = sum(1 for k in rep["verified"] if "/step_" in k)
    rep["max_action_difference"] = worst
    if a.continuation_all_conditions:
        for cond in sorted(by_cond):
            run = sorted(runs[cond], key=lambda p: p.name)[0]
            per = sorted((run / "checkpoints").glob("step_*"))
            snap = per[len(per) // 2]
            out = work / f"continuation_{cond}.json"
            rc = subprocess.run([sys.executable, str(ROOT / "scripts/rl_continuation_check.py"), "--snapshot", str(snap),
                                 "--feedback-dir", a.feedback_dir, "--steps", str(a.continuation_steps),
                                 "--work", str(work / f"cont_{cond}"), "--out", str(out), "--checkout", str(ROOT)],
                                cwd=ROOT, capture_output=True, text=True).returncode
            r = json.loads(out.read_text()) if out.exists() else {"ok": False}
            rep["continuation"][cond] = {"run": run.name, "snapshot": snap.name, "returncode": rc, "ok": bool(r.get("ok")),
                                         **{k: r.get(k) for k in ("snapshot_steps", "snapshot_updates", "snapshot_replay_size",
                                                                  "restored_transitions_unchanged", "steps_increased",
                                                                  "updates_increased", "replay_grew", "checkout_git_sha")}}
            if not r.get("ok"):
                rep["problems"].append(f"continuation {cond}")
    rep["problems"] += [f"hash {h}" for h in rep["hash_failures"]]
    rep["ok"] = not rep["problems"] and bool(targets)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("n_snapshots", "n_final_verified", "n_periodic_verified", "max_action_difference", "ok")}),
          {c: v["ok"] for c, v in rep["continuation"].items()}, rep["problems"][:5])
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
