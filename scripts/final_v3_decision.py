#!/usr/bin/env python
"""Append one entry to reports/final_v3/decisions.jsonl (append-only; never edit earlier lines).

Usage: python scripts/final_v3_decision.py --kind rule|freeze|deviation|decision|selection --stage S --text "..." [--evidence path ...]
The entry records the UTC time and the current HEAD commit (the entry itself lands in the next commit).
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports/final_v3/decisions.jsonl"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=["rule", "freeze", "deviation", "decision", "selection", "cut"])
    p.add_argument("--stage", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--evidence", nargs="*", default=[])
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    entry = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "head_commit": head, "kind": a.kind, "stage": a.stage, "text": a.text, "evidence": a.evidence}
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))
