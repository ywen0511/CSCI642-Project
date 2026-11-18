#!/usr/bin/env python
"""Compute usage per suite from the run records: number of runs, completed runs, total training wall time
(status.json elapsed_seconds = core-hours, one core per run), per-run min/median/max, the suite wall-clock span
(earliest run.json mtime to latest status.json mtime), hosts (from the LSF job logs when present) and disk usage.

Usage: python scripts/final_v3_usage.py --runs-root DIR --suites NAME [...] --out usage.json
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--suites", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = {}
    for name in a.suites:
        root = Path(a.runs_root) / name
        runs = sorted(r.parent for r in root.glob("*/run.json"))
        el, starts, ends, done = [], [], [], 0
        for r in runs:
            starts.append((r / "run.json").stat().st_mtime)
            st = r / "status.json"
            if st.exists():
                s = json.loads(st.read_text())
                ends.append(st.stat().st_mtime)
                if s.get("completed"):
                    done += 1
                    el.append(s["elapsed_seconds"])
        du = subprocess.run(["du", "-sb", str(root)], capture_output=True, text=True).stdout.split()
        out[name] = {"runs": len(runs), "completed": done, "core_hours": sum(el) / 3600 if el else 0.0,
                     "per_run_minutes": {"min": min(el) / 60, "median": statistics.median(el) / 60,
                                         "max": max(el) / 60} if el else None,
                     "wall_clock_hours": (max(ends) - min(starts)) / 3600 if ends else None,
                     "disk_bytes": int(du[0]) if du else None}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(k, v["runs"], v["completed"], round(v["core_hours"], 1), v["wall_clock_hours"] and round(v["wall_clock_hours"], 2))


if __name__ == "__main__":
    main()
