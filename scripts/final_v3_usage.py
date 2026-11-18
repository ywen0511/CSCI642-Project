#!/usr/bin/env python
"""Compute usage per suite from the run records: number of runs, completed runs, total training wall time
(status.json elapsed_seconds = core-hours, one core per run), per-run min/median/max, the suite wall-clock span
(earliest run.json mtime to latest status.json mtime), hosts (from the LSF job logs when present) and disk usage.

Killed / aborted attempts (no status.json) are counted from their LSF job logs ("Run time : N sec.") with
--aborted NAME=LSF_LOG_DIR. --markdown writes the execution-record table.

Usage: python scripts/final_v3_usage.py --runs-root DIR --suites NAME [...] [--aborted NAME=DIR ...]
       --out usage.json [--markdown usage.md]
"""
import argparse
import re
import json
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--suites", nargs="+", required=True)
    p.add_argument("--aborted", action="append", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--markdown")
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
    for item in a.aborted:
        name, d = item.split("=", 1)
        secs = [int(m.group(1)) for f in sorted(Path(d).glob("*.out"))
                for m in [re.search(r"Run time :\s+(\d+) sec", f.read_text(errors="replace"))] if m]
        out[name] = {"runs": len(secs), "completed": 0, "core_hours": sum(secs) / 3600, "aborted": True,
                     "per_run_minutes": {"min": min(secs) / 60, "median": statistics.median(secs) / 60,
                                         "max": max(secs) / 60} if secs else None,
                     "wall_clock_hours": None, "disk_bytes": None}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    if a.markdown:
        rows = ["| suite | runs | completed | core-hours | per-run minutes (median) | suite wall-clock hours | disk GB |",
                "|---|---|---|---|---|---|---|"]
        fmt = lambda x, f: "n/a" if x is None else format(x, f)
        for k, v in out.items():
            rows.append(f"| `{k}`{' (aborted)' if v.get('aborted') else ''} | {v['runs']} | {v['completed']} | "
                        f"{v['core_hours']:.1f} | {fmt(v['per_run_minutes'] and v['per_run_minutes']['median'], '.1f')} | "
                        f"{fmt(v['wall_clock_hours'], '.2f')} | {fmt(v['disk_bytes'] and v['disk_bytes'] / 1e9, '.1f')} |")
        tot = sum(v["core_hours"] for v in out.values())
        rows.append(f"| **total** | {sum(v['runs'] for v in out.values())} | {sum(v['completed'] for v in out.values())} | "
                    f"**{tot:.1f}** | | | |")
        Path(a.markdown).write_text("\n".join(rows) + "\n")
    for k, v in out.items():
        print(k, v["runs"], v["completed"], round(v["core_hours"], 1), v["wall_clock_hours"] and round(v["wall_clock_hours"], 2))


if __name__ == "__main__":
    main()
