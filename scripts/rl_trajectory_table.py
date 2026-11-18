#!/usr/bin/env python
"""Summarize deterministic rollouts of ALL final policies on the shared goal seeds (rl.trajectories output).

For preference runs (episodes with a "preference" block, rl/preference.py) the table adds preferred-pass,
violation-free, pillar-entry and right-corridor rates.

Usage: python scripts/rl_trajectory_table.py reports/rl_v2_media/trajectories_all_final_runs/trajectories.json --out table.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["sparse", "dense", "eegnet", "cbramod", "ideal", "independent_eegnet", "independent_cbramod"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectories")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = json.loads(Path(a.trajectories).read_text())
    groups = defaultdict(list)
    for name, run in data.items():
        cond = re.sub(r"_(shared|s\d\d)_s\d+$", "", name)
        for e in run["episodes"]:
            path = np.array(e["gripper"])
            length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            straight = float(np.linalg.norm(path[0] - np.array(e["goal"])))
            row = {"success": e["final_success"], "distance": e["final_distance"],
                   "length": length, "ratio": length / straight}
            if "preference" in e:
                pr = e["preference"]
                row.update(preferred=pr["preferred_pass"], free=pr["violation_free"], pillar=pr["pillar_entry"],
                           right=pr["right_corridor_entry"])
            groups[cond].append(row)
    pref = any("preferred" in x for v in groups.values() for x in v)
    lines = ["| condition | episodes (runs x 3 goals) | final success | final distance (m) | mean path length (m) | "
             "median path / straight-line distance |" + (" preferred pass | violation-free | pillar entry | right corridor |"
                                                          if pref else ""), "|---|---|---|---|---|---|" + ("---|" * 4 if pref else "")]
    for c in ORDER:
        v = groups[c]
        if not v:
            continue
        lines.append(f"| {c} | {len(v)} | {np.mean([x['success'] for x in v]):.3f} | "
                     f"{np.mean([x['distance'] for x in v]):.4f} | {np.mean([x['length'] for x in v]):.3f} | "
                     f"{np.median([x['ratio'] for x in v]):.2f} |" +
                     ("".join(f" {np.mean([x[k] for x in v]):.3f} |" for k in ("preferred", "free", "pillar", "right"))
                      if pref else ""))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
