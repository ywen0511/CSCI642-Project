#!/usr/bin/env python
"""Preference pilot report: final-policy metrics of every pilot run and the declared gate per variant.

Gate (reports/final_v3/decisions.jsonl, declared before any pilot result): mean final preferred-pass rate of ideal
minus that of sparse >= 0.20 AND ideal > sparse on each pilot seed. EEG pilots are listed but never used for the
gate.

Usage: python scripts/final_v3_pilot_report.py --variant NAME=RUNS_DIR [...] --out report.json
"""
import argparse
import json
import statistics
from pathlib import Path

KEYS = ("success_rate", "preferred_pass_rate", "violation_free_rate", "pillar_entry_rate",
        "right_corridor_entry_rate", "mean_path_length")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = {"gate_rule": "ideal - sparse mean final preferred-pass >= 0.20 and ideal > sparse on every pilot seed",
              "variants": {}}
    for item in a.variant:
        name, d = item.split("=", 1)
        runs = {}
        for run in sorted(Path(d).glob("*/run.json")):
            r = run.parent
            status = json.loads((r / "status.json").read_text()) if (r / "status.json").exists() else {}
            spec = json.loads(run.read_text())
            entry = {"condition": spec["condition"], "subject": spec["subject"], "seed": spec["seed"],
                     "completed": bool(status.get("completed")), "final_steps": status.get("final_steps"),
                     "config": {k: spec["config"][k] for k in ("eeg_weight", "total_timesteps", "preference")}}
            if (r / "evaluation.json").exists():
                ev = json.loads((r / "evaluation.json").read_text())
                entry.update({k: ev[k] for k in KEYS})
            runs[r.name] = entry
        by = lambda c: {x["seed"]: x["preferred_pass_rate"] for x in runs.values()
                        if x["condition"] == c and "preferred_pass_rate" in x}
        ideal, sparse = by("ideal"), by("sparse")
        seeds = sorted(set(ideal) & set(sparse))
        gate = None
        if seeds and len(seeds) == len(ideal) == len(sparse):
            diff = statistics.mean(ideal.values()) - statistics.mean(sparse.values())
            gate = {"ideal_mean": statistics.mean(ideal.values()), "sparse_mean": statistics.mean(sparse.values()),
                    "difference": diff, "ideal_better_every_seed": all(ideal[s] > sparse[s] for s in seeds),
                    "passed": diff >= 0.20 and all(ideal[s] > sparse[s] for s in seeds)}
        report["variants"][name] = {"runs_dir": d, "runs": runs, "gate": gate}
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    for n, v in report["variants"].items():
        print(n, v["gate"])


if __name__ == "__main__":
    main()
