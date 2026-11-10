#!/usr/bin/env python
"""Summarize individual pilot runs (not a suite): curves, AUC, feedback diagnostics, runtime.

Usage: python scripts/rl_pilot_report.py <pilot_dir> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.analysis import feedback_diagnostics  # noqa: E402
from rl.metrics import dedupe, normalized_auc  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilot_dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = {}
    for run in sorted(d for d in Path(a.pilot_dir).iterdir() if (d / "run.json").exists()):
        spec = json.loads((run / "run.json").read_text())
        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        item = {"condition": spec["condition"], "subject": spec["subject"], "seed": spec["seed"],
                "eeg_weight": spec["config"]["eeg_weight"], "completed": status.get("completed", False),
                "error": status.get("error"), "elapsed_seconds": status.get("elapsed_seconds"),
                "device": json.loads((run / "environment.json").read_text()).get("device")
                if (run / "environment.json").exists() else None}
        if item["completed"]:
            records = [json.loads(x) for x in (run / "evaluations.jsonl").read_text().splitlines() if x.strip()]
            if len(records) > 1 and records[-1]["steps"] == records[-2]["steps"] and records[-1] != records[-2] \
                    and "n_updates" not in records[-1]:
                # runs from before 'skip periodic eval at the final boundary': the earlier record
                # is the pre-final-update policy; keep the final policy's evaluation
                item["dropped_preupdate_final_eval"] = records[-2]["success_rate"]
                records.pop(-2)
            curve = dedupe(records)
            budget = status["final_steps"]
            item["curve"] = [(r["steps"], r["success_rate"], round(r["mean_final_distance"], 4)) for r in curve]
            item["final_success"] = curve[-1]["success_rate"]
            item["final_distance"] = curve[-1]["mean_final_distance"]
            item["normalized_auc_success"] = normalized_auc(curve, budget)
            item["steps_per_second"] = budget / status["elapsed_seconds"]
            item["feedback"], _ = feedback_diagnostics(run, 10000)
        report[run.name] = item
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    for name, r in report.items():
        f = r.get("feedback", {})
        print(f"{name:40s} done={r['completed']} succ={r.get('final_success')} "
              f"dist={r.get('final_distance', float('nan')):.3f} auc={r.get('normalized_auc_success', float('nan')):.3f} "
              f"sps={r.get('steps_per_second', float('nan')):.1f} neutral={f.get('neutral_fraction', float('nan')):.3f} "
              f"err_frac={f.get('error_label_fraction', float('nan')):.3f} ev/ep={f.get('events_per_episode', float('nan')):.2f} "
              f"p0={f.get('mean_p_error_label0', float('nan')):.3f} p1={f.get('mean_p_error_label1', float('nan')):.3f} "
              f"evAUROC={f.get('event_auroc', float('nan')):.3f} eegR={f.get('eeg_reward_sum', 0):.1f} task={f.get('task_reward_sum', 0):.0f}")


if __name__ == "__main__":
    main()
