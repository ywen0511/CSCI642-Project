"""Per-run learning-curve metrics computed from evaluations.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

CURVE_METRICS = ("success_rate", "ever_success_rate", "mean_final_distance", "mean_task_return")


def load_curve(path):
    """Evaluation records sorted by step, with identical duplicate records removed.

    The runner evaluates at the last periodic boundary and again after saving the final
    snapshot, so the final step can appear twice. Identical duplicates are dropped; a
    duplicate step with different results is an error.
    """
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return dedupe(records)


def dedupe(records):
    by_step = {}
    for record in records:
        step = record["steps"]
        if step in by_step:
            if by_step[step] != record:
                raise ValueError(f"Inconsistent duplicate evaluation at step {step}")
            continue
        by_step[step] = record
    return [by_step[s] for s in sorted(by_step)]


def normalized_auc(curve, budget, metric="success_rate"):
    """Trapezoidal area under metric(step) from step 0 to the full budget, divided by the budget.

    Requires an evaluation at step 0 and at the final budget step, so a run that learns
    earlier gets a larger value; for success rate the result lies in [0, 1].
    """
    if budget <= 0:
        raise ValueError("Budget must be positive")
    steps = [r["steps"] for r in curve]
    if not steps or steps[0] != 0 or steps[-1] != budget:
        raise ValueError(f"Curve must start at step 0 and end at the budget {budget}: {steps[:1]}..{steps[-1:]}")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Evaluation steps must be strictly increasing")
    values = [float(r[metric]) for r in curve]
    area = sum((s1 - s0) * (v0 + v1) / 2 for s0, s1, v0, v1 in zip(steps, steps[1:], values, values[1:]))
    return area / budget


def run_metrics(run_dir, budget):
    """Final-policy metrics (from evaluation.json) plus learning-speed AUC (from the curve)."""
    run_dir = Path(run_dir)
    final = json.loads((run_dir / "evaluation.json").read_text())
    curve = load_curve(run_dir / "evaluations.jsonl")
    if curve[-1] != final:
        raise ValueError(f"Final evaluation differs from the last curve record: {run_dir}")
    out = {k: final[k] for k in CURVE_METRICS}
    out["normalized_auc_success"] = normalized_auc(curve, budget)
    return out, curve
