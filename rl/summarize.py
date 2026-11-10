"""Validate suite completeness and report seed variation within each participant."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .feedback import PROFILE, sha256
from .metrics import run_metrics
from .suite import jobs


def summarize(directory):
    root = Path(directory)
    plan = json.loads((root / "plan.json").read_text())
    expected = list(jobs(plan["phase"], json.loads(PROFILE.read_text()), plan["config"]))
    if plan["jobs"] != expected:
        raise ValueError("Suite plan does not match the declared full comparison")
    rows, evidence, git_shas, sources = [], {}, set(), set()
    for job in expected:
        run = root / job["name"]
        spec = json.loads((run / "run.json").read_text())
        status = json.loads((run / "status.json").read_text())
        result = json.loads((run / "evaluation.json").read_text())
        if spec["config"] != plan["config"] or spec["phase"] != plan["phase"]:
            raise ValueError(f"Protocol differs: {run}")
        if any(spec[k] != job[k] for k in ("condition", "subject", "seed")):
            raise ValueError(f"Run identity differs: {run}")
        if not status["completed"] or status["resume_from"] is not None or status["initial_steps"] != 0:
            raise ValueError(f"Incomplete or resumed formal run: {run}")
        if status["final_steps"] != plan["config"]["total_timesteps"] or result["steps"] != status["final_steps"]:
            raise ValueError(f"Different training budget: {run}")
        cfg = plan["config"]
        if [e["seed"] for e in result["episodes"]] != list(range(cfg["eval_seed_base"], cfg["eval_seed_base"] + cfg["eval_episodes"])):
            raise ValueError(f"Different evaluation episodes: {run}")
        metrics, _ = run_metrics(run, cfg["total_timesteps"])
        env_info = json.loads((run / "environment.json").read_text())
        git_shas.add(env_info["git_sha"])
        sources.add(json.dumps(env_info["source_sha256"], sort_keys=True))
        row = {**job, **metrics}
        rows.append(row)
        evidence[job["name"]] = {name: sha256(run / name) for name in
                                  ("run.json", "status.json", "evaluation.json", "environment.json")}
    if len(sources) != 1:
        raise ValueError("Runs used different rl/*.py training sources")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["subject"] or "shared")].append(row)
    summary = []
    for (condition, subject), values in groups.items():
        item = {"condition": condition, "subject": subject, "n_seeds": len(values)}
        for metric in ("success_rate", "ever_success_rate", "mean_final_distance", "mean_task_return",
                       "normalized_auc_success"):
            v = [r[metric] for r in values]
            item[metric + "_mean"] = statistics.mean(v)
            item[metric + "_sd"] = statistics.stdev(v) if len(v) > 1 else None
        summary.append(item)
    return {"complete": True, "split": "validation" if plan["phase"] == "validation" else "test",
            "config": plan["config"], "training_git_shas": sorted(git_shas),
            "training_source_sha256": json.loads(sources.pop()), "runs": rows, "summary": summary, "evidence_sha256": evidence,
            "interpretation": "SD over RL seeds within participant; shared controls counted once, not as independent participant runs"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    a = p.parse_args()
    result = summarize(a.directory)
    root = Path(a.directory)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["summary"][0]))
        w.writeheader()
        w.writerows(result["summary"])
    print(json.dumps(result["summary"], indent=2))
