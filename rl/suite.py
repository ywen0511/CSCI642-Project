"""Plan or execute a common-budget multi-seed comparison; default is plan only.

Modes (all use the same plan, built from the protocol's declared seed lists):
  (default)      print the plan
  --execute      create the suite directory and run every job serially, then summarize
  --prepare      create the suite directory and plan.json only (for external schedulers)
  --run-job NAME run one job of an existing prepared suite into <out>/<NAME>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .environment import CONDITIONS
from .feedback import PROFILE, sha256
from .train import read_config

SHARED = ("sparse", "dense", "ideal")


def jobs(phase, profile, cfg):
    """Sparse/dense/ideal are trained once per seed; EEG conditions once per participant and seed."""
    if phase not in ("validation", "final"):
        raise ValueError(phase)
    seeds = cfg[phase + "_seeds"]
    split = "validation" if phase == "validation" else "test"
    for condition in CONDITIONS:
        subjects = [None] if condition in SHARED else profile["splits"][split]
        for subject in subjects:
            for seed in seeds:
                yield {"condition": condition, "subject": subject, "seed": seed,
                       "name": f"{condition}_{subject or 'shared'}_s{seed}"}


def make_plan(config, phase):
    cfg = read_config(config, phase)
    profile = json.loads(PROFILE.read_text())
    return {"phase": phase, "config": cfg, "config_sha256": sha256(config),
            "jobs": list(jobs(phase, profile, cfg))}


def command(plan_job, phase, config, feedback_dir, device, out):
    cmd = [sys.executable, "-m", "rl.train", "--config", str(Path(config).resolve()),
           "--phase", phase, "--condition", plan_job["condition"], "--seed", str(plan_job["seed"]),
           "--feedback-dir", str(Path(feedback_dir).resolve()), "--device", device,
           "--out", str((Path(out) / plan_job["name"]).resolve())]
    if plan_job["subject"]:
        cmd += ["--subject", plan_job["subject"]]
    return cmd


def run_job(plan_job, a, out):
    with (out / (plan_job["name"] + ".log")).open("x") as log:
        subprocess.run(command(plan_job, a.phase, a.config, a.feedback_dir, a.device, out),
                       check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="rl/protocol.json")
    p.add_argument("--phase", choices=["validation", "final"], required=True)
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-job", metavar="NAME")
    a = p.parse_args()
    plan = make_plan(a.config, a.phase)
    out = Path(a.out)
    if a.run_job:
        existing = json.loads((out / "plan.json").read_text())
        if existing != plan:
            raise ValueError("Prepared plan differs from this config/phase")
        job = next((j for j in plan["jobs"] if j["name"] == a.run_job), None)
        if job is None:
            raise ValueError(f"Job not in plan: {a.run_job}")
        run_job(job, a, out)
        return
    print(json.dumps({"phase": a.phase, "runs": len(plan["jobs"]),
                      "total_environment_steps": len(plan["jobs"]) * plan["config"]["total_timesteps"],
                      "seeds": plan["config"][a.phase + "_seeds"],
                      "execute": a.execute, "prepare": a.prepare}, indent=2))
    if not (a.execute or a.prepare):
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if a.prepare:
        return
    for job in plan["jobs"]:
        run_job(job, a, out)
    subprocess.run([sys.executable, "-m", "rl.summarize", str(out)], check=True)


if __name__ == "__main__":
    main()
