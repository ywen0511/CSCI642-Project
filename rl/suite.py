"""Plan or execute a common-budget multi-seed comparison; default is plan only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .environment import CONDITIONS
from .feedback import PROFILE, sha256
from .train import read_config


def jobs(phase, profile):
    seeds = [100, 101, 102] if phase == "validation" else [0, 1, 2, 3, 4]
    split = "validation" if phase == "validation" else "test"
    for condition in CONDITIONS:
        subjects = [None] if condition in ("sparse", "dense", "ideal") else profile["splits"][split]
        for subject in subjects:
            for seed in seeds:
                yield {"condition": condition, "subject": subject, "seed": seed,
                       "name": f"{condition}_{subject or 'shared'}_s{seed}"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rl/protocol.json")
    p.add_argument("--phase", choices=["validation", "final"], required=True)
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    cfg = read_config(a.config, a.phase)
    profile = json.loads(PROFILE.read_text())
    plan = {"phase": a.phase, "config": cfg, "config_sha256": sha256(a.config),
            "jobs": list(jobs(a.phase, profile))}
    print(json.dumps({"phase": a.phase, "runs": len(plan["jobs"]),
                      "total_environment_steps": len(plan["jobs"]) * cfg["total_timesteps"],
                      "execute": a.execute}, indent=2))
    if not a.execute:
        print(json.dumps(plan, indent=2))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for job in plan["jobs"]:
        command = [sys.executable, "-m", "rl.train", "--config", str(Path(a.config).resolve()),
                   "--phase", a.phase, "--condition", job["condition"], "--seed", str(job["seed"]),
                   "--feedback-dir", str(Path(a.feedback_dir).resolve()), "--device", a.device,
                   "--out", str((out / job["name"]).resolve())]
        if job["subject"]:
            command += ["--subject", job["subject"]]
        with (out / (job["name"] + ".log")).open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, "-m", "rl.summarize", str(out)], check=True)


if __name__ == "__main__":
    main()
