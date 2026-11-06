"""Actual FetchReach integration: all conditions, checkpoint reload, and continuation."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from .environment import CONDITIONS, distance, make_task
from .train import read_config
from .verify_checkpoint import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"ok": False, "scope": "CPU integration only; not a trained-policy performance result", "conditions": {}}
    try:
        cfg = copy.deepcopy(read_config("rl/protocol.json", "pilot"))
        env = make_task(cfg)
        try:
            obs, _ = env.reset(seed=0)
            env.action_space.seed(0)
            assert env.action_space.shape == (4,)
            for step in range(1, 51):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                d = distance(obs)
                assert reward == (-1.0 if d > cfg["goal_tolerance_m"] else 0.0)
                assert bool(info["is_success"]) == (d < cfg["goal_tolerance_m"])
                assert not terminated and bool(truncated) == (step == 50)
            report["environment_checks"] = {"steps": 50, "ok": True}
        finally:
            env.close()
        cfg.update(total_timesteps=160, checkpoint_every_steps=80, eval_every_steps=80, eval_episodes=2)
        cfg["sac"].update(learning_starts=20, batch_size=16, buffer_size=500, policy_kwargs={"net_arch": [32, 32]})
        config = out / "smoke_protocol.json"
        config.write_text(json.dumps(cfg, indent=2) + "\n")
        for condition in CONDITIONS:
            cmd = [sys.executable, "-m", "rl.train", "--config", str(config.resolve()), "--condition", condition,
                   "--feedback-dir", str(Path(a.feedback_dir).resolve()), "--out", str((out / condition).resolve())]
            if condition not in ("sparse", "dense", "ideal"):
                cmd += ["--subject", "s11"]
            with (out / f"{condition}.log").open("w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            status = json.loads((out / condition / "status.json").read_text())
            report["conditions"][condition] = {"steps": status["final_steps"], "updates": status["updates"],
                                               "verification": verify(status["final_checkpoint"])}
            if condition == "eegnet":
                # Resume a periodic snapshot, not just the easier final snapshot.
                checkpoint = out / condition / "checkpoints" / "step_000000080"
                report["periodic_checkpoint"] = verify(checkpoint)
                resume_cmd = [sys.executable, "-m", "rl.train", "--config", str(config.resolve()),
                              "--condition", "eegnet", "--subject", "s11", "--feedback-dir", str(Path(a.feedback_dir).resolve()),
                              "--resume", str(checkpoint.resolve()), "--steps", "40", "--out", str((out / "eegnet_resumed").resolve())]
                with (out / "eegnet_resumed.log").open("w") as f:
                    subprocess.run(resume_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                resumed = json.loads((out / "eegnet_resumed/status.json").read_text())
                assert resumed["initial_steps"] == 80 and resumed["final_steps"] == 120
                assert resumed["updates"] > resumed["restored_updates"]
                assert resumed["replay_size"] > resumed["restored_replay_size"]
                report["continuation"] = {"initial_steps": 80, "final_steps": 120,
                                          "verification": verify(resumed["final_checkpoint"]), "ok": True}
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
