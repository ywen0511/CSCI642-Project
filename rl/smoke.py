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

FEEDBACK_FIELDS = {"simulated_label", "p_error", "trial_id", "source_label", "event_index"}


def check_transitions(path, condition, cfg):
    """Neutral segments skip feedback; ideal signs follow the label; baselines get no EEG reward."""
    import csv
    w = cfg["eeg_weight"]
    counts = {"rows": 0, "segment_end": 0, "eligible": 0, "events": 0, "neutral": 0, "label0": 0, "label1": 0}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            counts["rows"] += 1
            seg, elig, ev = int(r["segment_end"]), int(r["feedback_eligible"]), int(r["feedback_event"])
            counts["segment_end"] += seg
            counts["eligible"] += elig
            counts["events"] += ev
            assert not (elig and not seg) and not (ev and not elig)
            if seg and not elig:
                counts["neutral"] += 1
                assert abs(float(r["segment_progress"])) <= cfg["error_epsilon_m"] + 1e-12
            if not elig:
                assert all(r[k] == "" for k in FEEDBACK_FIELDS) and float(r["eeg_reward"]) == 0.0
                continue
            label = int(r["simulated_label"])
            counts[f"label{label}"] += 1
            assert (float(r["segment_progress"]) < 0) == (label == 1)
            if condition == "ideal":
                assert float(r["p_error"]) == label
                assert abs(float(r["eeg_reward"]) - (w / 2 if label == 0 else -w / 2)) < 1e-12
            elif condition in ("sparse", "dense"):
                assert r["p_error"] == "" and float(r["eeg_reward"]) == 0.0 and not ev
            else:
                p = float(r["p_error"])
                assert abs(float(r["eeg_reward"]) - w * (0.5 - p)) < 1e-12
                if not condition.startswith("independent_"):
                    assert int(r["source_label"]) == label
    if condition in ("sparse", "dense"):
        assert counts["events"] == 0
    else:
        assert counts["events"] == counts["eligible"] > 0
    return counts


def scripted_feedback_check(cfg):
    """Real-simulator check: hold still (neutral), move toward the goal (progress, label 0)
    and away from it (retreat, label 1) under ideal feedback."""
    import numpy as np
    from .environment import FeedbackReward
    env = FeedbackReward(make_task(cfg), cfg, "ideal", 0)
    obs, _ = env.reset()
    w, n = cfg["eeg_weight"], cfg["feedback_interval_steps"]
    results = {}
    try:
        for phase in ("hold", "toward", "away"):
            for _ in range(n):
                direction = obs["desired_goal"] - obs["achieved_goal"]
                unit = direction / max(np.linalg.norm(direction), 1e-9)
                move = {"hold": np.zeros(3), "toward": 0.3 * unit, "away": -0.3 * unit}[phase]
                obs, reward, _, _, info = env.step(np.append(move, 0.0).astype(np.float32))
            assert info["segment_end"] == 1, phase
            results[phase] = {"segment_progress_m": info["segment_progress"], "label": info["simulated_label"],
                              "eeg_reward": info["eeg_reward"], "feedback_event": info["feedback_event"]}
        assert results["hold"]["feedback_event"] == 0 and results["hold"]["label"] == "" and results["hold"]["eeg_reward"] == 0
        assert abs(results["hold"]["segment_progress_m"]) <= cfg["error_epsilon_m"]
        assert results["toward"]["label"] == 0 and abs(results["toward"]["eeg_reward"] - w / 2) < 1e-12
        assert results["away"]["label"] == 1 and abs(results["away"]["eeg_reward"] + w / 2) < 1e-12
        assert env.counters()["event_count"] == 2
    finally:
        env.close()
    return {**results, "ok": True}


def check_observation_space(run_dir):
    """The policy's observation space must be the plain FetchReach dict (no EEG/label fields)."""
    from stable_baselines3 import SAC
    final = sorted((Path(run_dir) / "checkpoints").glob("final_*"))[-1]
    model = SAC.load(final / "model.zip", device="cpu")
    keys = sorted(model.observation_space.spaces)
    assert keys == ["achieved_goal", "desired_goal", "observation"], keys
    return {"policy_observation_keys": keys,
            "shapes": {k: list(model.observation_space.spaces[k].shape) for k in keys}}


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
        report["scripted_feedback_check"] = scripted_feedback_check(cfg)
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
                                               "verification": verify(status["final_checkpoint"]),
                                               "transitions": check_transitions(out / condition / "transitions.csv", condition, cfg),
                                               "observation": check_observation_space(out / condition)}
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
