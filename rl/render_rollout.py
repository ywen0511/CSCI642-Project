"""Two-stage GIF rendering that keeps torch and OpenGL in separate processes.

On the cluster nodes, loading torch and the software OpenGL (OSMesa) library in one process crashed, so:

  record:  python -m rl.render_rollout record CHECKPOINT --seeds 30000 30001 30002 --out actions.json
           (deterministic policy rollout, no rendering; stores actions and gripper positions)
  render:  python -m rl.render_rollout render actions.json --out policy.gif
           (no torch/SB3 import; replays the stored actions in MuJoCo, renders frames, and checks that the
           replayed gripper positions match the recorded ones)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def record(checkpoint, seeds, out):
    from stable_baselines3 import SAC

    from .checkpoint import validate_snapshot
    from .environment import make_task

    meta = validate_snapshot(checkpoint)
    cfg = meta["run_spec"]["config"]
    model = SAC.load(Path(checkpoint) / "model.zip", device="cpu")
    env = make_task(cfg)
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            actions, positions, done, info = [], [obs["achieved_goal"].tolist()], False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append(np.asarray(action, dtype=np.float64).tolist())
                positions.append(obs["achieved_goal"].tolist())
                done = terminated or truncated
            episodes.append({"seed": seed, "goal": obs["desired_goal"].tolist(), "actions": actions,
                             "positions": positions, "final_success": bool(info["is_success"])})
    finally:
        env.close()
    Path(out).write_text(json.dumps({"checkpoint": str(checkpoint), "config": cfg,
                                     "run_spec": {k: meta["run_spec"][k] for k in ("condition", "subject", "seed", "phase")},
                                     "episodes": episodes}) + "\n")


def render(actions_file, out):
    if "torch" in sys.modules:
        raise RuntimeError("render must run without torch loaded")
    import gymnasium as gym
    import gymnasium_robotics
    import imageio.v2 as imageio

    data = json.loads(Path(actions_file).read_text())
    cfg = data["config"]
    gym.register_envs(gymnasium_robotics)
    env = gym.make(cfg["env_id"], reward_type="sparse", render_mode="rgb_array", max_episode_steps=cfg["max_episode_steps"])
    frames, max_dev = [], 0.0
    try:
        for ep in data["episodes"]:
            obs, _ = env.reset(seed=ep["seed"])
            frames.append(env.render())
            for t, action in enumerate(ep["actions"]):
                obs, *_ = env.step(np.asarray(action, dtype=np.float32))
                frames.append(env.render())
                max_dev = max(max_dev, float(np.max(np.abs(obs["achieved_goal"] - np.asarray(ep["positions"][t + 1])))))
    finally:
        env.close()
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=40, loop=0)
    report = {"gif": str(out), "frames": len(frames), "max_replay_position_deviation_m": max_dev,
              "replay_matches_recording": max_dev <= 1e-6}
    Path(str(out) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if max_dev > 1e-6:
        raise SystemExit("Replayed trajectory deviates from the recorded policy rollout")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("checkpoint")
    r.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    r.add_argument("--out", required=True)
    g = sub.add_parser("render")
    g.add_argument("actions")
    g.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "record":
        record(a.checkpoint, a.seeds, a.out)
    else:
        render(a.actions, a.out)
