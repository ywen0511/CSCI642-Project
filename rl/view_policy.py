"""Render a trusted saved checkpoint on the shared sparse-only FetchReach task."""
import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
from stable_baselines3 import SAC

from .checkpoint import validate_snapshot


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=30000)
    a = p.parse_args()
    meta = validate_snapshot(a.checkpoint)
    cfg = meta["run_spec"]["config"]
    gym.register_envs(gymnasium_robotics)
    env = gym.make(cfg["env_id"], reward_type="sparse", render_mode="rgb_array",
                   max_episode_steps=cfg["max_episode_steps"])
    model = SAC.load(Path(a.checkpoint) / "model.zip", device="cpu")
    frames = []
    try:
        for episode in range(a.episodes):
            obs, _ = env.reset(seed=a.seed + episode)
            done = False
            while not done:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    finally:
        env.close()
    if not frames:
        raise RuntimeError("No frames rendered")
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=40, loop=0)
