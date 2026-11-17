"""Deterministic rollouts of saved policies on shared, prespecified goal seeds.

Usage: python -m rl.trajectories --run LABEL=CHECKPOINT_DIR [...] --seeds 30000 30001 30002 --out <new dir>

The goal seeds come from the render range (30000+), which is disjoint from training episode
seeds and from the evaluation goals. Writes trajectories.json (gripper and goal positions per
step, success, final distance) and a projection figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .checkpoint import validate_snapshot
from .environment import distance
from .train import PREF_SCHEMA, make_env


def rollout(checkpoint, seeds):
    meta = validate_snapshot(checkpoint)
    cfg = meta["run_spec"]["config"]
    model = SAC.load(Path(checkpoint) / "model.zip", device="cpu")
    env = make_env(cfg)
    pref = cfg["schema"] == PREF_SCHEMA
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            path, done, info = [obs["achieved_goal"].tolist()], False, {}
            goal = obs["desired_goal"].tolist()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                path.append(obs["achieved_goal"].tolist())
                done = terminated or truncated
            episodes.append({"seed": seed, "goal": goal, "gripper": path, "final_success": bool(info["is_success"]),
                             "final_distance": distance(obs),
                             "first_success_step": next((i for i, p in enumerate(path)
                                                         if np.linalg.norm(np.array(p) - goal) < cfg["goal_tolerance_m"]), None)})
            if pref:
                from .preference import Geometry, episode_metrics
                geo = Geometry(path[0], goal, cfg["preference"])
                episodes[-1]["preference"] = {
                    "pillar_center_xy": geo.m.tolist(), "pillar_radius_m": geo.R, "corridor_a": [geo.a0, geo.a1],
                    "start_xy": geo.s0.tolist(), "goal_xy": geo.g.tolist(),
                    **episode_metrics(geo, path, bool(info["is_success"]))}
    finally:
        env.close()
    return {"checkpoint": str(checkpoint), "run_spec": {k: meta["run_spec"][k] for k in ("condition", "subject", "seed", "phase")},
            "num_timesteps": meta["num_timesteps"], "episodes": episodes}


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(result)
    seeds = [e["seed"] for e in result[labels[0]]["episodes"]]
    fig, axes = plt.subplots(2, len(seeds), figsize=(4.2 * len(seeds), 7.5), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, seed in enumerate(seeds):
        for row, (a, b, name) in enumerate(((0, 1, "x-y"), (0, 2, "x-z"))):
            ax = axes[row][j]
            for k, label in enumerate(labels):
                ep = next(e for e in result[label]["episodes"] if e["seed"] == seed)
                g = np.array(ep["gripper"])
                status = "success" if ep["final_success"] else "%.1f cm" % (ep["final_distance"] * 100)
                ax.plot(g[:, a], g[:, b], color=colors[k % 10], lw=1.4, label=f"{label} ({status})")
                ax.scatter(g[0, a], g[0, b], color=colors[k % 10], marker="o", s=12)
            goal = ep["goal"]
            pref = ep.get("preference")
            if pref and name == "x-y":
                # virtual pillar (no collision) and the dispreferred right-hand corridor
                s0, g = np.array(pref["start_xy"]), np.array(pref["goal_xy"])
                u = (g - s0) / np.linalg.norm(g - s0)
                right = np.array([u[1], -u[0]])
                a0, a1 = pref["corridor_a"]
                D = np.linalg.norm(g - s0)
                corners = [s0 + a0 * D * u, s0 + a1 * D * u, s0 + a1 * D * u + 0.12 * right, s0 + a0 * D * u + 0.12 * right]
                ax.add_patch(plt.Polygon(corners, color="tab:red", alpha=0.12, label="right corridor (dispreferred)"))
                ax.add_patch(plt.Circle(tuple(pref["pillar_center_xy"]), pref["pillar_radius_m"], color="tab:red",
                                        alpha=0.5, label="virtual pillar"))
            ax.add_patch(plt.Circle((goal[a], goal[b]), 0.05, color="k", fill=False, ls=":"))
            ax.scatter(goal[a], goal[b], marker="*", color="k", s=80)
            ax.set_title(f"goal seed {seed}, {name} projection")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)
    fig.savefig(out / "trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="LABEL=CHECKPOINT_DIR")
    p.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    result = {}
    for item in a.run:
        label, path = item.split("=", 1)
        result[label] = rollout(path, a.seeds)
    (out / "trajectories.json").write_text(json.dumps(result, indent=2) + "\n")
    if not a.no_plot:
        plot(result, out)
    print(json.dumps({k: [(e["seed"], e["final_success"], round(e["final_distance"], 4)) for e in v["episodes"]]
                      for k, v in result.items()}, indent=2))
