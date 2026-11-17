"""Route-preference extension (FINAL_V3_HANDOFF.md, Section 7; schema csci642-rl-pref/v1).

FetchReach-v4 with unchanged physics, policy observation and sparse task reward. Two additions:

* Goal filter: after the seeded reset, goals are redrawn with the environment's own sampler and RNG until the
  horizontal start-goal distance D = ||g_xy - s0_xy|| >= min_goal_distance_xy_m; the observation is then
  recomputed by the environment's own observation function.
* Virtual pillar and simulated user preference "pass the pillar on the left": a vertical cylinder of radius R at
  the x-y midpoint m of start and goal (no collision). With u = (g_xy - s0_xy)/D, a(p) = ((p_xy - s0_xy).u)/D and
  l(p) = u_x (p_y - s0_y) - u_y (p_x - s0_x) (positive = left of the start-to-goal direction seen from above),
  a step is a violation if at its end the gripper is inside the pillar (||p_xy - m|| < R) or in the right-hand
  corridor (a0 <= a <= a1 and l < 0). A 5-step segment containing any violation is labeled 1 (error); otherwise
  the v2 distance-change rule applies (progress 0, retreat 1, deadband -> no feedback).

The preference never enters the task reward or the policy observation; sparse and dense remain task-only
baselines. Evaluation uses no feedback and no shaping.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from .environment import LOG_FIELDS, FeedbackReward, distance, error_label, make_task

PREF_FIELDS = ["px", "py", "start_x", "start_y", "goal_x", "goal_y", "a", "l",
               "pillar", "right_corridor", "violation", "segment_violation"]
PREF_KEYS = {"min_goal_distance_xy_m", "pillar_radius_m", "corridor_a", "preferred_side", "max_goal_redraws"}
EVAL_RATE_KEYS = ("preferred_pass", "violation_free", "pillar_entry", "right_corridor_entry")


def validate_preference_config(cfg):
    pref = cfg.get("preference")
    if not isinstance(pref, dict) or set(pref) != PREF_KEYS:
        raise ValueError(f"preference must define exactly {sorted(PREF_KEYS)}")
    d, r = pref["min_goal_distance_xy_m"], pref["pillar_radius_m"]
    a0, a1 = pref["corridor_a"]
    if not (0 < r and 0 < d and 0 <= a0 < 0.5 < a1 <= 1):
        raise ValueError("Invalid preference geometry")
    if pref["preferred_side"] != "left":
        raise ValueError("The simulated user prefers passing on the left")
    if not isinstance(pref["max_goal_redraws"], int) or pref["max_goal_redraws"] < 1:
        raise ValueError("Invalid max_goal_redraws")
    # the pillar edge must lie at least the goal tolerance from both start and goal (x-y)
    if d / 2 - r < cfg["goal_tolerance_m"] - 1e-12:
        raise ValueError("Pillar too close to start/goal for the minimum goal distance")
    return pref


class Geometry:
    """Start/goal geometry of one episode (x-y plane)."""

    def __init__(self, start, goal, pref):
        self.s0 = np.asarray(start, float)[:2].copy()
        self.g = np.asarray(goal, float)[:2].copy()
        self.D = float(np.linalg.norm(self.g - self.s0))
        if self.D <= 0:
            raise ValueError("Degenerate start-goal geometry")
        self.u = (self.g - self.s0) / self.D
        self.m = (self.s0 + self.g) / 2
        self.R = float(pref["pillar_radius_m"])
        self.a0, self.a1 = (float(x) for x in pref["corridor_a"])

    def coords(self, p):
        q = np.asarray(p, float)[:2] - self.s0
        a = float(q @ self.u) / self.D
        lat = float(self.u[0] * q[1] - self.u[1] * q[0])
        return a, lat

    def classify(self, p):
        a, lat = self.coords(p)
        pillar = bool(np.linalg.norm(np.asarray(p, float)[:2] - self.m) < self.R)
        right = bool(self.a0 <= a <= self.a1 and lat < 0)
        return {"a": a, "l": lat, "pillar": pillar, "right_corridor": right, "violation": pillar or right}

    def left_normal(self):
        return np.array([-self.u[1], self.u[0]])


class GoalFilter(gym.Wrapper):
    """Seeded reset, then redraw the goal with the environment's own sampler/RNG until D >= the minimum."""

    def __init__(self, env, pref):
        super().__init__(env)
        self.pref = pref
        self.start = self.goal_redraws = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        base = self.env.unwrapped
        start = np.array(obs["achieved_goal"], copy=True)
        redraws = 0
        while np.linalg.norm(np.asarray(base.goal)[:2] - start[:2]) < self.pref["min_goal_distance_xy_m"]:
            if redraws >= self.pref["max_goal_redraws"]:
                raise RuntimeError("Goal filter exceeded max_goal_redraws")
            base.goal = base._sample_goal().copy()
            redraws += 1
        if redraws:
            obs = base._get_obs()
        if not np.array_equal(obs["desired_goal"], base.goal):
            raise RuntimeError("Filtered goal is not in the returned observation")
        self.start, self.goal_redraws = start, redraws
        return obs, {**info, "goal_redraws": redraws}


def make_preference_task(cfg):
    return GoalFilter(make_task(cfg), validate_preference_config(cfg))


class PreferenceFeedbackReward(FeedbackReward):
    """FeedbackReward whose segment label is 1 whenever the segment contains a preference violation."""
    log_fields = LOG_FIELDS + PREF_FIELDS

    def __init__(self, env, config, *args, **kwargs):
        if not isinstance(env, GoalFilter):
            raise ValueError("The preference wrapper requires the goal-filtered task")
        self.pref = validate_preference_config(config)
        self.geometry = None
        self.segment_violation = False
        super().__init__(env, config, *args, **kwargs)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.geometry = Geometry(self.env.start, obs["desired_goal"], self.pref)
        self.segment_violation = False
        return obs, info

    def annotate(self, obs):
        p = obs["achieved_goal"]
        c = self.geometry.classify(p)
        self.segment_violation |= c["violation"]
        return {"px": float(p[0]), "py": float(p[1]), "start_x": float(self.geometry.s0[0]),
                "start_y": float(self.geometry.s0[1]), "goal_x": float(self.geometry.g[0]),
                "goal_y": float(self.geometry.g[1]), "a": c["a"], "l": c["l"], "pillar": int(c["pillar"]),
                "right_corridor": int(c["right_corridor"]), "violation": int(c["violation"]),
                "segment_violation": int(self.segment_violation)}

    def segment_label(self, start_distance, end_distance):
        if self.segment_violation:
            return 1
        return error_label(start_distance, end_distance, self.cfg["error_epsilon_m"])

    def close_segment(self):
        self.segment_violation = False


def crossings_left(geometry, positions):
    """Lateral offsets where the path crosses a = 0.5 (linear interpolation between step-end positions)."""
    out = []
    coords = [geometry.coords(p) for p in positions]
    for (a_prev, l_prev), (a_next, l_next) in zip(coords, coords[1:]):
        if (a_prev < 0.5 <= a_next) or (a_next < 0.5 <= a_prev):
            t = (0.5 - a_prev) / (a_next - a_prev)
            out.append(l_prev + t * (l_next - l_prev))
    return out


def episode_metrics(geometry, positions, final_success):
    """positions: gripper positions from reset (index 0) to the last step end.

    preferred pass = task success, no violation at any step end, and every crossing of a = 0.5 has l > 0
    (at least one crossing)."""
    classes = [geometry.classify(p) for p in positions[1:]]
    cross = crossings_left(geometry, positions)
    violation_free = not any(c["violation"] for c in classes)
    preferred = bool(final_success and violation_free and cross and all(x > 0 for x in cross))
    steps = np.diff(np.asarray(positions, float), axis=0)
    return {"preferred_pass": preferred, "violation_free": violation_free,
            "pillar_entry": any(c["pillar"] for c in classes),
            "right_corridor_entry": any(c["right_corridor"] for c in classes),
            "violation_steps": sum(c["violation"] for c in classes),
            "crossing_l": cross, "path_length": float(np.linalg.norm(steps, axis=1).sum()),
            "goal_distance_xy": geometry.D}


def evaluate_preference(model, cfg):
    """Held-out filtered goals (eval seeds), deterministic policy, no feedback and no shaping."""
    pref = validate_preference_config(cfg)
    env = make_preference_task(cfg)
    episodes = []
    try:
        for ep in range(cfg["eval_episodes"]):
            seed = cfg["eval_seed_base"] + ep
            obs, info = env.reset(seed=seed)
            geometry = Geometry(env.start, obs["desired_goal"], pref)
            positions = [np.array(obs["achieved_goal"], copy=True)]
            done, total, any_success, length = False, 0.0, False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                positions.append(np.array(obs["achieved_goal"], copy=True))
                total += float(reward)
                length += 1
                any_success |= bool(info["is_success"])
                done = terminated or truncated
            final_success = bool(info["is_success"])
            episodes.append({"seed": seed, "task_return": total, "final_distance": distance(obs),
                             "final_success": final_success, "ever_success": any_success, "length": length,
                             "goal_redraws": env.goal_redraws, **episode_metrics(geometry, positions, final_success)})
    finally:
        env.close()
    result = {"steps": model.num_timesteps, "n_updates": model._n_updates, "episodes": episodes,
              "success_rate": float(np.mean([e["final_success"] for e in episodes])),
              "ever_success_rate": float(np.mean([e["ever_success"] for e in episodes])),
              "mean_final_distance": float(np.mean([e["final_distance"] for e in episodes])),
              "mean_task_return": float(np.mean([e["task_return"] for e in episodes]))}
    for key in EVAL_RATE_KEYS:
        result[key + "_rate"] = float(np.mean([e[key] for e in episodes]))
    result["mean_path_length"] = float(np.mean([e["path_length"] for e in episodes]))
    if not all(math.isfinite(v) for k, v in result.items() if isinstance(v, float)):
        raise ValueError("Nonfinite evaluation summary")
    return result
