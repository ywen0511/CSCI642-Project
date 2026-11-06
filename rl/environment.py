"""Shared task environment and explicit project-specific feedback rule."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

CONDITIONS = ("sparse", "dense", "ideal", "eegnet", "cbramod",
              "independent_eegnet", "independent_cbramod")
LOG_FIELDS = ["step", "episode", "episode_step", "distance", "segment_progress",
              "simulated_label", "feedback_event", "event_index", "trial_id", "source_label",
              "p_error", "task_reward", "dense_reward", "eeg_reward", "reward",
              "terminated", "truncated"]


def distance(obs):
    return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))


def error_label(start_distance, end_distance, epsilon_m, goal_tolerance_m):
    # A stalled segment is non-error in this pilot definition; it still incurs task cost.
    return int(end_distance > goal_tolerance_m and end_distance - start_distance > epsilon_m)


class FeedbackReward(gym.Wrapper):
    def __init__(self, env, config, condition, seed, library=None, subject=None, log_path=None):
        super().__init__(env)
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.cfg, self.condition = config, condition
        self.library, self.subject = library, subject
        self.seed_base = int(seed)
        self.feedback_seed = int(seed) + int(config["feedback_seed_offset"])
        self.step_count = self.episode_count = self.event_count = 0
        self.episode_step = 0
        self.previous = self.segment_start = None
        self.segment_steps = 0
        self.handle = self.writer = None
        if condition in ("eegnet", "cbramod", "independent_eegnet", "independent_cbramod"):
            if library is None or subject not in library.profile["splits"][library.split]:
                raise ValueError("This condition requires a participant-specific feedback library")
        if log_path:
            self.handle = Path(log_path).open("x", newline="")
            self.writer = csv.DictWriter(self.handle, fieldnames=LOG_FIELDS)
            self.writer.writeheader()

    def reset(self, *, seed=None, options=None):
        # Independent episode seeds make reset boundaries reproducible on continuation.
        chosen = self.seed_base + self.episode_count if seed is None else seed
        obs, info = self.env.reset(seed=chosen, options=options)
        self.episode_count += 1
        self.episode_step = self.segment_steps = 0
        self.previous = self.segment_start = distance(obs)
        return obs, info

    def step(self, action):
        if self.previous is None:
            raise RuntimeError("Call reset first")
        obs, task, terminated, truncated, info = self.env.step(action)
        d = distance(obs)
        self.step_count += 1
        self.episode_step += 1
        self.segment_steps += 1
        dense = self.cfg["dense_weight"] * (self.previous - d) if self.condition == "dense" else 0.0
        emit = self.segment_steps >= self.cfg["feedback_interval_steps"] or terminated or truncated
        label = p = source_label = ""
        trial = ""
        event = self.event_count
        segment_progress = self.segment_start - d
        eeg = 0.0
        if emit:
            label = error_label(self.segment_start, d, self.cfg["error_epsilon_m"], self.cfg["goal_tolerance_m"])
            if self.condition == "ideal":
                p = float(label)
            elif self.library is not None:
                model = self.condition.removeprefix("independent_")
                row = self.library.sample(model, self.subject, label, self.feedback_seed,
                                          event, self.condition.startswith("independent_"))
                p, trial, source_label = row["p_error"], row["trial_id"], row["label"]
            if p != "":
                eeg = self.cfg["eeg_weight"] * (1.0 - p)
            self.event_count += 1
            self.segment_start, self.segment_steps = d, 0
        reward = float(task) + dense + eeg
        if not math.isfinite(reward):
            raise ValueError("Nonfinite reward")
        record = dict(step=self.step_count, episode=self.episode_count, episode_step=self.episode_step,
                      distance=d, segment_progress=segment_progress, simulated_label=label,
                      feedback_event=int(emit), event_index=event if emit else "", trial_id=trial,
                      source_label=source_label, p_error=p, task_reward=float(task),
                      dense_reward=dense, eeg_reward=eeg, reward=reward,
                      terminated=int(terminated), truncated=int(truncated))
        if self.writer:
            self.writer.writerow(record)
        self.previous = d
        return obs, reward, terminated, truncated, {**info, **record}

    def counters(self):
        return {"step_count": self.step_count, "episode_count": self.episode_count,
                "event_count": self.event_count}

    def restore_counters(self, counters):
        for key in ("step_count", "episode_count", "event_count"):
            setattr(self, key, int(counters[key]))

    def close(self):
        if self.handle:
            self.handle.flush()
            self.handle.close()
            self.handle = None
        self.env.close()


def make_task(config):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(config["env_id"], reward_type="sparse",
                   max_episode_steps=config["max_episode_steps"])
    actual = float(env.unwrapped.distance_threshold)
    if not math.isclose(actual, config["goal_tolerance_m"], abs_tol=1e-12):
        env.close()
        raise ValueError(f"Goal tolerance mismatch: environment {actual}")
    return env
