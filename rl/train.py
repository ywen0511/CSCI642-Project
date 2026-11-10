"""Train one fixed-budget, participant-specific offline-feedback SAC run."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from .checkpoint import restore_snapshot, save_snapshot
from .environment import CONDITIONS, FeedbackReward, distance, make_task
from .feedback import FeedbackLibrary, PROFILE, ROOT, sha256


def read_config(path, phase):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("schema") != "csci642-rl-stage2/v2":
        raise ValueError("Expected protocol v2; legacy v1 runs use different reward/feedback semantics")
    if cfg.get("feedback_rule") != "distance_change_deadband" or cfg.get("reward_center") != 0.5:
        raise ValueError("Protocol v2 requires the distance-change deadband and reward_center=0.5")
    for name in ("feedback_interval_steps", "max_episode_steps", "eval_episodes", "eval_every_steps",
                 "checkpoint_every_steps", "total_timesteps", "torch_threads"):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f"Invalid {name}")
    for name in ("eeg_weight", "dense_weight", "error_epsilon_m", "goal_tolerance_m"):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"Invalid {name}")
    validate_seed_lists(cfg)
    if phase == "final":
        evidence = Path(path).parent / cfg.get("validation_evidence_file", "MISSING")
        if cfg.get("status") != "frozen" or not evidence.is_file():
            raise ValueError("Final test feedback requires a frozen protocol and validation evidence")
        if sha256(evidence) != cfg.get("validation_evidence_sha256"):
            raise ValueError("Validation evidence checksum mismatch")
        ev = json.loads(evidence.read_text())
        if ev.get("split") != "validation" or not ev.get("complete"):
            raise ValueError("Incomplete/wrong-split validation evidence")
        candidate = {k: v for k, v in cfg.items() if k not in
                     ("validation_evidence_file", "validation_evidence_sha256")}
        candidate["status"] = "pilot"
        if candidate != ev.get("config"):
            raise ValueError("Frozen protocol differs from the validated configuration")
    return cfg


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_seed_lists(cfg):
    """Validation/final RL seeds are part of the protocol identity (and of the frozen config)."""
    lists = {}
    for name in ("validation_seeds", "final_seeds"):
        seeds = cfg.get(name)
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{name} must be a non-empty list")
        if not all(_is_int(s) and s >= 0 for s in seeds):
            raise ValueError(f"{name} must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{name} contains duplicates")
        lists[name] = seeds
    if set(lists["validation_seeds"]) & set(lists["final_seeds"]):
        raise ValueError("validation_seeds and final_seeds overlap")
    stride = cfg.get("training_episode_seed_stride")
    if not _is_int(stride) or stride < 1:
        raise ValueError("Invalid training_episode_seed_stride")
    return lists


RENDER_SEED_RANGE = (30000, 31000)  # rl.view_policy default goal seeds


def check_episode_seed_range(cfg, seed, total_steps):
    """Training episode seeds seed*stride + k must not reach held-out evaluation/render goals
    or the range of another RL seed."""
    stride = cfg["training_episode_seed_stride"]
    episodes = total_steps // cfg["max_episode_steps"] + 2
    lo, hi = seed * stride, seed * stride + episodes
    if episodes > stride:
        raise ValueError("Training episodes exceed the per-seed episode-seed stride")
    for a, b in ((cfg["eval_seed_base"], cfg["eval_seed_base"] + cfg["eval_episodes"]), RENDER_SEED_RANGE):
        if lo < b and a < hi:
            raise ValueError("Training episode seed range overlaps held-out evaluation/render goal seeds")


def evaluate(model, cfg):
    """Identical held-out goal seeds and sparse-only evaluation for every condition."""
    env = make_task(cfg)
    episodes = []
    try:
        for ep in range(cfg["eval_episodes"]):
            obs, _ = env.reset(seed=cfg["eval_seed_base"] + ep)
            done, total, any_success, length = False, 0.0, False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                length += 1
                any_success |= bool(info["is_success"])
                done = terminated or truncated
            episodes.append({"seed": cfg["eval_seed_base"] + ep, "task_return": total,
                             "final_distance": distance(obs), "final_success": bool(info["is_success"]),
                             "ever_success": any_success, "length": length})
    finally:
        env.close()
    return {"steps": model.num_timesteps, "n_updates": model._n_updates, "episodes": episodes,
            "success_rate": float(np.mean([e["final_success"] for e in episodes])),
            "ever_success_rate": float(np.mean([e["ever_success"] for e in episodes])),
            "mean_final_distance": float(np.mean([e["final_distance"] for e in episodes])),
            "mean_task_return": float(np.mean([e["task_return"] for e in episodes]))}


class RunCallback(BaseCallback):
    def __init__(self, wrapper, cfg, out, spec):
        super().__init__()
        self.wrapper, self.cfg, self.out, self.spec = wrapper, cfg, out, spec
        self.last_eval = self.last_save = 0

    def record_eval(self):
        result = evaluate(self.model, self.cfg)
        with (self.out / "evaluations.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        self.last_eval = self.num_timesteps = self.model.num_timesteps
        return result

    def _on_training_start(self):
        self.last_save = self.model.num_timesteps
        self.record_eval()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # SB3 calls _on_step before storing the transition; snapshot only after storage.
        step = self.model.num_timesteps
        if step - self.last_save >= self.cfg["checkpoint_every_steps"]:
            save_snapshot(self.model, self.wrapper, self.out / "checkpoints" / f"step_{step:09d}", self.spec)
            self.last_save = step
        # SB3 performs one more gradient update after the last rollout, and the runner then
        # evaluates the final policy; a periodic evaluation at the final boundary would be a
        # different (pre-update) policy recorded at the same step, so it is skipped.
        final_boundary = step >= self.model._total_timesteps
        if step - self.last_eval >= self.cfg["eval_every_steps"] and not final_boundary:
            self.record_eval()


def run(args):
    cfg = read_config(args.config, args.phase)
    if args.phase == "final" and args.steps is not None:
        raise ValueError("Final run length must come from the frozen protocol")
    split = "test" if args.phase == "final" else "validation"
    feedback_condition = args.condition in ("eegnet", "cbramod", "independent_eegnet", "independent_cbramod")
    library = FeedbackLibrary(args.feedback_dir, split) if feedback_condition else None
    if not feedback_condition and args.subject is not None:
        raise ValueError("Sparse/dense/ideal controls are shared across participants; omit --subject")
    if library and args.subject not in library.profile["splits"][split]:
        raise ValueError("Participant does not belong to the selected phase")
    budget = args.steps if args.steps is not None else cfg["total_timesteps"]
    if budget <= 0:
        raise ValueError("Step budget must be positive")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    if args.phase in ("validation", "final") and args.seed not in cfg[args.phase + "_seeds"]:
        raise ValueError(f"Seed {args.seed} is not declared in {args.phase}_seeds")
    check_episode_seed_range(cfg, args.seed, max(cfg["total_timesteps"], budget) + (
        json.loads((Path(args.resume) / "metadata.json").read_text())["num_timesteps"] if args.resume else 0))
    spec = {"config": cfg, "condition": args.condition, "subject": args.subject,
            "phase": args.phase, "seed": args.seed, "feedback_split": split if library else None,
            "feedback_table_sha256": library.table_sha256 if library else None,
            "feedback_profile_sha256": sha256(PROFILE),
            "training_source_sha256": {name: sha256(ROOT / "rl" / name) for name in
                                       ("train.py", "environment.py", "feedback.py", "checkpoint.py")}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(spec, indent=2) + "\n")
    torch.set_num_threads(cfg["torch_threads"])
    env = make_task(cfg)
    wrapper = FeedbackReward(env, cfg, args.condition, args.seed, library, args.subject, out / "transitions.csv")
    monitored = Monitor(wrapper, str(out / "monitor.csv"))
    status = {"completed": False, "requested_additional_steps": budget, "resume_from": args.resume}
    try:
        versions = {n: importlib.metadata.version(n) for n in
                    ("gymnasium", "gymnasium-robotics", "mujoco", "stable-baselines3", "torch", "numpy")}
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        env_info = {"python": sys.version, "platform": platform.platform(), "versions": versions,
                    "device": args.device, "git_sha": git_sha,
                    "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "rl").glob("*.py"))}}
        (out / "environment.json").write_text(json.dumps(env_info, indent=2) + "\n")
        if args.resume:
            model, meta = restore_snapshot(args.resume, monitored, wrapper, spec, args.device)
            status["restored_replay_size"] = meta["replay_size"]
            status["restored_updates"] = meta["n_updates"]
        else:
            model = SAC("MultiInputPolicy", monitored, seed=args.seed, device=args.device,
                        verbose=0, **copy.deepcopy(cfg["sac"]))
        model.set_logger(configure(str(out), ["csv"]))
        start = model.num_timesteps
        callback = RunCallback(wrapper, cfg, out, spec)
        t0 = time.monotonic()
        model.learn(total_timesteps=budget, callback=callback, reset_num_timesteps=not bool(args.resume))
        if model.num_timesteps < start + budget:
            raise RuntimeError("Training stopped before its budget")
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            raise RuntimeError("Nonfinite policy parameters")
        snapshot = save_snapshot(model, wrapper, out / "checkpoints" / f"final_{model.num_timesteps:09d}", spec)
        result = callback.record_eval()
        (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        status.update(completed=True, initial_steps=start, final_steps=model.num_timesteps,
                      updates=model._n_updates, replay_size=model.replay_buffer.size(),
                      elapsed_seconds=time.monotonic() - t0, final_checkpoint=str(snapshot.resolve()))
        print(json.dumps(status, indent=2))
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        monitored.close()
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rl/protocol.json")
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--phase", choices=["pilot", "validation", "final"], default="pilot")
    p.add_argument("--subject")
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, help="Pilot/validation additional steps; default is protocol budget")
    p.add_argument("--resume", help="Trusted checkpoint directory; output must be a new directory")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
