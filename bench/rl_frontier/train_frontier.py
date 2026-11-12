"""One run of the ErrP-Bench decoder-quality -> RL-benefit frontier.

Reuses the existing pipeline pieces unchanged: ``rl.environment.make_task`` /
``FeedbackReward`` (reward = sparse task reward [+ dense] [+ w*(0.5 - p_error) at labelled
5-step segments]) and ``rl.train.evaluate`` (100 fixed held-out goals, deterministic policy,
sparse success), plus the frozen SAC hyper-parameters from the task protocol JSON.

Conditions: sparse | dense | auroc0.55 ... auroc0.90 (synthetic calibrated decoder,
``rl.synthetic_feedback``) | auroc1.00 (= existing ``ideal`` condition, p = label).

Differences from ``rl.train`` (storage only): no periodic replay/optimizer snapshots and no
per-step transitions.csv; we keep the final SB3 model (``final_model.zip``, no replay buffer),
the evaluation curve (``evaluations.jsonl`` every eval_every_steps + final, and
``evaluation.json``), and a compact feedback log (``feedback_events.npz``: step, label,
p_error of every delivered feedback event) to verify the realised AUROC per run.

FetchPush: ``achieved_goal`` is the OBJECT position, so the existing distance(obs) already
measures object-goal distance; the error label is therefore the change of object-goal
distance (the quantity the task reward depends on), and the dense control is object-goal
progress shaping. No code change is needed for that.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from rl.environment import FeedbackReward, make_task  # noqa: E402
from rl.feedback import sha256  # noqa: E402
from rl.synthetic_feedback import AUROC_LEVELS, DEFAULT_BASE_RATE, SyntheticFeedbackSource  # noqa: E402
from rl.train import check_episode_seed_range, evaluate, read_config  # noqa: E402

TASKS = {"reach": "protocol_reach.json", "push": "protocol_push.json",
         "push_staged": "protocol_push.json"}


def staged_push_potential(obs):
    """SUPPLEMENTARY push_staged rule: ||gripper - object|| + ||object - goal|| (metres).

    FetchPush observation[0:3] = gripper position, observation[3:6] = object position,
    desired_goal = target. Under the object-goal rule the object almost never moves under
    early exploration (pilot: ~170 labelled segments per 30k steps), so feedback is nearly
    absent; this staged rule also labels approaching/retreating from the object. It replaces
    the distance used by FeedbackReward for labels (and for the dense control), never the
    sparse task reward or the evaluation metrics (rl.train.evaluate keeps object-goal
    distance via its own import).
    """
    o = obs["observation"]
    return float(np.linalg.norm(o[0:3] - o[3:6]) + np.linalg.norm(o[3:6] - obs["desired_goal"]))
CONDITIONS = ("sparse", "dense") + tuple(f"auroc{a:.2f}" for a in AUROC_LEVELS)


class FrontierCallback(BaseCallback):
    def __init__(self, cfg, out):
        super().__init__()
        self.cfg, self.out = cfg, out
        self.last_eval = 0
        self.ev_step, self.ev_label, self.ev_p = [], [], []
        self.eligible = 0

    def record_eval(self):
        result = evaluate(self.model, self.cfg)
        result["wall_seconds"] = time.monotonic() - self.t0
        with (self.out / "evaluations.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        self.last_eval = self.model.num_timesteps
        print(f"[eval] step={result['steps']} success={result['success_rate']:.2f} "
              f"dist={result['mean_final_distance']:.4f} t={result['wall_seconds']:.0f}s", flush=True)
        return result

    def _on_training_start(self):
        self.t0 = time.monotonic()
        self.record_eval()

    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("feedback_eligible"):
                self.eligible += 1
            if info.get("feedback_event"):
                self.ev_step.append(info["step"])
                self.ev_label.append(int(info["simulated_label"]))
                self.ev_p.append(float(info["p_error"]))
        return True

    def _on_rollout_end(self):
        step = self.model.num_timesteps
        # Same convention as rl.train: skip a periodic eval at the final boundary (the final
        # policy, after SB3's last gradient update, is evaluated by the runner).
        if step - self.last_eval >= self.cfg["eval_every_steps"] and step < self.model._total_timesteps:
            self.record_eval()


def run(args):
    cfg = read_config(Path(__file__).parent / TASKS[args.task], "pilot")
    if args.task == "push_staged":
        if args.condition == "sparse":
            raise ValueError("sparse is identical to push/sparse (label rule has no effect); reuse it")
        import rl.environment as rl_env
        rl_env.distance = staged_push_potential  # FeedbackReward looks the name up at call time
        cfg = {**cfg, "feedback_rule_variant": "staged_gripper_object_plus_object_goal"}
    if args.base_rate != DEFAULT_BASE_RATE:
        cfg = {**cfg, "synthetic_base_rate": args.base_rate}
    budget = args.steps if args.steps is not None else cfg["total_timesteps"]
    check_episode_seed_range(cfg, args.seed, budget)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source, condition, subject = None, args.condition, None
    if args.condition.startswith("auroc"):
        level = float(args.condition[5:])
        if level >= 1.0:
            condition = "ideal"
        else:
            source = SyntheticFeedbackSource(level, args.base_rate)
            condition, subject = "eegnet", source.name  # duck-typed library, see synthetic_feedback
    spec = {"task": args.task, "condition": args.condition, "wrapper_condition": condition,
            "synthetic_source": None if source is None else
            {"auroc": source.auroc, "dprime": source.dprime, "base_rate": source.base_rate},
            "seed": args.seed, "budget": budget, "smoke": args.steps is not None, "config": cfg,
            "source_sha256": {p: sha256(ROOT / p) for p in
                              ("rl/environment.py", "rl/train.py", "rl/synthetic_feedback.py",
                               "bench/rl_frontier/train_frontier.py")}}
    (out / "run.json").write_text(json.dumps(spec, indent=2) + "\n")
    torch.set_num_threads(cfg["torch_threads"])
    env = make_task(cfg)
    wrapper = FeedbackReward(env, cfg, condition, args.seed, source, subject, None)
    monitored = Monitor(wrapper, str(out / "monitor.csv"))
    status = {"completed": False, "budget": budget}
    try:
        status["versions"] = {n: importlib.metadata.version(n) for n in
                              ("gymnasium", "gymnasium-robotics", "mujoco", "stable-baselines3", "torch", "numpy")}
        status["host"] = platform.node()
        status["lsf_jobid"] = os.environ.get("LSB_JOBID")
        model = SAC("MultiInputPolicy", monitored, seed=args.seed, device="cpu", verbose=0,
                    **copy.deepcopy(cfg["sac"]))
        model.set_logger(configure(str(out), ["csv"]))
        cb = FrontierCallback(cfg, out)
        t0 = time.monotonic()
        model.learn(total_timesteps=budget, callback=cb)
        if model.num_timesteps < budget:
            raise RuntimeError("Training stopped before its budget")
        if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
            raise RuntimeError("Nonfinite policy parameters")
        model.save(out / "final_model.zip")
        result = cb.record_eval()
        (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        lab, p = np.asarray(cb.ev_label, dtype=np.int8), np.asarray(cb.ev_p, dtype=np.float32)
        np.savez_compressed(out / "feedback_events.npz", step=np.asarray(cb.ev_step, dtype=np.int32),
                            label=lab, p_error=p)
        fb = {"eligible_segments": cb.eligible, "feedback_events": int(len(lab)),
              "error_rate": float(lab.mean()) if len(lab) else None,
              "mean_p_error_label1": float(p[lab == 1].mean()) if (lab == 1).any() else None,
              "mean_p_error_label0": float(p[lab == 0].mean()) if (lab == 0).any() else None}
        if (lab == 1).any() and (lab == 0).any():
            neg = np.sort(p[lab == 0].astype(np.float64))
            pos = p[lab == 1].astype(np.float64)
            lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
            fb["realised_auroc"] = float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))
        status.update(completed=True, final_steps=model.num_timesteps, updates=model._n_updates,
                      elapsed_seconds=time.monotonic() - t0, feedback=fb,
                      final_success=result["success_rate"])
        print(json.dumps(status, indent=2), flush=True)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        monitored.close()
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--base-rate", type=float, default=DEFAULT_BASE_RATE)
    p.add_argument("--steps", type=int, help="SMOKE/PILOT ONLY: override the protocol budget")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
