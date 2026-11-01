"""Training loop with resumable, atomically written checkpoints.

Checkpoint contents (best.ckpt / last.ckpt / epoch_XXXX.ckpt):
  model_state (full: backbone + head), optimizer_state, scheduler_state,
  amp_scaler_state (explicitly 'not_used': training runs in fp32), epoch, global_step,
  best_metric/best_epoch, early-stopping counter, history, RNG states (python, numpy,
  torch CPU, torch CUDA), DataLoader shuffle-generator state, run config, input scaler,
  label mapping, split + preprocessing fingerprints, environment info and source SHA.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import InputScaler, SplitData
from .metrics import binary_metrics
from .models import build_model, param_groups
from .utils import (atomic_torch_save, env_info, get_rng_states, git_sha, set_rng_states,
                    set_seed, write_json)

LABEL_MAP = {"0": "non-error", "1": "error"}


def _logits_to_margin(logits: torch.Tensor) -> torch.Tensor:
    # error-logit margin z = l1 - l0 ; P(error) = sigmoid(z) = softmax(l)[1]
    return logits[:, 1] - logits[:, 0]


@torch.no_grad()
def predict_margin(model: nn.Module, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(_logits_to_margin(model(xb)).double().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def _class_weights(y: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "inverse_frequency":
        counts = np.bincount(y, minlength=2).astype(np.float64)
        w = len(y) / (2.0 * counts)
        return torch.tensor(w, dtype=torch.float32)
    raise ValueError(mode)


class Trainer:
    def __init__(self, cfg: dict, run_dir: str | Path, train: SplitData, val: SplitData,
                 device: str = "cuda", meta: dict | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.meta = meta or {}
        tcfg = cfg["train"]
        set_seed(cfg["seed"], deterministic=tcfg.get("deterministic", True))

        self.scaler = InputScaler(cfg["input"]["scaling"], dict(cfg["input"].get("scaling_params", {}))).fit(train.X)
        self.Xtr = torch.from_numpy(self.scaler.transform(train.X)).to(self.device)
        self.ytr = torch.from_numpy(train.y).to(self.device)
        self.Xva = self.scaler.transform(val.X)
        self.yva = val.y

        self.model = build_model(cfg["model"], input_shape=tuple(train.X.shape[1:])).to(self.device)
        self.opt = torch.optim.AdamW(param_groups(self.model, cfg), weight_decay=tcfg["weight_decay"]) \
            if tcfg["optimizer"] == "adamw" else torch.optim.Adam(param_groups(self.model, cfg),
                                                                  weight_decay=tcfg["weight_decay"])
        self.steps_per_epoch = math.ceil(len(self.ytr) / tcfg["batch_size"])
        if tcfg.get("scheduler", "none") == "cosine":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=tcfg["max_epochs"] * self.steps_per_epoch, eta_min=tcfg.get("eta_min", 1e-6))
        else:
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: 1.0)
        w = _class_weights(train.y, tcfg["class_weighting"])
        self.loss_w = w.to(self.device) if w is not None else None
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(cfg["seed"])

        self.epoch = 0
        self.global_step = 0
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.history: list[dict] = []
        self.class_counts_train = np.bincount(train.y, minlength=2).tolist()

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.sched.state_dict(),
            "amp_scaler_state": "not_used (fp32 training)",
            "epoch": self.epoch,                    # number of completed epochs
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "early_stopping_bad_epochs": self.bad_epochs,
            "history": self.history,
            "rng_states": get_rng_states(),
            "sampler_generator_state": self.gen.get_state(),
            "config": self.cfg,
            "input_scaler": self.scaler.state(),
            "label_mapping": LABEL_MAP,
            "class_counts_train": self.class_counts_train,
            "meta": self.meta,
            "source_sha": git_sha(),
            "env": env_info(),
        }

    def load_state(self, st: dict[str, Any]) -> None:
        self.model.load_state_dict(st["model_state"])
        self.opt.load_state_dict(st["optimizer_state"])
        self.sched.load_state_dict(st["scheduler_state"])
        self.epoch = st["epoch"]
        self.global_step = st["global_step"]
        self.best_metric = st["best_metric"]
        self.best_epoch = st["best_epoch"]
        self.bad_epochs = st["early_stopping_bad_epochs"]
        self.history = st["history"]
        self.gen.set_state(st["sampler_generator_state"])
        set_rng_states(st["rng_states"])
        assert self.scaler.state() == st["input_scaler"], "input scaler mismatch on resume"

    def try_resume(self) -> bool:
        p = self.ckpt_dir / "last.ckpt"
        if not p.exists():
            return False
        # load on CPU: RNG / generator states must stay CPU ByteTensors; model and optimizer
        # tensors are moved to the parameters' device by load_state_dict.
        st = torch.load(p, map_location="cpu", weights_only=False)
        self.load_state(st)
        print(f"[resume] from {p} at epoch {self.epoch}, step {self.global_step}", flush=True)
        return True

    def _save_epoch_checkpoints(self, improved: bool) -> None:
        st = self.state_dict()
        atomic_torch_save(st, self.ckpt_dir / "last.ckpt")
        if improved:
            atomic_torch_save(st, self.ckpt_dir / "best.ckpt")
        k = self.cfg["train"]["periodic_ckpt_every"]
        keep = self.cfg["train"]["periodic_ckpt_keep"]
        if k and self.epoch % k == 0:
            atomic_torch_save(st, self.ckpt_dir / f"epoch_{self.epoch:04d}.ckpt")
            periodic = sorted(self.ckpt_dir.glob("epoch_*.ckpt"))
            for old in periodic[:-keep] if keep > 0 else []:
                old.unlink()

    # ------------------------------------------------------------------ loop
    def train_one_epoch(self, max_steps: int | None = None) -> dict:
        tcfg = self.cfg["train"]
        self.model.train()
        n = len(self.ytr)
        perm = torch.randperm(n, generator=self.gen).to(self.device)
        bs = tcfg["batch_size"]
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            xb, yb = self.Xtr[idx], self.ytr[idx]
            if tcfg.get("augment", {}).get("time_shift_samples", 0):
                s = tcfg["augment"]["time_shift_samples"]
                shift = int(torch.randint(-s, s + 1, (1,), generator=self.gen))
                xb = torch.roll(xb, shifts=shift, dims=-1)
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb, weight=self.loss_w,
                                   label_smoothing=tcfg.get("label_smoothing", 0.0))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg["grad_clip"])
            self.opt.step()
            self.sched.step()
            if hasattr(self.model, "apply_constraints"):
                self.model.apply_constraints()
            self.global_step += 1
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
            if max_steps and self.global_step >= max_steps:
                break
        return {"train_loss": tot / max(cnt, 1)}

    def validate(self) -> dict:
        z = predict_margin(self.model, self.Xva, self.device)
        p = 1 / (1 + np.exp(-z))
        m = binary_metrics(self.yva, p)
        yt = torch.from_numpy(self.yva)
        lw = self.loss_w.cpu() if self.loss_w is not None else None
        logits2 = torch.stack([torch.zeros_like(torch.from_numpy(z)), torch.from_numpy(z)], 1).float()
        m["val_loss"] = float(F.cross_entropy(logits2, yt, weight=lw))
        return m

    def fit(self, max_epochs: int | None = None, max_steps: int | None = None) -> dict:
        tcfg = self.cfg["train"]
        max_epochs = max_epochs or tcfg["max_epochs"]
        log_path = self.run_dir / "train_log.jsonl"
        monitor = tcfg["monitor"]  # e.g. 'auroc'
        stopped_early = False
        while self.epoch < max_epochs:
            if self.bad_epochs >= tcfg["patience"]:
                stopped_early = True
                break
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tr = self.train_one_epoch(max_steps=max_steps)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_train = time.time() - t0
            vm = self.validate()
            self.epoch += 1
            score = vm[monitor]
            improved = bool(np.isfinite(score) and score > self.best_metric + tcfg.get("min_delta", 0.0))
            if improved:
                self.best_metric, self.best_epoch, self.bad_epochs = float(score), self.epoch, 0
            else:
                self.bad_epochs += 1
            rec = {"epoch": self.epoch, "global_step": self.global_step, **tr,
                   "val_loss": vm["val_loss"], "val_auroc": vm["auroc"],
                   "val_balanced_accuracy": vm["balanced_accuracy"], "val_brier": vm["brier"],
                   "improved": improved, "best_metric": self.best_metric, "best_epoch": self.best_epoch,
                   "epoch_time_s": round(time.time() - t0, 3), "train_time_s": round(t_train, 3),
                   "train_samples_per_s": round(len(self.ytr) / max(t_train, 1e-9), 1),
                   "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                   if torch.cuda.is_available() else None,
                   "lr": [g["lr"] for g in self.opt.param_groups]}
            self.history.append(rec)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            self._save_epoch_checkpoints(improved)
            if max_steps and self.global_step >= max_steps:
                break
        if self.bad_epochs >= tcfg["patience"]:
            stopped_early = True
        summary = {"epochs_completed": self.epoch, "global_step": self.global_step,
                   "best_metric": self.best_metric, "best_epoch": self.best_epoch,
                   "monitor": monitor, "stopped_early": stopped_early,
                   "total_epoch_time_s": round(sum(h["epoch_time_s"] for h in self.history), 2)}
        write_json(summary, self.run_dir / "train_summary.json")
        return summary


def load_model_from_ckpt(path: str | Path, device="cpu"):
    st = torch.load(path, map_location=device, weights_only=False)
    cfg = st["config"]
    model = build_model(cfg["model"], input_shape=tuple(st["meta"]["input_shape"]), load_pretrained=False)
    model.load_state_dict(st["model_state"])
    model.to(device).eval()
    scaler = InputScaler(st["input_scaler"]["kind"], st["input_scaler"]["params"])
    return model, scaler, st
