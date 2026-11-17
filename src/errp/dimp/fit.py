"""One decoder fit of the improvement study (CV fold with early stopping, or fixed-epoch final refit).

Checkpoints (all with full resumable state): best.ckpt (CV only: highest inner-validation participant-macro AUROC,
earliest epoch on exact ties), last.ckpt, epoch_XXXX.ckpt every `periodic_every_epochs` (never pruned), and for
final refits final.ckpt (= the fixed-epoch snapshot used for evaluation). Contents: model (backbone + head),
optimizer, scheduler, RNG (python/numpy/torch CPU/CUDA), sampler generator, scaler ('not_used', fp32), epoch and
step counters (zero-based last index and one-based completed count), early-stopping counter, history,
trainable-parameter names and module modes, class weights, input scaler, initialization provenance, fit spec.
Resumption is allowed only when the fit identity hash, source training-file hashes and input hashes match.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..data import InputScaler
from ..utils import atomic_torch_save, env_info, get_rng_states, git_sha, set_rng_states, set_seed, sha256_file, write_json
from . import models
from .data import SubjectData

TRAINING_FILES = ("src/errp/dimp/fit.py", "src/errp/dimp/models.py", "src/errp/dimp/data.py", "src/errp/dimp/spec.py",
                  "src/errp/models/eegnet.py", "src/errp/models/cbramod_official/cbramod.py",
                  "src/errp/models/cbramod_official/criss_cross_transformer.py", "src/errp/data.py", "src/errp/utils.py")
PROBE_N = 16


def training_source_hashes(root: Path) -> dict:
    return {f: sha256_file(root / f) for f in TRAINING_FILES}


def class_weights(y, kind):
    if kind == "ce":
        return None
    if kind == "weighted_ce":
        c = np.bincount(y, minlength=2).astype(np.float64)
        return torch.tensor(len(y) / (2.0 * c), dtype=torch.float32)
    raise ValueError(kind)


def per_participant_auroc(subjects, y, score):
    out = {}
    for s in sorted(set(subjects)):
        m = subjects == s
        if len(set(y[m])) < 2:
            out[s] = float("nan")
        else:
            out[s] = float(roc_auc_score(y[m], score[m]))
    return out


class Fit:
    def __init__(self, spec: dict, run_dir, train: SubjectData, val: SubjectData | None, root: Path,
                 pretrained_path: str | None, init_sha256: str | None, input_hashes: dict, device="cuda"):
        self.spec, self.recipe = spec, spec["recipe"]
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.root = root
        r = self.recipe
        set_seed(spec["seed"], deterministic=True)
        sc = r["scaling"]
        self.scaler = InputScaler(sc["kind"], {k: v for k, v in sc.items() if k != "kind"}).fit(train.X)
        self.Xtr = torch.from_numpy(self.scaler.transform(train.X)).to(self.device)
        self.ytr = torch.from_numpy(train.y).to(self.device)
        self.train_ids = train.trial_ids
        self.val = val
        if val is not None:
            self.Xva = self.scaler.transform(val.X)
        self.model = models.build(r, tuple(train.X.shape[1:]),
                                  pretrained_path if r["arch"] == "cbramod" else None).to(self.device)
        if r["arch"] == "cbramod":
            assert self.model.load_report and not self.model.load_report["missing"] and \
                not self.model.load_report["unexpected"] and not self.model.load_report["shape_mismatch"]
        lr_head = r["head_lr"] if r["arch"] == "cbramod" else r["lr"]
        groups = self.model.param_groups(lr_head, r.get("backbone_lr"))
        if r["optimizer"] == "adamw":
            self.opt = torch.optim.AdamW(groups, weight_decay=r["weight_decay"])
        else:
            self.opt = torch.optim.Adam(groups, weight_decay=r["weight_decay"])
        self.steps_per_epoch = math.ceil(len(self.ytr) / r["batch_size"])
        if r["scheduler"] == "cosine":
            # horizon = configured maximum epochs (not the selected final epoch count)
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=r["scheduler_horizon_epochs"] * self.steps_per_epoch, eta_min=r["eta_min"])
        else:
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: 1.0)
        w = class_weights(train.y, r["loss"])
        self.loss_w = w.to(self.device) if w is not None else None
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(spec["seed"])
        self.epoch = 0  # completed epochs (one-based count of the last completed epoch)
        self.global_step = 0
        self.best_metric, self.best_epoch, self.bad_epochs = -float("inf"), 0, 0
        self.history = []
        self.provenance = {"pretrained_path_name": Path(pretrained_path).name if pretrained_path else None,
                           "initialization_sha256": init_sha256, "input_hashes": input_hashes,
                           "training_source_sha256": training_source_hashes(root),
                           "class_counts_train": np.bincount(train.y, minlength=2).tolist(),
                           "n_train": int(len(train.y)), "n_val": int(len(val.y)) if val is not None else 0,
                           "train_participants": sorted(set(train.subject_ids)),
                           "val_participants": sorted(set(val.subject_ids)) if val is not None else [],
                           "load_report": getattr(self.model, "load_report", None)}

    # ------------------------------------------------------------- state
    def state_dict(self):
        return {"format": "csci642-dimp-ckpt/v1", "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(), "scheduler_state": self.sched.state_dict(),
                "amp_scaler_state": "not_used (fp32 training)",
                "epoch_completed": self.epoch, "epoch_index_zero_based": self.epoch - 1,
                "global_step": self.global_step, "best_metric": self.best_metric, "best_epoch": self.best_epoch,
                "early_stopping_bad_epochs": self.bad_epochs, "history": self.history,
                "rng_states": get_rng_states(), "sampler_generator_state": self.gen.get_state(),
                "trainable_names": self.model.trainable_names(), "module_modes_train": self._train_modes(),
                "class_weights": None if self.loss_w is None else self.loss_w.cpu().tolist(),
                "input_scaler": self.scaler.state(), "label_mapping": {"0": "non-error", "1": "error"},
                "spec": self.spec, "provenance": self.provenance, "source_sha": git_sha(self.root), "env": env_info()}

    def _train_modes(self):
        was = self.model.training
        self.model.train()
        modes = self.model.module_modes()
        self.model.train(was)
        return modes

    def load_state(self, st):
        if st["spec"]["identity_hash"] != self.spec["identity_hash"]:
            raise ValueError("resume identity mismatch")
        if st["provenance"]["training_source_sha256"] != self.provenance["training_source_sha256"]:
            raise ValueError("resume training-source mismatch")
        if st["provenance"]["input_hashes"] != self.provenance["input_hashes"]:
            raise ValueError("resume input mismatch")
        if st["trainable_names"] != self.model.trainable_names():
            raise ValueError("trainability mask mismatch")
        self.model.load_state_dict(st["model_state"])
        self.opt.load_state_dict(st["optimizer_state"])
        self.sched.load_state_dict(st["scheduler_state"])
        self.epoch, self.global_step = st["epoch_completed"], st["global_step"]
        self.best_metric, self.best_epoch = st["best_metric"], st["best_epoch"]
        self.bad_epochs, self.history = st["early_stopping_bad_epochs"], st["history"]
        self.gen.set_state(st["sampler_generator_state"])
        set_rng_states(st["rng_states"])
        assert self.scaler.state() == st["input_scaler"]

    def try_resume(self):
        p = self.ckpt_dir / "last.ckpt"
        if not p.exists():
            return None
        st = torch.load(p, map_location="cpu", weights_only=False)
        self.load_state(st)
        return {"resumed_from_epoch": self.epoch, "global_step": self.global_step}

    # ------------------------------------------------------------- loop
    def train_one_epoch(self):
        r = self.recipe
        self.model.train()
        n = len(self.ytr)
        perm = torch.randperm(n, generator=self.gen).to(self.device)
        tot, cnt = 0.0, 0
        for i in range(0, n, r["batch_size"]):
            idx = perm[i:i + r["batch_size"]]
            if len(idx) < 2:
                continue
            logits = self.model(self.Xtr[idx])
            loss = F.cross_entropy(logits, self.ytr[idx], weight=self.loss_w, label_smoothing=r["label_smoothing"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.global_step}")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if r.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], r["grad_clip"])
            self.opt.step()
            self.sched.step()
            if hasattr(self.model, "apply_constraints"):
                self.model.apply_constraints()
            self.global_step += 1
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        return tot / max(cnt, 1)

    @torch.no_grad()
    def margins(self, X):
        self.model.eval()
        out = []
        for i in range(0, len(X), 256):
            lg = self.model(torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(self.device))
            out.append((lg[:, 1] - lg[:, 0]).double().cpu())
        return torch.cat(out).numpy()

    def validate(self):
        z = self.margins(self.Xva)
        y, s = self.val.y, self.val.subject_ids
        pp = per_participant_auroc(s, y, z)
        from .spec import participant_macro_auroc
        macro = participant_macro_auroc(pp)
        p = 1 / (1 + np.exp(-z))
        lw = self.loss_w.cpu() if self.loss_w is not None else None
        lg = torch.stack([torch.zeros(len(z)), torch.from_numpy(z).float()], 1)
        vloss = float(F.cross_entropy(lg, torch.from_numpy(y), weight=lw))
        return {"val_macro_auroc": macro, "val_pooled_auroc": float(roc_auc_score(y, z)),
                "val_participant_auroc": pp, "val_loss": vloss, "val_brier": float(np.mean((p - y) ** 2))}

    def save(self, improved):
        st = self.state_dict()
        atomic_torch_save(st, self.ckpt_dir / "last.ckpt")
        if improved:
            atomic_torch_save(st, self.ckpt_dir / "best.ckpt")
        k = self.recipe["periodic_every_epochs"]
        if k and self.epoch % k == 0:
            atomic_torch_save(st, self.ckpt_dir / f"epoch_{self.epoch:04d}.ckpt")  # never pruned

    def fit(self, stop_after_epochs: int | None = None):
        """CV: until max_epochs or patience. Final: exactly spec['epochs'] epochs.
        stop_after_epochs (testing only) interrupts the loop to exercise resumption."""
        r, final = self.recipe, self.spec["mode"] == "final"
        target = self.spec["epochs"] if final else r["max_epochs"]
        log = self.run_dir / "train_log.jsonl"
        n_run = 0
        while self.epoch < target and (final or self.bad_epochs < r["patience"]):
            if stop_after_epochs is not None and n_run >= stop_after_epochs:
                return None
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss = self.train_one_epoch()
            self.epoch += 1
            rec = {"epoch": self.epoch, "epoch_index_zero_based": self.epoch - 1, "global_step": self.global_step,
                   "train_loss": loss, "lr": [g["lr"] for g in self.opt.param_groups]}
            improved = False
            if not final:
                vm = self.validate()
                rec.update(vm)
                score = vm["val_macro_auroc"]
                improved = score > self.best_metric  # strict: earliest epoch wins exact ties
                if improved:
                    self.best_metric, self.best_epoch, self.bad_epochs = score, self.epoch, 0
                else:
                    self.bad_epochs += 1
                rec.update(best_metric=self.best_metric, best_epoch=self.best_epoch, improved=improved)
            rec.update(epoch_time_s=round(time.time() - t0, 3),
                       peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None)
            self.history.append(rec)
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.save(improved)
            n_run += 1
        if final:
            atomic_torch_save(self.state_dict(), self.ckpt_dir / "final.ckpt")
        return {"epochs_completed": self.epoch, "global_step": self.global_step,
                "best_epoch": None if final else self.best_epoch,
                "best_metric": None if final else self.best_metric,
                "stopped_early": (not final) and self.bad_epochs >= r["patience"] and self.epoch < target,
                "total_epoch_time_s": round(sum(h["epoch_time_s"] for h in self.history), 2)}
