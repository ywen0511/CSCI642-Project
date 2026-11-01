#!/usr/bin/env python
"""Resumption check.

Mode 'equivalence': train K epochs uninterrupted vs (K-1 epochs, stop, resume from
last.ckpt, 1 more epoch) and compare final weights / validation margins.

Mode 'continue': copy an existing last.ckpt (e.g. downloaded from the GitHub Release) into a
scratch run directory (the original file is never modified), resume, train N more epochs and
report that the epoch/step counters advanced and the loss is finite.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from errp.data import get_split, load_splits  # noqa: E402
from errp.train import Trainer, predict_margin  # noqa: E402
from errp.utils import load_yaml, sha256_file, write_json  # noqa: E402


def make_trainer(cfg, run_dir, prep, splits, meta_shape=None):
    view = cfg["input"]["view"]
    tr = get_split(prep, view, splits, "train")
    va = get_split(prep, view, splits, "validation")
    meta = {"input_shape": list(tr.X.shape[1:]), "resume_check": True}
    return Trainer(cfg, run_dir, tr, va, meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["equivalence", "continue"], required=True)
    ap.add_argument("--config")
    ap.add_argument("--ckpt", help="last.ckpt to continue from (mode=continue)")
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    splits = load_splits(a.splits)
    work = Path(a.work_dir)
    if work.exists():
        shutil.rmtree(work)
    if a.mode == "equivalence":
        cfg = load_yaml(a.config)
        K = a.epochs
        t_full = make_trainer(cfg, work / "full", a.prep_dir, splits)
        t_full.fit(max_epochs=K)
        t_a = make_trainer(cfg, work / "split", a.prep_dir, splits)
        t_a.fit(max_epochs=K - 1)
        del t_a
        t_b = make_trainer(cfg, work / "split", a.prep_dir, splits)
        assert t_b.try_resume()
        t_b.fit(max_epochs=K)
        sa, sb = t_full.model.state_dict(), t_b.model.state_dict()
        maxdiff = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
        za = predict_margin(t_full.model, t_full.Xva, t_full.device)
        zb = predict_margin(t_b.model, t_b.Xva, t_b.device)
        res = {"mode": "equivalence", "config": a.config, "epochs": K,
               "max_abs_weight_diff": maxdiff, "max_abs_val_margin_diff": float(np.abs(za - zb).max()),
               "history_full_last": t_full.history[-1], "history_resumed_last": t_b.history[-1],
               "global_step_full": t_full.global_step, "global_step_resumed": t_b.global_step}
        res["bitwise_identical_weights"] = maxdiff == 0.0
    else:
        src = Path(a.ckpt)
        sha_before = sha256_file(src)
        st = torch.load(src, map_location="cpu", weights_only=False)
        cfg = st["config"]
        run = work / "continued"
        (run / "checkpoints").mkdir(parents=True)
        shutil.copy2(src, run / "checkpoints" / "last.ckpt")
        t = make_trainer(cfg, run, a.prep_dir, splits)
        assert t.try_resume()
        e0, s0 = t.epoch, t.global_step
        stopped = t.bad_epochs >= cfg["train"]["patience"]
        if stopped:  # the run had early-stopped; allow extra epochs for the check only
            t.cfg = json.loads(json.dumps(cfg))
            t.cfg["train"]["patience"] = t.bad_epochs + a.epochs + 1
        t.fit(max_epochs=e0 + a.epochs)
        res = {"mode": "continue", "source_ckpt": str(src), "source_sha256_before": sha_before,
               "source_sha256_after": sha256_file(src), "resumed_epoch": e0, "resumed_global_step": s0,
               "final_epoch": t.epoch, "final_global_step": t.global_step,
               "source_run_had_early_stopped": stopped,
               "new_epochs": t.history[-a.epochs:],
               "all_losses_finite": all(np.isfinite(h["train_loss"]) for h in t.history[-a.epochs:])}
        res["ok"] = (res["final_epoch"] == e0 + a.epochs and res["final_global_step"] > s0
                     and res["all_losses_finite"] and sha_before == res["source_sha256_after"])
    write_json(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if "history" not in k and k != "new_epochs"}))


if __name__ == "__main__":
    main()
