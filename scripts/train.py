#!/usr/bin/env python
"""Train one decoder run from a config file. Resumes automatically from last.ckpt.

Example:
  python scripts/train.py --config configs/models/eegnet.yaml --run-id eegnet_s42 \
      --prep-dir data/derived/prep_v1 --splits configs/splits.json --out-root artifacts/runs
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from errp.data import SplitData, get_split, load_splits  # noqa: E402
from errp.train import Trainer  # noqa: E402
from errp.utils import env_info, git_sha, load_yaml, sha256_file, write_json  # noqa: E402


def subset(d: SplitData, n: int, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.where(d.y == c)[0], n // 2, replace=False) for c in (0, 1)]))
    return SplitData(d.X[idx], d.y[idx], d.trial_ids[idx], d.subject_ids[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--out-root", default="artifacts/runs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="dotted key=json, e.g. train.lr=0.0003")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs (smoke/benchmark/resume check)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke-train-subset", type=int, default=0,
                    help="train on a balanced subset of N training trials and validate on the same subset "
                         "(overfit sanity check only)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    for ov in args.override:
        k, v = ov.split("=", 1)
        node = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            node = node[kk]
        node[ks[-1]] = json.loads(v)
    if args.seed is not None:
        cfg["seed"] = args.seed
    splits = load_splits(args.splits)
    view = cfg["input"]["view"]
    train = get_split(args.prep_dir, view, splits, "train")
    val = get_split(args.prep_dir, view, splits, "validation")
    if args.smoke_train_subset:
        train = subset(train, args.smoke_train_subset, cfg["seed"])
        val = train
    run_dir = Path(args.out_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prep_cfg = json.loads((Path(args.prep_dir) / "preprocessing_config.json").read_text())
    meta = {
        "run_id": args.run_id,
        "input_shape": list(train.X.shape[1:]),
        "splits_sha256": sha256_file(args.splits),
        "splits": splits,
        "preprocessing_id": prep_cfg["preprocessing_id"],
        "trial_manifest_sha256": sha256_file(Path(args.prep_dir) / "trial_manifest.csv"),
        "n_train": int(len(train.y)), "n_val": int(len(val.y)),
        "train_class_counts": np.bincount(train.y, minlength=2).tolist(),
        "val_class_counts": np.bincount(val.y, minlength=2).tolist(),
        "smoke_train_subset": args.smoke_train_subset,
    }
    write_json({"config": cfg, "meta": meta, "env": env_info(), "source_sha": git_sha(),
                "argv": sys.argv}, run_dir / "run_config.json")
    trainer = Trainer(cfg, run_dir, train, val, device=args.device, meta=meta)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    load_report = getattr(trainer.model, "pretrained_load_report", None)
    if cfg["model"]["name"] == "cbramod":
        load_report = {"path": cfg["model"]["pretrained_path"], "strict": True,
                       "loaded": bool(trainer.model.pretrained_loaded)}
    if load_report is not None:
        write_json(load_report, run_dir / "pretrained_load_report.json")
        print("[pretrained]", json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in load_report.items()}))
    print(f"[run] {args.run_id} params={n_params} train={meta['train_class_counts']} "
          f"val={meta['val_class_counts']} input={meta['input_shape']}", flush=True)
    if not args.no_resume:
        trainer.try_resume()
    t0 = time.time()
    summary = trainer.fit(max_epochs=args.max_epochs, max_steps=args.max_steps)
    summary["wall_time_s_this_invocation"] = round(time.time() - t0, 2)
    summary["n_params"] = n_params
    write_json(summary, run_dir / "train_summary.json")
    print("[done]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
