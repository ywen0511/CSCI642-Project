"""Deterministic candidate enumeration for the declared search budget (config 'tuning.grid')."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

SHORT = {"train.lr": "lr", "train.backbone_lr": "bblr"}


def _fmt(v) -> str:
    return f"{v:.0e}".replace("e-0", "e-") if isinstance(v, float) else str(v)


def candidates(config_path: str | Path, seed: int | None = None) -> list[dict]:
    cfg = load_yaml(config_path)
    seed = cfg["seed"] if seed is None else seed
    model = cfg["model"]["name"]
    grid = (cfg.get("tuning") or {}).get("grid") or {}
    keys = list(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        tag = "_".join(f"{SHORT.get(k, k.split('.')[-1])}{_fmt(v)}" for k, v in zip(keys, vals))
        run_id = f"{model}_{tag}_s{seed}" if tag else f"{model}_s{seed}"
        out.append({"run_id": run_id, "seed": seed,
                    "overrides": [f"{k}={json.dumps(v)}" for k, v in zip(keys, vals)]})
    return out
