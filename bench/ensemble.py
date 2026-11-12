#!/usr/bin/env python
"""ErrP-Ens: label-free ensembles of aligned decoders, built from saved per-trial predictions (no training).

  python bench/ensemble.py --runs RUNS_ROOT [--k 3]

For every dataset, seed and fold: members = all decoders run with Euclidean alignment for that (seed, fold).
Each member's test margins are rank-transformed WITHIN each test participant (label-free; ranks in [0, 1]) and
averaged. Variants written as pseudo-models into RUNS_ROOT/<ds>/<name>/ea/s<seed>/fold<i>/{metrics.json,predictions.npz}:
  ens_all : all aligned members
  ens_top{k}: the k members with the highest macro-participant AUROC on that fold's VALIDATION participants
              (validation participants are disjoint from the fold's test participants).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fold import metrics_for  # noqa: E402

DETERMINISTIC = {"xdawn_rg"}
BASE = ["xdawn_rg", "eegnet", "eeginception_erp", "eegnex", "eegconformer", "ctnet", "labram", "eegpt", "cbramod"]


def subj_rank(z, subj):
    r = np.empty(len(z))
    for s in np.unique(subj):
        m = subj == s
        r[m] = (rankdata(z[m]) - 1) / max(1, m.sum() - 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    root = Path(a.runs)
    n = 0
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        nfold = len({p.name for m in BASE for p in (ds_dir / m / "ea").glob("s*/fold*")})
        seeds = sorted({p.name for m in BASE if m not in DETERMINISTIC for p in (ds_dir / m / "ea").glob("s*")})

        def run_dir(m, seed, fold):
            d = ds_dir / m / "ea" / seed / fold
            if not (d / "metrics.json").exists() and m in DETERMINISTIC:  # identical for every seed: reuse any
                d = next((q / fold for q in sorted((ds_dir / m / "ea").glob("s*")) if (q / fold / "metrics.json").exists()), d)
            return d

        for seed in seeds:
            # members fixed per (dataset, seed): models with ALL folds of this seed, so every test participant of a
            # seed is scored by the same ensemble
            folds = [f"fold{i}" for i in range(nfold)]
            members_seed = [m for m in BASE if all((run_dir(m, seed, f) / "metrics.json").exists() for f in folds)]
            for fold in folds:
                mem = {}
                for m in members_seed:
                    d = run_dir(m, seed, fold)
                    mem[m] = (json.loads((d / "metrics.json").read_text()), np.load(d / "predictions.npz", allow_pickle=True))
                if len(mem) < 2:
                    continue
                ref = next(iter(mem.values()))[1]
                tid, subj, y = ref["test_trial"], ref["test_subject"], ref["test_label"]
                for m, (_, p) in mem.items():
                    assert (p["test_trial"] == tid).all(), (ds_dir.name, m, seed, fold)
                ranks = {m: subj_rank(p["test_margin"], subj) for m, (_, p) in mem.items()}
                val = {m: r["validation"]["macro_subject_auroc"] for m, (r, _) in mem.items()}
                top = sorted(mem, key=lambda m: -val[m])[:a.k]
                for name, members in (("ens_all", sorted(mem)), (f"ens_top{a.k}", top)):
                    z = np.mean([ranks[m] for m in members], axis=0)
                    out = ds_dir / name / "ea" / seed / fold
                    out.mkdir(parents=True, exist_ok=True)
                    res = {"model": name, "align": "ea", "members": members, "member_val_auroc": val,
                           "test": metrics_for(y, z, subj), "wall_s": 0.0}
                    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
                    np.savez_compressed(out / "predictions.npz", test_subject=subj, test_trial=tid, test_label=y, test_margin=z)
                    n += 1
    print("wrote", n, "ensemble fold results")


if __name__ == "__main__":
    main()
