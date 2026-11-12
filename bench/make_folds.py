#!/usr/bin/env python
"""Cross-subject folds for one prepared dataset: K disjoint test-subject groups; inside each fold the validation
subjects (early stopping / calibration only) are a fixed share of the remaining subjects, never tested in that fold.

  python bench/make_folds.py --prep-dir DIR --k 5 --out DIR/folds   (writes fold_<i>.json + folds.json)
Deterministic: subjects sorted, shuffled with seed 0; fold i tests group i; validation = the next
max(1, round(0.15 * n_train_pool)) subjects of the shuffled order after the test group (cyclic).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--prep-dir", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
man = pd.read_csv(Path(a.prep_dir) / "trial_manifest.csv", dtype={"subject_id": str})
subs = sorted(man.loc[man.retained == 1, "subject_id"].unique())
order = list(np.random.default_rng(0).permutation(subs))
k = min(a.k, len(order))
groups = [list(g) for g in np.array_split(np.array(order, dtype=object), k)]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
folds = []
for i, test in enumerate(groups):
    rest = [s for s in order[order.index(test[-1]) + 1:] + order[:order.index(test[0])] if s not in test]
    n_val = max(1, round(0.15 * len(rest)))
    val, train = rest[:n_val], rest[n_val:]
    f = {"fold": i, "train": sorted(train), "validation": sorted(val), "test": sorted(test)}
    (out / f"fold_{i}.json").write_text(json.dumps(f, indent=1) + "\n")
    folds.append(f)
(out / "folds.json").write_text(json.dumps({"subjects": subs, "k": k, "folds": folds}, indent=1) + "\n")
print(k, "folds;", [(len(f["train"]), len(f["validation"]), len(f["test"])) for f in folds])
