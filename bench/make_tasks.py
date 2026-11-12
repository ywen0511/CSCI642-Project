#!/usr/bin/env python
"""Write one shell command per (dataset, model, align, seed, fold) fit that is not finished yet.

  python bench/make_tasks.py --B $B --datasets d1 d2 --models m1 m2 --aligns none ea --seeds 42 43 44 \
      --device cuda --out tasks.txt
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--B", required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--aligns", nargs="+", default=["none", "ea"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()
B = Path(a.B)
lines = []
for ds in a.datasets:
    folds = json.loads((B / "data/folds" / ds / "folds.json").read_text())["k"]
    for m in a.models:
        for al in a.aligns:
            for s in a.seeds:
                for i in range(folds):
                    o = B / "runs" / ds / m / al / f"s{s}" / f"fold{i}"
                    if (o / "metrics.json").exists():
                        continue
                    lines.append(f"cd {B}/code && {B}/env/bin/python bench/run_fold.py --prep-dir {B}/data/prep/{ds} "
                                 f"--config bench/configs/{m}.yaml --splits {B}/data/folds/{ds}/fold_{i}.json --seed {s} "
                                 f"--align {al} --out {o} --device {a.device} > {o}.log 2>&1 || echo FAIL {o}")
Path(a.out).write_text("\n".join(lines) + ("\n" if lines else ""))
print(len(lines), "tasks ->", a.out)
