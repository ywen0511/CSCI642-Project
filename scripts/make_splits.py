#!/usr/bin/env python
"""Generate the frozen participant split (run once, after the data audit).

Rule (README Section 4): sort eligible participant IDs, permute with numpy
``default_rng(42)``, take the first 3 as test, the next 2 as validation, the rest as train.
Eligible = participants with a verified robot-condition file and both classes retained.
Writes configs/splits.json and fills the 'split' column of the trial manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from errp.utils import sha256_file, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out", default="configs/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        sp = json.loads(out.read_text())
        print("splits already frozen; not regenerating:", sp)
    else:
        man = pd.read_csv(Path(args.prep_dir) / "trial_manifest.csv", dtype={"subject_id": str})
        kept = man[man.retained == 1]
        per = kept.groupby("subject_id")["label"].agg(lambda s: set(s.astype(int)))
        eligible = sorted(s for s, labs in per.items() if labs == {0, 1})
        rng = np.random.default_rng(args.seed)
        perm = [eligible[i] for i in rng.permutation(len(eligible))]
        sp = {
            "test": sorted(perm[:args.n_test]),
            "validation": sorted(perm[args.n_test:args.n_test + args.n_val]),
            "train": sorted(perm[args.n_test + args.n_val:]),
            "eligible_sorted": eligible,
            "permutation": perm,
            "procedure": f"numpy.random.default_rng({args.seed}).permutation(len(eligible_sorted)); "
                         f"first {args.n_test} -> test, next {args.n_val} -> validation, rest -> train",
            "rng_library": f"numpy {np.__version__} (PCG64 via default_rng)",
            "seed": args.seed,
        }
        write_json(sp, out)
        print("wrote", out, sp)
    man_path = Path(args.prep_dir) / "trial_manifest.csv"
    man = pd.read_csv(man_path, dtype={"subject_id": str, "trial_id": str})
    lut = {s: k for k in ("train", "validation", "test") for s in sp[k]}
    man["split"] = man["subject_id"].map(lut).astype(object).where(man["subject_id"].isin(lut), "unassigned")
    man.to_csv(man_path, index=False)
    print("splits sha256", sha256_file(out))
    print(man[man.retained == 1].groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
