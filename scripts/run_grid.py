#!/usr/bin/env python
"""Train every candidate of a config's declared search grid (sequentially, resumable).

  python scripts/run_grid.py --config configs/models/eegnet.yaml --prep-dir ... --out-root ...
  python scripts/run_grid.py --config ... --seed 43 --only-run-id <selected run id pattern>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from errp.tuning import candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--overrides", nargs="*", default=None,
                    help="train only the candidate with exactly these overrides (used for extra seeds)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cands = candidates(a.config, a.seed)
    if a.overrides is not None:
        cands = [c for c in cands if sorted(c["overrides"]) == sorted(a.overrides)]
        assert len(cands) == 1, cands
    for c in cands:
        cmd = [sys.executable, str(Path(__file__).parent / "train.py"), "--config", a.config, "--run-id", c["run_id"],
               "--prep-dir", a.prep_dir, "--splits", a.splits, "--out-root", a.out_root, "--seed", str(c["seed"])]
        for o in c["overrides"]:
            cmd += ["--override", o]
        print("[grid]", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
