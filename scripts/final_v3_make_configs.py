#!/usr/bin/env python
"""Write the declared final-v3 candidate protocols (FINAL_V3_HANDOFF.md, Sections 6.3 and 7.3).

Every candidate copies the v2 validation candidate protocol (same SAC, budget, seeds, evaluation goals and
feedback mechanics) and changes only what the handoff declares:

* protocol v3 (configs/final_v3/v3_candidates/): schema csci642-rl-stage2/v3, per-decoder reward centers from
  configs/final_v3/centers.json (identified by SHA-256), eeg_weight in {0.5, 1.0, 2.0}, dense_weight 10.
* route preference (configs/final_v3/pref_candidates/): schema csci642-rl-pref/v1, the same centers,
  the preference geometry below, eeg_weight in {1.0, 2.0, 4.0}.

Usage: python scripts/final_v3_make_configs.py [--pillar-radius R] [--corridor A0 A1] [--pref-steps N]
(the preference options exist only for the declared pilot gate, Section 7.3; every change is logged in
reports/final_v3/decisions.jsonl before validation).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.feedback import sha256  # noqa: E402

BASE = ROOT / "configs/rl_v2/candidate_w100.json"
CENTERS = "configs/final_v3/centers.json"
V3_WEIGHTS = {"v3_w050": 0.5, "v3_w100": 1.0, "v3_w200": 2.0}
PREF_WEIGHTS = {"pref_w100": 1.0, "pref_w200": 2.0, "pref_w400": 4.0}


def base_v3():
    cfg = json.loads(BASE.read_text())
    assert cfg["schema"] == "csci642-rl-stage2/v2" and cfg["status"] == "pilot"
    centers = json.loads((ROOT / CENTERS).read_text())["centers"]
    out = {}
    for k, v in cfg.items():
        out[k] = v
        if k == "reward_center":
            out[k] = centers
            out["centers_file"] = CENTERS
            out["centers_sha256"] = sha256(ROOT / CENTERS)
    out["schema"] = "csci642-rl-stage2/v3"
    return out


def write(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path.relative_to(ROOT), sha256(path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pillar-radius", type=float, default=0.025)
    p.add_argument("--corridor", type=float, nargs=2, default=[0.25, 0.75])
    p.add_argument("--pref-steps", type=int, default=150000)
    p.add_argument("--only", choices=["v3", "pref"], help="write only one family")
    p.add_argument("--pref-dir", default="configs/final_v3/pref_candidates",
                   help="output directory for the preference configs (pilot-gate variants use their own directory)")
    p.add_argument("--pref-names", nargs="*", help="subset of preference candidates to write")
    a = p.parse_args()
    v3 = base_v3()
    if a.only in (None, "v3"):
        for name, w in V3_WEIGHTS.items():
            cfg = copy.deepcopy(v3)
            cfg.update(eeg_weight=w, dense_weight=10.0)
            write(ROOT / f"configs/final_v3/v3_candidates/{name}.json", cfg)
    if a.only in (None, "pref"):
        for name, w in PREF_WEIGHTS.items():
            if a.pref_names and name not in a.pref_names:
                continue
            cfg = copy.deepcopy(v3)
            cfg.update(schema="csci642-rl-pref/v1",
                       feedback_rule="distance_change_deadband_with_route_preference",
                       eeg_weight=w, dense_weight=10.0, total_timesteps=a.pref_steps,
                       preference={"min_goal_distance_xy_m": 0.15, "pillar_radius_m": a.pillar_radius,
                                   "corridor_a": list(a.corridor), "preferred_side": "left",
                                   "max_goal_redraws": 10000})
            write(ROOT / a.pref_dir / f"{name}.json", cfg)


if __name__ == "__main__":
    main()
