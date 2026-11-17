#!/usr/bin/env python
"""Pre-run geometry check of the route-preference task (FINAL_V3_HANDOFF.md, Section 7.1; no learning).

On the 100 evaluation goals (eval seeds) and the 3 trajectory goals:
  1. every filtered goal satisfies D = ||g_xy - s0_xy|| >= min_goal_distance_xy_m;
  2. a straight-line scripted controller (proportional, direction-preserving, at most 1.5 cm per step and axis) violates on every goal;
  3. a scripted left-detour controller (waypoint m + 0.06 m * left normal, then the goal) reaches the goal
     without any violation on every goal (preferred pass);
  4. on both scripted paths the logged feedback labels (ideal condition, PreferenceFeedbackReward) equal the
     definitions recomputed from the logged positions: label 1 if any step of the segment is a violation,
     otherwise the v2 distance-change rule (progress 0, retreat 1, deadband -> no label).

Usage: python scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_candidates/pref_w100.json --out DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.environment import error_label  # noqa: E402
from rl.preference import Geometry, PreferenceFeedbackReward, episode_metrics, make_preference_task  # noqa: E402
from rl.train import read_config  # noqa: E402

STEP_LIMIT = 0.3  # action units; FetchReach moves the mocap 0.05 m per unit -> at most 1.5 cm per step
WAYPOINT_OFFSET = 0.06


def controller(kind, geo, goal):
    waypoint = None
    if kind == "left_detour":
        waypoint = np.append(geo.m + WAYPOINT_OFFSET * geo.left_normal(), goal[2])
    state = {"phase": 0}

    def act(obs):
        p = obs["achieved_goal"]
        target = goal
        if waypoint is not None and state["phase"] == 0:
            target = np.array([waypoint[0], waypoint[1], p[2] + 0.5 * (goal[2] - p[2])])
            if np.linalg.norm(p[:2] - waypoint[:2]) < 0.01:
                state["phase"] = 1
                target = goal
        v = (target - p) / 0.05
        v = v * min(1.0, STEP_LIMIT / max(np.abs(v).max(), 1e-12))  # scale the whole vector: keeps the direction
        return np.append(v, 0.0).astype(np.float32)
    return act


def run_episode(cfg, seed, kind, log_path):
    env = PreferenceFeedbackReward(make_preference_task(cfg), cfg, "ideal", 0, log_path=log_path)
    # the wrapper owns training episode seeds; force the evaluation/trajectory goal seed for this check
    env.episode_seed_base, env.episode_count = seed, 0
    obs, _ = env.reset()
    goal = np.array(obs["desired_goal"])
    geo = env.geometry
    act = controller(kind, geo, goal)
    positions = [np.array(obs["achieved_goal"])]
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        positions.append(np.array(obs["achieved_goal"]))
        done = term or trunc
    env.close()
    m = episode_metrics(geo, positions, bool(info["is_success"]))
    return geo, m, bool(info["is_success"])


def check_labels(log_path, cfg):
    """Recompute violations from the logged positions and segment labels from the definitions."""
    n = bad = 0
    seg_viol = False
    with open(log_path, newline="") as f:
        for r in csv.DictReader(f):
            geo = Geometry([float(r["start_x"]), float(r["start_y"])], [float(r["goal_x"]), float(r["goal_y"])],
                           cfg["preference"])
            c = geo.classify([float(r["px"]), float(r["py"])])
            d = float(r["distance"])
            if int(r["episode_step"]) == 1:
                seg_viol = False
            seg_viol |= c["violation"]
            bad += int(int(r["violation"]) != int(c["violation"]))
            if int(r["segment_end"]):
                seg_start = d + float(r["segment_progress"])  # segment_progress = start distance - end distance
                expected = 1 if seg_viol else error_label(seg_start, d, cfg["error_epsilon_m"])
                got = None if r["simulated_label"] == "" else int(r["simulated_label"])
                n += 1
                bad += int(expected != got)
                if got is not None:
                    w = cfg["eeg_weight"]
                    bad += int(abs(float(r["eeg_reward"]) - (w / 2 if got == 0 else -w / 2)) > 1e-12)
                seg_viol = False
    return n, bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = read_config(a.config, "pilot")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    seeds = list(range(cfg["eval_seed_base"], cfg["eval_seed_base"] + cfg["eval_episodes"])) + [30000, 30001, 30002]
    rows, report = [], {"config": a.config, "preference": cfg["preference"], "n_goals": len(seeds),
                        "step_limit_action_units": STEP_LIMIT, "waypoint_offset_m": WAYPOINT_OFFSET}
    for seed in seeds:
        row = {"seed": seed}
        for kind in ("straight", "left_detour"):
            log = out / f"{kind}_{seed}.csv"
            geo, m, success = run_episode(cfg, seed, kind, log)
            n, bad = check_labels(log, cfg)
            row.update({"D": geo.D, f"{kind}_success": success, f"{kind}_violation_free": m["violation_free"],
                        f"{kind}_preferred_pass": m["preferred_pass"], f"{kind}_pillar": m["pillar_entry"],
                        f"{kind}_right": m["right_corridor_entry"], f"{kind}_segments": n,
                        f"{kind}_label_mismatches": bad})
            log.unlink()  # per-goal logs are checked here; the summary below is the committed evidence
        rows.append(row)
    D_min = cfg["preference"]["min_goal_distance_xy_m"]
    checks = {
        "all_goals_satisfy_distance_rule": all(r["D"] >= D_min for r in rows),
        "straight_violates_on_every_goal": all(not r["straight_violation_free"] for r in rows),
        "left_detour_reaches_goal_without_violation_on_every_goal": all(
            r["left_detour_success"] and r["left_detour_violation_free"] and r["left_detour_preferred_pass"] for r in rows),
        "labels_match_definitions": all(r["straight_label_mismatches"] == 0 and r["left_detour_label_mismatches"] == 0
                                        for r in rows),
    }
    report.update(checks=checks, ok=all(checks.values()),
                  D_range=[min(r["D"] for r in rows), max(r["D"] for r in rows)],
                  straight={"success": sum(r["straight_success"] for r in rows),
                            "pillar_entry": sum(r["straight_pillar"] for r in rows),
                            "right_corridor_entry": sum(r["straight_right"] for r in rows)},
                  left_detour={"success": sum(r["left_detour_success"] for r in rows),
                               "preferred_pass": sum(r["left_detour_preferred_pass"] for r in rows)},
                  labeled_segments_checked=sum(r["straight_segments"] + r["left_detour_segments"] for r in rows))
    with (out / "per_goal.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "geometry_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("geometry check failed")


if __name__ == "__main__":
    main()
