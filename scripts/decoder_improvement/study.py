#!/usr/bin/env python
"""Plan and summarize the stages of the decoder-improvement study.

  plan      --stage {A,B,C,D,E,F,G} --decisions-dir DIR --prep-root PREP --out PLAN.json
  summarize --plan PLAN.json --runs-root RUNS --out DECISION.json

Plans use only study.json and the committed decisions of earlier stages (winners), never test data. A fit already
planned in an earlier stage with the identical identity (recipe, preprocessing, fold, seed, mode) is reused, not
re-planned. Selection (stages A-E): per inner-validation participant, mean AUROC over seeds at each fit's selected
checkpoint, then the mean over the 7 participants; ties within 1e-6 -> fewer trainable parameters -> lexicographic
candidate ID. Stage F (confirmation) and G (final refit) never reselect.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from errp.dimp.spec import (BAND_DIR, cbramod_recipe, eegnet_recipe, fit_spec, load_study, recipe_id,  # noqa: E402
                            selection_score, short_hash)

TOL = 1e-6
BAND_NAMES = {(1, 20): "1-20", (0.3, 40): "0.3-40", (0.3, 75): "0.3-75"}


def prep_id(prep_root, band):
    return json.loads((Path(prep_root) / BAND_DIR[band] / "preprocessing_config.json").read_text())["preprocessing_id"]


def load_decision(ddir, stage):
    p = Path(ddir) / f"stage_{stage}.json"
    if not p.exists():
        raise FileNotFoundError(f"decision for stage {stage} not committed: {p}")
    return json.loads(p.read_text())


def winner_recipe(ddir, stage, arch=None):
    d = load_decision(ddir, stage)
    w = d["winners"][arch] if arch else d["winner"]
    return w["recipe"], w["candidate_id"]


def previous_fit_ids(ddir, stage):
    out = set()
    for s in "ABCDEFG":
        if s == stage:
            break
        p = Path(ddir).parent / "plans" / f"stage_{s}.json"
        if p.exists():
            out |= {f["fit_id"] for f in json.loads(p.read_text())["fits"]}
    return out


def candidates_for(stage, study, ddir):
    st = {s["id"]: s for s in study["stages"]}
    if stage == "A":
        return [(c["id"], cbramod_recipe(study, c["head"], c["adaptation"], c["backbone_lr"], "1-20",
                                         head_lr=c["head_lr"])) for c in st["A"]["candidates"]], {}
    if stage == "B":
        r, cid = winner_recipe(ddir, "A")
        if r["adaptation"] == "frozen":
            return [], {"skipped": True, "reason": f"Stage A winner {cid} has a frozen backbone"}
        out = []
        for lr in st["B"]["values"]:
            rr = dict(r, backbone_lr=lr)
            out.append((f"{cid}@bb{lr:g}", rr))
        return out, {"parent": cid}
    if stage == "C":
        out = []
        for lr in st["C"]["lr"]:
            for do in st["C"]["dropout"]:
                out.append((f"eeg_lr{lr:g}_do{do:g}", eegnet_recipe(study, lr, do, "1-20")))
        return out, {"reference": "eeg_lr0.001_do0.5"}
    if stage == "D":
        cb = load_decision(ddir, "B")
        cb_r, cb_id = (cb["winner"]["recipe"], cb["winner"]["candidate_id"]) if not cb.get("skipped") \
            else winner_recipe(ddir, "A")
        eg_r, eg_id = winner_recipe(ddir, "C")
        out = []
        for lo, hi in st["D"]["bands_hz"]:
            b = BAND_NAMES[(lo, hi)]
            out.append((f"cbramod:{cb_id}@{b}", dict(cb_r, band=b)))
            out.append((f"eegnet:{eg_id}@{b}", dict(eg_r, band=b)))
        return out, {"parents": {"cbramod": cb_id, "eegnet": eg_id}}
    if stage == "E":
        d = load_decision(ddir, "D")
        out = []
        for arch in ("cbramod", "eegnet"):
            w = d["winners"][arch]
            for loss in st["E"]["losses"]:
                key = {"inverse_frequency_weighted_cross_entropy": "weighted_ce", "unweighted_cross_entropy": "ce"}[loss]
                out.append((f"{arch}:{w['candidate_id']}#{key}", dict(w["recipe"], loss=key)))
        return out, {"parents": {a: d["winners"][a]["candidate_id"] for a in ("cbramod", "eegnet")}}
    raise ValueError(stage)


def recipes_FG(study, ddir):
    """Selected (Stage E winner per architecture) and reference recipes."""
    e = load_decision(ddir, "E")
    a = {s["id"]: s for s in study["stages"]}["A"]
    ref_cb = next(c for c in a["candidates"] if c["id"] == "cb_reference")
    refs = {"cbramod": cbramod_recipe(study, ref_cb["head"], ref_cb["adaptation"], ref_cb["backbone_lr"], "1-20",
                                      head_lr=ref_cb["head_lr"]),
            "eegnet": eegnet_recipe(study, 0.001, 0.5, "1-20")}
    out = []
    for arch in ("cbramod", "eegnet"):
        sel = e["winners"][arch]["recipe"]
        out.append({"arch": arch, "role": "selected", "recipe": sel, "candidate_id": e["winners"][arch]["candidate_id"]})
        if short_hash(sel) == short_hash(refs[arch]):
            out[-1]["identical_to_reference"] = True
        else:
            out.append({"arch": arch, "role": "reference", "recipe": refs[arch], "candidate_id": f"{arch}_reference"})
    return out


def plan(stage, ddir, prep_root, out):
    study = load_study()
    folds = [f["id"] for f in study["inner_folds"]]
    meta = {}
    fits, candidates = [], []
    if stage in "ABCDE":
        cands, meta = candidates_for(stage, study, ddir)
        for cid, r in cands:
            pid = prep_id(prep_root, r["band"])
            candidates.append({"candidate_id": cid, "recipe_id": recipe_id(r), "recipe": r,
                               "fit_ids": [fit_spec(study, r, f, s, pid)["fit_id"] for f in folds
                                           for s in study["seeds"]["screening"]]})
            fits += [fit_spec(study, r, f, s, pid) for f in folds for s in study["seeds"]["screening"]]
    elif stage == "F":
        for item in recipes_FG(study, ddir):
            r = item["recipe"]
            pid = prep_id(prep_root, r["band"])
            ids = [fit_spec(study, r, f, s, pid) for f in folds for s in study["seeds"]["confirmation_and_final"]]
            candidates.append({**item, "recipe_id": recipe_id(r), "fit_ids": [x["fit_id"] for x in ids]})
            fits += ids
    elif stage == "G":
        freeze = json.loads((Path(ddir) / "freeze.json").read_text())
        for item in freeze["recipes"]:
            r = item["recipe"]
            pid = prep_id(prep_root, r["band"])
            ids = [fit_spec(study, r, "outer", s, pid, mode="final", epochs=item["final_epochs"])
                   for s in study["seeds"]["confirmation_and_final"]]
            candidates.append({**{k: item[k] for k in ("arch", "role", "candidate_id")}, "recipe_id": recipe_id(r),
                               "recipe": r, "final_epochs": item["final_epochs"], "fit_ids": [x["fit_id"] for x in ids]})
            fits += ids
    unique = {}
    for f in fits:
        if f["fit_id"] in unique and unique[f["fit_id"]]["identity_hash"] != f["identity_hash"]:
            raise ValueError("fit id collision with different identity")
        unique[f["fit_id"]] = f
    prev = previous_fit_ids(ddir, stage)
    new = [f for f in unique.values() if f["fit_id"] not in prev]
    result = {"stage": stage, "study_sha": short_hash(study, 16), **meta, "candidates": candidates,
              "fits": list(unique.values()), "new_fit_ids": [f["fit_id"] for f in new],
              "reused_fit_ids": sorted(set(unique) & prev)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"stage": stage, "candidates": len(candidates), "fits": len(unique), "new": len(new),
                      "reused": len(result["reused_fit_ids"]), **meta}))


def fit_status(runs_root, fid):
    p = Path(runs_root) / fid / "status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    if st.get("state") != "completed":
        raise ValueError(f"fit not completed: {fid}")
    return st


def prob_quality(runs_root, fids):
    """Pooled Brier/NLL/separation of raw p over all inner-validation predictions of a candidate (per seed mean)."""
    import pandas as pd
    by_seed = {}
    for fid in fids:
        df = pd.read_csv(Path(runs_root) / fid / "inner_val_predictions.csv")
        seed = int(fid.rsplit("__s", 1)[1].split("__")[0])
        by_seed.setdefault(seed, []).append(df)
    rows = []
    for seed, dfs in by_seed.items():
        d = pd.concat(dfs)
        p, y = d["p_error"].values, d["label"].values
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        from sklearn.metrics import roc_auc_score
        rows.append({"pooled_auroc": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2)),
                     "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                     "mean_p_label0": float(p[y == 0].mean()), "mean_p_label1": float(p[y == 1].mean())})
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def summarize(plan_path, runs_root, out):
    plan = json.loads(Path(plan_path).read_text())
    stage = plan["stage"]
    table = []
    for c in plan["candidates"]:
        sts = [fit_status(runs_root, f) for f in c["fit_ids"]]
        sel = selection_score([{"seed": s["seed"], "per_participant_auroc": s["per_participant_auroc"]} for s in sts])
        epochs = [s["best_epoch_one_based"] for s in sts]
        row = {k: c[k] for k in c if k != "fit_ids"}
        row.update(score=sel["score"], per_participant_seed_mean=sel["per_participant_seed_mean"],
                   n_fits=len(sts), params_trainable=sts[0]["params_trainable"],
                   best_epochs_one_based=epochs, median_best_epoch=statistics.median(epochs),
                   fit_ids=c["fit_ids"], **{"cv_" + k: v for k, v in prob_quality(runs_root, c["fit_ids"]).items()})
        table.append(row)
    result = {"stage": stage, "plan_sha": short_hash(plan, 16), "rule": "participant-macro of seed-mean AUROC; "
              "ties<=1e-6 -> fewer trainable params -> lexicographic candidate id", "candidates": table}
    if stage in "ABC":
        result["winner"] = choose(table)
    elif stage in "DE":
        result["winners"] = {arch: choose([t for t in table if t["recipe"]["arch"] == arch]) for arch in ("cbramod", "eegnet")}
    elif stage == "F":
        result["note"] = "confirmation only; no reselection"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for t in sorted(table, key=lambda t: -t["score"]):
        print(f"{t['candidate_id']:45s} score={t['score']:.4f} pooled={t['cv_pooled_auroc']:.4f} "
              f"brier={t['cv_brier']:.4f} params={t['params_trainable']} medEp={t['median_best_epoch']}")
    w = result.get("winner") or result.get("winners")
    print("WINNER", json.dumps({k: v["candidate_id"] for k, v in w.items()} if "winners" in result else w["candidate_id"]
                               if w else None))


def choose(table):
    best = max(t["score"] for t in table)
    tied = [t for t in table if best - t["score"] <= TOL]
    tied.sort(key=lambda t: (t["params_trainable"], t["candidate_id"]))
    w = tied[0]
    return {"candidate_id": w["candidate_id"], "recipe": w["recipe"], "recipe_id": w["recipe_id"], "score": w["score"],
            "tied_within_tolerance": [t["candidate_id"] for t in tied]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stage", required=True)
    p.add_argument("--decisions-dir", required=True)
    p.add_argument("--prep-root", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--plan", required=True)
    s.add_argument("--runs-root", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.stage, a.decisions_dir, a.prep_root, a.out)
    else:
        summarize(a.plan, a.runs_root, a.out)
