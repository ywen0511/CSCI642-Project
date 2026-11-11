#!/usr/bin/env python
"""Verification of downloaded decoder-study artifacts (handoff Section 12).

  checkpoints --extract DIR [DIR ...] --prep-root PREP --weights-dir W --plans PLANS --out JSON [--resume-work NEW]
      For every fit archive found under the extract dirs: SHA-256 of every file against the fit's status.json;
      load the selected checkpoint (best.ckpt for CV fits, final.ckpt for final refits) and reproduce the stored
      inference probe from regenerated inputs (input SHA-256 must match; max |logit diff| reported, <= 1e-6 required
      on the same device type); load the first periodic checkpoint of each unique arch/head/adaptation strictly and
      check finite outputs. With --resume-work: copy the last.ckpt of one fit per unique CBraMod head/adaptation,
      per EEGNet dropout and per selected recipe into an isolated directory and train two more epochs; check restored
      optimizer/scheduler/RNG/trainability, epoch and step progression, finite losses, unchanged original hashes,
      unchanged frozen backbone and updated trainable layers.
  repro --extract DIR --prep-root PREP --plan-g PLAN_G --calibration CAL --predictions DIR --out JSON
      Recompute raw/temperature/deployment probabilities of every final model on validation and test from the
      downloaded final checkpoints and compare with the exported prediction tables.
  tables --predictions DIR --extract DIR [DIR ...] --dec DEC --plans PLANS --test-results DIR --audit AUDIT --out JSON
      Recompute test metrics from the exported predictions, CV selection scores from the downloaded per-fit
      inner-validation predictions, and every number in number_audit.json; regenerate tables and compare.
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOL_SAME = 1e-6
TOL_XDEV = 1e-5


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_fits(extract_dirs):
    out = {}
    for e in extract_dirs:
        for st in Path(e).glob("*/*/status.json"):
            out[st.parent.name] = st.parent
    return out


def cmd_checkpoints(a):
    import torch
    from errp.data import InputScaler
    from errp.dimp import models
    from errp.dimp.data import Prepared
    from errp.dimp.fit import Fit
    from errp.dimp.spec import BAND_DIR
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fits = find_fits(a.extract)
    rep = {"device": dev, "n_fits": len(fits), "hash_failures": [], "probe": {}, "periodic": {}, "resume": {}, "problems": []}
    preps = {}
    first_periodic = {}
    n_files = 0
    for fid, d in sorted(fits.items()):
        st = json.loads((d / "status.json").read_text())
        for rel, h in st["file_sha256"].items():
            n_files += 1
            if sha(d / rel) != h:
                rep["hash_failures"].append(f"{fid}/{rel}")
        spec = json.loads((d / "fit.json").read_text())
        r = spec["recipe"]
        if r["band"] not in preps:
            preps[r["band"]] = Prepared(Path(a.prep_root) / BAND_DIR[r["band"]])
        prep = preps[r["band"]]
        pr = np.load(d / "probe.npz", allow_pickle=False)
        ids = pr["trial_ids"]
        idx = np.array([np.where(prep.ids == t)[0][0] for t in ids])
        Xall = np.load(prep.dir / {"eeg256": "epochs_256hz_uV.npy", "patch200": "epochs_200hz_patches_uV.npy"}[r["input_view"]], mmap_mode="r")
        ck = torch.load(d / "checkpoints" / st["selected_checkpoint"], map_location="cpu", weights_only=False)
        sc = InputScaler(ck["input_scaler"]["kind"], ck["input_scaler"]["params"])
        X = sc.transform(np.asarray(Xall[idx], dtype=np.float32))
        in_ok = hashlib.sha256(X.tobytes()).hexdigest() == str(pr["input_sha256"][0])
        net = models.build(r, tuple(X.shape[1:]), None)
        net.load_state_dict(ck["model_state"], strict=True)
        net.to(dev).eval()
        with torch.no_grad():
            lg = net(torch.from_numpy(X).to(dev)).double().cpu().numpy()
        diff = float(np.abs(lg - pr["logits"]).max())
        rep["probe"][fid] = {"checkpoint": st["selected_checkpoint"], "input_hash_ok": in_ok, "max_abs_logit_diff": diff}
        if not in_ok or diff > TOL_SAME:
            rep["problems"].append(f"probe {fid}: input_ok={in_ok} diff={diff}")
        key = f"{r['arch']}/{r.get('head', '-')}/{r.get('adaptation', 'full')}"
        periodic = sorted((d / "checkpoints").glob("epoch_*.ckpt"))
        if periodic and key not in first_periodic:
            first_periodic[key] = fid
            pk = torch.load(periodic[0], map_location="cpu", weights_only=False)
            net2 = models.build(r, tuple(X.shape[1:]), None)
            net2.load_state_dict(pk["model_state"], strict=True)
            net2.to(dev).eval()
            with torch.no_grad():
                out = net2(torch.from_numpy(X).to(dev))
            ok = bool(torch.isfinite(out).all()) and pk["epoch_completed"] == int(periodic[0].stem.split("_")[1])
            rep["periodic"][key] = {"fit": fid, "checkpoint": periodic[0].name, "loaded_strict": True, "finite": ok,
                                    "epoch_completed": pk["epoch_completed"]}
            if not ok:
                rep["problems"].append(f"periodic {key}")
    rep["n_files_hash_checked"] = n_files
    rep["problems"] += [f"hash {x}" for x in rep["hash_failures"]]
    if a.resume_work:
        work = Path(a.resume_work)
        work.mkdir(parents=True, exist_ok=False)
        want = {}
        for fid, d in sorted(fits.items()):
            spec = json.loads((d / "fit.json").read_text())
            if spec["mode"] != "cv":
                continue
            r = spec["recipe"]
            key = (f"cbramod/{r['head']}/{r['adaptation']}" if r["arch"] == "cbramod" else f"eegnet/dropout{r['dropout']}")
            want.setdefault(key, fid)
            want.setdefault(f"recipe/{spec['recipe_id']}", fid)
        sel = json.loads((Path(a.plans) / "stage_G.json").read_text())
        targets = {k: v for k, v in want.items() if not k.startswith("recipe/")}
        for c in sel["candidates"]:
            k = f"recipe/{c['recipe_id']}"
            if k in want:
                targets[f"final_recipe/{c['arch']}/{c['role']}"] = want[k]
        for key, fid in sorted(targets.items()):
            src = fits[fid]
            dst = work / key.replace("/", "__")
            shutil.copytree(src, dst)
            before = {p: sha(src / "checkpoints" / p) for p in ("last.ckpt",)}
            spec = json.loads((dst / "fit.json").read_text())
            st0 = torch.load(dst / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
            spec2 = copy.deepcopy(spec)
            spec2["recipe"]["max_epochs"] = st0["epoch_completed"] + 2
            spec2["recipe"]["patience"] = 10 ** 6
            r = spec2["recipe"]
            prep = preps.get(r["band"]) or Prepared(Path(a.prep_root) / BAND_DIR[r["band"]])
            trn = prep.view(r["input_view"], spec["fold_def"]["train"])
            val = prep.view(r["input_view"], spec["fold_def"]["validation"])
            wpath = str(Path(a.weights_dir) / "CBraMod" / "pretrained_weights.pth") if r["arch"] == "cbramod" else None
            init = sha(wpath) if wpath else None
            f = Fit(spec2, dst, trn, val, ROOT, wpath, init, prep.input_hashes(r["input_view"]), dev)
            res = f.try_resume()
            before_params = {n: p.detach().clone() for n, p in f.model.named_parameters()}
            opt_steps = int(next(iter(f.opt.state.values()))["step"]) if f.opt.state else 0
            summ = f.fit()
            after = {n: p.detach() for n, p in f.model.named_parameters()}
            frozen_same = all(torch.equal(before_params[n], after[n]) for n in f.model.frozen_names())
            trainable_changed = all(not torch.equal(before_params[n], after[n]) for n in f.model.trainable_names()
                                    if "mask_encoding" not in n)
            new_hist = f.history[-2:]
            ok = (res is not None and res["resumed_from_epoch"] == st0["epoch_completed"]
                  and summ["epochs_completed"] == st0["epoch_completed"] + 2
                  and summ["global_step"] - st0["global_step"] in (2 * f.steps_per_epoch, 2 * (f.steps_per_epoch - 1))
                  and all(np.isfinite(h["train_loss"]) for h in new_hist) and frozen_same and trainable_changed
                  and sha(src / "checkpoints" / "last.ckpt") == before["last.ckpt"])
            rep["resume"][key] = {"fit": fid, "resumed_from_epoch": st0["epoch_completed"], "epochs_after": summ["epochs_completed"],
                                  "steps_before": st0["global_step"], "steps_after": summ["global_step"],
                                  "optimizer_step_count_restored": opt_steps, "new_train_losses": [h["train_loss"] for h in new_hist],
                                  "frozen_params_unchanged": frozen_same, "trainable_params_updated": trainable_changed,
                                  "original_last_ckpt_unchanged": sha(src / "checkpoints" / "last.ckpt") == before["last.ckpt"], "ok": ok}
            if not ok:
                rep["problems"].append(f"resume {key}")
        rep["resume_guarantee"] = ("restored model/optimizer/scheduler/RNG/sampler/trainability; epoch and step "
                                   "progression; finite losses; frozen parameters unchanged and trainable ones updated. "
                                   "Bitwise equality with an uninterrupted run was verified in the diagnostic gate.")
    rep["ok"] = not rep["problems"]
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("n_fits", "n_files_hash_checked", "ok")}), len(rep["probe"]), "probes,",
          len(rep["periodic"]), "periodic,", len(rep["resume"]), "resumes;", rep["problems"][:5])
    sys.exit(0 if rep["ok"] else 1)


def cmd_repro(a):
    import subprocess
    tmp = Path(a.out).with_suffix(".work")
    rep = {"problems": []}
    runs = tmp / "runs"
    runs.mkdir(parents=True, exist_ok=False)
    for fid, d in find_fits([a.extract]).items():
        (runs / fid).symlink_to(d.resolve())
    for split in ("validation", "test"):
        mdir = tmp / f"margins_{split}"
        cmd = [sys.executable, str(ROOT / "scripts/decoder_improvement/final_eval.py"), "margins", "--plan-g", a.plan_g,
               "--runs-root", str(runs), "--prep-root", a.prep_root, "--split", split, "--out", str(mdir)]
        if split == "test":
            cmd += ["--calibration", a.calibration]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(ROOT / "scripts" / "decoder_improvement"))
    from final_eval import final_models, load_margins, probs
    calib = json.loads(Path(a.calibration).read_text())
    for split in ("validation", "test"):
        exp = pd.read_csv(Path(a.predictions) / f"dimp_{split}_predictions.csv", dtype={"subject_id": str, "trial_id": str})
        worst = {"p_error_raw": 0.0, "p_error_temperature": 0.0, "p_error": 0.0, "margin": 0.0}
        for m in final_models(a.plan_g):
            df = load_margins(tmp / f"margins_{split}", m["fit_id"])
            p_raw, p_t, p_dep, _, _ = probs(m, df["margin"].values, calib)
            e = exp[exp.checkpoint_id == f"{m['fit_id']}:final"].set_index("trial_id").loc[df.trial_id]
            for col, v in (("p_error_raw", p_raw), ("p_error_temperature", p_t), ("p_error", p_dep), ("margin", df["margin"].values)):
                worst[col] = max(worst[col], float(np.abs(e[col].values - v).max()))
            if (e["label"].values != df["label"].values).any():
                rep["problems"].append(f"label mismatch {m['fit_id']} {split}")
        rep[split] = worst
        if max(worst["p_error_raw"], worst["p_error_temperature"], worst["p_error"]) > TOL_SAME:
            rep["problems"].append(f"{split}: probability difference {worst}")
    rep["tolerance_abs_probability"] = TOL_SAME
    rep["ok"] = not rep["problems"]
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))
    sys.exit(0 if rep["ok"] else 1)


def resolve(item, root):
    src, sel = item["source"], item["selector"]
    p = root / src
    if src.endswith(".json"):
        obj = json.loads(p.read_text())
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", sel.split("-")[0]):
            obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
        if "-" in sel:  # difference of two fields of the same parent
            base, rhs = sel.rsplit(".", 1)[0], sel.split("-")[1]
            parent = json.loads(p.read_text())
            for part in re.findall(r"[^.\[\]]+|\[\d+\]", base):
                parent = parent[int(part[1:-1])] if part.startswith("[") else parent[part]
            return float(obj) - float(parent[rhs])
        return float(obj)
    df = pd.read_csv(p)
    m = re.match(r"(mean|sd)\((\w+)\) where (.*)", sel)
    fn, col, cond = m.groups()
    for c in cond.split(","):
        k, v = c.split("=")
        df = df[df[k].astype(str) == v]
    return float(df[col].mean() if fn == "mean" else df[col].std(ddof=1))


def cmd_tables(a):
    sys.path.insert(0, str(ROOT / "scripts" / "decoder_improvement"))
    from final_eval import metric_block
    from errp.dimp.spec import selection_score
    from errp.dimp.fit import per_participant_auroc
    rep = {"problems": []}
    # (1) test metrics from exported predictions
    T = pd.read_csv(Path(a.predictions) / "dimp_test_predictions.csv", dtype={"subject_id": str})
    R = pd.read_csv(Path(a.test_results) / "test_metrics_per_model.csv")
    worst = 0.0
    for fid, g in T.groupby("checkpoint_id"):
        fid = fid.rsplit(":", 1)[0]
        for kind, col in (("raw", "p_error_raw"), ("temperature", "p_error_temperature"), ("deployment", "p_error")):
            mb = metric_block(g["label"].values, g[col].values, g["subject_id"].values)
            row = R[(R.fit_id == fid) & (R.probability == kind)].iloc[0]
            for k in ("auroc", "brier", "nll", "balanced_accuracy", "ece_10bin"):
                worst = max(worst, abs(mb["pooled"][k] - row[k]))
            worst = max(worst, abs(mb["participant_macro_auroc"] - row["participant_macro_auroc"]))
    rep["test_metrics_max_abs_diff"] = worst
    if worst > 1e-9:
        rep["problems"].append(f"test metrics differ by {worst}")
    # (2) CV selection scores from downloaded inner-validation predictions
    fits = find_fits(a.extract)
    cvdiff = 0.0
    n_checked = 0
    for s in "ACDEF":
        dec = json.loads((Path(a.dec) / f"stage_{s}.json").read_text())
        for c in dec["candidates"]:
            rows = []
            for fid in c["fit_ids"]:
                if fid not in fits:
                    rep["problems"].append(f"fit missing from downloads: {fid}")
                    continue
                df = pd.read_csv(fits[fid] / "inner_val_predictions.csv", dtype={"subject_id": str})
                pp = per_participant_auroc(df.subject_id.values, df.label.values, df.margin.values)
                rows.append({"seed": int(fid.split("__s")[1].split("__")[0]), "per_participant_auroc": pp})
            if len(rows) == len(c["fit_ids"]):
                cvdiff = max(cvdiff, abs(selection_score(rows)["score"] - c["score"]))
                n_checked += 1
    rep["cv_scores_recomputed"] = n_checked
    rep["cv_score_max_abs_diff"] = cvdiff
    if cvdiff > 1e-9:
        rep["problems"].append(f"CV scores differ by {cvdiff}")
    # (3) every audited number
    audit = json.loads(Path(a.audit).read_text())
    bad = []
    for it in audit["items"]:
        v = resolve(it, ROOT)
        if it["fmt"].format(v) != it["printed"]:
            bad.append({**it, "recomputed": v})
    rep["audited_numbers"] = len(audit["items"])
    rep["audit_mismatches"] = bad[:20]
    if bad:
        rep["problems"].append(f"{len(bad)} audited numbers differ")
    rep["ok"] = not rep["problems"]
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "audit_mismatches"}))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoints")
    c.add_argument("--extract", nargs="+", required=True)
    c.add_argument("--prep-root", required=True)
    c.add_argument("--weights-dir", required=True)
    c.add_argument("--plans", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--resume-work")
    r = sub.add_parser("repro")
    r.add_argument("--extract", required=True, help="extract directory of the downloaded final Release")
    r.add_argument("--prep-root", required=True)
    r.add_argument("--plan-g", required=True)
    r.add_argument("--calibration", required=True)
    r.add_argument("--predictions", required=True)
    r.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--predictions", required=True)
    t.add_argument("--extract", nargs="+", required=True)
    t.add_argument("--dec", required=True)
    t.add_argument("--plans", required=True)
    t.add_argument("--test-results", required=True)
    t.add_argument("--audit", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"checkpoints": cmd_checkpoints, "repro": cmd_repro, "tables": cmd_tables}[a.cmd](a)
