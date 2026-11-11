#!/usr/bin/env python
"""Final-model margins, calibration, one-shot test evaluation and probability export (handoff Section 8).

  margins   --plan-g PLAN_G --runs-root RUNS --prep-root PREP --split {validation,test} --out DIR [--calibration CAL]
            (test requires the committed calibration file; its SHA-256 is recorded in every test margin file)
  calibrate --plan-g PLAN_G --margins DIR_VAL --freeze FREEZE --out CAL.json
  evaluate  --plan-g PLAN_G --margins DIR_TEST --calibration CAL.json --out DIR
  export    --plan-g PLAN_G --margins-val DIR_VAL --margins-test DIR_TEST --calibration CAL.json --runs-root RUNS --out DIR

Margin z = logit_error - logit_nonerror. Controlled comparison: identity (raw) and temperature per final model.
Deployment calibration: selected candidates only, method chosen per architecture by cross-fitting on s11/s13
(study.json), refit per seed with equal participant weighting; references use temperature.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from errp.calibration import fit_temperature  # noqa: E402
from errp.dimp import calibration as cal  # noqa: E402
from errp.dimp.spec import BAND_DIR, load_study  # noqa: E402
from errp.metrics import binary_metrics  # noqa: E402

SPLIT_PARTICIPANTS = {"validation": ["s11", "s13"], "test": ["s02", "s08", "s09"]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def final_models(plan_g):
    plan = json.loads(Path(plan_g).read_text())
    out = []
    for c in plan["candidates"]:
        for fid in c["fit_ids"]:
            seed = int(fid.split("__s")[1].split("__")[0])
            out.append({"fit_id": fid, "arch": c["arch"], "role": c["role"], "candidate_id": c["candidate_id"],
                        "recipe_id": c["recipe_id"], "recipe": c["recipe"], "seed": seed})
    return out


def cmd_margins(a):
    import torch
    from errp.dimp import models
    from errp.dimp.data import Prepared
    from errp.data import InputScaler
    if a.split == "test":
        if not a.calibration or not Path(a.calibration).exists():
            raise SystemExit("test margins require the committed calibration file")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in final_models(a.plan_g):
        dst = out / f"{m['fit_id']}.csv"
        if dst.exists():
            continue
        r = m["recipe"]
        prep = Prepared(Path(a.prep_root) / BAND_DIR[r["band"]])
        d = prep.view(r["input_view"], SPLIT_PARTICIPANTS[a.split])
        ck = Path(a.runs_root) / m["fit_id"] / "checkpoints" / "final.ckpt"
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["spec"]["fit_id"] != m["fit_id"]:
            raise ValueError("checkpoint identity mismatch")
        net = models.build(r, tuple(d.X.shape[1:]), None)
        net.load_state_dict(st["model_state"])
        net.to(dev).eval()
        sc = InputScaler(st["input_scaler"]["kind"], st["input_scaler"]["params"])
        X = sc.transform(d.X)
        zs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                lg = net(torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(dev)).double().cpu().numpy()
                zs.append(lg)
        lg = np.concatenate(zs)
        df = pd.DataFrame({"subject_id": d.subject_ids, "trial_id": d.trial_ids, "label": d.y,
                           "logit_nonerror": lg[:, 0], "logit_error": lg[:, 1], "margin": lg[:, 1] - lg[:, 0]})
        df.attrs = {}
        df.to_csv(dst, index=False)
        meta = {"fit_id": m["fit_id"], "split": a.split, "checkpoint": "final.ckpt", "checkpoint_sha256": sha(ck),
                "preprocessing_id": prep.prep_id, "scaler": st["input_scaler"], "device": dev,
                "calibration_sha256_at_test_access": sha(a.calibration) if a.split == "test" else None}
        (out / f"{m['fit_id']}.json").write_text(json.dumps(meta, indent=1) + "\n")
        print("margins", a.split, m["fit_id"], len(df), flush=True)


def load_margins(d, fid):
    return pd.read_csv(Path(d) / f"{fid}.csv", dtype={"subject_id": str, "trial_id": str})


def cmd_calibrate(a):
    study = load_study()
    freeze = json.loads(Path(a.freeze).read_text())
    models_ = final_models(a.plan_g)
    out = {"schema": "csci642-dimp-calibration/v1", "freeze_sha256": sha(a.freeze), "fit_split": "validation (s11, s13)",
           "note": "validation predictions are calibration-development outputs, not unbiased held-out feedback quality",
           "temperature_per_model": {}, "deployment": {}}
    for m in models_:
        df = load_margins(a.margins, m["fit_id"])
        t = fit_temperature(df["margin"].values, df["label"].values)
        t["calibration_id"] = f"{m['fit_id']}:temperature"
        out["temperature_per_model"][m["fit_id"]] = t
    for arch in ("cbramod", "eegnet"):
        sel = [m for m in models_ if m["arch"] == arch and m["role"] == "selected"]
        per_seed = {}
        for m in sel:
            df = load_margins(a.margins, m["fit_id"])
            per_seed[m["seed"]] = {"z": df["margin"].values, "y": df["label"].values, "subjects": df["subject_id"].values}
        choice = cal.crossfit_choice(per_seed)
        fitted, fallback_log = {}, []
        for method in choice["ranking"]:
            ok = True
            fitted = {}
            for m in sel:
                d = per_seed[m["seed"]]
                c = cal.fit(method, d["z"], d["y"], d["subjects"])
                if not c["success"]:
                    ok = False
                    fallback_log.append({"method": method, "seed": m["seed"], "reason": "fit failed"})
                    break
                c["calibration_id"] = f"{m['fit_id']}:deploy_{method}"
                fitted[m["fit_id"]] = c
            if ok:
                break
        else:
            raise RuntimeError(f"every calibration method failed for {arch}: {fallback_log}")
        out["deployment"][arch] = {"selection": {k: v for k, v in choice.items() if k != "detail"},
                                   "crossfit_detail": choice["detail"], "method": method, "fallbacks": fallback_log,
                                   "per_model": fitted}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({arch: {"method": v["method"], "heldout_nll": v["selection"]["mean_heldout_participant_nll"]}
                      for arch, v in out["deployment"].items()}, indent=1))


def probs(m, z, calib):
    t = calib["temperature_per_model"][m["fit_id"]]
    p_raw = cal.apply({"method": "identity", "params": {}}, z)
    p_t = cal.apply({"method": "temperature", "params": {"T": t["temperature"]}}, z)
    if m["role"] == "selected":
        dep = calib["deployment"][m["arch"]]["per_model"][m["fit_id"]]
        p_dep, dep_id = cal.apply(dep, z), dep["calibration_id"]
    else:
        p_dep, dep_id = p_t, t["calibration_id"]
    return p_raw, p_t, p_dep, dep_id, t["calibration_id"]


def metric_block(y, p, subjects):
    pooled = binary_metrics(y, p)
    per = {s: binary_metrics(y[subjects == s], p[subjects == s]) for s in sorted(set(subjects))}
    macro_auc = statistics.mean(v["auroc"] for v in per.values())
    extra = {"mean_p_label0": float(p[y == 0].mean()), "mean_p_label1": float(p[y == 1].mean()),
             "label_mean_gap": float(p[y == 1].mean() - p[y == 0].mean()),
             "frac_label0_below_0.5": float((p[y == 0] < 0.5).mean()), "frac_label1_below_0.5": float((p[y == 1] < 0.5).mean())}
    return {"pooled": {**pooled, **extra}, "participant_macro_auroc": macro_auc,
            "participant_macro_balanced_accuracy": statistics.mean(v["balanced_accuracy"] for v in per.values()),
            "per_participant": per}


def cmd_evaluate(a):
    calib = json.loads(Path(a.calibration).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, per_rows = [], []
    for m in final_models(a.plan_g):
        meta = json.loads((Path(a.margins) / f"{m['fit_id']}.json").read_text())
        if meta["calibration_sha256_at_test_access"] != sha(a.calibration):
            raise ValueError("test margins were not produced under this calibration file")
        df = load_margins(a.margins, m["fit_id"])
        y, z, s = df["label"].values, df["margin"].values, df["subject_id"].values
        p_raw, p_t, p_dep, dep_id, t_id = probs(m, z, calib)
        for kind, p in (("raw", p_raw), ("temperature", p_t), ("deployment", p_dep)):
            mb = metric_block(y, p, s)
            rows.append({"fit_id": m["fit_id"], "arch": m["arch"], "role": m["role"], "recipe_id": m["recipe_id"],
                         "seed": m["seed"], "probability": kind, "participant_macro_auroc": mb["participant_macro_auroc"],
                         "participant_macro_balanced_accuracy": mb["participant_macro_balanced_accuracy"],
                         **{k: v for k, v in mb["pooled"].items() if not isinstance(v, dict)},
                         **{f"cm_{k}": v for k, v in mb["pooled"]["confusion_matrix"].items()}})
            for sid, v in mb["per_participant"].items():
                per_rows.append({"fit_id": m["fit_id"], "arch": m["arch"], "role": m["role"], "seed": m["seed"],
                                 "probability": kind, "participant": sid,
                                 **{k: v[k] for k in ("n", "auroc", "balanced_accuracy", "sensitivity_tpr",
                                                      "specificity_tnr", "brier", "nll", "ece_10bin")}})
    R = pd.DataFrame(rows)
    P = pd.DataFrame(per_rows)
    R.to_csv(out / "test_metrics_per_model.csv", index=False)
    P.to_csv(out / "test_metrics_per_participant.csv", index=False)
    summ = (R.groupby(["arch", "role", "recipe_id", "probability"])
             [["auroc", "participant_macro_auroc", "balanced_accuracy", "brier", "nll", "ece_10bin", "label_mean_gap"]]
             .agg(["mean", "std", "count"]))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out / "test_summary_seed_mean_sd.csv", index=False)
    # paired selected - reference on the same participants and seeds (AUROC is calibration-invariant: use raw)
    paired = []
    for arch in ("cbramod", "eegnet"):
        Ps = P[(P.arch == arch) & (P.probability == "raw")]
        if set(Ps.role) != {"selected", "reference"}:
            continue
        piv = Ps.pivot_table(index=["seed", "participant"], columns="role", values="auroc")
        for (seed, part), r in piv.iterrows():
            paired.append({"arch": arch, "seed": seed, "participant": part,
                           "selected_minus_reference_auroc": r["selected"] - r["reference"]})
    pd.DataFrame(paired).to_csv(out / "test_paired_differences.csv", index=False)
    (out / "evaluation_meta.json").write_text(json.dumps({"calibration_sha256": sha(a.calibration),
                                                         "n_models": int(R.fit_id.nunique()),
                                                         "note": "single batch evaluation of all predeclared final models"}, indent=1) + "\n")
    print(summ.reset_index()[["arch", "role", "probability", "auroc_mean", "auroc_std", "participant_macro_auroc_mean",
                              "balanced_accuracy_mean", "brier_mean"]].to_string())


def cmd_export(a):
    calib = json.loads(Path(a.calibration).read_text())
    study = load_study()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"validation": [], "test": []}
    for m in final_models(a.plan_g):
        status = json.loads((Path(a.runs_root) / m["fit_id"] / "status.json").read_text())
        for split, d in (("validation", a.margins_val), ("test", a.margins_test)):
            df = load_margins(d, m["fit_id"])
            meta = json.loads((Path(d) / f"{m['fit_id']}.json").read_text())
            p_raw, p_t, p_dep, dep_id, t_id = probs(m, df["margin"].values, calib)
            tables[split].append(pd.DataFrame({
                "subject_id": df["subject_id"], "trial_id": df["trial_id"], "label": df["label"], "split": split,
                "model": m["arch"], "recipe_id": m["recipe_id"], "role": m["role"], "decoder_seed": m["seed"],
                "logit_nonerror": df["logit_nonerror"], "logit_error": df["logit_error"], "margin": df["margin"],
                "p_error_raw": p_raw, "p_error_temperature": p_t, "p_error": p_dep,
                "checkpoint_id": f"{m['fit_id']}:final", "preprocessing_id": meta["preprocessing_id"],
                "scaler_id": f"{m['fit_id']}:{meta['scaler']['kind']}", "calibration_id": dep_id,
                "temperature_calibration_id": t_id}))
    manifest = {"schema": "csci642-dimp-predictions/v1", "files": {}, "calibration_sha256": sha(a.calibration),
                "p_error": "selected candidates: selected deployment calibration; references: temperature",
                "validation_note": "validation rows are calibration-development outputs (s11/s13 were used to fit calibrators)"}
    for split, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        f = out / f"dimp_{split}_predictions.csv"
        df.to_csv(f, index=False, float_format="%.10g")
        manifest["files"][f.name] = {"sha256": sha(f), "bytes": f.stat().st_size, "rows": len(df)}
    # seed-42 feedback interface tables (selected candidates only, historical schema + extra columns)
    seed = study["seeds"]["feedback_export"]
    for split in ("validation", "test"):
        df = pd.read_csv(out / f"dimp_{split}_predictions.csv", dtype={"subject_id": str, "trial_id": str})
        fb = df[(df.role == "selected") & (df.decoder_seed == seed)].copy()
        cols = ["subject_id", "trial_id", "label", "split", "model", "decoder_seed", "p_error_raw", "p_error",
                "checkpoint_id", "preprocessing_id", "calibration_id", "recipe_id", "p_error_temperature", "margin"]
        f = out / f"vnext_{split}_predictions.csv"
        fb[cols].to_csv(f, index=False, float_format="%.10g")
        manifest["files"][f.name] = {"sha256": sha(f), "bytes": f.stat().st_size, "rows": len(fb)}
    (out / "predictions_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["files"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("margins")
    s.add_argument("--plan-g", required=True)
    s.add_argument("--runs-root", required=True)
    s.add_argument("--prep-root", required=True)
    s.add_argument("--split", choices=["validation", "test"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--calibration")
    c = sub.add_parser("calibrate")
    c.add_argument("--plan-g", required=True)
    c.add_argument("--margins", required=True)
    c.add_argument("--freeze", required=True)
    c.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--plan-g", required=True)
    e.add_argument("--margins", required=True)
    e.add_argument("--calibration", required=True)
    e.add_argument("--out", required=True)
    x = sub.add_parser("export")
    x.add_argument("--plan-g", required=True)
    x.add_argument("--margins-val", required=True)
    x.add_argument("--margins-test", required=True)
    x.add_argument("--calibration", required=True)
    x.add_argument("--runs-root", required=True)
    x.add_argument("--out", required=True)
    a = ap.parse_args()
    {"margins": cmd_margins, "calibrate": cmd_calibrate, "evaluate": cmd_evaluate, "export": cmd_export}[a.cmd](a)
