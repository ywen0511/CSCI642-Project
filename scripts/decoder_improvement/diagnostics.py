#!/usr/bin/env python
"""Pre-training diagnostic gate for the decoder-improvement study (handoff Section 6). Writes one JSON report.

Usage: python scripts/decoder_improvement/diagnostics.py --prep-root PREP --weights-dir W --work NEW_DIR --out report.json
Checks: (1) counts/labels/trial identities/split disjointness/identical windows across bands; (2) finite voltages,
per-participant distributions, tensor shapes, resampling consistency; (3) official weight loading (keys, shapes,
removed reconstruction head, parameter counts); (4) trainable names per adaptation mode, frozen parameters
byte-identical after an optimizer step, unfrozen ones finite-gradient and changed; (5) frozen modules in eval
mode during training; (6) small training-only subset overfitting (reference CBraMod, EEGNet; frozen disclosed);
(7) save/reload probe reproduction and interrupted-vs-uninterrupted resume equivalence; (8) selection inputs
contain no outer-validation/test participant.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from errp.dimp import models  # noqa: E402
from errp.dimp.data import Prepared, SubjectData  # noqa: E402
from errp.dimp.fit import Fit  # noqa: E402
from errp.dimp.spec import BAND_DIR, cbramod_recipe, eegnet_recipe, fit_spec, folds, load_study  # noqa: E402
from errp.utils import set_seed, sha256_file  # noqa: E402


def check(report, name, ok, **info):
    report["checks"][name] = {"ok": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:400], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-root", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    study = load_study()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"checks": {}, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}
    W = Path(a.weights_dir) / "CBraMod" / "pretrained_weights.pth"
    rep["initialization"] = {"sha256": sha256_file(W), "bytes": W.stat().st_size}

    # ---------------- (1) data identity
    preps = {b: Prepared(Path(a.prep_root) / d) for b, d in BAND_DIR.items()}
    ref = preps["1-20"]
    same = all((p.ids == ref.ids).all() and (p.y == ref.y).all() and (p.subjects == ref.subjects).all() for p in preps.values())
    counts = {"n": int(len(ref.y)), "label0": int((ref.y == 0).sum()), "label1": int((ref.y == 1).sum())}
    exp = study["inputs"]["dataset"]
    osplit = study["outer_split"]
    split_counts = {k: int(np.isin(ref.subjects, osplit[k]).sum()) for k in ("train", "validation", "test")}
    fd = folds(study)
    inner_ok = all(not set(f["train"]) & set(f["validation"]) for k, f in fd.items() if k != "outer")
    covered = sorted(p for k, f in fd.items() if k != "outer" for p in f["validation"])
    check(rep, "1_data_identity_and_splits", same and counts["n"] == exp["expected_trials"]
          and counts["label0"] == exp["class_counts"]["0"] and counts["label1"] == exp["class_counts"]["1"]
          and split_counts == osplit["trial_counts"] and inner_ok and covered == sorted(osplit["train"])
          and not set(osplit["train"]) & (set(osplit["validation"]) | set(osplit["test"])),
          counts=counts, split_counts=split_counts, prep_ids={b: p.prep_id for b, p in preps.items()},
          identical_trials_labels_order_across_bands=bool(same), inner_folds_disjoint=inner_ok,
          each_training_participant_held_out_once=covered == sorted(osplit["train"]))
    # historical prep_v1 equality for the 1-20 band (if available next to the prep root)
    hist = Path(a.prep_root).parents[1] / "prep_v1"
    if (hist / "epochs_256hz_uV.npy").exists():
        eq = {v: sha256_file(hist / f) == sha256_file(ref.dir / f) for v, f in
              (("eeg256", "epochs_256hz_uV.npy"), ("patch200", "epochs_200hz_patches_uV.npy"), ("ids", "epoch_trial_ids.npy"))}
        rep["band_1_20_vs_historical_prep_v1_bytes_equal"] = eq

    # ---------------- (2) voltages, shapes, resampling
    stats = {}
    ok2 = True
    for b, p in preps.items():
        X256 = np.load(p.dir / "epochs_256hz_uV.npy", mmap_mode="r")
        X200 = np.load(p.dir / "epochs_200hz_patches_uV.npy", mmap_mode="r")
        ok2 &= X256.shape == (len(p.y), 27, 512) and X200.shape == (len(p.y), 27, 2, 200)
        per = {}
        for s in sorted(set(p.subjects)):
            x = np.asarray(X256[p.subjects == s], dtype=np.float64)
            ok2 &= bool(np.isfinite(x).all())
            per[s] = {"mean_uV": float(x.mean()), "std_uV": float(x.std()),
                      "p99_abs_uV": float(np.percentile(np.abs(x), 99))}
        idx = np.arange(0, len(p.y), 97)
        back = resample_poly(np.asarray(X256[idx], dtype=np.float64), 25, 32, axis=-1)
        maxdiff = float(np.abs(back - np.asarray(X200[idx]).reshape(len(idx), 27, 400)).max())
        ok2 &= maxdiff < 1e-3
        stats[b] = {"shapes": [list(X256.shape), list(X200.shape)], "per_participant": per,
                    "patch200_vs_resampled_eeg256_max_abs_uV": maxdiff}
    check(rep, "2_voltages_shapes_resampling", ok2, bands=stats)

    # ---------------- (3) weights
    recs = {cid: cbramod_recipe(study, h, ad, lr, "1-20") for cid, h, ad, lr in (
        ("flat_frozen", "historical_flatten_mlp", "frozen", None), ("flat_last2", "historical_flatten_mlp", "last2", 3e-5),
        ("flat_full", "historical_flatten_mlp", "full", 3e-5), ("timepool_frozen", "temporal_mean_linear", "frozen", None),
        ("timepool_last2", "temporal_mean_linear", "last2", 3e-5), ("timepool_full", "temporal_mean_linear", "full", 3e-5))}
    set_seed(0)
    m = models.build(recs["flat_full"], (27, 2, 200), str(W))
    lr_ = m.load_report
    head_flat = sum(p.numel() for p in m.classifier.parameters())
    m2 = models.build(recs["timepool_full"], (27, 2, 200), str(W))
    head_tp = sum(p.numel() for p in m2.classifier.parameters())
    bb = sum(p.numel() for p in m.backbone.parameters())
    check(rep, "3_official_weights", not lr_["missing"] and not lr_["unexpected"] and not lr_["shape_mismatch"]
          and head_flat == study["heads"]["historical_flatten_mlp"]["parameter_count"]
          and head_tp == study["heads"]["temporal_mean_linear"]["parameter_count"]
          and isinstance(m.backbone.proj_out, torch.nn.Identity),
          load_report={k: (v if not isinstance(v, list) or len(v) < 20 else len(v)) for k, v in lr_.items()},
          head_params={"historical_flatten_mlp": head_flat, "temporal_mean_linear": head_tp},
          backbone_params_after_head_removal=bb, total_flat=bb + head_flat, total_timepool=bb + head_tp)

    # ---------------- (4,5) trainability and modes
    tr = preps["1-20"].view("patch200", ["s03"])
    xb = torch.from_numpy(tr.X[:16] / 100.0).float().to(dev)
    yb = torch.from_numpy(tr.y[:16]).to(dev)
    trainable_report, ok4, ok5 = {}, True, True
    for cid, r in recs.items():
        set_seed(1)
        mm = models.build(r, (27, 2, 200), str(W)).to(dev)
        names = mm.trainable_names()
        before = {n: p.detach().clone() for n, p in mm.named_parameters()}
        opt = torch.optim.AdamW(mm.param_groups(5e-4, r["backbone_lr"]), weight_decay=0.05)
        mm.train()
        modes = mm.module_modes()
        loss = F.cross_entropy(mm(xb), yb)
        loss.backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in mm.named_parameters() if p.requires_grad)
        opt.step()
        changed = {n for n, p in mm.named_parameters() if not torch.equal(p.detach(), before[n])}
        frozen_unchanged = all(n not in changed for n in mm.frozen_names())
        trainable_changed = all(n in changed for n in names if "mask_encoding" not in n)
        bb_tr = [n for n in names if n.startswith("backbone.")]
        expect_bb = {"frozen": lambda n: False, "full": lambda n: True,
                     "last2": lambda n: n.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11."))}[r["adaptation"]]
        mask_ok = all(expect_bb(n) for n in bb_tr) and (r["adaptation"] != "last2" or len(bb_tr) > 0)
        frozen_mods_eval = True
        if r["adaptation"] in ("frozen", "last2"):
            for n, md in mm.named_modules():
                if n.startswith("backbone") and not n.startswith(("backbone.encoder.layers.10", "backbone.encoder.layers.11")) \
                        and n not in ("backbone", "backbone.encoder", "backbone.encoder.layers") and md.training:
                    frozen_mods_eval = False
                if r["adaptation"] == "frozen" and n.startswith("backbone") and md.training:
                    frozen_mods_eval = False
        ok4 &= grads_ok and frozen_unchanged and trainable_changed and mask_ok
        ok5 &= frozen_mods_eval and mm.classifier.training
        trainable_report[cid] = {"n_trainable_params": models.count(mm, True), "n_trainable_tensors": len(names),
                                 "backbone_trainable_tensor_names": bb_tr, "head_tensor_names": [n for n in names if n.startswith("classifier")],
                                 "frozen_tensors_byte_identical_after_step": frozen_unchanged,
                                 "trainable_tensors_changed": trainable_changed, "finite_grads": bool(grads_ok),
                                 "frozen_modules_in_eval_mode": frozen_mods_eval,
                                 "modules_in_train_mode": sorted(n for n, v in modes.items() if v)[:60]}
    check(rep, "4_trainability_masks", ok4, modes={k: {kk: v[kk] for kk in ("n_trainable_params", "frozen_tensors_byte_identical_after_step",
                                                                            "trainable_tensors_changed", "finite_grads")} for k, v in trainable_report.items()})
    check(rep, "5_frozen_module_modes", ok5)
    rep["trainable_names"] = trainable_report

    # ---------------- (6) overfit a training-only subset
    sub = preps["1-20"].view("patch200", ["s04", "s05"])
    sel = np.concatenate([np.where(sub.y == 0)[0][:32], np.where(sub.y == 1)[0][:32]])
    of = {}
    for cid, r in (("cb_reference", cbramod_recipe(study, "historical_flatten_mlp", "full", 1e-4, "1-20")),
                   ("cb_flat_frozen", recs["flat_frozen"])):
        set_seed(2)
        mm = models.build(r, (27, 2, 200), str(W)).to(dev)
        x = torch.from_numpy(sub.X[sel] / 100.0).float().to(dev)
        y = torch.from_numpy(sub.y[sel]).to(dev)
        opt = torch.optim.AdamW(mm.param_groups(5e-4, r["backbone_lr"]), weight_decay=0.0)
        for _ in range(150):
            mm.train()
            loss = F.cross_entropy(mm(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        mm.eval()
        with torch.no_grad():
            lg = mm(x)
        of[cid] = {"final_train_loss": float(F.cross_entropy(lg, y)),
                   "train_auroc": float(roc_auc_score(y.cpu(), (lg[:, 1] - lg[:, 0]).cpu())),
                   "train_acc": float((lg.argmax(1) == y).float().mean())}
    sub256 = preps["1-20"].view("eeg256", ["s04", "s05"])
    set_seed(2)
    er = eegnet_recipe(study, 1e-3, 0.5, "1-20")
    mm = models.build(er, (27, 512)).to(dev)
    sd = float(np.std(sub256.X[sel]))
    x = torch.from_numpy(sub256.X[sel] / sd).float().to(dev)
    y = torch.from_numpy(sub256.y[sel]).to(dev)
    opt = torch.optim.Adam(mm.parameters(), lr=1e-2)
    for _ in range(400):
        mm.train()
        loss = F.cross_entropy(mm(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mm.apply_constraints()
    mm.eval()
    with torch.no_grad():
        lg = mm(x)
    of["eegnet"] = {"final_train_loss": float(F.cross_entropy(lg, y)), "train_acc": float((lg.argmax(1) == y).float().mean()),
                    "train_auroc": float(roc_auc_score(y.cpu(), (lg[:, 1] - lg[:, 0]).cpu())), "lr_used_for_check": 1e-2}
    check(rep, "6_overfit_training_subset", of["cb_reference"]["train_acc"] >= 0.99 and of["eegnet"]["train_acc"] >= 0.95,
          subset="64 balanced trials of s04/s05 (outer training participants)", results=of,
          note="frozen-backbone result disclosed; not required to separate the subset")

    # ---------------- (7) save/reload probe + resume equivalence (small real fits)
    res7 = {}
    ok7 = True
    for cid, r in (("cb_timepool_last2", recs["timepool_last2"]), ("eegnet_ref", er)):
        r = copy.deepcopy(r)
        r["max_epochs"], r["patience"], r["periodic_every_epochs"] = 3, 99, 1
        if r["arch"] == "cbramod":
            r["scheduler_horizon_epochs"] = 3
        view = r["input_view"]
        p = preps["1-20"]
        trn = p.view(view, ["s04"])
        val = p.view(view, ["s03", "s06"])
        spec = fit_spec(study, r, "cv0", 42, p.prep_id)
        hashes = p.input_hashes(view)
        init = rep["initialization"]["sha256"] if r["arch"] == "cbramod" else None
        wpath = str(W) if r["arch"] == "cbramod" else None
        f1 = Fit(spec, work / f"{cid}_uninterrupted", trn, val, ROOT, wpath, init, hashes, dev)
        f1.fit()
        f2 = Fit(spec, work / f"{cid}_interrupted", trn, val, ROOT, wpath, init, hashes, dev)
        f2.fit(stop_after_epochs=1)
        f3 = Fit(spec, work / f"{cid}_interrupted", trn, val, ROOT, wpath, init, hashes, dev)
        resumed = f3.try_resume()
        f3.fit()
        a_sd, b_sd = f1.model.state_dict(), f3.model.state_dict()
        maxdiff = max(float((a_sd[k].double() - b_sd[k].double()).abs().max()) for k in a_sd)
        # reload last.ckpt into a fresh model and compare with the in-memory model after the final epoch
        st = torch.load(f1.ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)
        mr = models.build(r, tuple(trn.X.shape[1:]), None)
        mr.load_state_dict(st["model_state"])
        mr.to(dev).eval()
        f1.model.eval()
        xs = torch.from_numpy(f1.scaler.transform(val.X[:16])).to(dev)
        with torch.no_grad():
            d_probe = float((mr(xs) - f1.model(xs)).abs().max())
        ck = sorted(x.name for x in f1.ckpt_dir.iterdir())
        keys = sorted(st.keys())
        need = {"model_state", "optimizer_state", "scheduler_state", "rng_states", "sampler_generator_state",
                "trainable_names", "early_stopping_bad_epochs", "epoch_completed", "class_weights", "input_scaler"}
        ok = maxdiff == 0.0 and d_probe == 0.0 and need <= set(keys) and resumed and resumed["resumed_from_epoch"] == 1
        ok7 &= ok
        res7[cid] = {"resume_vs_uninterrupted_max_abs_weight_diff": maxdiff, "reload_probe_max_abs_logit_diff": d_probe,
                     "resumed": resumed, "checkpoint_files": ck, "checkpoint_keys": keys,
                     "frozen_params_unchanged_vs_init": None}
        if r["arch"] == "cbramod":
            init_m = models.build(r, tuple(trn.X.shape[1:]), wpath)
            fsd = init_m.state_dict()
            res7[cid]["frozen_params_unchanged_vs_init"] = all(
                torch.equal(fsd[n].cpu(), f3.model.state_dict()[n].cpu()) for n in f3.model.frozen_names())
            ok7 &= res7[cid]["frozen_params_unchanged_vs_init"]
    check(rep, "7_reload_and_resume", ok7, results=res7)

    # ---------------- (8) no outer validation/test participants in selection inputs
    banned = set(osplit["validation"]) | set(osplit["test"])
    used = {p for k, f in fd.items() if k != "outer" for p in f["train"] + f["validation"]}
    check(rep, "8_no_test_or_outer_validation_in_selection", not used & banned, cv_participants=sorted(used))
    rep["all_ok"] = all(c["ok"] for c in rep["checks"].values())
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    shutil.rmtree(work / "cb_timepool_last2_interrupted", ignore_errors=True)
    print("ALL_OK", rep["all_ok"])


if __name__ == "__main__":
    main()
