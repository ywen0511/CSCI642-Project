#!/usr/bin/env python
"""Execute ONE planned fit of the decoder-improvement study (idempotent, resumable, ledgered).

Usage: python scripts/decoder_improvement/run_fit.py --plan PLAN.json --fit-id FIT_ID \
          --runs-root RUNS --prep-root PREP_ROOT [--weights-dir W] [--ledger LEDGER.jsonl] [--device cuda]

A fit directory is reused only if status.json says completed AND every recorded checkpoint/prediction hash
still matches. An existing incomplete directory is resumed from last.ckpt when its identity, training-source and
input hashes match (recorded as a resumed attempt); otherwise the run fails loudly.
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from errp.dimp import models  # noqa: E402
from errp.dimp.data import Prepared  # noqa: E402
from errp.dimp.fit import PROBE_N, Fit, per_participant_auroc  # noqa: E402
from errp.dimp.spec import BAND_DIR, participant_macro_auroc  # noqa: E402
from errp.utils import sha256_file, write_json  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def ledger(path, rec):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"utc": now(), "host": socket.gethostname(), "lsf_job": os.environ.get("LSB_JOBID"),
                                **rec}) + "\n")


def verify_completed(run_dir):
    st = json.loads((run_dir / "status.json").read_text())
    if st.get("state") != "completed":
        return False
    for rel, h in st["file_sha256"].items():
        p = run_dir / rel
        if not p.is_file() or sha256_file(p) != h:
            raise ValueError(f"completed fit has changed/missing file: {p}")
    return True


def load_eval_model(ckpt_path, recipe, input_shape, device):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = models.build(recipe, input_shape, None)
    m.load_state_dict(st["model_state"])
    return m.to(device).eval(), st


@torch.no_grad()
def margins(model, X, device):
    out = []
    for i in range(0, len(X), 256):
        lg = model(torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(device))
        out.append((lg[:, 1] - lg[:, 0]).double().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--fit-id", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prep-root", required=True)
    ap.add_argument("--weights-dir", default=os.environ.get("ERRP_WEIGHTS_DIR"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stop-after-epochs", type=int, default=None, help="testing only: interrupt to test resume")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    spec = next(s for s in plan["fits"] if s["fit_id"] == a.fit_id)
    r = spec["recipe"]
    run_dir = Path(a.runs_root) / spec["fit_id"]
    if (run_dir / "status.json").exists() and verify_completed(run_dir):
        ledger(a.ledger, {"fit_id": spec["fit_id"], "event": "reuse_verified_completed"})
        print("REUSED", spec["fit_id"])
        return
    prep = Prepared(Path(a.prep_root) / BAND_DIR[r["band"]])
    if prep.prep_id != spec["prep_id"]:
        raise ValueError(f"prep id mismatch {prep.prep_id} != {spec['prep_id']}")
    fd = spec["fold_def"]
    train = prep.view(r["input_view"], fd["train"])
    val = prep.view(r["input_view"], fd["validation"]) if spec["mode"] == "cv" else None
    if val is not None:
        assert not set(train.subject_ids) & set(val.subject_ids)
    init = None
    pretrained = None
    if r["arch"] == "cbramod":
        pretrained = str(Path(a.weights_dir) / "CBraMod" / "pretrained_weights.pth")
        init = sha256_file(pretrained)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(spec, run_dir / "fit.json")
    fit = Fit(spec, run_dir, train, val, ROOT, pretrained, init, prep.input_hashes(r["input_view"]), a.device)
    resumed = fit.try_resume()
    ledger(a.ledger, {"fit_id": spec["fit_id"], "event": "start", "resumed": resumed, "plan": Path(a.plan).name})
    t0 = time.time()
    try:
        summary = fit.fit(stop_after_epochs=a.stop_after_epochs)
    except Exception as exc:
        write_json({"state": "failed", "error": f"{type(exc).__name__}: {exc}", "utc": now(), "resumed": resumed},
                   run_dir / "status.json")
        ledger(a.ledger, {"fit_id": spec["fit_id"], "event": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    if summary is None:
        write_json({"state": "interrupted_for_test", "utc": now(), "epoch_completed": fit.epoch}, run_dir / "status.json")
        ledger(a.ledger, {"fit_id": spec["fit_id"], "event": "interrupted_for_test", "epoch": fit.epoch})
        return
    device = fit.device
    eval_ckpt = run_dir / "checkpoints" / ("best.ckpt" if spec["mode"] == "cv" else "final.ckpt")
    model, st = load_eval_model(eval_ckpt, r, tuple(train.X.shape[1:]), device)
    outputs = {}
    probe_src = val if val is not None else train
    Xp = fit.scaler.transform(probe_src.X[:PROBE_N])
    probe_logits = []
    with torch.no_grad():
        probe_logits = model(torch.from_numpy(Xp).to(device)).double().cpu().numpy()
    np.savez(run_dir / "probe.npz", trial_ids=probe_src.trial_ids[:PROBE_N], logits=probe_logits,
             input_sha256=np.array([__import__("hashlib").sha256(Xp.tobytes()).hexdigest()]))
    if val is not None:
        z = margins(model, fit.Xva, device)
        pp = per_participant_auroc(val.subject_ids, val.y, z)
        macro = participant_macro_auroc(pp)
        if abs(macro - st["best_metric"]) > 1e-9:
            raise ValueError(f"reloaded best checkpoint macro AUROC {macro} != recorded {st['best_metric']}")
        import pandas as pd
        pd.DataFrame({"subject_id": val.subject_ids, "trial_id": val.trial_ids, "label": val.y, "margin": z,
                      "p_error": 1 / (1 + np.exp(-z))}).to_csv(run_dir / "inner_val_predictions.csv", index=False)
        outputs = {"per_participant_auroc": pp, "macro_auroc": macro}
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name not in ("status.json",))
    status = {"state": "completed", "utc": now(), "fit_id": spec["fit_id"], "identity_hash": spec["identity_hash"],
              "recipe_id": spec["recipe_id"], "mode": spec["mode"], "fold": spec["fold"], "seed": spec["seed"],
              "resumed": resumed, "summary": summary, "selected_checkpoint": eval_ckpt.name,
              "best_epoch_one_based": summary["best_epoch"], "best_epoch_zero_based":
                  None if summary["best_epoch"] is None else summary["best_epoch"] - 1,
              "epochs_completed": summary["epochs_completed"], **outputs,
              "params_total": models.count(fit.model), "params_trainable": models.count(fit.model, True),
              "params_head": sum(p.numel() for p in fit.model.classifier.parameters()) if r["arch"] == "cbramod" else None,
              "trainable_names": fit.model.trainable_names(), "wall_time_s": round(time.time() - t0, 1),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "peak_gpu_mem_mb": max([h.get("peak_gpu_mem_mb") or 0 for h in fit.history] or [0]),
              "provenance": fit.provenance, "source_sha": st["source_sha"],
              "file_sha256": {str(p.relative_to(run_dir)): sha256_file(p) for p in files}}
    write_json(status, run_dir / "status.json")
    ledger(a.ledger, {"fit_id": spec["fit_id"], "event": "completed", "epochs": summary["epochs_completed"],
                      "best_epoch": summary["best_epoch"], "macro_auroc": outputs.get("macro_auroc"),
                      "wall_time_s": status["wall_time_s"]})
    print("COMPLETED", spec["fit_id"], json.dumps({k: status[k] for k in ("epochs_completed", "best_epoch_one_based",
                                                                         "wall_time_s")}), outputs.get("macro_auroc"))


if __name__ == "__main__":
    main()
