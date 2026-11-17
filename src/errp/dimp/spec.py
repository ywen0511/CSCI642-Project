"""Decoder-improvement study specification: recipes, bands, folds and canonical fit identities.

All numbers come from ``configs/decoder_improvement/study.json`` (the plan mirrored from the handoff). A recipe is
the complete set of scientific settings of one decoder; its content hash identifies it. A fit is one recipe trained
on one fold (inner CV fold or the outer training set) with one seed; its identity is derived from the recipe hash,
the preprocessing ID of its band, the fold definition and the seed (plus the fixed epoch count for final refits).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "configs" / "decoder_improvement" / "study.json"
BANDS = {"1-20": (1.0, 20.0), "0.3-40": (0.3, 40.0), "0.3-75": (0.3, 75.0)}
BAND_DIR = {"1-20": "band_1_20", "0.3-40": "band_0p3_40", "0.3-75": "band_0p3_75"}


def load_study(path=STUDY) -> dict:
    return json.loads(Path(path).read_text())


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def short_hash(obj, n=12) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:n]


def _fmt(x) -> str:
    return "none" if x is None else f"{x:.0e}".replace("e-0", "e-").replace("+0", "")


def cbramod_recipe(study, head, adaptation, backbone_lr, band="1-20", loss="weighted_ce", head_lr=None) -> dict:
    d = study["defaults"]["cbramod"]
    if adaptation == "frozen":
        backbone_lr = None
    elif backbone_lr is None:
        raise ValueError("non-frozen adaptation needs a backbone learning rate")
    return {"arch": "cbramod", "head": head, "adaptation": adaptation, "backbone_lr": backbone_lr,
            "head_lr": head_lr if head_lr is not None else d["head_lr"], "band": band, "loss": loss,
            "dropout": d["dropout"], "optimizer": d["optimizer"], "weight_decay": d["weight_decay"],
            "batch_size": d["batch_size"], "max_epochs": d["max_epochs"], "patience": d["patience"],
            "scheduler": d["scheduler"], "scheduler_horizon_epochs": d["scheduler_horizon"], "eta_min": d["eta_min"],
            "grad_clip": d["grad_clip"], "label_smoothing": d["label_smoothing"],
            "periodic_every_epochs": d["periodic_every_epochs"], "input_view": "patch200",
            "scaling": {"kind": "fixed_divide", "divisor": 100.0},
            "initialization": "pinned_official_cbramod"}


def eegnet_recipe(study, lr, dropout, band="1-20", loss="weighted_ce") -> dict:
    d = study["defaults"]["eegnet"]
    return {"arch": "eegnet", "lr": lr, "dropout": dropout, "band": band, "loss": loss,
            "optimizer": d["optimizer"], "weight_decay": d["weight_decay"], "batch_size": d["batch_size"],
            "max_epochs": d["max_epochs"], "patience": d["patience"], "scheduler": d["scheduler"],
            "grad_clip": None, "label_smoothing": d["label_smoothing"],
            "periodic_every_epochs": d["periodic_every_epochs"], "model": d["model"], "input_view": "eeg256",
            "scaling": {"kind": "train_global_std"}, "initialization": "seeded_random"}


def recipe_id(r: dict) -> str:
    """Readable, unique ID; the content hash suffix guarantees uniqueness."""
    loss = {"weighted_ce": "wce", "ce": "ce"}[r["loss"]]
    band = r["band"].replace(".", "p").replace("-", "_")
    if r["arch"] == "cbramod":
        head = {"historical_flatten_mlp": "flat", "temporal_mean_linear": "timepool"}[r["head"]]
        name = f"cb-{head}-{r['adaptation']}-bb{_fmt(r['backbone_lr'])}-hl{_fmt(r['head_lr'])}-b{band}-{loss}"
    else:
        name = f"eeg-lr{_fmt(r['lr'])}-do{r['dropout']}-b{band}-{loss}"
    return f"{name}-{short_hash(r, 8)}"


def folds(study) -> dict:
    out = {f["id"]: {"train": f["train"], "validation": f["validation"]} for f in study["inner_folds"]}
    out["outer"] = {"train": study["outer_split"]["train"], "validation": []}
    return out


def fit_spec(study, recipe: dict, fold: str, seed: int, prep_id: str, mode: str = "cv", epochs: int | None = None) -> dict:
    """mode 'cv': inner fold, early stopping on inner-validation participant-macro AUROC.
    mode 'final': all seven training participants, fixed epoch count, final snapshot is the model."""
    f = folds(study)[fold]
    if mode == "cv":
        assert fold != "outer" and epochs is None
    elif mode == "final":
        assert fold == "outer" and isinstance(epochs, int) and 1 <= epochs <= recipe["max_epochs"]
    else:
        raise ValueError(mode)
    identity = {"recipe_hash": short_hash(recipe, 16), "prep_id": prep_id, "fold": fold, "fold_def": f,
                "seed": seed, "mode": mode, "epochs": epochs}
    rid = recipe_id(recipe)
    fid = f"{rid}__{fold}__s{seed}" + (f"__e{epochs}" if mode == "final" else "")
    return {"fit_id": fid, "identity_hash": short_hash(identity, 16), "identity": identity, "recipe_id": rid,
            "recipe": recipe, "fold": fold, "fold_def": f, "seed": seed, "mode": mode, "epochs": epochs,
            "prep_id": prep_id}


def participant_macro_auroc(per_participant: dict) -> float:
    """Equal weight per participant; an undefined (single-class) participant is a protocol failure."""
    vals = list(per_participant.values())
    if not vals or any(v != v for v in vals):
        raise ValueError(f"undefined per-participant AUROC: {per_participant}")
    return float(sum(vals) / len(vals))


def selection_score(fit_rows: list[dict]) -> dict:
    """Stage selection score: per inner-validation participant, mean AUROC over seeds; then mean over participants.
    fit_rows: [{"seed": s, "per_participant_auroc": {pid: auc}}] covering all folds and seeds of one candidate."""
    by_p: dict[str, list] = {}
    for r in fit_rows:
        for pid, a in r["per_participant_auroc"].items():
            by_p.setdefault(pid, []).append(a)
    if any(v != v for vs in by_p.values() for v in vs):
        raise ValueError("undefined participant AUROC in candidate fits")
    per_p = {p: sum(v) / len(v) for p, v in sorted(by_p.items())}
    return {"score": sum(per_p.values()) / len(per_p), "per_participant_seed_mean": per_p,
            "n_participants": len(per_p), "n_fits": len(fit_rows)}
