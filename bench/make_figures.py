#!/usr/bin/env python
"""Measured figures of the ErrP-Bench paper (no synthetic numbers).

  python bench/make_figures.py --b $B --results $B/results/final --out paper/errp_bench/figures

Writes: waveforms.pdf (grand-average error-minus-correct ERP per dataset), scatter_participants.pdf (ErrP-Ens vs
best single decoder per held-out participant), roc.pdf (ROC of ErrP-Ens vs baselines, seed 42, scores rank-normalised
within participant), score_hist.pdf (decoder outputs on one held-out robot participant), rl_curves.pdf (FetchReach
success curves), ea_heatmap.pdf (EA effect per model x dataset), ranks.pdf (mean rank over datasets).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

DS = ["errphri_robot", "errphri_cursor", "bnci2015_013", "erpcore_ern", "interhri"]
DSN = {"errphri_robot": "ErrP-HRI robot", "errphri_cursor": "ErrP-HRI cursor", "bnci2015_013": "BNCI 2015-013",
       "erpcore_ern": "ERP CORE ERN", "interhri": "IntEr-HRI"}
NICE = {"xdawn_rg": "xDAWN+RG", "eegnet": "EEGNet", "eeginception_erp": "EEG-Inception", "eegnex": "EEGNeX",
        "eegconformer": "EEG-Conformer", "ctnet": "CTNet", "labram": "LaBraM", "eegpt": "EEGPT", "cbramod": "CBraMod",
        "ens_mix": "ErrP-Ens", "ens_all": "ErrP-Ens-EA", "ens_top3": "ErrP-Ens-EA (top-3)"}
FM = {"labram", "eegpt", "cbramod"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "pdf.fonttype": 42})


def subj_rank(z, subj):
    r = np.empty(len(z))
    for s in np.unique(subj):
        m = subj == s
        r[m] = (rankdata(z[m]) - 1) / max(1, m.sum() - 1)
    return r


def pooled_preds(runs, ds, model, align, seed="s42"):
    for sd in [seed] + sorted(q.name for q in (runs / ds / model / align).glob("s*") if q.name != seed):
        out = _pooled(runs, ds, model, align, sd)
        if out is not None:
            return out
    return None


def _pooled(runs, ds, model, align, seed):
    zs, ys, ss = [], [], []
    for f in range(5):
        p = runs / ds / model / align / seed / f"fold{f}" / "predictions.npz"
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        zs.append(d["test_margin"]); ys.append(d["test_label"]); ss.append(d["test_subject"])
    return np.concatenate(zs), np.concatenate(ys), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None, help="runs root (default: B/runs)")
    ap.add_argument("--rl-curves", default=None, help="compact rl_curves.csv (used if B/rl_runs/sweep is absent)")
    a = ap.parse_args()
    B, R, O = Path(a.b), Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs) if a.runs else B / "runs"
    V = [v for v in json.loads((R / "decoding/variants.json").read_text()) if v["complete"]]
    comp = {(v["dataset"], v["model"], v["align"]): v for v in V}

    def best(d, pred):
        ks = [k for k in comp if k[0] == d and pred(k)]
        return max(ks, key=lambda k: comp[k]["macro_auroc_mean"]) if ks else None

    # 1. grand-average waveforms (needs the prepared EEG; skipped if absent)
    have_prep = all((B / "data/prep" / d / "epochs_256hz_uV.npy").exists() for d in DS)
    fig, axs = plt.subplots(1, 5, figsize=(10, 1.9), sharey=False)
    for ax, d in (zip(axs, DS) if have_prep else []):
        P = B / "data/prep" / d
        cfg = json.loads((P / "preprocessing_config.json").read_text())
        ch = cfg["channels"] if "channels" in cfg else cfg["config"]["channels"]["order"]
        t0 = cfg["window_s"][0] if "window_s" in cfg else cfg["config"]["epoch"]["tmin_s"]
        ci = next(ch.index(c) for c in ("FCz", "Cz", "FC1", "Fz") if c in ch)
        X = np.load(P / "epochs_256hz_uV.npy", mmap_mode="r")
        rows = {r["trial_id"]: r for r in csv.DictReader(open(P / "trial_manifest.csv"))}
        man = [rows[t] for t in np.load(P / "epoch_trial_ids.npy")]  # array order
        y = np.array([int(float(r["label"])) for r in man]); s = np.array([r["subject_id"] for r in man])
        x = np.asarray(X[:, ci, :], dtype=np.float64)
        t = t0 + np.arange(x.shape[1]) / 256.0
        err, cor = [], []
        for u in np.unique(s):
            m = s == u
            if (y[m] == 1).sum() and (y[m] == 0).sum():
                err.append(x[m & (y == 1)].mean(0)); cor.append(x[m & (y == 0)].mean(0))
        err, cor = np.array(err), np.array(cor)
        dif = err - cor
        ax.plot(t, cor.mean(0), color="0.55", lw=1, label="correct")
        ax.plot(t, err.mean(0), color="C3", lw=1, label="error")
        se = dif.std(0, ddof=1) / np.sqrt(len(dif))
        ax.plot(t, dif.mean(0), color="k", lw=1.2, label="error $-$ correct")
        ax.fill_between(t, dif.mean(0) - se, dif.mean(0) + se, color="k", alpha=0.15, lw=0)
        ax.axvline(0, color="0.7", lw=0.6); ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"{DSN[d]} ({ch[ci]}, n={len(dif)})"); ax.set_xlabel("time (s)")
    if have_prep:
        axs[0].set_ylabel("amplitude ($\\mu$V)"); axs[0].legend(loc="lower left", frameon=False)
        fig.tight_layout(); fig.savefig(O / "waveforms.pdf")
    else:
        print("prepared EEG not found: waveforms.pdf skipped")
    plt.close(fig)

    # 2. per-participant scatter ErrP-Ens (mixed-alignment rank ensemble) vs best single decoder (hindsight best on the test participants)
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    lo = 1
    for i, d in enumerate(DS):
        ke, kb = (d, "ens_mix", "both"), best(d, lambda k: not k[1].startswith("ens"))
        if ke not in comp or kb is None:
            continue
        u = sorted(set(comp[ke]["per_subject"]) & set(comp[kb]["per_subject"]))
        x = np.array([comp[kb]["per_subject"][q] for q in u]); yv = np.array([comp[ke]["per_subject"][q] for q in u])
        lo = min(lo, x.min(), yv.min())
        ax.scatter(x, yv, s=10, color=f"C{i}", label=f"{DSN[d]} ({(yv > x).sum()}/{len(u)} above)", alpha=0.8, lw=0)
    ax.plot([lo, 1], [lo, 1], color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("best single decoder (per dataset), AUROC"); ax.set_ylabel("ErrP-Ens, AUROC")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(O / "scatter_participants.pdf"); plt.close(fig)

    # 3. ROC curves, seed 42, scores rank-normalised within participant (as the ensemble does)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.2))
    for ax, d in zip(axs, DS):
        cands = [("ens_mix", "both"), best(d, lambda k: k[1] in {"eegnet", "eeginception_erp", "eegnex", "eegconformer", "ctnet"}),
                 best(d, lambda k: k[1] in FM), best(d, lambda k: k[1] == "xdawn_rg")]
        for j, k in enumerate(cands):
            if k is None:
                continue
            k = k[-2:] if len(k) == 3 else k
            pr = pooled_preds(runs, d, k[0], k[1])
            if pr is None:
                continue
            z, yv, s = pr
            r = subj_rank(z, s)
            fpr, tpr, _ = roc_curve(yv, r)
            ax.plot(fpr, tpr, lw=1.3 if j == 0 else 0.9, color=["C3", "C0", "C2", "0.4"][j],
                    label=f"{NICE[k[0]]}{'+EA' if k[1] == 'ea' and not k[0].startswith('ens') else ''} ({roc_auc_score(yv, r):.3f})")
        ax.plot([0, 1], [0, 1], color="0.8", lw=0.6)
        ax.set_title(DSN[d]); ax.set_xlabel("false positive rate"); ax.legend(frameon=False, loc="lower right", fontsize=5.5)
    axs[0].set_ylabel("true positive rate")
    fig.tight_layout(); fig.savefig(O / "roc.pdf"); plt.close(fig)

    # 4. decoder outputs on one held-out robot participant (median participant of ErrP-Ens), seed 42
    d = "errphri_robot"
    ke = (d, "ens_mix", "both")
    if ke in comp:
        ps = comp[ke]["per_subject"]
        med = sorted(ps, key=ps.get)[len(ps) // 2]
        kf = best(d, lambda k: k[1] in FM)
        fig, axs = plt.subplots(1, 2, figsize=(5.2, 1.9), sharey=True)
        for ax, k in zip(axs, [kf, ke]):
            z, yv, s = pooled_preds(runs, d, k[1], k[2])
            m = s == med
            r = subj_rank(z[m], s[m])
            bins = np.linspace(0, 1, 21)
            ax.hist(r[yv[m] == 0], bins, color="0.6", alpha=0.7, density=True, label="correct trials")
            ax.hist(r[yv[m] == 1], bins, color="C3", alpha=0.6, density=True, label="error trials")
            ax.set_title(f"{NICE[k[1]]}{'+EA' if k[2] == 'ea' and not k[1].startswith('ens') else ''}: "
                         f"AUROC {roc_auc_score(yv[m], r):.3f}")
            ax.set_xlabel("within-participant score rank")
        axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(O / "score_hist.pdf"); plt.close(fig)
        (O / "score_hist_participant.txt").write_text(f"participant {med} of {d}, seed s42\n")

    # 5. FetchReach learning curves (mean +- s.e.m. over seeds)
    sweep = B / "rl_runs/sweep/reach"
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    sty = [("sparse", "sparse", "0.3", "--"), ("dense", "dense", "0.3", ":"), ("auroc0.55", "AUROC 0.55", "C3", "-"),
           ("auroc0.70", "AUROC 0.70", "C1", "-"), ("auroc0.90", "AUROC 0.90", "C2", "-"), ("auroc1.00", "AUROC 1.00", "C0", "-")]
    rows_csv = None
    if not sweep.exists() and a.rl_curves:
        rows_csv = list(csv.DictReader(open(a.rl_curves)))
    for c, lab, col, ls in sty:
        curves = []
        if rows_csv is not None:
            by = {}
            for r in rows_csv:
                if r["task"] == "reach" and r["condition"] == c:
                    by.setdefault(r["seed"], []).append((int(r["steps"]), float(r["success_rate"])))
            curves = [sorted(v) for _, v in sorted(by.items())]
        for sd in (sorted(sweep.glob(f"{c}/seed*")) if rows_csv is None else []):
            if not (sd / "status.json").exists():
                continue
            rows = [json.loads(line) for line in open(sd / "evaluations.jsonl")]
            curves.append([(r["steps"], np.mean([e["final_success"] for e in r["episodes"]])) for r in rows])
        if not curves:
            continue
        n = min(len(cv) for cv in curves)
        st = np.array([p[0] for p in curves[0][:n]]); Y = np.array([[p[1] for p in cv[:n]] for cv in curves])
        m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
        ax.plot(st / 1000, m, color=col, ls=ls, lw=1.1, label=f"{lab} (n={len(Y)})")
        ax.fill_between(st / 1000, m - se, m + se, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("environment steps (k)"); ax.set_ylabel("success rate (100 held-out goals)")
    ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(O / "rl_curves.pdf"); plt.close(fig)

    # 6. EA effect heatmap
    S = json.loads((R / "decoding/stats.json").read_text())
    models = [m for m in NICE if not m.startswith("ens") and any(e["model"] == m for e in S["ea_vs_none"])]
    M = np.full((len(models), len(DS)), np.nan); P = np.full_like(M, np.nan)
    for e in S["ea_vs_none"]:
        if e["model"] in models and e["dataset"] in DS:
            M[models.index(e["model"]), DS.index(e["dataset"])] = e["delta"]; P[models.index(e["model"]), DS.index(e["dataset"])] = e["p"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(models)):
        for j in range(len(DS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([DSN[d] for d in DS], rotation=25, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NICE[m] for m in models])
    fig.colorbar(im, ax=ax, label="$\\Delta$ AUROC (EA $-$ none)"); fig.tight_layout(); fig.savefig(O / "ea_heatmap.pdf"); plt.close(fig)

    # 7. mean rank over datasets (complete variants on all 5 datasets)
    variants = sorted({(k[1], k[2]) for k in comp if k[0] in DS})
    rk = {v: [] for v in variants}
    for d in DS:
        order = sorted([v for v in variants if (d, *v) in comp], key=lambda v: -comp[(d, *v)]["macro_auroc_mean"])
        for i, v in enumerate(order):
            rk[v].append(i + 1)
    full = [v for v in variants if len(rk[v]) == len(DS)]
    full.sort(key=lambda v: np.mean(rk[v]))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    col = ["C3" if v[0].startswith("ens") else ("C2" if v[0] in FM else ("0.4" if v[0] == "xdawn_rg" else "C0")) for v in full]
    ax.barh(range(len(full)), [np.mean(rk[v]) for v in full], color=col)
    ax.set_yticks(range(len(full)))
    ax.set_yticklabels([NICE[v[0]] + (" + EA" if v[1] == "ea" and not v[0].startswith("ens") else "") for v in full], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("mean rank over 5 datasets (lower is better)")
    fig.tight_layout(); fig.savefig(O / "ranks.pdf"); plt.close(fig)
    print("figures written to", O)


if __name__ == "__main__":
    main()
