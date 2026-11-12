#!/usr/bin/env python
"""Aggregate ErrP-Bench cross-subject runs into tables + statistics.

  python bench/aggregate.py --runs RUNS_ROOT --out OUT_DIR

Layout: RUNS_ROOT/<dataset>/<model>/<align>/s<seed>/fold<i>/{metrics.json,predictions.npz}.
Unit of analysis = held-out subject (each subject is tested in exactly one fold). For every
dataset x model x align x seed: per-subject AUROC; the variant score is the mean over subjects (macro-subject
AUROC); reported as mean ± SD over seeds. Paired tests over subjects (seed-averaged per-subject AUROC):
EA vs no alignment per model; best variant vs every other variant per dataset (Wilcoxon signed-rank,
two-sided, Holm-corrected within dataset). Only complete variants (all folds x all seeds present) are tabulated.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

NICE = {"xdawn_rg": "xDAWN+RG", "eegnet": "EEGNet", "eeginception_erp": "EEG-Inception", "eegnex": "EEGNeX",
        "eegconformer": "EEG-Conformer", "ctnet": "CTNet", "labram": "LaBraM", "eegpt": "EEGPT", "cbramod": "CBraMod",
        "ens_all": "ErrP-Ens (all)", "ens_top3": "ErrP-Ens (val-top3)"}
PRETRAINED = {"labram", "eegpt", "cbramod"}
DS_NICE = {"errphri_robot": "ErrP-HRI robot", "errphri_cursor": "ErrP-HRI cursor", "bnci2015_013": "BNCI 2015-013",
           "erpcore_ern": "ERP CORE ERN", "interhri": "IntEr-HRI"}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-folds", type=json.loads, default="{}", help='{"dataset": n_folds}')
    a = ap.parse_args()
    root, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # per (ds, model, align) -> seed -> subject -> auroc
    data = defaultdict(lambda: defaultdict(dict))
    folds_seen = defaultdict(lambda: defaultdict(set))
    walls = defaultdict(list)
    for m in root.glob("*/*/*/s*/fold*/metrics.json"):
        fold_dir = m.parent
        ds, model, align, seed = fold_dir.parts[-5], fold_dir.parts[-4], fold_dir.parts[-3], fold_dir.parts[-2]
        r = json.loads(m.read_text())
        for s, v in r["test"]["per_subject_auroc"].items():
            data[(ds, model, align)][seed][s] = v
        folds_seen[(ds, model, align)][seed].add(fold_dir.name)
        walls[(ds, model, align)].append(r.get("wall_s", 0))
    rows = []
    complete = {}
    for key, seeds in sorted(data.items()):
        ds = key[0]
        nf = a.expect_folds.get(ds)
        # a seed counts only if all its folds finished; partial seeds are dropped (listed in variants.json)
        partial = sorted(s for s in seeds if nf and len(folds_seen[key][s]) != nf)
        seeds = {s: v for s, v in seeds.items() if s not in partial}
        ok = bool(seeds)
        if not seeds:
            seeds = data[key]
        subs = sorted(set.intersection(*[set(v) for v in seeds.values()]))
        per_seed = [np.mean([seeds[s][u] for u in subs]) for s in sorted(seeds)]
        per_subject = {u: float(np.mean([seeds[s][u] for s in seeds])) for u in subs}
        row = {"dataset": ds, "model": key[1], "align": key[2], "n_seeds": len(seeds), "n_subjects": len(subs),
               "complete": ok, "seeds": sorted(seeds), "partial_seeds_dropped": partial, "macro_auroc_mean": float(np.mean(per_seed)),
               "macro_auroc_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
               "per_seed": per_seed, "per_subject": per_subject,
               "mean_wall_s": float(np.mean(walls[key])), "total_wall_s": float(np.sum(walls[key]))}
        rows.append(row)
        if ok:
            complete[key] = row
    (out / "variants.json").write_text(json.dumps(rows, indent=1) + "\n")
    stats = {"ea_vs_none": [], "best_vs_others": []}
    for (ds, model, align), r in complete.items():
        if align == "ea" and (ds, model, "none") in complete:
            b = complete[(ds, model, "none")]
            subs = sorted(set(r["per_subject"]) & set(b["per_subject"]))
            x = np.array([r["per_subject"][u] for u in subs]); y = np.array([b["per_subject"][u] for u in subs])
            p = float(wilcoxon(x, y).pvalue) if len(subs) >= 5 and np.any(x != y) else float("nan")
            stats["ea_vs_none"].append({"dataset": ds, "model": model, "delta": float(np.mean(x - y)),
                                        "n_pos": int((x > y).sum()), "n": len(subs), "p": p})
    for ds in sorted({k[0] for k in complete}):
        cand = {k: v for k, v in complete.items() if k[0] == ds}
        best = max(cand, key=lambda k: cand[k]["macro_auroc_mean"])
        comps = []
        for k, v in cand.items():
            if k == best:
                continue
            subs = sorted(set(v["per_subject"]) & set(cand[best]["per_subject"]))
            x = np.array([cand[best]["per_subject"][u] for u in subs]); y = np.array([v["per_subject"][u] for u in subs])
            p = float(wilcoxon(x, y).pvalue) if len(subs) >= 5 and np.any(x != y) else float("nan")
            comps.append({"dataset": ds, "best": f"{best[1]}/{best[2]}", "other": f"{k[1]}/{k[2]}",
                          "delta": float(np.mean(x - y)), "n_pos": int((x > y).sum()), "n": len(subs), "p": p})
        ps = np.array([c["p"] if np.isfinite(c["p"]) else 1.0 for c in comps])
        for c, adj in zip(comps, holm(ps) if len(ps) else []):
            c["p_holm"] = float(adj)
        stats["best_vs_others"] += comps
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    # main LaTeX table: rows = model x align, columns = datasets
    dss = [d for d in DS_NICE if any(k[0] == d for k in complete)]
    best_by_ds = {d: max((v["macro_auroc_mean"] for k, v in complete.items() if k[0] == d), default=None) for d in dss}
    lines = ["\\begin{tabular}{ll" + "c" * len(dss) + "c}", "\\toprule",
             "Model & Align & " + " & ".join(DS_NICE[d] for d in dss) + " & Mean rank \\\\", "\\midrule"]
    variants = sorted({(k[1], k[2]) for k in complete}, key=lambda v: (list(NICE).index(v[0]) if v[0] in NICE else 99, v[1]))
    ranks = defaultdict(list)
    for d in dss:
        order = sorted([v for v in variants if (d, *v) in complete], key=lambda v: -complete[(d, *v)]["macro_auroc_mean"])
        for i, v in enumerate(order):
            ranks[v].append(i + 1)
    for v in variants:
        cells = []
        for d in dss:
            r = complete.get((d, *v))
            if r is None:
                cells.append("--")
                continue
            s = f"{r['macro_auroc_mean']:.3f}$\\pm${r['macro_auroc_sd']:.3f}"
            cells.append(f"\\textbf{{{s}}}" if abs(r["macro_auroc_mean"] - best_by_ds[d]) < 1e-12 else s)
        name = NICE.get(v[0], v[0]) + ("$^\\dagger$" if v[0] in PRETRAINED else "")
        mr = f"{np.mean(ranks[v]):.1f}" if len(ranks[v]) == len(dss) else "--"
        lines.append(f"{name} & {'EA' if v[1] == 'ea' else '--'} & " + " & ".join(cells) + f" & {mr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")
    txt = []
    for r in rows:
        txt.append(f"{r['dataset']:16s} {r['model']:17s} {r['align']:4s} seeds={r['n_seeds']} subj={r['n_subjects']:3d} "
                   f"{'complete' if r['complete'] else 'PARTIAL '} macroAUROC={r['macro_auroc_mean']:.4f}±{r['macro_auroc_sd']:.4f} "
                   f"wall/fit={r['mean_wall_s']:.0f}s")
    (out / "variants.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
