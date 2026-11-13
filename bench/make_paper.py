#!/usr/bin/env python
"""Generate every measured number/table of the ErrP-Bench paper from result files (no hand-typed numbers).

  python bench/make_paper.py --results DIR --out paper/errp_bench/generated

DIR contains: dataset_stats.json, decoding/{variants.json,stats.json}, rl_frontier/summary.json (optional),
fits_count.json. Writes numbers.tex (\\newcommand macros), table_datasets.tex, table_main.tex, table_ea.tex,
table_rl.tex and the figures' source CSVs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DS = ["errphri_robot", "errphri_cursor", "bnci2015_013", "erpcore_ern", "interhri"]
DS_NICE = {"errphri_robot": "ErrP-HRI robot", "errphri_cursor": "ErrP-HRI cursor", "bnci2015_013": "BNCI 2015-013",
           "erpcore_ern": "ERP CORE ERN", "interhri": "IntEr-HRI", "interhri_early": "IntEr-HRI 0--0.5\\,s"}
PARADIGM = {"errphri_robot": "observed robot error", "errphri_cursor": "observed cursor error",
            "bnci2015_013": "cursor monitoring", "erpcore_ern": "own response error (flanker)",
            "interhri": "orthosis-introduced error"}
LOCK = {"errphri_robot": "machine action", "errphri_cursor": "machine action", "bnci2015_013": "cursor step",
        "erpcore_ern": "response", "interhri": "perturbation"}
MODELS = ["xdawn_rg", "eegnet", "eeginception_erp", "eegnex", "eegconformer", "ctnet", "labram", "eegpt", "cbramod",
          "ens_mix", "ens_all", "ens_top3"]
NICE = {"xdawn_rg": "xDAWN+RG", "eegnet": "EEGNet", "eeginception_erp": "EEG-Inception", "eegnex": "EEGNeX",
        "eegconformer": "EEG-Conformer", "ctnet": "CTNet", "labram": "LaBraM", "eegpt": "EEGPT", "cbramod": "CBraMod",
        "ens_mix": "ErrP-Ens", "ens_all": "ErrP-Ens-EA", "ens_top3": "ErrP-Ens-EA (top-3)"}
FAMILY = {"xdawn_rg": "Riemannian", "eegnet": "compact", "eeginception_erp": "compact", "eegnex": "compact",
          "eegconformer": "compact", "ctnet": "compact", "labram": "foundation", "eegpt": "foundation",
          "cbramod": "foundation", "ens_mix": "ours", "ens_all": "ours", "ens_top3": "ours"}


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def f3(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    nums = []
    # ---------------- datasets
    st = {r["dataset"]: r for r in json.loads((R / "dataset_stats.json").read_text())}
    lines = ["\\begin{tabular}{llrrrrrl}", "\\toprule",
             "Dataset & Paradigm & Subj. & Ch. & Epochs & Error \\% & Min err./subj. & Window (s) \\\\", "\\midrule"]
    for d in DS:
        r = st[d]
        w = r["window_s"]
        lines.append(f"{DS_NICE[d]} & {PARADIGM[d]} & {r['n_subjects']} & {r['n_channels']} & {r['n_epochs']:,} & "
                     f"{100 * r['error_rate']:.1f} & {r['min_errors_per_subject']} & [{w[0]:.1f}, {w[1]:.1f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (O / "table_datasets.tex").write_text("\n".join(lines) + "\n")
    uniq_subj = st["errphri_robot"]["n_subjects"] + sum(st[d]["n_subjects"] for d in DS[2:])  # robot/cursor: same people
    nums += [macro("NSubjectsTotal", uniq_subj), macro("NEpochsTotal", f"{sum(st[d]['n_epochs'] for d in DS):,}")]
    # ---------------- decoding
    V = json.loads((R / "decoding/variants.json").read_text())
    S = json.loads((R / "decoding/stats.json").read_text())
    comp = {(v["dataset"], v["model"], v["align"]): v for v in V if v["complete"]}
    fits = json.loads((R / "fits_count.json").read_text())
    nums += [macro("NFitsTotal", f"{fits['n_fits']:,}"), macro("NSeedsMain", fits["seeds_text"]),
             macro("GPUHours", f"{fits['gpu_hours']:.0f}")]
    dss = [d for d in DS if any(k[0] == d for k in comp)]
    best = {d: max(v["macro_auroc_mean"] for k, v in comp.items() if k[0] == d) for d in dss}
    ranks = defaultdict(list)
    for d in dss:
        order = sorted([k for k in comp if k[0] == d], key=lambda k: -comp[k]["macro_auroc_mean"])
        for i, k in enumerate(order):
            ranks[(k[1], k[2])].append(i + 1)
    lines = ["\\begin{tabular}{lll" + "c" * len(dss) + "c}", "\\toprule",
             "Family & Model & EA & " + " & ".join(DS_NICE[d] for d in dss) + " & Rank \\\\", "\\midrule"]
    prev_family = None
    for m in MODELS:
        for al in ("none", "ea", "both"):
            if not any((d, m, al) in comp for d in dss):
                continue
            if FAMILY[m] != prev_family and prev_family is not None:
                lines.append("\\midrule")
            prev_family = FAMILY[m]
            cells = []
            for d in dss:
                v = comp.get((d, m, al))
                if v is None:
                    cells.append("--")
                    continue
                s = f"{v['macro_auroc_mean']:.3f}" + (f"\\scriptsize{{$\\pm${v['macro_auroc_sd']:.3f}}}" if v["n_seeds"] > 1 else "\\scriptsize{$^{\\ast}$}")
                cells.append(f"\\textbf{{{s}}}" if abs(v["macro_auroc_mean"] - best[d]) < 1e-12 else s)
            rk = ranks[(m, al)]
            mark = {"ea": "\\checkmark", "both": "both"}.get(al, "")
            lines.append(f"{FAMILY[m]} & {NICE[m]} & {mark} & " + " & ".join(cells)
                         + f" & {np.mean(rk):.1f}" + ("" if len(rk) == len(dss) else "$^{\\dagger}$") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (O / "table_main.tex").write_text("\n".join(lines) + "\n")
    # EA effect table: per model, mean delta over datasets, n datasets improved, per-dataset delta (+p)
    ea = [e for e in S["ea_vs_none"] if e["model"] in MODELS]
    lines = ["\\begin{tabular}{l" + "c" * len(dss) + "c}", "\\toprule",
             "Model & " + " & ".join(DS_NICE[d] for d in dss) + " & Mean $\\Delta$ \\\\", "\\midrule"]
    all_d = []
    fm_d, sc_d = [], []
    for m in MODELS[:9]:
        row = {e["dataset"]: e for e in ea if e["model"] == m}
        if not row:
            continue
        cells = []
        for d in dss:
            e = row.get(d)
            if e is None:
                cells.append("--")
                continue
            star = "$^{\\ast}$" if np.isfinite(e["p"]) and e["p"] < 0.05 else ""
            cells.append(f"{e['delta']:+.3f}{star}")
            all_d.append(e["delta"])
            (fm_d if FAMILY[m] == "foundation" else sc_d).append(e["delta"])
        md = np.mean([row[d]["delta"] for d in dss if d in row])
        lines.append(f"{NICE[m]} & " + " & ".join(cells) + f" & {md:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (O / "table_ea.tex").write_text("\n".join(lines) + "\n")
    nums += [macro("EAMeanDelta", f"{np.mean(all_d):+.3f}"), macro("EAPosFrac", f"{sum(x > 0 for x in all_d)}/{len(all_d)}"),
             macro("EAFMDelta", f"{np.mean(fm_d):+.3f}" if fm_d else "--"),
             macro("EAScratchDelta", f"{np.mean(sc_d):+.3f}" if sc_d else "--")]
    # per-dataset best and family bests
    for d in dss:
        tag = "".join(ch for ch in DS_NICE[d].title() if ch.isalpha())
        k = max((k for k in comp if k[0] == d), key=lambda k: comp[k]["macro_auroc_mean"])
        nums += [macro(f"Best{tag}", f3(comp[k]["macro_auroc_mean"])),
                 macro(f"BestModel{tag}", NICE[k[1]] + (" + EA" if k[2] == "ea" and not k[1].startswith("ens") else ""))]
        for fam in ("Riemannian", "compact", "foundation", "ours"):
            ks = [k for k in comp if k[0] == d and FAMILY[k[1]] == fam]
            if ks:
                kb = max(ks, key=lambda k: comp[k]["macro_auroc_mean"])
                nums.append(macro(f"Best{fam.title()}{tag}", f3(comp[kb]["macro_auroc_mean"])))
    # best-vs-others significance summary
    bvo = S["best_vs_others"]
    (O / "best_vs_others.json").write_text(json.dumps(bvo, indent=1) + "\n")
    lines = ["\\begin{tabular}{llcp{0.45\\linewidth}}", "\\toprule",
             "Dataset & Best variant & Sig.\\ worse / others & Not significantly different (Holm $p\\geq0.05$) \\\\", "\\midrule"]
    for d in dss:
        cs = [c for c in bvo if c["dataset"] == d]
        if not cs:
            continue
        nm = lambda v: NICE[v.split("/")[0]] + ("+EA" if v.endswith("/ea") and not v.startswith("ens") else "")
        ns = [c for c in cs if not c["p_holm"] < 0.05]
        lines.append(f"{DS_NICE[d]} & {nm(cs[0]['best'])} & {len(cs) - len(ns)}/{len(cs)} & "
                     + (", ".join(nm(c["other"]) for c in ns) or "none") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (O / "table_bvo.tex").write_text("\n".join(lines) + "\n")
    # ---------------- paired comparisons over participants
    from scipy.stats import wilcoxon, spearmanr
    allv = {(v["dataset"], v["model"], v["align"]): v for v in V}
    txt = ["ErrP-Bench results summary (generated by bench/make_paper.py; every number below is measured)", ""]

    def best_key(d, fam=None, align=None):
        ks = [k for k in comp if k[0] == d and (fam is None or FAMILY[k[1]] == fam) and (align is None or k[2] == align)]
        return max(ks, key=lambda k: comp[k]["macro_auroc_mean"]) if ks else None

    def paired(k1, k2):
        a1, a2 = comp[k1]["per_subject"], comp[k2]["per_subject"]
        u = sorted(set(a1) & set(a2))
        x, y = np.array([a1[i] for i in u]), np.array([a2[i] for i in u])
        p = float(wilcoxon(x, y).pvalue) if len(u) >= 5 and np.any(x != y) else float("nan")
        return float(np.mean(x - y)), int((x > y).sum()), len(u), p

    def fp(p):
        return "n/a" if not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else f"{p:.0e}")

    def fpl(p):  # LaTeX (math mode) version
        if not np.isfinite(p):
            return "\\text{n/a}"
        if p >= 0.001:
            return f"{p:.3f}"
        m, e = f"{p:.0e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"

    # robot vs cursor, same participants: best variant of each condition, and per model family
    if "errphri_robot" in dss and "errphri_cursor" in dss:
        kr, kc = best_key("errphri_robot"), best_key("errphri_cursor")
        a1, a2 = comp[kc]["per_subject"], comp[kr]["per_subject"]
        u = sorted(set(a1) & set(a2))
        x, y = np.array([a1[i] for i in u]), np.array([a2[i] for i in u])
        p = float(wilcoxon(x, y).pvalue)
        nums += [macro("CursorMinusRobot", f3(np.mean(x - y))), macro("CursorGtRobot", f"{int((x > y).sum())}/{len(u)}"),
                 macro("CursorRobotP", fpl(p))]
        txt.append(f"robot vs cursor (same {len(u)} participants, best variant each): cursor-robot = {np.mean(x-y):+.3f}, "
                   f"cursor>robot in {int((x>y).sum())}/{len(u)}, Wilcoxon p={fp(p)}")
    # foundation vs compact: best foundation minus best compact per dataset
    gaps = []
    fm_lines = ["\\begin{tabular}{llcllcccc}", "\\toprule",
                "Dataset & Best foundation & AUROC & Best compact & & AUROC & $\\Delta$ & FM higher & $p$ \\\\", "\\midrule"]
    for d in dss:
        kf, kc_ = best_key(d, "foundation"), best_key(d, "compact")
        if kf and kc_:
            dl, npos, n, p = paired(kf, kc_)
            gaps.append(dl)
            ea_ = lambda k: " + EA" if k[2] == "ea" else ""
            fm_lines.append(f"{DS_NICE[d]} & {NICE[kf[1]]}{ea_(kf)} & {comp[kf]['macro_auroc_mean']:.3f} & "
                            f"{NICE[kc_[1]]}{ea_(kc_)} & & {comp[kc_]['macro_auroc_mean']:.3f} & {dl:+.3f} & {npos}/{n} & ${fpl(p)}$ \\\\")
            txt.append(f"{d}: best foundation {NICE[kf[1]]}/{kf[2]} {comp[kf]['macro_auroc_mean']:.3f} vs best compact "
                       f"{NICE[kc_[1]]}/{kc_[2]} {comp[kc_]['macro_auroc_mean']:.3f}: delta {dl:+.3f} ({npos}/{n} participants higher, p={fp(p)})")
    fm_lines += ["\\bottomrule", "\\end{tabular}"]
    (O / "table_fm.tex").write_text("\n".join(fm_lines) + "\n")
    if gaps:
        nums += [macro("FMMinusCompactMean", f"{np.mean(gaps):+.3f}"), macro("FMWinsDatasets", f"{sum(g > 0 for g in gaps)}/{len(gaps)}")]
    # ensemble vs best single (paired, Holm over datasets)
    ENSAL = {"ens_mix": "both", "ens_all": "ea", "ens_top3": "ea"}
    ENSTAG = {"ens_mix": "EnsMix", "ens_all": "EnsAll", "ens_top3": "EnsTop"}
    for ens in ("ens_mix", "ens_all", "ens_top3"):
        rows_e, ps = [], []
        for d in dss:
            ke = (d, ens, ENSAL[ens])
            ks = best_key(d) if False else max((k for k in comp if k[0] == d and not k[1].startswith("ens")),
                                               key=lambda k: comp[k]["macro_auroc_mean"], default=None)
            if ke in comp and ks:
                dl, npos, n, p = paired(ke, ks)
                rows_e.append((d, dl, npos, n, p, ks)); ps.append(p if np.isfinite(p) else 1.0)
        tag = ENSTAG[ens]
        if rows_e:
            from aggregate import holm
            adj = holm(np.array(ps))
            nums += [macro(f"{tag}VsBestMean", f"{np.mean([r[1] for r in rows_e]):+.3f}"),
                     macro(f"{tag}BeatsBest", f"{sum(r[1] > 0 for r in rows_e)}/{len(rows_e)}"),
                     macro(f"{tag}SigBetter", f"{sum(r[1] > 0 and q < 0.05 for r, q in zip(rows_e, adj))}"),
                     macro(f"{tag}SigWorse", f"{sum(r[1] < 0 and q < 0.05 for r, q in zip(rows_e, adj))}")]
            for r, q in zip(rows_e, adj):
                txt.append(f"{ens} vs best single on {r[0]} ({NICE[r[5][1]]}/{r[5][2]}): {r[1]:+.3f} ({r[2]}/{r[3]} participants), "
                           f"p={fp(r[4])}, Holm p={fp(q)}")
    # rank of each ensemble among all variants
    for ens in ("ens_mix", "ens_all", "ens_top3"):
        rk = ranks.get((ens, ENSAL[ens]))
        if rk:
            nums.append(macro(ENSTAG[ens] + "MeanRank", f"{np.mean(rk):.1f}"))
        vals = {d: comp[(d, ens, ENSAL[ens])]["macro_auroc_mean"] for d in dss if (d, ens, ENSAL[ens]) in comp}
        for d, v in vals.items():
            tg = "".join(ch for ch in DS_NICE[d].title() if ch.isalpha())
            nums.append(macro(f"{ENSTAG[ens]}{tg}", f3(v)))
    # best single (non-ensemble) decoder per dataset
    for d in dss:
        tg = "".join(ch for ch in DS_NICE[d].title() if ch.isalpha())
        ks = max((k for k in comp if k[0] == d and not k[1].startswith("ens")), key=lambda k: comp[k]["macro_auroc_mean"])
        nums += [macro(f"BestSingle{tg}", f3(comp[ks]["macro_auroc_mean"])),
                 macro(f"BestSingleModel{tg}", NICE[ks[1]] + (" + EA" if ks[2] == "ea" else ""))]
    nums.append(macro("NVariantsMax", str(max(sum(1 for k in comp if k[0] == d) for d in dss))))
    # EA: per dataset mean delta and n significant
    for d in dss:
        es = [e for e in ea if e["dataset"] == d]
        tag = "".join(ch for ch in DS_NICE[d].title() if ch.isalpha())
        if es:
            nums += [macro(f"EADelta{tag}", f"{np.mean([e['delta'] for e in es]):+.3f}"),
                     macro(f"EASigPos{tag}", str(sum(e["delta"] > 0 and np.isfinite(e["p"]) and e["p"] < 0.05 for e in es))),
                     macro(f"EASigNeg{tag}", str(sum(e["delta"] < 0 and np.isfinite(e["p"]) and e["p"] < 0.05 for e in es))),
                     macro(f"EANModels{tag}", str(len(es)))]
            txt.append(f"EA on {d}: mean delta {np.mean([e['delta'] for e in es]):+.3f} over {len(es)} models; " +
                       ", ".join(f"{NICE[e['model']]} {e['delta']:+.3f} (p={fp(e['p'])})" for e in es))
    # early-window control (IntEr-HRI first 0.5 s)
    early = [v for v in V if v["dataset"] == "interhri_early" and v["complete"]]
    for v in early:
        nm = {"eegnet": "EEGNet", "xdawn_rg": "Xdawn"}.get(v["model"])
        if nm:
            nums.append(macro(f"Early{nm}{'EA' if v['align'] == 'ea' else 'None'}", f3(v["macro_auroc_mean"])))
            full = allv.get(("interhri", v["model"], v["align"]))
            txt.append(f"interhri_early {v['model']}/{v['align']}: {v['macro_auroc_mean']:.3f} (full window: "
                       f"{full['macro_auroc_mean']:.3f})" if full else f"interhri_early {v['model']}/{v['align']}: {v['macro_auroc_mean']:.3f}")
    # ---------------- RL frontier
    rl = R / "rl_frontier/summary.json"
    best_ds = {d: comp[best_key(d)]["macro_auroc_mean"] for d in dss}
    if rl.exists():
        (O / "rl_summary_copy.json").write_text(rl.read_text())
        import csv
        runs = list(csv.DictReader(open(R / "rl_frontier/per_run.csv")))
        T = json.loads(rl.read_text())["table"]
        tasks = [t for t in ("reach", "push", "push_staged") if any(e["task"] == t and e["condition"] != "sparse" for e in T)]
        TN = {"reach": "FetchReach", "push": "FetchPush", "push_staged": "FetchPush (staged)"}
        conds = ["sparse", "dense"] + [f"auroc{a:.2f}" for a in (0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0)]
        CN = {"sparse": "sparse only", "dense": "dense (reference)"}
        lines = ["\\begin{tabular}{l" + "cc" * len(tasks) + "}", "\\toprule",
                 "Feedback & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TN[t]}}}" for t in tasks) + " \\\\",
                 " & " + " & ".join("norm.\\ AUC & $\\Delta$ vs sparse" for _ in tasks) + " \\\\", "\\midrule"]
        for c in conds:
            cells = []
            for t in tasks:
                e = next((e for e in T if e["task"] == t and e["condition"] == c), None)
                if e is None:
                    cells += ["--", "--" if c != "sparse" else "ref"]
                    continue
                cells.append(f"{e['normalized_auc_mean']:.3f}\\scriptsize{{$\\pm${e['normalized_auc_sd']:.3f}}}")
                cells.append("ref" if c == "sparse" else
                             f"{e['d_normalized_auc_mean']:+.3f} ({e['d_normalized_auc_npos']}/{e['d_normalized_auc_npaired']})")
            lab = CN[c] if c in CN else ("AUROC 1.00 (exact)" if c == "auroc1.00" else f"AUROC {float(c[5:]):.2f}")
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (O / "table_rl.tex").write_text("\n".join(lines) + "\n")
        nseeds = {t: len({r["seed"] for r in runs if r["task"] == t}) for t in tasks}
        nums.append(macro("RLSeedsText", ", ".join(f"{nseeds[t]} ({TN[t]})" for t in tasks)))
        # trend test: Spearman of paired AUC gain on AUROC level; break-even AUROC by linear interpolation of the mean
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.2), squeeze=False)
        for t, ax in zip(tasks, axs[0]):
            sp_ref = "push" if t == "push_staged" else t
            sparse = {r["seed"]: float(r["normalized_auc"]) for r in runs if r["task"] == sp_ref and r["condition"] == "sparse"}
            xs, gs = [], []
            for r in runs:
                if r["task"] == t and r["condition"].startswith("auroc") and r["seed"] in sparse:
                    xs.append(float(r["condition"][5:])); gs.append(float(r["normalized_auc"]) - sparse[r["seed"]])
            if len(set(xs)) >= 3:
                rho, p = spearmanr(xs, gs)
                lv = sorted(set(xs)); mg = [np.mean([g for x, g in zip(xs, gs) if x == l]) for l in lv]
                be = next((lv[i - 1] + (lv[i] - lv[i - 1]) * (0 - mg[i - 1]) / (mg[i] - mg[i - 1])
                           for i in range(1, len(lv)) if mg[i - 1] < 0 <= mg[i]), None)
                tg = {"reach": "Reach", "push": "Push", "push_staged": "PushStaged"}[t]
                nums += [macro(f"RLRho{tg}", f"{rho:.2f}"), macro(f"RLRhoP{tg}", fpl(p)),
                         macro(f"RLBreakEven{tg}", f"{be:.2f}" if be is not None else
                               ("$<$0.55" if mg[0] >= 0 else "$>$1.0"))]
                txt.append(f"RL {t}: Spearman(AUROC, paired AUC gain vs sparse) rho={rho:.2f} p={fp(p)} (n={len(xs)} runs); "
                           f"mean gain per level " + ", ".join(f"{l:.2f}:{m:+.3f}" for l, m in zip(lv, mg)) +
                           f"; break-even AUROC {be if be is None else round(be, 3)}")
                CM = {"sparse": "Sparse", "dense": "Dense", "auroc1.00": "Exact", "auroc0.55": "Low", "auroc0.90": "Ninety"}
                for e in T:
                    if e["task"] == t and e["condition"] in CM:
                        nums += [macro(f"RLAUC{tg}{CM[e['condition']]}", f"{e['normalized_auc_mean']:.3f}"),
                                 macro(f"RLSucc{tg}{CM[e['condition']]}", f"{e['final_success_mean']:.2f}")]
                        if e["condition"] != "sparse":
                            nums += [macro(f"RLGain{tg}{CM[e['condition']]}", f"{e['d_normalized_auc_mean']:+.3f}"),
                                     macro(f"RLPos{tg}{CM[e['condition']]}", f"{e['d_normalized_auc_npos']}/{e['d_normalized_auc_npaired']}"),
                                     macro(f"RLSignP{tg}{CM[e['condition']]}", f"{e['d_normalized_auc_sign_p']:.3f}")]
                for e in T:
                    if e["task"] == t:
                        txt.append(f"   {t} {e['condition']:10s} n={e['n']} normAUC={e['normalized_auc_mean']:.3f}±{e['normalized_auc_sd']:.3f} "
                                   f"final success={e['final_success_mean']:.3f}")
                # figure
                ax.axhline(np.mean(list(sparse.values())), color="0.4", ls="--", lw=1, label="sparse")
                dn = [float(r["normalized_auc"]) for r in runs if r["task"] == sp_ref and r["condition"] == "dense"]
                if dn:
                    ax.axhline(np.mean(dn), color="0.4", ls=":", lw=1, label="dense")
                ys = [[float(r["normalized_auc"]) for r in runs if r["task"] == t and r["condition"] == f"auroc{l:.2f}"] for l in lv]
                ax.errorbar(lv, [np.mean(y) for y in ys], yerr=[np.std(y, ddof=1) for y in ys], marker="o", color="C0",
                            capsize=3, label="ErrP feedback")
                for i, d in enumerate(dss):
                    ax.axvline(best_ds[d], color=f"C{i + 1}", alpha=0.6, lw=1.5, label=f"best decoder: {DS_NICE[d].replace(chr(92) + ',', ' ')}")
                ax.set_title(TN[t]); ax.set_xlabel("feedback AUROC"); ax.set_ylabel("normalized success AUC")
        axs[0][0].legend(fontsize=6, loc="best")
        fig.tight_layout()
        (O.parent / "figures").mkdir(exist_ok=True)
        fig.savefig(O.parent / "figures/frontier.pdf"); fig.savefig(O.parent / "figures/frontier.png", dpi=150)
    for d in dss:
        k = best_key(d)
        txt.append(f"best on {d}: {NICE[k[1]]}/{k[2]} macro-participant AUROC {comp[k]['macro_auroc_mean']:.3f}")
    txt += ["", "All variants (dataset, model, alignment, seeds, participants, macro AUROC mean±SD over seeds):"]
    for v in V:
        txt.append(f"  {v['dataset']:16s} {v['model']:17s} {v['align']:4s} seeds={v['n_seeds']} subj={v['n_subjects']:3d} "
                   f"{'complete' if v['complete'] else 'PARTIAL'} {v['macro_auroc_mean']:.4f}±{v['macro_auroc_sd']:.4f}")
    (O / "results_summary.txt").write_text("\n".join(txt) + "\n")
    (O / "numbers.tex").write_text("% generated by bench/make_paper.py -- do not edit\n" + "\n".join(nums) + "\n")
    print("\n".join(nums))


if __name__ == "__main__":
    main()
