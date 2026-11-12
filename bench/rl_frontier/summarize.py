"""Summarize the ErrP-Bench decoder-quality -> RL-benefit frontier sweep.

Usage (cluster, RL venv):
  python bench/rl_frontier/summarize.py \
      --runs /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/rl_runs/sweep \
      --out  /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/results/rl_frontier

Expects <runs>/<task>/<condition>/seed<k>/ with status.json (completed, non-smoke),
evaluation.json and evaluations.jsonl. Incomplete/missing runs are listed, never imputed.
Outputs: summary_table.md, summary_table.csv, frontier.csv, per_run.csv, frontier.pdf/.png,
summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rl.metrics import load_curve, normalized_auc  # noqa: E402

TASKS = ("reach", "push", "push_staged")
TASK_LABEL = {"reach": "FetchReach-v4", "push": "FetchPush-v4 (object-goal rule)",
              "push_staged": "FetchPush-v4 (staged rule, suppl.)"}
# push_staged has no own sparse arm: the label rule does not affect sparse training, so its
# sparse reference (and paired differences) are the push/sparse runs with the same seeds.
SPARSE_FROM = {"push_staged": "push"}
LEVELS = (0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0)
CONDS = ("sparse", "dense") + tuple(f"auroc{a:.2f}" for a in LEVELS)
SEEDS = range(5)


def load_runs(root, allow_smoke=False):
    rows, missing = [], []
    for task in TASKS:
        for cond in CONDS:
            for seed in SEEDS:
                d = Path(root) / task / cond / f"seed{seed}"
                if cond == "sparse" and task in SPARSE_FROM:
                    continue
                st = d / "status.json"
                ok = st.is_file() and json.loads(st.read_text()).get("completed")
                if not ok:
                    missing.append(f"{task}/{cond}/seed{seed}")
                    continue
                status, spec = json.loads(st.read_text()), json.loads((d / "run.json").read_text())
                if spec.get("smoke") and not allow_smoke:
                    raise ValueError(f"Smoke run inside the sweep directory: {d}")
                curve = load_curve(d / "evaluations.jsonl")
                final = json.loads((d / "evaluation.json").read_text())
                if curve[-1] != final:
                    raise ValueError(f"Final evaluation differs from curve end: {d}")
                budget = spec["budget"]
                fb = status.get("feedback", {})
                rows.append(dict(task=task, condition=cond, seed=seed, budget=budget,
                                 final_success=final["success_rate"],
                                 normalized_auc=normalized_auc(curve, budget),
                                 final_distance=final["mean_final_distance"],
                                 feedback_events=fb.get("feedback_events"),
                                 eligible_segments=fb.get("eligible_segments"),
                                 error_rate=fb.get("error_rate"),
                                 realised_auroc=fb.get("realised_auroc"),
                                 elapsed_min=status.get("elapsed_seconds", float("nan")) / 60,
                                 curve=[(r["steps"], r["success_rate"]) for r in curve]))
    return rows, missing


def msd(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def sign_test_p(npos, nneg):
    """Exact two-sided sign test (ties dropped)."""
    n = npos + nneg
    if n == 0:
        return float("nan")
    k = min(npos, nneg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["condition"]), {})[r["seed"]] = r
    table = []
    for task in TASKS:
        base = by.get((SPARSE_FROM.get(task, task), "sparse"), {})
        for cond in CONDS:
            runs = by.get((task, cond), {})
            if cond == "sparse" and task in SPARSE_FROM:
                runs = base
            if not runs:
                continue
            ent = dict(task=task, condition=cond, n=len(runs),
                       auroc=float(cond[5:]) if cond.startswith("auroc") else None)
            for m in ("final_success", "normalized_auc", "final_distance", "realised_auroc",
                      "feedback_events", "error_rate"):
                ent[m + "_mean"], ent[m + "_sd"] = msd([r[m] for r in runs.values()])
            for m in ("final_success", "normalized_auc"):
                diffs = [runs[s][m] - base[s][m] for s in sorted(runs) if s in base]
                ent[f"d_{m}_mean"], ent[f"d_{m}_sd"] = msd(diffs)
                ent[f"d_{m}_npos"] = sum(d > 0 for d in diffs)
                ent[f"d_{m}_nneg"] = sum(d < 0 for d in diffs)
                ent[f"d_{m}_npaired"] = len(diffs)
                ent[f"d_{m}_sign_p"] = sign_test_p(ent[f"d_{m}_npos"], ent[f"d_{m}_nneg"])
            table.append(ent)
    return table


def f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_outputs(rows, missing, table, out):
    out.mkdir(parents=True, exist_ok=True)
    per_run_fields = [k for k in rows[0] if k != "curve"] if rows else []
    with (out / "per_run.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=per_run_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "summary_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with (out / "frontier.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "auroc", "condition", "n_paired", "auc_gain_vs_sparse_mean", "auc_gain_sd",
                    "auc_gain_npos", "final_success_gain_mean", "final_success_gain_sd",
                    "final_success_gain_npos", "normalized_auc_mean", "sparse_auc_mean", "dense_auc_mean"])
        for task in TASKS:
            ref = {e["condition"]: e for e in table if e["task"] == task}
            ref.setdefault("sparse", next((e for e in table if e["task"] == SPARSE_FROM.get(task)
                                           and e["condition"] == "sparse"), {}))
            for e in table:
                if e["task"] == task and e["auroc"] is not None:
                    w.writerow([task, e["auroc"], e["condition"], e["d_normalized_auc_npaired"],
                                e["d_normalized_auc_mean"], e["d_normalized_auc_sd"], e["d_normalized_auc_npos"],
                                e["d_final_success_mean"], e["d_final_success_sd"], e["d_final_success_npos"],
                                e["normalized_auc_mean"],
                                ref.get("sparse", {}).get("normalized_auc_mean"),
                                ref.get("dense", {}).get("normalized_auc_mean")])
    lines = ["# Decoder-quality -> RL-benefit frontier", "",
             "Mean ± SD over RL seeds (0-4). Δ = paired (same seed) difference vs sparse; "
             "`+k/n` = seeds with a positive difference; p = exact two-sided sign test. "
             "AUROC 1.00 = ideal feedback (p = label). Realised AUROC = AUROC of delivered p_error "
             "vs simulated label over all feedback events of the run.", ""]
    for task in TASKS:
        ents = [e for e in table if e["task"] == task]
        if not any(e["condition"] != "sparse" for e in ents):
            continue
        lines += [f"## {TASK_LABEL[task]} (budget {rows_budget(rows, task)} env steps)", "",
                  "| condition | n | final success | norm. AUC | final dist (m) | Δ final success (+k/n, p) "
                  "| Δ norm. AUC (+k/n, p) | realised AUROC | feedback events |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for e in ents:
            def d(m):
                if e["condition"] == "sparse":
                    return "ref"
                return (f"{f(e[f'd_{m}_mean'])} ± {f(e[f'd_{m}_sd'])} "
                        f"(+{e[f'd_{m}_npos']}/{e[f'd_{m}_npaired']}, p={f(e[f'd_{m}_sign_p'], 2)})")
            lines.append(f"| {e['condition']} | {e['n']} | {f(e['final_success_mean'])} ± {f(e['final_success_sd'])} "
                         f"| {f(e['normalized_auc_mean'])} ± {f(e['normalized_auc_sd'])} "
                         f"| {f(e['final_distance_mean'], 4)} ± {f(e['final_distance_sd'], 4)} "
                         f"| {d('final_success')} | {d('normalized_auc')} "
                         f"| {f(e['realised_auroc_mean'])} | {f(e['feedback_events_mean'], 0)} |")
        lines.append("")
    if missing:
        lines += [f"**Missing/incomplete runs ({len(missing)}):** " + ", ".join(missing), ""]
    (out / "summary_table.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"n_runs": len(rows), "missing": missing, "table": table},
                                                 indent=2, default=float) + "\n")
    print("\n".join(lines))


def rows_budget(rows, task):
    b = {r["budget"] for r in rows if r["task"] == task}
    return "/".join(str(x) for x in sorted(b))


def plot(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey, ink, ink2 = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#52514e"
    tasks = [t for t in TASKS if any(e["task"] == t and e["condition"] != "sparse" for e in table)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        ents = {e["condition"]: e for e in table if e["task"] == task}
        pts = sorted((e["auroc"], e["normalized_auc_mean"], e["normalized_auc_sd"])
                     for e in ents.values() if e["auroc"] is not None)
        for name, color, ls in (("sparse", grey, "--"), ("dense", orange, ":")):
            if name in ents:
                m, s = ents[name]["normalized_auc_mean"], ents[name]["normalized_auc_sd"]
                ax.axhline(m, color=color, ls=ls, lw=1.5, zorder=1, label=f"{name} (mean ± SD band)")
                if not math.isnan(s):
                    ax.axhspan(m - s, m + s, color=color, alpha=0.10, lw=0, zorder=0)
        if pts:
            x, y, s = map(np.array, zip(*pts))
            ax.errorbar(x, y, yerr=np.nan_to_num(s), color=blue, lw=2, marker="o", ms=5,
                        capsize=3, elinewidth=1, zorder=3, label="synthetic decoder (mean ± SD)")
        ax.set_title(TASK_LABEL[task], fontsize=10, color=ink)
        ax.set_xlabel("decoder AUROC (1.0 = ideal)", fontsize=9, color=ink2)
        ax.set_ylabel("normalized success AUC", fontsize=9, color=ink2)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"{v:.2f}" for v in LEVELS], fontsize=7)
        ax.tick_params(colors=ink2, labelsize=7)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"frontier.{ext}", dpi=200, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/rl_runs/sweep")
    p.add_argument("--out", default="/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/results/rl_frontier")
    p.add_argument("--allow-smoke", action="store_true", help="TESTING ONLY: accept shortened smoke runs")
    a = p.parse_args()
    rows, missing = load_runs(a.runs, a.allow_smoke)
    if not rows:
        raise SystemExit(f"No completed runs under {a.runs}")
    table = summarize(rows)
    out = Path(a.out)
    write_outputs(rows, missing, table, out)
    plot(table, out)


if __name__ == "__main__":
    main()
