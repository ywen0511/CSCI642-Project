"""Tables, learning curves, feedback diagnostics and figures for a complete suite.

Usage: python -m rl.analysis <suite_dir> --feedback-dir artifacts/rl_feedback --out <new dir>

Aggregation rule (declared in reports/rl_v2_experiment_plan.md):
- RL-seed mean and SD within each (condition, participant).
- EEG-assisted conditions: unweighted macro-average over participants of those seed means.
- Shared controls (sparse/dense/ideal) are trained once per seed and counted once; they are
  not three participant observations.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import load_curve
from .suite import SHARED
from .summarize import summarize

METRICS = ("success_rate", "ever_success_rate", "mean_final_distance", "mean_task_return", "normalized_auc_success")
MAIN = ("sparse", "dense", "eegnet", "cbramod")
DIAGNOSTIC = ("ideal", "independent_eegnet", "independent_cbramod")
ORDER = MAIN + DIAGNOSTIC
COLORS = {"sparse": "#7f7f7f", "dense": "#1f77b4", "ideal": "#2ca02c", "eegnet": "#d62728",
          "cbramod": "#9467bd", "independent_eegnet": "#ff9896", "independent_cbramod": "#c5b0d5"}


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties; nan if a class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def decoder_metrics(feedback_dir, split):
    """Fixed EEG decoder quality per participant (calibrated p_error, threshold 0.5)."""
    path = Path(feedback_dir) / f"{'validation' if split == 'validation' else 'test'}_predictions.csv"
    groups = defaultdict(list)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            groups[(r["model"], r["subject_id"])].append((float(r["p_error"]), int(r["label"])))
    out = []
    for (model, subject), rows in sorted(groups.items()):
        p = np.array([x for x, _ in rows])
        y = np.array([t for _, t in rows])
        pred = p >= 0.5
        bal = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
        out.append({"model": model, "subject": subject, "n": len(y), "error_rate": float(y.mean()),
                    "auroc": auroc(p, y), "balanced_accuracy": float(bal),
                    "brier": float(np.mean((p - y) ** 2)),
                    "mean_p_error_label0": float(p[y == 0].mean()), "mean_p_error_label1": float(p[y == 1].mean())})
    return out


def feedback_diagnostics(run_dir, bin_steps):
    """Feedback availability, label balance, probabilities and reward components from transitions.csv."""
    seg = elig = events = 0
    labels, probs, eeg, dense, task = [], [], [], [], []
    bins = defaultdict(lambda: {"segment_end": 0, "eligible": 0, "label1": 0, "eeg_reward": 0.0})
    episodes = set()
    with (Path(run_dir) / "transitions.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            step = int(r["step"])
            b = bins[(step - 1) // bin_steps]
            episodes.add(r["episode"])
            task.append(float(r["task_reward"]))
            dense.append(float(r["dense_reward"]))
            if r["segment_end"] == "1":
                seg += 1
                b["segment_end"] += 1
            if r["feedback_eligible"] == "1":
                elig += 1
                b["eligible"] += 1
                label = int(r["simulated_label"])
                b["label1"] += label
                labels.append(label)
            if r["feedback_event"] == "1":
                events += 1
                probs.append(float(r["p_error"]))
                eeg.append(float(r["eeg_reward"]))
                b["eeg_reward"] += float(r["eeg_reward"])
    labels_a = np.array(labels, int)
    probs_a = np.array(probs, float)
    out = {"steps": len(task), "episodes": len(episodes), "segment_ends": seg, "eligible": elig,
           "feedback_events": events, "neutral_fraction": 1 - elig / seg if seg else float("nan"),
           "error_label_fraction": float(labels_a.mean()) if len(labels_a) else float("nan"),
           "events_per_episode": events / len(episodes) if episodes else float("nan"),
           "task_reward_sum": float(np.sum(task)), "dense_reward_sum": float(np.sum(dense)),
           "eeg_reward_sum": float(np.sum(eeg)), "eeg_reward_mean_per_event": float(np.mean(eeg)) if eeg else 0.0}
    if events:
        if len(probs_a) != len(labels_a):
            raise ValueError(f"Events and eligible labels differ: {run_dir}")
        out.update({"mean_p_error_label0": float(probs_a[labels_a == 0].mean()) if (labels_a == 0).any() else float("nan"),
                    "mean_p_error_label1": float(probs_a[labels_a == 1].mean()) if (labels_a == 1).any() else float("nan"),
                    "event_auroc": auroc(probs_a, labels_a),
                    "mean_eeg_reward_label0": float(np.mean(np.array(eeg)[labels_a == 0])) if (labels_a == 0).any() else float("nan"),
                    "mean_eeg_reward_label1": float(np.mean(np.array(eeg)[labels_a == 1])) if (labels_a == 1).any() else float("nan")})
    timeline = [{"bin_start_step": k * bin_steps, **v,
                 "neutral_fraction": 1 - v["eligible"] / v["segment_end"] if v["segment_end"] else float("nan"),
                 "error_label_fraction": v["label1"] / v["eligible"] if v["eligible"] else float("nan")}
                for k, v in sorted(bins.items())]
    return out, timeline


def aggregate(rows):
    """Seed mean/SD per (condition, subject), then participant macro-average for EEG conditions."""
    by_cs = defaultdict(list)
    for r in rows:
        by_cs[(r["condition"], r["subject"] or "shared")].append(r)
    within = []
    for (condition, subject), values in sorted(by_cs.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        item = {"condition": condition, "subject": subject, "n_seeds": len(values),
                "seeds": sorted(v["seed"] for v in values)}
        for m in METRICS:
            v = [x[m] for x in values]
            item[m + "_mean"], item[m + "_sd"] = statistics.mean(v), _sd(v)
        within.append(item)
    overall = []
    for condition in ORDER:
        items = [w for w in within if w["condition"] == condition]
        if not items:
            continue
        runs = [r for r in rows if r["condition"] == condition]
        entry = {"condition": condition, "n_participants": 0 if condition in SHARED else len(items),
                 "n_runs": len(runs),
                 "aggregation": "mean/SD over RL seeds" if condition in SHARED else
                 "unweighted macro-average over participants of seed means; SD over participant means"}
        for m in METRICS:
            if condition in SHARED:
                v = [r[m] for r in runs]
            else:
                v = [w[m + "_mean"] for w in items]
            entry[m + "_mean"], entry[m + "_sd"] = statistics.mean(v), _sd(v)
            entry[m + "_min"], entry[m + "_max"] = min(v), max(v)
        overall.append(entry)
    return within, overall


def paired_differences(rows, a, b, metric="success_rate"):
    """Per (participant, seed) differences a - b; shared controls are matched by seed only."""
    index = {(r["condition"], r["subject"], r["seed"]): r[metric] for r in rows}
    out = []
    for (cond, subject, seed), value in sorted(index.items(), key=lambda kv: (str(kv[0][1]), kv[0][2])):
        if cond != a:
            continue
        other = index.get((b, subject, seed), index.get((b, None, seed)))
        if other is not None:
            out.append({"subject": subject or "shared", "seed": seed, "a": a, "b": b, "metric": metric,
                        "difference": value - other})
    return out


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_all(out, curves, rows, within, overall, timelines, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_curve(condition, metric, subject=None):
        series = defaultdict(list)
        for c in curves:
            if c["condition"] == condition and (subject is None or c["subject"] == subject):
                series[(c["subject"], c["seed"])].append((c["step"], c[metric]))
        if not series:
            return None
        if condition not in SHARED and subject is None:
            # participant macro: average seeds within participant, then participants
            per_subject = defaultdict(list)
            for (s, _), pts in series.items():
                per_subject[s].append([v for _, v in sorted(pts)])
            steps = [st for st, _ in sorted(next(iter(series.values())))]
            subj_means = np.array([np.mean(v, axis=0) for v in per_subject.values()])
            return np.array(steps), subj_means.mean(0), subj_means.std(0, ddof=1) if len(subj_means) > 1 else 0 * subj_means[0]
        mat = np.array([[v for _, v in sorted(pts)] for pts in series.values()])
        steps = [st for st, _ in sorted(next(iter(series.values())))]
        return np.array(steps), mat.mean(0), mat.std(0, ddof=1) if len(mat) > 1 else 0 * mat[0]

    for metric, label, fname in (("success_rate", "final-step success rate", "curves_success"),
                                 ("mean_final_distance", "final distance to goal (m)", "curves_distance")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
        for ax, group, title in ((axes[0], MAIN, "Main comparison"),
                                 (axes[1], ("sparse", "ideal") + DIAGNOSTIC[1:] + ("eegnet", "cbramod"),
                                  "Mechanism diagnostics")):
            for condition in group:
                res = mean_curve(condition, metric)
                if res is None:
                    continue
                steps, m, s = res
                ax.plot(steps, m, color=COLORS[condition], label=condition,
                        ls="--" if condition.startswith("independent") else "-")
                ax.fill_between(steps, m - s, m + s, color=COLORS[condition], alpha=0.15)
            ax.set_title(title)
            ax.set_xlabel("environment steps")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel(label + "\n(EEG: participant macro-average; band = SD)")
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=150)
        fig.savefig(out / f"{fname}.pdf")
        plt.close(fig)

    subjects = sorted({c["subject"] for c in curves if c["subject"]})
    if subjects:
        fig, axes = plt.subplots(1, len(subjects), figsize=(4.2 * len(subjects), 3.8), sharey=True, squeeze=False)
        for ax, subject in zip(axes[0], subjects):
            for condition in ORDER:
                if condition in SHARED:
                    res = mean_curve(condition, "success_rate")
                else:
                    res = mean_curve(condition, "success_rate", subject)
                if res is None:
                    continue
                steps, m, s = res
                ax.plot(steps, m, color=COLORS[condition], label=condition,
                        ls="--" if condition.startswith("independent") else "-")
                ax.fill_between(steps, m - s, m + s, color=COLORS[condition], alpha=0.1)
            ax.set_title(f"feedback participant {subject}")
            ax.set_xlabel("environment steps")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("final-step success (mean ± SD over RL seeds)")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "curves_success_by_participant.png", dpi=150)
        fig.savefig(out / "curves_success_by_participant.pdf")
        plt.close(fig)

    # final success per run (every seed and participant shown)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, condition in enumerate(ORDER):
        rs = [r for r in rows if r["condition"] == condition]
        if not rs:
            continue
        xs = i + np.linspace(-0.25, 0.25, len(rs)) if len(rs) > 1 else [i]
        ax.scatter(xs, [r["success_rate"] for r in rs], color=COLORS[condition], s=18)
        o = next(x for x in overall if x["condition"] == condition)
        ax.hlines(o["success_rate_mean"], i - 0.3, i + 0.3, color="k")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel("final-step success rate (each dot = one run)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "final_success_runs.png", dpi=150)
    fig.savefig(out / "final_success_runs.pdf")
    plt.close(fig)

    # feedback availability over training
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for condition in ("ideal", "eegnet", "cbramod", "independent_eegnet", "independent_cbramod"):
        tl = [t for t in timelines if t["condition"] == condition]
        if not tl:
            continue
        steps = sorted({t["bin_start_step"] for t in tl})
        for ax, key in zip(axes, ("neutral_fraction", "error_label_fraction")):
            ys = [np.nanmean([t[key] for t in tl if t["bin_start_step"] == s]) for s in steps]
            ax.plot(steps, ys, color=COLORS[condition], label=condition,
                    ls="--" if condition.startswith("independent") else "-")
    axes[0].set_ylabel("neutral (deadband) fraction of segment ends")
    axes[1].set_ylabel("error-label fraction of labeled segments")
    for ax in axes:
        ax.set_xlabel("environment steps (bin start)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "feedback_availability.png", dpi=150)
    fig.savefig(out / "feedback_availability.pdf")
    plt.close(fig)


def analyze(suite, feedback_dir, out, bin_steps=10000, plots=True):
    suite, out = Path(suite), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary = summarize(suite)
    budget = summary["config"]["total_timesteps"]
    rows = summary["runs"]
    curves, diagnostics, timelines = [], [], []
    for r in rows:
        run = suite / r["name"]
        for rec in load_curve(run / "evaluations.jsonl"):
            curves.append({"condition": r["condition"], "subject": r["subject"], "seed": r["seed"],
                           "step": rec["steps"], **{m: rec[m] for m in METRICS if m in rec}})
        d, tl = feedback_diagnostics(run, bin_steps)
        diagnostics.append({"condition": r["condition"], "subject": r["subject"], "seed": r["seed"], **d})
        timelines += [{"condition": r["condition"], "subject": r["subject"], "seed": r["seed"], **t} for t in tl]
    within, overall = aggregate(rows)
    decoders = decoder_metrics(feedback_dir, summary["split"])
    # relate fixed decoder quality to RL outcomes per participant (descriptive only)
    relation = []
    for d in decoders:
        if d["subject"] == "pooled":
            continue
        for cond in (d["model"], "independent_" + d["model"]):
            w = next((x for x in within if x["condition"] == cond and x["subject"] == d["subject"]), None)
            if w:
                relation.append({"decoder": d["model"], "condition": cond, "subject": d["subject"],
                                 "decoder_auroc": d["auroc"], "decoder_balanced_accuracy": d["balanced_accuracy"],
                                 "decoder_brier": d["brier"], "rl_success_mean": w["success_rate_mean"],
                                 "rl_success_sd": w["success_rate_sd"], "rl_auc_mean": w["normalized_auc_success_mean"],
                                 "rl_final_distance_mean": w["mean_final_distance_mean"]})
    paired = []
    for a, b in (("eegnet", "sparse"), ("cbramod", "sparse"), ("eegnet", "cbramod"), ("dense", "sparse"),
                 ("ideal", "sparse"), ("eegnet", "independent_eegnet"), ("cbramod", "independent_cbramod")):
        for metric in ("success_rate", "normalized_auc_success"):
            paired += paired_differences(rows, a, b, metric)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "by_condition_participant.csv", within)
    write_csv(out / "by_condition.csv", overall)
    write_csv(out / "curves.csv", curves)
    write_csv(out / "feedback_diagnostics.csv", diagnostics)
    write_csv(out / "feedback_timeline.csv", timelines)
    write_csv(out / "decoder_metrics.csv", decoders)
    write_csv(out / "decoder_vs_rl.csv", relation)
    write_csv(out / "paired_differences.csv", paired)
    result = {"suite": str(suite), "split": summary["split"], "budget": budget, "n_runs": len(rows),
              "training_git_shas": summary["training_git_shas"], "by_condition": overall,
              "by_condition_participant": within, "decoder_metrics": decoders, "decoder_vs_rl": relation}
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    if plots:
        plot_all(out, curves, rows, within, overall, timelines, budget)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite")
    p.add_argument("--feedback-dir", default="artifacts/rl_feedback")
    p.add_argument("--out", required=True)
    p.add_argument("--bin-steps", type=int, default=10000)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    res = analyze(a.suite, a.feedback_dir, a.out, a.bin_steps, not a.no_plots)
    print(json.dumps(res["by_condition"], indent=2))
