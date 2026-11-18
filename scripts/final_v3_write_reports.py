#!/usr/bin/env python
"""Generate the final-v3 result tables (markdown) from committed rl.analysis outputs; no number is typed by hand.

Every number written into a table is also recorded in an audit file (number -> source file, row key, column),
like the decoder branch's number audit.

Usage:
  python scripts/final_v3_write_reports.py sensitivity --out reports/final_v3/generated/sensitivity_tables.md
  python scripts/final_v3_write_reports.py v3 --out reports/final_v3/generated/v3_tables.md
  python scripts/final_v3_write_reports.py preference --out reports/final_v3/generated/preference_tables.md
(each writes <out> and <out without .md>_audit.json; inputs are the fixed paths below)
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
ORDER = ["sparse", "dense", "eegnet", "cbramod", "ideal", "independent_eegnet", "independent_cbramod"]
PAIRS = [("eegnet", "sparse"), ("cbramod", "sparse"), ("eegnet", "cbramod"), ("dense", "sparse"),
         ("ideal", "sparse"), ("eegnet", "independent_eegnet"), ("cbramod", "independent_cbramod"),
         ("independent_eegnet", "sparse"), ("independent_cbramod", "sparse")]


class Doc:
    def __init__(self):
        self.lines, self.audit = [], []

    def num(self, value, source, key, column, digits=3, signed=False):
        if value in ("", None) or value != value:
            return "n/a"
        text = f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"
        self.audit.append({"text": text, "value": float(value), "source": str(Path(source).relative_to(ROOT)),
                           "key": key, "column": column})
        return text

    def add(self, *lines):
        self.lines.extend(lines)

    def write(self, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.lines) + "\n")
        out.with_name(out.stem + "_audit.json").write_text(json.dumps(self.audit, indent=1) + "\n")
        print(out, len(self.audit), "audited numbers")


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def by_condition(d, analysis_dir, label, pref=False):
    path = Path(analysis_dir) / "by_condition.csv"
    data = {r["condition"]: r for r in rows(path)}
    head = "| condition | runs | final success | final distance (m) | normalized success AUC |"
    sep = "|---|---|---|---|---|"
    if pref:
        head += " preferred pass | preferred-pass AUC | violation-free | pillar entry | right corridor |"
        sep += "---|---|---|---|---|"
    d.add(f"**{label}**", "", head, sep)
    for c in ORDER:
        r = data.get(c)
        if r is None:
            continue
        k = f"{label}:{c}"
        line = (f"| {c} | {r['n_runs']} | {d.num(r['success_rate_mean'], path, k, 'success_rate_mean')} ± "
                f"{d.num(r['success_rate_sd'], path, k, 'success_rate_sd')} | "
                f"{d.num(r['mean_final_distance_mean'], path, k, 'mean_final_distance_mean', 4)} | "
                f"{d.num(r['normalized_auc_success_mean'], path, k, 'normalized_auc_success_mean')} ± "
                f"{d.num(r['normalized_auc_success_sd'], path, k, 'normalized_auc_success_sd')} |")
        if pref:
            line += (f" {d.num(r['preferred_pass_rate_mean'], path, k, 'preferred_pass_rate_mean')} ± "
                     f"{d.num(r['preferred_pass_rate_sd'], path, k, 'preferred_pass_rate_sd')} | "
                     f"{d.num(r['normalized_auc_preferred_pass_mean'], path, k, 'normalized_auc_preferred_pass_mean')} | "
                     f"{d.num(r['violation_free_rate_mean'], path, k, 'violation_free_rate_mean')} | "
                     f"{d.num(r['pillar_entry_rate_mean'], path, k, 'pillar_entry_rate_mean')} | "
                     f"{d.num(r['right_corridor_entry_rate_mean'], path, k, 'right_corridor_entry_rate_mean')} |")
        d.add(line)
    d.add("")


def paired_from_runs(path, pairs, metrics):
    """Same pairing as rl.analysis.paired_differences, from runs.csv (used when an older analysis lacks a pair)."""
    index = {(r["condition"], r["subject"], r["seed"]): r for r in rows(path)}
    g = defaultdict(list)
    for (cond, subject, seed), r in sorted(index.items()):
        for a, b in pairs:
            if cond != a:
                continue
            other = index.get((b, subject, seed), index.get((b, "", seed)))
            if other is not None:
                for m in metrics:
                    g[(a, b, m)].append(float(r[m]) - float(other[m]))
    return g


def paired(d, analysis_dir, label, metrics, pairs=PAIRS):
    path = Path(analysis_dir) / "paired_differences.csv"
    g = defaultdict(list)
    for r in rows(path):
        g[(r["a"], r["b"], r["metric"])].append(float(r["difference"]))
    missing = [(a, b) for a, b in pairs if not any((a, b, m) in g for m in metrics)]
    if missing:
        path = Path(analysis_dir) / "runs.csv"
        g.update(paired_from_runs(path, missing, metrics))
    d.add(f"**{label}**", "", "| comparison | metric | pairs | mean difference | SD | first better / worse / equal |",
          "|---|---|---|---|---|---|")
    for a, b in pairs:
        for m in metrics:
            v = g.get((a, b, m))
            if not v:
                continue
            pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
            k = f"{label}:{a}-{b}:{m}"
            d.add(f"| {a} - {b} | {m} | {len(v)} | {d.num(statistics.mean(v), path, k, 'mean(difference)', 3, True)} | "
                  f"{d.num(statistics.stdev(v) if len(v) > 1 else float('nan'), path, k, 'sd(difference)')} | "
                  f"{pos} / {neg} / {len(v) - pos - neg} |")
    d.add("")


def diagnostics(d, analysis_dir, label):
    path = Path(analysis_dir) / "feedback_diagnostics.csv"
    g = defaultdict(list)
    for r in rows(path):
        g[r["condition"]].append(r)
    d.add(f"**{label}**", "", "| condition | error-label fraction | feedback events / episode | EEG reward per event | "
          "EEG reward per run | mean p_error (label 0 / 1) | event AUROC |", "|---|---|---|---|---|---|---|")
    for c in ORDER:
        rs = g.get(c)
        if not rs or c in ("sparse", "dense"):
            continue

        def m(col):
            vals = [float(x[col]) for x in rs if x.get(col) not in ("", None)]
            return statistics.mean(vals) if vals else float("nan")
        k = f"{label}:{c}"
        d.add(f"| {c} | {d.num(m('error_label_fraction'), path, k, 'mean error_label_fraction')} | "
              f"{d.num(m('events_per_episode'), path, k, 'mean events_per_episode', 2)} | "
              f"{d.num(m('eeg_reward_mean_per_event'), path, k, 'mean eeg_reward_mean_per_event', 4, True)} | "
              f"{d.num(m('eeg_reward_sum'), path, k, 'mean eeg_reward_sum', 1, True)} | "
              f"{d.num(m('mean_p_error_label0'), path, k, 'mean mean_p_error_label0')} / "
              f"{d.num(m('mean_p_error_label1'), path, k, 'mean mean_p_error_label1')} | "
              f"{d.num(m('event_auroc'), path, k, 'mean event_auroc')} |")
    d.add("")


def cross_suite(d, a_dir, b_dir, label_a, label_b, metrics=("success_rate", "normalized_auc_success")):
    """Per condition: run-level differences a - b matched by (participant, seed) across two suites."""
    pa, pb = Path(a_dir) / "runs.csv", Path(b_dir) / "runs.csv"
    A = {(r["condition"], r["subject"], r["seed"]): r for r in rows(pa)}
    B = {(r["condition"], r["subject"], r["seed"]): r for r in rows(pb)}
    d.add(f"| condition | metric | pairs | {label_a} mean | {label_b} mean | mean difference ({label_a} - {label_b}) | "
          "better / worse / equal |", "|---|---|---|---|---|---|---|")
    for c in ORDER:
        keys = sorted(k for k in A if k[0] == c and k in B)
        if not keys:
            continue
        for m in metrics:
            va = [float(A[k][m]) for k in keys]
            vb = [float(B[k][m]) for k in keys]
            diff = [x - y for x, y in zip(va, vb)]
            pos, neg = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
            k = f"{label_a}-{label_b}:{c}:{m}"
            d.add(f"| {c} | {m} | {len(keys)} | {d.num(statistics.mean(va), pa, k, m)} | {d.num(statistics.mean(vb), pb, k, m)} | "
                  f"{d.num(statistics.mean(diff), pa, k, 'mean difference', 3, True)} | {pos} / {neg} / {len(diff) - pos - neg} |")
    d.add("")


def selection_table(d, selection_json, primary, label):
    path = Path(selection_json)
    s = json.loads(path.read_text())
    d.add(f"**{label}** (rule: {s.get('reason', '')}; selected EEG weight {s['selected_eeg_weight']})", "",
          "| candidate | EEG weight | mean final success (12 runs) | mean normalized AUC (12 runs) | mean preferred pass (12 runs) |",
          "|---|---|---|---|---|")
    for c in sorted(s["candidates"], key=lambda c: c["eeg_weight"]):
        k = f"{label}:w={c['eeg_weight']}"
        pp = c.get("selection_preferred_pass")
        d.add(f"| `{Path(c['suite']).name}` | {c['eeg_weight']} | {d.num(c['selection_success'], path, k, 'selection_success', 4)} | "
              f"{d.num(c['selection_auc'], path, k, 'selection_auc', 4)} | "
              f"{'n/a' if pp is None else d.num(pp, path, k, 'selection_preferred_pass', 4)} |")
    d.add("")


def sensitivity(out):
    d = Doc()
    F = {"0.1": R / "final_v3/sensitivity_w010_final", "0.5": R / "final_v3/sensitivity_w050_final", "1.0": R / "rl_v2_final"}
    d.add("# Weight sensitivity (w = 0.1, 0.5, 1.0): generated tables", "",
          "Generated by `scripts/final_v3_write_reports.py sensitivity` from `rl.analysis` outputs (w = 1.0: the v2 "
          "final suite, `reports/rl_v2_final/`). **Status: post-hoc sensitivity analysis requested after the v2 final "
          "results were seen. The v2 primary result (w = 1.0) is unchanged; these suites are not a re-selection.** "
          "Shared controls: mean ± SD over 5 RL seeds; EEG conditions: macro-average over participants s02/s08/s09 of "
          "RL-seed means, ± SD over participants.", "", "## Final comparison by EEG weight", "")
    for w, p in F.items():
        by_condition(d, p, f"w = {w}")
    d.add("## Paired differences (per participant and RL seed; shared controls matched by seed)", "")
    for w, p in F.items():
        paired(d, p, f"w = {w}", ("success_rate", "normalized_auc_success"), PAIRS[:3] + PAIRS[5:7])
    d.add("## Feedback diagnostics during training (mean over runs)", "")
    for w, p in F.items():
        diagnostics(d, p, f"w = {w}")
    d.add("## Validation (v2, participants s11/s13, seeds 100-102)", "")
    selection_table(d, ROOT / "configs/rl_v2/frozen/selection.json", "selection_success", "v2 validation candidates")
    det = R / "final_v3/sensitivity_determinism.json"
    if det.exists():
        dj = json.loads(det.read_text())
        d.add("## Determinism check (sparse/dense vs the v2 final runs)", "", "| suite | runs | metrics and evaluation "
              "histories identical | transition logs identical | final inference probes identical |", "|---|---|---|---|---|")
        for name, rep in dj.items():
            d.add(f"| {name} | {rep['n_runs']} | {rep['all_metrics_and_histories_identical']} | "
                  f"{rep['all_transition_logs_identical']} | {rep['all_final_probes_identical']} |")
        d.add("")
    d.write(out)


def v3(out):
    d = Doc()
    V = R / "final_v3/v3_validation"
    F3, F2 = R / "final_v3/v3_final", R / "rl_v2_final"
    d.add("# Protocol v3 (centered EEG reward): generated tables", "",
          "Generated by `scripts/final_v3_write_reports.py v3`. **Status: post-hoc follow-up with its own declared "
          "validation selection and freeze; test participants s02/s08/s09 were already used in v2, so the v3 final "
          "suite is a follow-up evaluation, not an untouched confirmatory test.**", "",
          "## Validation candidates (s11/s13, seeds 100-102)", "")
    if (V / "selection.json").exists():
        selection_table(d, V / "selection.json", "selection_auc", "v3 validation candidates")
    for cand in sorted(p for p in V.iterdir() if p.is_dir()) if V.exists() else []:
        by_condition(d, cand, cand.name)
    if F3.exists():
        d.add("## Final comparison (s02/s08/s09, seeds 0-4, selected weight)", "")
        by_condition(d, F3, "v3 final")
        paired(d, F3, "v3 final", ("success_rate", "normalized_auc_success"))
        d.add("## Feedback diagnostics: v3 final vs v2 final (w = 1.0)", "")
        diagnostics(d, F3, "v3 final")
        diagnostics(d, F2, "v2 final (w = 1.0)")
        d.add("## Direct comparison v3 final - v2 final (same participants and seeds)", "")
        cross_suite(d, F3, F2, "v3", "v2")
        d.add("## Label-independent controls relative to sparse", "")
        paired(d, F3, "v3 final", ("success_rate", "normalized_auc_success"), PAIRS[7:])
        paired(d, F2, "v2 final (w = 1.0)", ("success_rate", "normalized_auc_success"), PAIRS[7:])
    d.write(out)


def preference(out):
    d = Doc()
    V = R / "final_v3/preference/validation"
    F = R / "final_v3/preference/final"
    d.add("# Route-preference extension: generated tables", "",
          "Generated by `scripts/final_v3_write_reports.py preference`. **Status: extension (milestone preference "
          "item), with its own pilots, declared validation selection and freeze; evaluated on test participants "
          "already used in v2/v3.** Preferred pass = task success, no violation at any step, and the gripper passes "
          "a = 0.5 on the left of the pillar.", "")
    pr = R / "final_v3/preference/pilot_report.json"
    if pr.exists():
        rep = json.loads(pr.read_text())
        d.add("## Pilots (validation participants, seeds 900/901, w = 1.0) and the declared gate", "",
              f"Gate: {rep['gate_rule']}. Final-policy rates per pilot run (seed 900 / 901).", "",
              "| variant | condition | preferred pass | task success | violation-free | pillar entry | right corridor |",
              "|---|---|---|---|---|---|---|")
        for vname, v in rep["variants"].items():
            by = defaultdict(dict)
            for rname, x in v["runs"].items():
                by[x["condition"]][x["seed"]] = (rname, x)
            for cond in ORDER:
                if cond not in by:
                    continue
                cells = []
                for k in ("preferred_pass_rate", "success_rate", "violation_free_rate", "pillar_entry_rate",
                          "right_corridor_entry_rate"):
                    cells.append(" / ".join(d.num(x.get(k), pr, f"{vname}/{rn}", k, 2) for s, (rn, x) in sorted(by[cond].items())))
                d.add(f"| {vname} | {cond} | " + " | ".join(cells) + " |")
        d.add("", "| variant | ideal mean | sparse mean | difference | ideal > sparse on every seed | gate passed |",
              "|---|---|---|---|---|---|")
        for vname, v in rep["variants"].items():
            g = v["gate"]
            if g:
                d.add(f"| {vname} | {d.num(g['ideal_mean'], pr, vname, 'gate.ideal_mean')} | "
                      f"{d.num(g['sparse_mean'], pr, vname, 'gate.sparse_mean')} | "
                      f"{d.num(g['difference'], pr, vname, 'gate.difference', signed=True)} | "
                      f"{g['ideal_better_every_seed']} | **{g['passed']}** |")
        d.add("")
    if (V / "selection.json").exists():
        d.add("## Validation candidates (s11/s13, seeds 100-102)", "")
        selection_table(d, V / "selection.json", "selection_preferred_pass", "preference validation candidates")
    for cand in sorted(p for p in V.iterdir() if p.is_dir()) if V.exists() else []:
        by_condition(d, cand, cand.name, pref=True)
    if F.exists():
        d.add("## Final comparison (s02/s08/s09, seeds 0-4, selected weight)", "")
        by_condition(d, F, "preference final", pref=True)
        paired(d, F, "preference final", ("preferred_pass_rate", "normalized_auc_preferred_pass", "violation_free_rate",
                                          "success_rate", "normalized_auc_success"))
        diagnostics(d, F, "preference final")
    d.write(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["sensitivity", "v3", "preference"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    {"sensitivity": sensitivity, "v3": v3, "preference": preference}[a.which](a.out)
