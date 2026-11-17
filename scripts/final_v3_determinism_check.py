#!/usr/bin/env python
"""Determinism check (FINAL_V3_HANDOFF.md, Section 5 step 2): sparse/dense runs of a new final suite versus the
restored v2 final suite (same seeds; these conditions do not use the EEG weight).

Compared per run: evaluation.json and the parsed evaluation history (evaluations.jsonl) for equality, and the
SHA-256 of transitions.csv, evaluations.jsonl and the final snapshot's model.zip / replay_buffer.pkl /
probe.npz. Any mismatch is reported (not hidden).

Usage: python scripts/final_v3_determinism_check.py --new SUITE --reference V2_FINAL_SUITE --out JSON
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def final_snapshot(run):
    snaps = sorted((run / "checkpoints").glob("final_*"))
    return snaps[-1] if snaps else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    new, ref = Path(a.new), Path(a.reference)
    names = [j["name"] for j in json.loads((new / "plan.json").read_text())["jobs"]
             if j["condition"] in ("sparse", "dense")]
    rows = []
    for name in names:
        n, r = new / name, ref / name
        ev_new = json.loads((n / "evaluation.json").read_text())
        ev_ref = json.loads((r / "evaluation.json").read_text())
        hist_new = [json.loads(x) for x in (n / "evaluations.jsonl").read_text().splitlines() if x.strip()]
        hist_ref = [json.loads(x) for x in (r / "evaluations.jsonl").read_text().splitlines() if x.strip()]
        row = {"run": name, "evaluation_identical": ev_new == ev_ref, "history_identical": hist_new == hist_ref,
               "history_records": len(hist_new),
               "final_success_new": ev_new["success_rate"], "final_success_ref": ev_ref["success_rate"]}
        for f in ("transitions.csv", "evaluations.jsonl"):
            row[f + "_sha256_identical"] = sha(n / f) == sha(r / f)
        sn, sr = final_snapshot(n), final_snapshot(r)
        for f in ("model.zip", "replay_buffer.pkl", "probe.npz"):
            row["final_" + f + "_sha256_identical"] = sn is not None and sr is not None and sha(sn / f) == sha(sr / f)
        row["identical"] = row["evaluation_identical"] and row["history_identical"]
        rows.append(row)
    report = {"new_suite": str(new), "reference_suite": str(ref), "runs": rows, "n_runs": len(rows),
              "all_metrics_and_histories_identical": all(r["identical"] for r in rows),
              "all_transition_logs_identical": all(r["transitions.csv_sha256_identical"] for r in rows),
              "all_final_probes_identical": all(r["final_probe.npz_sha256_identical"] for r in rows),
              "note": "model.zip/replay_buffer.pkl hashes can differ by archive metadata even when weights are equal; "
                      "metrics, evaluation histories, transition logs and inference probes are the determinism criteria"}
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
