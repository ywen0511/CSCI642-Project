#!/bin/bash
# Stage 1 step 1: freeze w=0.1 and w=0.5 from their restored, complete v2 validation suites (no rerun).
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
cd $C/repo && git fetch -q origin final-v3-experiments && git checkout -q -B final-v3-experiments FETCH_HEAD
echo "HEAD $(git rev-parse HEAD)"
mkdir -p $C/stage1
for w in w010 w050; do
  rm -rf artifacts/final_v3/sensitivity_${w}_frozen
  $PY -m rl.freeze --validation-suite $C/restore/suites/rl_v2_validation/candidate_$w \
    --rationale "Post-hoc weight-sensitivity suite (not selected by the declared v2 rule; selected protocol remains w=1.0). Requested after v2 final results." \
    --out artifacts/final_v3/sensitivity_${w}_frozen && echo "FROZEN $w"
  # consistency with the committed v2 validation analysis (reports/rl_v2_validation/candidate_$w/runs.csv)
  $PY - $w <<'PY'
import csv, json, sys
w = sys.argv[1]
ev = json.load(open(f"artifacts/final_v3/sensitivity_{w}_frozen/validation_evidence.json"))
ref = {r["name"]: r for r in csv.DictReader(open(f"reports/rl_v2_validation/candidate_{w}/runs.csv"))}
bad = [r["name"] for r in ev["runs"] if any(abs(float(ref[r["name"]][k]) - r[k]) > 0 for k in
       ("success_rate", "mean_final_distance", "normalized_auc_success"))]
print("EVIDENCE_MATCHES_COMMITTED_V2_ANALYSIS", w, len(ev["runs"]), "runs, mismatches:", bad)
PY
done
echo FREEZE_DONE
