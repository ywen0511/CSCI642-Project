#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
cd $C/dev; rm -rf /tmp/kz_geo2
$PY scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_candidates/pref_w100.json --out /tmp/kz_geo2 2>&1 | grep -v Adroit | head -45
$PY -c "
import csv;r=list(csv.DictReader(open(\"/tmp/kz_geo2/per_goal.csv\")))
print([x[\"seed\"] for x in r if x[\"straight_violation_free\"]==\"True\"])" 2>&1
echo DEVGEO_DONE
