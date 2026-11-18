#!/bin/bash
# package_prefval.sh <tool_commit>: rl.package of the 3 preference validation suites (final snapshots + all logs),
# plus one package per candidate with the logs of the externally killed first attempt (no checkpoints; their
# periodic snapshots stay on the filer and are listed with sizes).
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
T=$1
for w in w100 w200 w400; do
  $C/ops/package.sh $T suite $C/runs/pref_validation_$w pref_validation_$w
  A=$C/aborted/pref_validation_${w}_killed_20261118T0805Z; P=$C/packages/pref_validation_${w}_killed_attempt
  rm -rf $P; mkdir -p $P
  tar -C $A --exclude='*/checkpoints' -czf $P/killed_attempt_logs.tar.gz runs logs
  (cd $A/runs && find . -path '*/checkpoints/*' -type f -printf '%s %p\n' | sort -k2) > $P/killed_attempt_checkpoints_on_filer.txt
  cat > $P/README.txt <<TXT
First attempt of pref_validation_$w (launched 2026-11-18 06:20Z from ae633ea), killed by a signal external to LSF
at 08:04-08:05Z at 260k-290k of 300k steps (no status.json / final evaluation). Not used for any result; all runs
were rerun unchanged (main package of this Release). Contents: run directories without checkpoints (configs,
progress, evaluations, transition logs, monitor) and LSF job logs. Periodic snapshots remain on the project filer
($A), listed with sizes in killed_attempt_checkpoints_on_filer.txt.
TXT
  (cd $P && sha256sum killed_attempt_logs.tar.gz killed_attempt_checkpoints_on_filer.txt README.txt > SHA256SUMS)
  echo "KILLED_PKG $w $(du -sh $P | cut -f1)"
done
echo PACKAGE_PREFVAL_DONE
