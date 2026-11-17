# Reproducing the decoder-improvement study

All commands run from the repository root of branch `decoder-improvement-v1-experiments`. `<DIR>` is any working
directory you choose (a local path argument, not part of the repository). Every command below exists in the delivered
source; nothing needs to be edited.

## 0. Environment

```bash
conda create -y -p <DIR>/env python=3.11
<DIR>/env/bin/python -m pip install -r requirements-lock.txt
# the lock contains nvidia-nccl-cu12 and nvidia-nccl-cu13, which write the same libnccl.so.2; make the CUDA 13 build win:
<DIR>/env/bin/python -m pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.30.7
export PY=<DIR>/env/bin/python ERRP_WEIGHTS_DIR=<DIR>/weights CUBLAS_WORKSPACE_CONFIG=:4096:8
```
Exact lock used: `reports/decoder_improvement/environment_lock.txt` (torch 2.15.0+cu132, NVIDIA driver 580, A100).

## 1. Inputs (external, pinned, verified)

```bash
$PY scripts/decoder_improvement/acquire_inputs.py --out <DIR>            # raw ErrP-HRI @4bc3e524 + CBraMod weights
for b in band_1_20 band_0p3_40 band_0p3_75; do
  $PY scripts/prepare_data.py --raw-root <DIR>/data/raw/dataset-ErrP-HRI \
      --config configs/decoder_improvement/preprocessing_$b.yaml --out-dir <DIR>/prep/$b
done
```

## 2. Tests and diagnostic gate

```bash
$PY -m unittest tests.test_dimp -v
$PY scripts/decoder_improvement/diagnostics.py --prep-root <DIR>/prep --weights-dir <DIR>/weights \
    --work <DIR>/diag_work --out <DIR>/diagnostics.json            # GPU recommended
```

## 3. Re-running one fit or a stage (uses the committed plans)

```bash
FIT=$($PY -c "import json;print(json.load(open('reports/decoder_improvement/plans/stage_A.json'))['fits'][0]['fit_id'])")
$PY scripts/decoder_improvement/run_fit.py --plan reports/decoder_improvement/plans/stage_A.json --fit-id $FIT \
    --runs-root <DIR>/runs --prep-root <DIR>/prep --weights-dir <DIR>/weights --ledger <DIR>/ledger.jsonl
# every fit of a stage:
for f in $($PY -c "import json;print(' '.join(x['fit_id'] for x in json.load(open('reports/decoder_improvement/plans/stage_A.json'))['fits']))"); do
  $PY scripts/decoder_improvement/run_fit.py --plan reports/decoder_improvement/plans/stage_A.json --fit-id $f \
      --runs-root <DIR>/runs --prep-root <DIR>/prep --weights-dir <DIR>/weights --ledger <DIR>/ledger.jsonl
done
$PY scripts/decoder_improvement/study.py summarize --plan reports/decoder_improvement/plans/stage_A.json \
    --runs-root <DIR>/runs --out <DIR>/stage_A.json
```
Plans for later stages are derived from the committed decisions (`study.py plan --stage D --decisions-dir
reports/decoder_improvement/decisions --prep-root <DIR>/prep --out <DIR>/stage_D.json`). Freeze, refits and final
evaluation:

```bash
$PY scripts/decoder_improvement/freeze.py --plan-f reports/decoder_improvement/plans/stage_F.json \
    --decisions-dir reports/decoder_improvement/decisions --runs-root <DIR>/runs --out <DIR>/freeze.json
# Stage G fits: run_fit.py with plans/stage_G.json as above
$PY scripts/decoder_improvement/final_eval.py margins --plan-g reports/decoder_improvement/plans/stage_G.json \
    --runs-root <DIR>/runs --prep-root <DIR>/prep --split validation --out <DIR>/margins_validation
$PY scripts/decoder_improvement/final_eval.py calibrate --plan-g reports/decoder_improvement/plans/stage_G.json \
    --margins <DIR>/margins_validation --freeze reports/decoder_improvement/decisions/freeze.json --out <DIR>/calibration.json
$PY scripts/decoder_improvement/final_eval.py margins --plan-g reports/decoder_improvement/plans/stage_G.json \
    --runs-root <DIR>/runs --prep-root <DIR>/prep --split test --out <DIR>/margins_test \
    --calibration reports/decoder_improvement/decisions/calibration.json
$PY scripts/decoder_improvement/final_eval.py evaluate --plan-g reports/decoder_improvement/plans/stage_G.json \
    --margins <DIR>/margins_test --calibration reports/decoder_improvement/decisions/calibration.json --out <DIR>/test
$PY scripts/decoder_improvement/make_report_tables.py --dec reports/decoder_improvement/decisions \
    --test <DIR>/test --out <DIR>/generated
```

## 4. Using the published checkpoints instead of retraining

```bash
$PY scripts/decoder_improvement/release_tools.py fetch --tag <final Release tag> --out <DIR>/final_release
# reproduce validation/test probabilities from the downloaded final checkpoints and compare with the exports,
# load/probe checkpoints, and resume training for two epochs:
$PY scripts/decoder_improvement/verify_study.py checkpoints --extract <DIR>/final_release/extract --prep-root <DIR>/prep \
    --weights-dir <DIR>/weights --plans reports/decoder_improvement/plans --out <DIR>/verify_ckpt.json --resume-work <DIR>/resume
$PY scripts/decoder_improvement/verify_study.py repro --extract <DIR>/final_release/extract --prep-root <DIR>/prep \
    --plan-g reports/decoder_improvement/plans/stage_G.json --calibration reports/decoder_improvement/decisions/calibration.json \
    --predictions <DIR>/final_release/download --out <DIR>/verify_repro.json
```

## 5. New feedback profile (RL environment, CPU)

```bash
python scripts/download_feedback.py --profile configs/decoder_improvement/rl_feedback_vnext.json --out-dir <DIR>/fb_vnext
VNEXT_DIR=<DIR>/fb_vnext python -m unittest tests.test_feedback_vnext rl.test_stage2
```
