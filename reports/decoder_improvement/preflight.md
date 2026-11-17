# Decoder-improvement study: preflight

Date: 2026-11-11 (UTC). Planning commit `c459bd52a5536cd22628b37e04e635b11e270eaa` (`decoder-improvement-v1`);
execution branch `decoder-improvement-v1-experiments` (worktree from that commit; no clone, connected history).
Every run records its exact source commit and the SHA-256 of its training files.

## Repository and access
- GitHub repository `ywen0511/CSCI642-Project`, account `zhengtaoyao` with push access (verified by pushing the
  execution branch). Release upload: draft-first with digest round trip (`scripts/decoder_improvement/upload_release.py`);
  the first real round trip is the recovery/smoke publication logged in `upload_ledger.jsonl`. No token is printed or stored.

## Compute and scheduler
- GM Warren LSF. GPU jobs: queue `gpu_large_shared`, hosts gc175 or gc176 (4x NVIDIA A100-SXM4-40GB each, driver
  580.126.20), exactly one GPU per job, at most four concurrent jobs, each job runs its fits sequentially and exits.
  gc175's four GPUs were reserved in LSF by another user's job since 2026-11-09 (physically idle), so the gate and
  the stages ran on gc176 (same hardware); gc216/gc217 belong to other active work and are not used.
- CPU jobs (data preparation, audit, acquisition test, packaging): queue `cpu_ubuntu_wrn`.

## Environment
- New conda environment (Python 3.11) at `<PROJECT_ROOT>/env`, installed from the exact historical lock
  `requirements-lock.txt`; torch 2.15.0+cu132, CUDA 13.2, mne 1.13.2, numpy 2.2.6, scipy 1.17.1, scikit-learn 1.9.1.
  Full lock: [`environment_lock.txt`](environment_lock.txt).
- Fix recorded: the lock lists `nvidia-nccl-cu12==2.21.5` and `nvidia-nccl-cu13==2.30.7`, which write the same file
  `nvidia/nccl/lib/libnccl.so.2`; the install order left the cu12 library and `import torch` failed
  (`undefined symbol: ncclCommResume`). Reinstalling `nvidia-nccl-cu13==2.30.7 --no-deps` restored a library
  byte-identical to the historical environment (SHA-256 618968ee...); `pip freeze` is unchanged.
- The historical EEG environment was not modified.

## Inputs
- Raw ErrP-HRI at commit `4bc3e52469934aa0551a2261a6e4e899d156c28f`: reused the verified historical acquisition
  (read-only). A fresh acquisition with `scripts/decoder_improvement/acquire_inputs.py` into an empty directory
  verified all 49 files (size + SHA-256) and the pinned weights; regenerating
  the 1-20 Hz band from that fresh copy reproduced every prepared file byte for byte
  ([`acquisition_test_report.json`](acquisition_test_report.json)).
- CBraMod initialization: HF `weighting666/CBraMod` revision `500543c7e30bda1b22bfd51a49301b238dee21fd`,
  `pretrained_weights.pth`, 19,775,842 bytes, SHA-256 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`
  (identical to the historical acquisition record).
- Prepared bands (new IDs; `prep_v1` untouched): `1-20` -> `dimp_band_1_20-9bb0159f0ffd`, `0.3-40` -> `dimp_band_0p3_40-bef6c26cca50`, `0.3-75` -> `dimp_band_0p3_75-2eed075a6d61`.
  The new 1-20 Hz arrays are byte-identical to the historical `prep_v1` arrays.

## Diagnostic gate (all passed; [`diagnostics/diagnostics.json`](diagnostics/diagnostics.json))
- 1_data_identity_and_splits: PASS
- 2_voltages_shapes_resampling: PASS
- 3_official_weights: PASS
- 4_trainability_masks: PASS
- 5_frozen_module_modes: PASS
- 6_overfit_training_subset: PASS
- 7_reload_and_resume: PASS
- 8_no_test_or_outer_validation_in_selection: PASS

Key facts: 5,822 retained trials (3,767 / 2,055); outer split 3,434 / 959 / 1,429; inner folds disjoint and each
training participant held out once; official weights: 211 tensors, no missing/unexpected/shape-mismatched keys,
reconstruction head (`proj_out`) removed; backbone 4,883,800 parameters; heads 4,401,002 (historical flatten MLP)
and 10,802 (temporal-mean linear); trainable parameters: flat frozen 4,401,002, flat last-2 5,206,202, flat full
9,284,602; frozen tensors byte-identical after an optimizer step; frozen modules stay in eval mode; a 64-trial
training subset is overfit by the reference CBraMod and EEGNet (and, disclosed, by the frozen flat head); reload
reproduces logits exactly; resuming an interrupted fit gives bitwise-identical weights.

## Storage
- Project filer quota for this account: 1 TB in total. At preflight the account used ~535 GB after other
  projects were purged. Measured per fit: ~0.7 GB (CBraMod flat head, all best/last/periodic checkpoints),
  smaller for the temporal-mean head and EEGNet. Estimated study total <= 120 GB plus packaging copies.

## Benchmark (first Stage A/C fits on one A100 40GB)
- CBraMod historical recipe, one inner fold: ~60-75 s training (22-27 epochs with early stopping), plus data
  loading, hashing and checkpoint writes (~2-3 min per fit end to end).
- EEGNet, one inner fold: 28-75 s training. Estimated total for <= 215 fits: < 15 GPU-hours.

## Historical audit
See [`historical_delivery_audit.md`](historical_delivery_audit.md).
