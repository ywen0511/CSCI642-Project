# CSCI642: EEG error-potential feedback for robot reinforcement learning

Repository: **ywen0511/CSCI642-Project**. This is branch **final-v3-experiments**, the execution of
[FINAL_V3_HANDOFF.md](FINAL_V3_HANDOFF.md) (planning branch `final-v3`). It merges the RL v2 execution branch
(`rl-v2-experiments`) and the decoder-improvement execution branch (`decoder-improvement-v1-experiments`).

## Status

| study | branch | status | main result | reports |
|---|---|---|---|---|
| EEG stage 1 (EEGNet / CBraMod decoders, calibration, frozen feedback probabilities) | `eeg-stage1` | done (2026-11-01) | test AUROC EEGNet 0.637, CBraMod 0.573 (decoder seed 42) | [eeg_results.md](reports/eeg_results.md) |
| RL protocol v2 (FetchReach, SAC, replayed EEG feedback, w = 1.0) | `rl-v2-experiments` | done (2026-11-10), declared primary result | normalized success AUC: dense 0.915 > ideal 0.764 > sparse 0.659 ≈ EEGNet 0.641 > CBraMod 0.571 | [rl_v2_results.md](reports/rl_v2_results.md), [final_project_report.md](reports/final_project_report.md) |
| Decoder improvement (CBraMod freezing / last-layer fine-tuning, bands, loss) | `decoder-improvement-v1-experiments` | done (2026-11-11), negative | no change improved held-out decoding: CBraMod ~0.564, EEGNet 0.599 test AUROC (five controlled refits) | [decoder_improvement/final_report.md](reports/decoder_improvement/final_report.md), [artifact index](reports/decoder_improvement/artifact_index.md) |
| Final round v3: weight sensitivity, centered reward (v3), route preference | `final-v3-experiments` (this branch) | done (2026-11-19); sensitivity = post-hoc, v3 = post-hoc follow-up with its own selection, preference = extension | sensitivity: at w = 0.5 both decoders lead sparse in AUC (+0.060 / +0.039); v3 (centered reward, selected w = 0.5): EEGNet 0.688 / CBraMod 0.690 vs sparse 0.659 AUC, bias removed; preference: EEG feedback did not teach the preference (preferred pass 0.045-0.048 vs sparse 0.044; ideal 0.082) | [final_project_report.md](reports/final_project_report.md), [reports/final_v3/](reports/final_v3/) ([sensitivity](reports/final_v3/sensitivity_results.md), [v3](reports/final_v3/v3_results.md), [preference](reports/final_v3/preference_results.md), [execution](reports/final_v3/execution.md)) |

The historical feedback profile [`configs/rl_feedback.json`](configs/rl_feedback.json) is unchanged and is the only
profile used for RL. `configs/decoder_improvement/rl_feedback_vnext.json` (from the decoder study) is a separate,
explicitly unused experimental profile.

Decoder-improvement study (merged): executed from [`DECODER_IMPROVEMENT_HANDOFF.md`](DECODER_IMPROVEMENT_HANDOFF.md);
status [`reports/decoder_improvement/status.json`](reports/decoder_improvement/status.json), results
[`results.md`](reports/decoder_improvement/results.md). Its Releases are listed in its artifact index.

RL v2 (merged): frozen protocol and evidence [configs/rl_v2/frozen/](configs/rl_v2/frozen/), execution record
[rl_v2_execution.md](reports/rl_v2_execution.md); Releases in the [final project report, Section 8](reports/final_project_report.md#8-artifacts).

The sections below are the historical README text of `rl-v2-experiments` (original `rl-proposal-v2` handoff text),
kept for reference; they are not instructions for this round.

## Start here

**Executing agent: read [GPU_AGENT_HANDOFF.md](GPU_AGENT_HANDOFF.md) in full, then complete its required implementation, validation, experiments, analysis, report and verified Release delivery.** That document is the complete current assignment; it takes precedence over historical stage instructions. Use this repository for source and new deliverables. No access to the private Savaw repository is needed. Do not retrain the frozen EEG decoders.

The user only needs to provide the handoff MD link. No user-side cloning or setup is required, and the executing agent must not run `git clone`. Reuse an existing workspace or retrieve the exact source archive/API content yourself. The agent must execute the work and directly upload code, results, all checkpoints and Release assets, then verify fresh downloads; do not stop at an upload-ready folder.

## What is ready

- EEGNet and CBraMod training, calibration, evaluation, feedback probabilities and EEG checkpoints were completed previously. Original assets remain in this repository's Releases.
- FetchReach/SAC training, evaluation, checkpoint/replay/RNG saving, continuation, suite planning and packaging are implemented under `rl/`.
- Protocol **v2** uses `r_task + w_eeg * (0.5 - p_error)` at eligible feedback events.
- Distance changes inside the inclusive deadband receive **no EEG feedback**, no sampled EEG trial and no binary label. Task reward is retained.
- Neutral segments still end on schedule. Logs distinguish segment boundaries, eligibility and actual feedback.

The v2 preparation performed unit/static checks only. It did **not** run SAC training, simulator smoke runs, parameter selection or final experiments. Historical v1 smoke results do not validate v2. The default 50,000 steps and five final seeds are still the old planner defaults; the next agent must implement the explicit seed/budget plan in the handoff before formal runs.

## Proposal stages and current status

| Proposal stage | Status |
|---|---|
| 1: reaching/SAC and sparse/dense baselines | **Done** on this branch: formal sparse/dense baselines in every validation and final suite (150k steps) |
| 2: EEG audit, training, calibration and evaluation | Completed earlier; weights, splits and predictions kept frozen |
| 3: EEG integration and main comparison | **Done**: smoke, pilots, 3 validation candidates, freeze, 75-run final comparison; optional route preference not performed |
| 4: analysis, curves, trajectories and report | **Done**: curves, AUC, diagnostics, trajectories/GIFs, reports, verified Releases |

Historical names `eeg-stage1` and `rl-stage2` are delivery names, not the proposal's numbered stages.

## Useful files

- [Complete next-agent assignment](GPU_AGENT_HANDOFF.md)
- [Current RL mechanics and commands](rl/README.md)
- [v2 change and verification record](reports/rl_protocol_v2.md)
- [Frozen EEG inputs](docs/rl_handoff.md) and [machine-readable profile](configs/rl_feedback.json)
- [Historical EEG specification](EEG_README.md), [measured EEG results](reports/eeg_results.md)
- [Historical v1 execution report](rl/execution_report.md)

Small feedback files can be downloaded and verified using only Python's standard library:

```bash
python3 scripts/download_feedback.py --out-dir artifacts/rl_feedback
```

This downloads from Releases in **this same repository**. Reading probabilities requires neither raw EEG nor GPU inference. For training setup use `rl/requirements.txt`, not the EEG dependencies at the repository root.

All new documentation, reports, code comments, commit messages and Release notes must be in English. Preserve every periodic/final checkpoint and replay buffer until the new Release has been freshly downloaded and verified. Do not overwrite old tags or upload raw third-party EEG.
