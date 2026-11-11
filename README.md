# CSCI642: decoder improvement and verified delivery

> **Decoder-improvement study (branch `decoder-improvement-v1-experiments`).** Executed from
> [`DECODER_IMPROVEMENT_HANDOFF.md`](DECODER_IMPROVEMENT_HANDOFF.md). Result: no tested change (smaller head,
> frozen / last-two-block / full fine-tuning, 0.3-40 / 0.3-75 Hz input, unweighted loss; the backbone learning-rate
> stage was skipped because the selected backbone was frozen)
> improved held-out decoding; the selected CBraMod and a controlled refit of the historical recipe both reached test
> AUROC ~0.56, EEGNet 0.60. Reports: [`reports/decoder_improvement/final_report.md`](reports/decoder_improvement/final_report.md),
> [`results.md`](reports/decoder_improvement/results.md), status
> [`status.json`](reports/decoder_improvement/status.json), Releases and verification:
> [`artifact_index.md`](reports/decoder_improvement/artifact_index.md). The new seed-42 probabilities are an
> experimental feedback profile (`configs/decoder_improvement/rl_feedback_vnext.json`); the historical feedback and
> RL results are unchanged and did not use them. The text below is the original planning README.


This branch prepares a new EEG decoder study. **No new training or improvement result is claimed by this handoff.**

## Single entry point for the executing agent

Read **[DECODER_IMPROVEMENT_HANDOFF.md](DECODER_IMPROVEMENT_HANDOFF.md)** in full and execute it through implementation, experiments, analysis, publication and verification. The user only needs to provide that link. It supplies the task, exact data sources, historical workspace hints, fixed experiment plan, selection rules, output requirements and completion criteria.

The assignment is to investigate and improve CBraMod and EEGNet, preserve the historical study, repair incomplete historical artifact delivery where possible, and publish a reproducible new study in **ywen0511/CSCI642-Project**. A negative result is an acceptable scientific outcome; incomplete uploads are not a completed delivery.

- Prepared from `rl-v2-experiments` at `f6decb7a99628638743a4c126d83668149d5c5c8`.
- Planning branch: `decoder-improvement-v1`.
- Machine-readable plan: [configs/decoder_improvement/study.json](configs/decoder_improvement/study.json).
- Initial status: [reports/decoder_improvement/status.json](reports/decoder_improvement/status.json).
- Historical results: [EEG](reports/eeg_results.md), [RL v2](reports/rl_v2_results.md).

This assignment supersedes the execution instructions in `GPU_AGENT_HANDOFF.md`, `EEG_README.md` and historical README files for the new work. Those files remain source/provenance references. Do not accidentally execute the old RL assignment again.

The new study has a bounded plan: at most **215 distinct formal EEG training fits**, excluding explicitly logged smoke checks and retries. Independent fits can use the existing cluster scheduler. It does not authorize buying compute or starting a new full RL experiment suite.

Raw third-party EEG and upstream initialization weights retain their original redistribution conditions. Reproduction must include pinned acquisition instructions and checksums for any externally obtained inputs, and all project-generated artifacts that can be distributed. No credentials belong in the repository or Releases.
