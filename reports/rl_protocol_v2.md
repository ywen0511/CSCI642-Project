# Proposal-aligned RL protocol v2 preparation

Date: 2026-11-08. Repository: `ywen0511/CSCI642-Project`, branch `rl-proposal-v2`.
Base: `rl-stage2` at `073f16a4b6a4f33997eae1bf3549c04d09a568e9`.

## Changes implemented

1. Center ideal and decoded feedback at 0.5: `r_task + w_eeg * (0.5 - p_error)`.
2. Label distance decreases beyond epsilon non-error and increases beyond epsilon error. Inclusive-deadband changes have no label, sample or EEG reward. Apply this rule irrespective of the goal region; success stays in the task environment.
3. Reset segment state even when feedback is skipped. Count only labeled opportunities for deterministic sampling. Distinguish segment boundaries, feedback eligibility and actual feedback in transition logs.
4. Require protocol schema v2 with explicit reward/label semantics. Old v1 configs are rejected; existing config/source identity checks prevent old checkpoints being silently resumed under the new model.
5. Add regression coverage and an English end-to-end execution/delivery assignment. Preserve historical EEG instructions separately and label old v1 smoke evidence as historical.

Sparse/dense task rewards, EEG weights/calibration/splits/probability files, default training budgets and planner seed lists were not changed. The next agent must implement the seed/budget preparation described in the handoff before formal experiments.

## Verification performed during preparation

Command: `/private/tmp/csci642-rl-stage2-env/bin/python -m unittest rl.test_stage2 -v`.

Result: **21 tests passed**, including signed ideal/decoded/control rewards, probability 0.5, inclusive threshold boundaries, no sampling or counter advancement in neutral segments, blank neutral CSV fields, final partial segments, skipped-segment reset, unchanged sparse/dense rewards and legacy protocol rejection. Tests use a small deterministic toy environment; they do not perform SAC optimization or run MuJoCo.

The existing EEG closeout/feedback suite also passed: **20 tests** via `/private/tmp/eeg-closeout-env/bin/python -m unittest discover -s tests -v`. Total: **41 non-training tests passed**. Python AST parsing, JSON parsing, changed-document local links and `git diff --check` passed. The simulated failure cases in the closeout tests are intentional assertions, not failed project runs.

No v2 simulator smoke run, model training, checkpoint inference, validation selection or final comparison was performed during this preparation. Historical v1 execution reports and Releases remain unchanged in meaning. They cannot establish learning performance for v2.

## Next action

Execute [GPU_AGENT_HANDOFF.md](../GPU_AGENT_HANDOFF.md) from the repository root. CPU is supported; GPU execution is optional and should be benchmarked. Deliver future checkpoints, replay buffers and evidence through new Releases in this repository, with fresh-download verification.
