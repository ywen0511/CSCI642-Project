# Decoder-improvement study: results

Branch `decoder-improvement-v1-experiments`. Plan: `DECODER_IMPROVEMENT_HANDOFF.md` / `configs/decoder_improvement/study.json`.
All tables below are copied verbatim from [`results/generated/results_tables.md`](results/generated/results_tables.md),
generated from committed decision files and test outputs (number-to-source map: `results/generated/number_audit.json`).

**Short answer.** None of the tested changes improved held-out decoding. Train-only cross-validation preferred a
frozen CBraMod backbone with the historical flatten head and unweighted loss (CV score 0.578 vs 0.567 for the
historical CBraMod recipe), but on the previously evaluated test participants the selected CBraMod and its
controlled reference refit were indistinguishable (pooled AUROC 0.564 ± 0.005 vs 0.563 ± 0.006; paired difference
-0.003, positive in 5 of 15 participant-seed pairs). EEGNet's historical recipe won its own search and stayed the
better decoder (test AUROC 0.599 ± 0.014). Filtering 0.3-40 or 0.3-75 Hz did not help either model. The selected
affine calibration improved probability quality (test Brier 0.227 vs 0.240 for CBraMod-selected with temperature;
0.223 vs 0.240 for EEGNet) but not ranking.

## Train-only cross-validation (7 training participants, 3 fixed inner folds, seeds 42-44)

## Stage A: CBraMod head x adaptation (1-20 Hz, weighted CE, seeds 42-44, 3 folds)

| candidate | recipe | trainable params | CV score | pooled AUROC | Brier (raw) | NLL (raw) | p gap (err-nonerr) | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `cb_flat_frozen` | historical_flatten_mlp / frozen / bb None / 1-20 Hz / weighted_ce | 4401002 | 0.5787 | 0.5626 | 0.2781 | 0.9782 | 0.0445 | 3 |
| `cb_flat_last2` | historical_flatten_mlp / last2 / bb 3e-05 / 1-20 Hz / weighted_ce | 5206202 | 0.5750 | 0.5726 | 0.2573 | 0.7378 | 0.0422 | 3 |
| `cb_reference` | historical_flatten_mlp / full / bb 0.0001 / 1-20 Hz / weighted_ce | 9284602 | 0.5675 | 0.5626 | 0.3154 | 1.4557 | 0.0715 | 12 |
| `cb_timepool_full` | temporal_mean_linear / full / bb 3e-05 / 1-20 Hz / weighted_ce | 4894402 | 0.5666 | 0.5580 | 0.2953 | 0.8956 | 0.0548 | 31 |
| `cb_flat_full` | historical_flatten_mlp / full / bb 3e-05 / 1-20 Hz / weighted_ce | 9284602 | 0.5588 | 0.5503 | 0.2981 | 1.1064 | 0.0516 | 19 |
| `cb_timepool_frozen` | temporal_mean_linear / frozen / bb None / 1-20 Hz / weighted_ce | 10802 | 0.5575 | 0.5498 | 0.2442 | 0.6816 | 0.0176 | 29 |
| `cb_timepool_last2` | temporal_mean_linear / last2 / bb 3e-05 / 1-20 Hz / weighted_ce | 816002 | 0.5546 | 0.5496 | 0.2529 | 0.7128 | 0.0270 | 8 |

Selected: cb_flat_frozen

## Stage B: backbone learning-rate refinement

Skipped: Stage A winner cb_flat_frozen has a frozen backbone (handoff 7.3: skip with recorded reason); no backbone learning rate to refine

## Stage C: EEGNet learning rate x dropout (1-20 Hz)

| candidate | recipe | trainable params | CV score | pooled AUROC | Brier (raw) | NLL (raw) | p gap (err-nonerr) | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `eeg_lr0.001_do0.5` | lr 0.001 / dropout 0.5 / 1-20 Hz / weighted_ce | 2562 | 0.6118 | 0.5985 | 0.2490 | 0.7068 | 0.0645 | 47 |
| `eeg_lr0.001_do0.25` | lr 0.001 / dropout 0.25 / 1-20 Hz / weighted_ce | 2562 | 0.6085 | 0.6016 | 0.2560 | 0.7388 | 0.0841 | 85 |
| `eeg_lr0.0003_do0.5` | lr 0.0003 / dropout 0.5 / 1-20 Hz / weighted_ce | 2562 | 0.5962 | 0.5815 | 0.2567 | 0.7310 | 0.0521 | 138 |
| `eeg_lr0.0003_do0.25` | lr 0.0003 / dropout 0.25 / 1-20 Hz / weighted_ce | 2562 | 0.5935 | 0.5841 | 0.2592 | 0.7369 | 0.0662 | 120 |

Selected: eeg_lr0.001_do0.5

## Stage D: band comparison for the selected recipes

| candidate | recipe | trainable params | CV score | pooled AUROC | Brier (raw) | NLL (raw) | p gap (err-nonerr) | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `eegnet:eeg_lr0.001_do0.5@1-20` | lr 0.001 / dropout 0.5 / 1-20 Hz / weighted_ce | 2562 | 0.6118 | 0.5985 | 0.2490 | 0.7068 | 0.0645 | 47 |
| `eegnet:eeg_lr0.001_do0.5@0.3-40` | lr 0.001 / dropout 0.5 / 0.3-40 Hz / weighted_ce | 2562 | 0.6030 | 0.5917 | 0.2530 | 0.7201 | 0.0622 | 61 |
| `eegnet:eeg_lr0.001_do0.5@0.3-75` | lr 0.001 / dropout 0.5 / 0.3-75 Hz / weighted_ce | 2562 | 0.5974 | 0.5835 | 0.2591 | 0.7416 | 0.0544 | 61 |
| `cbramod:cb_flat_frozen@1-20` | historical_flatten_mlp / frozen / bb None / 1-20 Hz / weighted_ce | 4401002 | 0.5787 | 0.5626 | 0.2781 | 0.9782 | 0.0445 | 3 |
| `cbramod:cb_flat_frozen@0.3-40` | historical_flatten_mlp / frozen / bb None / 0.3-40 Hz / weighted_ce | 4401002 | 0.5667 | 0.5550 | 0.3007 | 1.5074 | 0.0561 | 13 |
| `cbramod:cb_flat_frozen@0.3-75` | historical_flatten_mlp / frozen / bb None / 0.3-75 Hz / weighted_ce | 4401002 | 0.5666 | 0.5521 | 0.3041 | 1.5104 | 0.0504 | 12 |

Selected: {"cbramod": "cbramod:cb_flat_frozen@1-20", "eegnet": "eegnet:eeg_lr0.001_do0.5@1-20"}

## Stage E: loss comparison (weighted vs unweighted cross-entropy)

| candidate | recipe | trainable params | CV score | pooled AUROC | Brier (raw) | NLL (raw) | p gap (err-nonerr) | median best epoch |
|---|---|---|---|---|---|---|---|---|
| `eegnet:eegnet:eeg_lr0.001_do0.5@1-20#weighted_ce` | lr 0.001 / dropout 0.5 / 1-20 Hz / weighted_ce | 2562 | 0.6118 | 0.5985 | 0.2490 | 0.7068 | 0.0645 | 47 |
| `eegnet:eegnet:eeg_lr0.001_do0.5@1-20#ce` | lr 0.001 / dropout 0.5 / 1-20 Hz / ce | 2562 | 0.6106 | 0.5990 | 0.2337 | 0.6691 | 0.0632 | 59 |
| `cbramod:cbramod:cb_flat_frozen@1-20#ce` | historical_flatten_mlp / frozen / bb None / 1-20 Hz / ce | 4401002 | 0.5794 | 0.5695 | 0.2539 | 0.8085 | 0.0486 | 3 |
| `cbramod:cbramod:cb_flat_frozen@1-20#weighted_ce` | historical_flatten_mlp / frozen / bb None / 1-20 Hz / weighted_ce | 4401002 | 0.5787 | 0.5626 | 0.2781 | 0.9782 | 0.0445 | 3 |

Selected: {"cbramod": "cbramod:cbramod:cb_flat_frozen@1-20#ce", "eegnet": "eegnet:eegnet:eeg_lr0.001_do0.5@1-20#weighted_ce"}

## Stage F: five-seed confirmation and freeze

| arch | role | candidate | CV score (5 seeds) | screening seeds 42-44 | new seeds 45-46 | final epochs (fixed) | identical to reference |
|---|---|---|---|---|---|---|---|
| cbramod | selected | `cbramod:cbramod:cb_flat_frozen@1-20#ce` | 0.5783 | 0.5794 | 0.5766 | 5 |  |
| cbramod | reference | `cbramod_reference` | 0.5667 | 0.5675 | 0.5655 | 17 |  |
| eegnet | selected | `eegnet:eegnet:eeg_lr0.001_do0.5@1-20#weighted_ce` | 0.6103 | 0.6118 | 0.6080 | 47 | yes |

cbramod: selected minus reference per participant and seed: mean +0.0116, positive in 19 of 35 pairs; selection optimism (screening minus new seeds) +0.0028.

eegnet: selected recipe identical to the reference; no distinct candidate.

## Calibration and test

## Calibration (validation participants s11/s13 only; committed before test access)

| arch | held-out NLL identity | temperature | affine | selected | per-seed parameters (seeds 42-46) |
|---|---|---|---|---|---|
| cbramod | 0.7440 | 0.6601 | 0.6392 | affine | a=0.332, b=-0.555, a=0.325, b=-0.735, a=0.325, b=-0.172, a=0.332, b=-0.282, a=0.355, b=-0.160 |
| eegnet | 0.6844 | 0.6785 | 0.6415 | affine | a=0.553, b=-0.553, a=0.476, b=-0.536, a=0.535, b=-0.557, a=0.533, b=-0.571, a=0.523, b=-0.480 |

Controlled-comparison temperatures (per final model): cb-flat-frozen-bbnone-hl5e-4 s42: T=3.06, cb-flat-frozen-bbnone-hl5e-4 s43: T=18.03, cb-flat-frozen-bbnone-hl5e-4 s44: T=2.58, cb-flat-frozen-bbnone-hl5e-4 s45: T=2.36, cb-flat-frozen-bbnone-hl5e-4 s46: T=2.41, cb-flat-full-bb1e-4-hl5e-4-b s42: T=6.08, cb-flat-full-bb1e-4-hl5e-4-b s43: T=6.46, cb-flat-full-bb1e-4-hl5e-4-b s44: T=5.82, cb-flat-full-bb1e-4-hl5e-4-b s45: T=6.72, cb-flat-full-bb1e-4-hl5e-4-b s46: T=7.26, eeg-lr1e-3-do0.5-b1_20-wce-a s42: T=1.76, eeg-lr1e-3-do0.5-b1_20-wce-a s43: T=1.92, eeg-lr1e-3-do0.5-b1_20-wce-a s44: T=1.85, eeg-lr1e-3-do0.5-b1_20-wce-a s45: T=1.99, eeg-lr1e-3-do0.5-b1_20-wce-a s46: T=1.49

## Outer test (s02, s08, s09; 1,429 trials; five final refits per recipe; mean ± sample SD over seeds)

| arch | role | seeds | pooled AUROC | participant-macro AUROC | bal. acc @0.5 raw | bal. acc temperature | bal. acc deployment | Brier raw | Brier temperature | Brier deployment | NLL temperature | NLL deployment | ECE temperature | ECE deployment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cbramod | selected | 5 | 0.564 ± 0.005 | 0.562 ± 0.006 | 0.529 ± 0.016 | 0.529 ± 0.016 | 0.513 ± 0.004 | 0.280 ± 0.035 | 0.240 ± 0.012 | 0.227 ± 0.001 | 0.673 ± 0.025 | 0.646 ± 0.001 | 0.101 ± 0.058 | 0.039 ± 0.006 |
| cbramod | reference | 5 | 0.563 ± 0.006 | 0.566 ± 0.006 | 0.544 ± 0.006 | 0.544 ± 0.006 | 0.544 ± 0.006 | 0.334 ± 0.004 | 0.244 ± 0.002 | 0.244 ± 0.002 | 0.682 ± 0.003 | 0.682 ± 0.003 | 0.126 ± 0.011 | 0.126 ± 0.011 |
| eegnet | selected | 5 | 0.599 ± 0.014 | 0.601 ± 0.015 | 0.571 ± 0.010 | 0.571 ± 0.010 | 0.522 ± 0.010 | 0.242 ± 0.006 | 0.240 ± 0.004 | 0.223 ± 0.002 | 0.676 ± 0.008 | 0.638 ± 0.006 | 0.135 ± 0.007 | 0.022 ± 0.004 |

AUROC is identical for raw, temperature and deployment probabilities (monotone calibration). For the EEGNet row the selected recipe equals the historical reference recipe. Deployment = affine calibration (selected candidates); the CBraMod reference uses temperature for both.

### Per participant (raw probabilities, mean over 5 seeds)

| arch | role | participant | AUROC | SD over seeds | bal. acc @0.5 |
|---|---|---|---|---|---|
| cbramod | reference | s02 | 0.547 | 0.015 | 0.537 |
| cbramod | reference | s08 | 0.531 | 0.008 | 0.511 |
| cbramod | reference | s09 | 0.619 | 0.025 | 0.589 |
| cbramod | selected | s02 | 0.602 | 0.012 | 0.536 |
| cbramod | selected | s08 | 0.520 | 0.008 | 0.501 |
| cbramod | selected | s09 | 0.565 | 0.011 | 0.541 |
| eegnet | selected | s02 | 0.598 | 0.011 | 0.563 |
| eegnet | selected | s08 | 0.569 | 0.019 | 0.544 |
| eegnet | selected | s09 | 0.635 | 0.029 | 0.597 |

Paired test differences, cbramod selected minus reference (same participants and seeds): mean -0.0033 AUROC, positive in 5 of 15 pairs.

### Historical archived results (Release eeg-stage1-20261101-bccbb8a; validation-selected historical runs, not controlled refits)

| model | seed-42 AUROC | five-seed AUROC | five-seed bal. acc | five-seed calibrated Brier |
|---|---|---|---|---|
| eegnet | 0.637 | 0.613 ± 0.016 | 0.577 ± 0.012 | 0.238 ± 0.006 |
| cbramod | 0.573 | 0.568 ± 0.007 | 0.546 ± 0.010 | 0.245 ± 0.004 |

Figures: [`fig_stage_A.png`](results/generated/fig_stage_A.png) (participant-level CV AUROC per Stage A candidate),
[`fig_test_auroc.png`](results/generated/fig_test_auroc.png) (test AUROC per seed with the historical five-seed means).
