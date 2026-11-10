# RL protocol v2: validation candidate w010 (EEG weight 0.1)

Scope: **complete validation suite** (33 runs) on validation participants s11/s13, RL seeds 100/101/102,
150,000 environment steps per run, dense weight 10.0, EEG weight 0.1.
Not selected; preserved with all outcomes. No test-participant feedback was used.

Training commit `cfe0305` (training sources `train.py`/`environment.py`/`feedback.py`/`checkpoint.py` with
SHA-256 recorded in every `run.json`); every run is a fresh, non-resumed, fixed-budget run and passed
`rl.summarize`'s completeness checks.

Selection metric (12 EEGNet + CBraMod runs): mean final success 0.9800, mean final distance
0.0278 m, mean normalized AUC 0.7622.

| condition | runs | final success (mean) | final distance (m) | normalized AUC |
|---|---|---|---|---|
| sparse | 3 | 1.000 | 0.0259 | 0.748 |
| dense | 3 | 1.000 | 0.0217 | 0.904 |
| ideal | 3 | 1.000 | 0.0183 | 0.811 |
| eegnet | 6 | 0.988 | 0.0276 | 0.760 |
| cbramod | 6 | 0.972 | 0.0280 | 0.765 |
| independent_eegnet | 6 | 0.987 | 0.0271 | 0.721 |
| independent_cbramod | 6 | 0.905 | 0.0291 | 0.756 |

Assets: one archive per run with every periodic (10k-step) and final snapshot (`model.zip` incl. optimizer and
entropy state, `replay_buffer.pkl`, `rng.pt`, `probe.npz`, `metadata.json`), `transitions.csv`,
`evaluations.jsonl`, SB3 logs, `run.json`, `environment.json`, `status.json` and the job log;
`shared_evidence.tar.gz` holds `plan.json`, `summary.json/csv` and job logs. Archives larger than 100 MB are split
losslessly into `<archive>.part-NN`; reassemble with `cat <archive>.part-* > <archive>` and check the full-archive
SHA-256 in `manifest.json`. `SHA256SUMS` covers every asset of this Release.
Selection evidence: `configs/rl_v2/frozen/`, analysis: `reports/rl_v2_validation/` on branch `rl-v2-experiments`.
