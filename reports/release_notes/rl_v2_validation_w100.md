# RL protocol v2: validation candidate w100 (EEG weight 1.0)

Scope: **complete validation suite** (33 runs) on validation participants s11/s13, RL seeds 100/101/102,
150,000 environment steps per run, dense weight 10.0, EEG weight 1.0.
**Selected** by the declared rule; its configuration is the frozen protocol (`configs/rl_v2/frozen/protocol.json`). No test-participant feedback was used.

Training commit `cfe0305` (training sources `train.py`/`environment.py`/`feedback.py`/`checkpoint.py` with
SHA-256 recorded in every `run.json`); every run is a fresh, non-resumed, fixed-budget run and passed
`rl.summarize`'s completeness checks.

Selection metric (12 EEGNet + CBraMod runs): mean final success 0.9967, mean final distance
0.0230 m, mean normalized AUC 0.7313.

| condition | runs | final success (mean) | final distance (m) | normalized AUC |
|---|---|---|---|---|
| sparse | 3 | 1.000 | 0.0259 | 0.748 |
| dense | 3 | 1.000 | 0.0217 | 0.904 |
| ideal | 3 | 1.000 | 0.0185 | 0.862 |
| eegnet | 6 | 1.000 | 0.0201 | 0.700 |
| cbramod | 6 | 0.993 | 0.0259 | 0.763 |
| independent_eegnet | 6 | 0.990 | 0.0270 | 0.638 |
| independent_cbramod | 6 | 0.990 | 0.0244 | 0.660 |

Assets: one archive per run with every periodic (10k-step) and final snapshot (`model.zip` incl. optimizer and
entropy state, `replay_buffer.pkl`, `rng.pt`, `probe.npz`, `metadata.json`), `transitions.csv`,
`evaluations.jsonl`, SB3 logs, `run.json`, `environment.json`, `status.json` and the job log;
`shared_evidence.tar.gz` holds `plan.json`, `summary.json/csv` and job logs. Archives larger than 100 MB are split
losslessly into `<archive>.part-NN`; reassemble with `cat <archive>.part-* > <archive>` and check the full-archive
SHA-256 in `manifest.json`. `SHA256SUMS` covers every asset of this Release.
Selection evidence: `configs/rl_v2/frozen/`, analysis: `reports/rl_v2_validation/` on branch `rl-v2-experiments`.
