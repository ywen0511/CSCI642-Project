# RL protocol v2: final comparison (frozen protocol, test participants)

Scope: **complete final suite** (75 runs): sparse, dense and ideal x RL seeds 0-4, and EEGNet, CBraMod and their
label-independent controls x participants s02/s08/s09 x RL seeds 0-4. 150,000 environment steps per run, every
run from scratch and not resumed. Frozen protocol `configs/rl_v2/frozen/protocol.json` (EEG weight 1.0, dense
weight 10.0), selected on validation participants only; its validation evidence hash is checked by the runner.
Training commit `25bb0f4` (recorded in every run; training-source SHA-256 in every `run.json`).

| condition | runs | final success | ever success | final distance (m) | sparse return | normalized AUC |
|---|---|---|---|---|---|---|
| sparse | 5 | 0.942 ± 0.083 | 0.980 | 0.0301 ± 0.0112 | -4.44 | 0.659 ± 0.131 |
| dense | 5 | 1.000 ± 0.000 | 1.000 | 0.0214 ± 0.0078 | -1.77 | 0.915 ± 0.033 |
| eegnet | 15 | 0.968 ± 0.034 | 0.995 | 0.0233 ± 0.0026 | -3.21 | 0.641 ± 0.041 |
| cbramod | 15 | 0.952 ± 0.063 | 0.983 | 0.0229 ± 0.0050 | -4.04 | 0.571 ± 0.115 |
| ideal | 5 | 0.986 ± 0.031 | 1.000 | 0.0213 ± 0.0080 | -2.65 | 0.764 ± 0.097 |
| independent_eegnet | 15 | 0.960 ± 0.049 | 0.985 | 0.0241 ± 0.0066 | -3.55 | 0.595 ± 0.036 |
| independent_cbramod | 15 | 0.940 ± 0.078 | 0.972 | 0.0385 ± 0.0280 | -4.54 | 0.602 ± 0.038 |

(± = SD over RL seeds for sparse/dense/ideal; SD over participants' seed means for EEG conditions.) Final success
is near ceiling; learning speed (AUC) separates dense > ideal > sparse, while decoded EEG feedback did not speed up
learning. Full results: `reports/rl_v2_results.md`; report: `reports/final_project_report.md` (branch
`rl-v2-experiments`).

Assets:
- `final__<run>.tar.gz[.part-NN]`: one archive per run with the final snapshot
  (`model.zip` incl. optimizer/entropy state, `replay_buffer.pkl`, `rng.pt`, `probe.npz`, `metadata.json`),
  `transitions.csv`, `evaluations.jsonl`, SB3 logs, `run.json`, `environment.json`, `status.json`, job log.
- `final__shared_evidence.tar.gz`: `plan.json`, `summary.json/csv`, job logs; `final__manifest.json`: every file
  of every run with SHA-256 and the full-archive hashes of split archives.
- `media__analysis_and_media.tar.gz`: frozen protocol and selection evidence, validation and final analysis outputs
  (CSV/JSON source data for every figure, figures), trajectory data, GIFs and their replay checks.
- `gif__*.gif`: representative rollouts (rule: RL seed 0; EEG participant s02; goal seeds 30000-30002).

Archives above 100 MB are split losslessly: `cat <archive>.part-* > <archive>` and check the full-archive SHA-256
in `final__manifest.json`. `SHA256SUMS` covers every asset of this Release.

**Checkpoint scope (user instruction: upload only the important checkpoints):** each run archive contains the run's **final snapshot** (`model.zip` with optimizer and entropy state, `replay_buffer.pkl`, `rng.pt`, `probe.npz`, `metadata.json`) and every log, evaluation record, transition log and config. The periodic 10k-step snapshots are not uploaded; each is listed with its SHA-256 in the manifest (`in_archive: false`) and retained on the project filer (`artifacts/rl_v2_final`, `artifacts/rl_v2_validation`). Periodic-snapshot reload and continuation were verified on the pilot and smoke Releases, which include all snapshots.
