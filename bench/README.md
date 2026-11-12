# ErrP-Bench

Cross-subject benchmark of error-related potential (ErrP) decoding on five public datasets, plus a
decoder-quality frontier for reinforcement learning from ErrP feedback. Paper: `paper/errp_bench/`.

## Datasets (public, not redistributed here)

| name | source | license | participants |
|---|---|---|---|
| `errphri_robot`, `errphri_cursor` | ErrP-HRI (Ehrlich & Cheng 2019) | none declared | 12 (same people) |
| `bnci2015_013` | BNCI Horizon 2020, 013-2015 (Chavarriaga & Millán 2010) | CC BY-NC-ND 4.0 | 6 |
| `erpcore_ern` | ERP CORE, ERN/flanker (Kappenman et al. 2021) | CC BY-SA 4.0 | 40 |
| `interhri` | IntEr-HRI, Zenodo 8345429 (Kueper et al. 2024) | CC BY 4.0 | 8 |

Download each dataset from its source, then prepare it with `bench/prep/<dataset>.py` (the configurations are
in `bench/prep/*.yaml`; the format is in `bench/DATA_FORMAT.md`). Run `bench/prep/validate_prepared.py` and
`bench/prep/sanity_check_prep.py` before training.

## Pipeline

```bash
python bench/make_folds.py ...          # 5 participant-disjoint folds per dataset (+ validation participants)
python bench/make_tasks.py ...          # one shell line per (dataset, model, alignment, seed, fold)
bash   bench/qworker.sh QUEUE SLOTS     # GPU worker: claims task files atomically, retries up to 3 times
python bench/run_fold.py --prep-dir P --config bench/configs/<model>.yaml --splits F --seed S --align {none,ea} --out O
python bench/ensemble.py --runs RUNS    # ErrP-Ens (label-free rank ensemble of aligned decoders), no training
python bench/aggregate.py --runs RUNS --out results/decoding --expect-folds '{"errphri_robot":5,...}'
python bench/dataset_stats.py ...       # results/dataset_stats.json
python bench/make_paper.py --results results --out paper/errp_bench/generated   # every number/table/figure
```

RL frontier (CPU): `bench/rl_frontier/README.md` (`submit_sweep.sh`, `train_frontier.py`, `summarize.py`).

## Conventions

- Unit of analysis: held-out participant. Score: macro-participant AUROC, averaged over training seeds.
- A seed counts only if all of its folds finished; partial seeds are dropped and listed, never imputed.
- Euclidean alignment (EA) uses only each participant's unlabeled trials (trace-preserving).
- Tests: two-sided Wilcoxon signed-rank over participants, Holm-corrected within dataset.
