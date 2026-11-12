# Decoder-quality -> RL-benefit frontier (ErrP-Bench)

How much RL benefit does ErrP feedback give as a function of decoder AUROC? Same SAC/reward
pipeline as `rl/` (protocol copied from `configs/rl_v2/frozen/protocol.json`, unchanged except
`env_id`/`total_timesteps`), with the replayed EEG decoder replaced by a synthetic decoder of
controlled quality.

## Feedback model (`rl/synthetic_feedback.py`)

Labelled 5-step segment with label y (deadband rule of `rl/environment.py`):
margin z = ±d'/2 + eps, eps ~ N(0,1), d' = sqrt(2)·Phi^-1(AUROC) (equal-variance binormal ->
population AUROC exactly the target); p_error = sigmoid(logit(pi) + d'·z) = P(y=1 | z), the
Bayes-calibrated probability at a FIXED base rate pi = 0.35. Reward add-on `w·(0.5 - p_error)`, w = 1.0.

Why fixed pi = 0.35: the real frozen decoders are calibrated to the EEG training prevalence
(~0.35-0.36), so the synthetic decoders live on the same probability scale and the frontier is
comparable with the real eegnet/cbramod points; the on-policy error rate is non-stationary and
condition-dependent, so calibrating to it would couple the feedback map to the learner. Side
effect (shared with the real decoders): an uninformative decoder gives p = 0.35, i.e. a constant
+0.15 per labelled segment. When the on-policy error rate differs from 0.35 (e.g. ~0.7 on Push
early in training) the probabilities are not calibrated w.r.t. that population; the AUROC is.

AUROC levels 0.55/0.60/0.65/0.70/0.80/0.90 synthetic; 1.00 = existing `ideal` condition
(p = label). eps is a SHA-256-keyed function of (seed + 1e6, event index): deterministic, and
common random numbers across AUROC levels. Unit test: `python -m unittest rl.test_synthetic_feedback -v`
(1e5 draws per level, |AUROC - target| < 0.005, decile calibration gap < 0.02; results in
`unit_test_report.json`).

## Tasks

- `reach`: FetchReach-v4, 150k steps (as the frozen protocol).
- `push`: FetchPush-v4, 500k steps (largest budget fitting the deadline at the measured
  ~17-18.5 s per 10k... see pilot below). `achieved_goal` is the object position, so the existing
  label rule = change of OBJECT-goal distance and `dense` = object-goal progress shaping.
- `push_staged` (supplementary, not in the original plan): FetchPush with the label/dense
  distance replaced by ||gripper-object|| + ||object-goal||, because under the object-goal rule
  the object rarely moves early in training (pilot: ~170 labelled segments in 30k steps vs ~4k
  on Reach), so feedback is almost absent. Its sparse reference is `push/sparse` (the label rule
  does not affect sparse training). Evaluation metrics are unchanged (object-goal success/distance).

Conditions: sparse, dense, auroc{0.55..1.00}; seeds 0-4; eval every 10k steps on 100 fixed goals
(seeds 20000-20099), deterministic policy, sparse success. Metrics: final success, normalized
success AUC (`rl.metrics.normalized_auc`), final object/gripper-goal distance.

## Run

```bash
# on gc211, code at /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/code_rl
bash bench/rl_frontier/submit_sweep.sh               # push + reach, 90 jobs
bash bench/rl_frontier/submit_sweep.sh push_staged   # supplementary, 40 jobs
/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-rl/repo/venv/rl/bin/python \
    bench/rl_frontier/summarize.py      # defaults: rl_runs/sweep -> results/rl_frontier
```

Per run (`rl_runs/sweep/<task>/<cond>/seed<k>/`): run.json, status.json (incl. realised AUROC,
feedback counts), evaluations.jsonl, evaluation.json, final_model.zip (no replay), feedback_events.npz,
progress.csv, monitor.csv. Smoke runs (shortened `--steps`) live in `rl_runs/smoke/` and are not results.
