# RL stage 2: execution and handoff

This branch integrates the supplied `CS642-EEG-Assisted-RL-main.zip` baseline with the frozen EEG
stage. Start here for RL work. The root README's original EEG instructions are historical and do
not authorize retraining the frozen EEG decoders as part of this stage.

## Current scope

Implemented: FetchReach/SAC, seven reward conditions, participant-specific frozen EEG replay,
validation/test separation, unified evaluation, periodic model/optimizer/replay/RNG snapshots,
checkpoint reload verification, continuation into a fresh episode, multi-seed suite planning,
validation-based protocol freezing, summaries and Release packaging.

Actual execution evidence is in [execution_report.md](execution_report.md). Small integration runs
are not evidence that a robot policy has learned the task. Full validation and final comparisons
remain to be run. The source archive's hashes and adaptation provenance are in
[baseline_provenance.json](baseline_provenance.json); the original archive was not changed.

## Frozen EEG inputs

Use [the original EEG Release](https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a)
and `configs/rl_feedback.json`, decoder seed 42. `FeedbackLibrary` checks the full table's size,
SHA-256, split, participants, checkpoint/preprocessing/calibration IDs, labels and trial pairing.
Calibration and network weights stay fixed. No embeddings or live neural-network inference are
needed for this offline replay experiment.

- Pilot and validation: s11/s13 only, 959 unique EEG trials per decoder in total.
- Final: s02/s08/s09, 1,429 trials per decoder in total, after freezing the RL protocol.
- One feedback participant per learned EEG-assisted policy; do not pool participants silently.
- Sample with replacement from that participant's appropriate label pool. Repeated EEG trials
  are replays, not new human observations. Use `p_error`, the calibrated error probability.

## Explicit project protocol (not an exact Kim reproduction)

The simulator is Gymnasium-Robotics `FetchReach-v4`, not Kim's Kinova pick-and-place task. It has
no added obstacle or route-preference objective. Keep the original sparse reward (typically -1
outside the goal threshold, 0 inside), the 5 cm success threshold and 50-step time limit.

Default feedback event: every 5 simulator steps, plus a final partial segment at episode end.
For segment-start distance d_start and segment-end distance d_end:

`simulated_error = (d_end > 0.05 m) and (d_end - d_start > 0.001 m)`.

Improvement, remaining within the goal, and changes inside the deadband are non-error. In
particular a stalled segment is non-error; this is a limitation of the pilot rule, not a claim
about human preferences. The mapping uses privileged simulator geometry to construct a noisy
feedback channel. Report it explicitly when interpreting results.

FetchReach runs at 25 control steps/s. Five steps are 0.2 s, whereas each source EEG epoch spans
2 s. This is an abstract offline feedback-frequency choice; it does not simulate real acquisition
latency or establish real-time deployment. The 5-step interval, 1 mm deadband and weights are
project defaults in `protocol.json`, not settings attributed to Kim.

| Condition | Training reward |
|---|---|
| sparse | r_task |
| dense | r_task + w_dense * (d_t - d_next), every step |
| ideal | r_task + w_eeg * (1 - simulated_error), at feedback events |
| eegnet / cbramod | r_task + w_eeg * (1 - p_error), at feedback events |
| independent_eegnet / independent_cbramod | same formula and schedule, but sample from the participant's full pool independently of the simulated label |

Outside feedback events the EEG reward is zero. Weights are nonnegative. `1-p` follows Kim's
reward direction; its positive offset and the segment frequency can themselves affect behavior,
which motivates the label-independent controls. The dense progress term is inherited from the
supplied baseline; with gamma=0.99 it is not claimed to be a policy-invariant potential shaping term.

EEGNet and CBraMod sampling uses the same deterministic key (feedback seed, split, participant,
event index, simulated label) and sorted paired trial IDs. Given the same key they select the same
trial; policies can diverge and produce different labels, so this does not promise identical
on-policy feedback sequences. Full-pool controls ignore the simulated label in their sampling key.
The policy observation remains the original robot observation/goal dictionary; true labels and
EEG probabilities affect rewards and logs only.

## Setup and first checks

Run from the repository root in a separate Python 3.12 environment (3.11 was used by the supplied
baseline; this integration was tested on 3.12). Do not upgrade the EEG training environment.

```bash
python3.12 -m venv venv/rl
. venv/rl/bin/activate
python -m pip install -r rl/requirements.txt
python scripts/download_feedback.py --out-dir artifacts/rl_feedback
python -m unittest rl.test_stage2 -v
python -m rl.smoke --out artifacts/rl_smoke
```

The smoke run uses the real FetchReach simulator and original validation feedback CSV, a small
network and 160 steps per condition. It performs gradient updates, reloads every final checkpoint,
checks inference probes and replay size, and resumes the EEGNet periodic checkpoint from step 80
to step 120. It does not use test feedback for policy training. A failed check returns nonzero.

An individual 50,000-step pilot (not automatically run by reading this document):

```bash
python -m rl.train --condition eegnet --subject s11 --phase pilot --seed 0 \
  --out artifacts/rl_runs/pilot_eegnet_s11_s0
```

Use `--condition sparse`, `dense`, or `ideal` without `--subject`. All command output directories
must be new, so earlier results and checkpoints are not overwritten. `--steps` is an additional
step budget for pilot/validation runs and is forbidden for final runs.

## Validation, freezing and final runs

Choose parameters using validation participants only. Initially evaluate the default protocol;
if changing weights, declare the candidate grid before running it (suggested small grid:
w_eeg in [0.1, 0.5, 1.0], w_dense in [1.0, 10.0]). Save each candidate JSON and all results.
Use one common EEG weight for both main decoders and their controls. Select primarily by mean
final-goal success across both validation participants, both main decoders and RL seeds; use mean
final distance as a tie-breaker, then prefer the smaller weight. Do not select on test feedback.
The freeze command records a rationale but does not automate or certify this scientific choice.

`suite` prints the run plan by default. `--execute` runs all jobs and stops on the first failure.
Validation uses RL seeds 100/101/102: 33 runs per candidate, 1,650,000 environment steps at the
default budget. Sparse/dense/ideal controls are each trained once per seed and shared across
participants, avoiding duplicate controls counted as independent evidence.

```bash
python -m rl.suite --phase validation --config rl/protocol.json --out artifacts/rl_validation
# Add --execute only when ready to run this declared budget.
python -m rl.suite --phase validation --config rl/protocol.json --out artifacts/rl_validation --execute
python -m rl.freeze --validation-suite artifacts/rl_validation \
  --rationale "Describe all validation candidates and the prespecified selection rule." \
  --out artifacts/rl_frozen
python -m rl.suite --phase final --config artifacts/rl_frozen/protocol.json \
  --out artifacts/rl_final
python -m rl.suite --phase final --config artifacts/rl_frozen/protocol.json \
  --out artifacts/rl_final --execute
```

Final runs use RL seeds 0-4: 75 runs, 3,750,000 environment steps. Each feedback participant gets
a policy trained using that participant's frozen feedback pool; generalization is across EEG
participants and fresh robot goal episodes, not a held-out subset of that participant's EEG trials.
Formal runs start from scratch and use the full frozen budget. Interrupted formal runs can be
continued for debugging, but the current suite summary deliberately rejects resumed formal runs
because continuation resets the simulator episode; rerun that formal job cleanly for comparability.

All conditions are evaluated without EEG/dense shaping, on the same deterministic held-out goal
seeds 20000-20019 at each checkpoint and at the end. Report final-step success as the primary
metric, plus ever-success, final distance, sparse task return and the full learning curve.
Do not select the best checkpoint from final/test curves; compare the fixed-budget final policies.
Report RL-seed variation within participant and participant results separately. Shared controls
are not three independent participant measurements. Do not treat these few participants as a
large population study or claim that simulator geometry defines actual human preferences.

## Checkpoints, continuation and delivery

Every checkpoint directory contains `model.zip` (policy/critics/target and optimizer/entropy state
as serialized by SB3), `replay_buffer.pkl`, `rng.pt`, fixed inference probes and hashed metadata.
Periodic snapshots are taken after transition storage and before the next gradient update;
final snapshots are taken after training. All periodic and final snapshots are retained.

Continuation restores model, optimizer, replay, RNG and feedback counters, then starts a new
simulator episode. It is not bitwise uninterrupted-trajectory continuation and may skip the
pending gradient update at a periodic snapshot. Only load trusted checkpoint pickle/torch files.

```bash
python -m rl.verify_checkpoint artifacts/rl_runs/pilot_eegnet_s11_s0/checkpoints/final_000050000 \
  --out artifacts/checkpoint_verification.json
python -m rl.train --condition eegnet --subject s11 --phase pilot --seed 0 \
  --resume artifacts/rl_runs/pilot_eegnet_s11_s0/checkpoints/final_000050000 \
  --steps 1000 --out artifacts/rl_runs/pilot_eegnet_s11_s0_continued
python -m rl.summarize artifacts/rl_final
python -m rl.package --runs artifacts/rl_final --out artifacts/rl_release
```

Logs include per-step task/dense/EEG reward components, simulated labels, sampled trial IDs and
probabilities, evaluations, resolved protocol, dependency versions, source hashes and exit status.
SB3 optimization metrics are in `progress.csv`. Preserve stdout/stderr and commands as the suite does.

Upload every packaged run archive, shared evidence, manifest and `SHA256SUMS` to a NEW Release in
`ywen0511/CSCI642-Project`. Point its tag at the actual source commit. Re-download all assets to a
fresh directory, verify SHA-256, then rerun checkpoint inference and a short continuation check.
Keep checkpoints on durable storage until upload verification passes; a Git ignore is not permission
to discard them. Keep original EEG Releases intact. Smoke archives must be explicitly labeled
integration-only, never as a completed policy-performance experiment.

## References

- [Kim: EEG-to-reward framework](https://arxiv.org/html/2507.13171v2)
- [FetchReach task and control frequency](https://robotics.farama.org/envs/fetch/reach/)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Frozen EEG handoff](../docs/rl_handoff.md)
