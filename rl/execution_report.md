# Historical v1 integration execution report

**Historical evidence only:** these runs used protocol v1 (`1-p`, with stalled segments treated as non-error). They do not validate the v2 reward or neutral-segment behavior. Rerun v2 smoke checks before new experiments; see [the active handoff](../GPU_AGENT_HANDOFF.md).

## Implemented and executed

- Read and adapted the supplied FetchReach/SAC archive; provenance is recorded in `baseline_provenance.json`.
- Downloaded the pinned original EEG feedback tables and manifest; all published sizes and SHA-256 checks passed.
- Ran `python -m unittest rl.test_stage2 -v`: 16 tests passed. Coverage includes label-conditioned trial pairing, label-independent controls, schema/identity/probability failures, reward accounting, terminal partial segments, final-phase gating and frozen-config drift rejection.
- Ran `python -m rl.smoke --out artifacts/rl_stage2_smoke_v2` on macOS arm64, Python 3.12.14, CPU, one torch thread. See `requirements-lock.txt` for the exact installed environment.
- All seven conditions completed 160 simulator steps and 140 SAC gradient updates, using a small [32, 32] policy, batch size 16 and validation participant s11 for feedback conditions. The environment contract check also passed for a complete 50-step episode.
- Reloaded each final checkpoint, restored its replay buffer, and reproduced fixed deterministic action probes with maximum absolute error 0.0 (tolerance 1e-6).
- Verified the EEGNet periodic checkpoint at step 80, restored its 80 replay transitions and 59 gradient updates, then continued to step 120 with 120 replay transitions and 99 updates. Simulator continuation begins a new episode; it is not bitwise uninterrupted-trajectory resumption.
- A first integration attempt exposed SB3 modifying `policy_kwargs` in place. Passing a deep copy fixed the protocol-identity mismatch; the complete subsequent integration run passed.
- Verified the validation suite planner: 33 runs per configuration, 1,650,000 environment steps at the 50,000-step default. The final plan has 75 runs / 3,750,000 steps and is gated on a frozen validation protocol.

Machine-readable actual integration evidence: [`smoke_report.json`](smoke_report.json).
All smoke checkpoints, replay buffers, logs, evaluation outputs and resolved configs are retained
in the integration-only Release assets. They must not be described as trained performance baselines.

## Remaining work

- Run the declared validation experiment budget, assess the distance-based label rule and feedback frequency, select common EEG weights using validation only, and freeze the protocol.
- Run the full five-seed final comparison after that freeze. No formal validation or final suite has been executed in this delivery.
- Rendered GIF output is implemented but has not been exercised in this headless integration check.
- No EEG retraining, new EEG calibration, real-time EEG collection, obstacle avoidance, route preference or exact Kinova task reproduction was performed.

The development smoke run does not establish learning success or that EEG feedback improves RL.
