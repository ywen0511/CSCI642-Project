# RL protocol v2: integration smoke evidence (not a performance result)

Scope: **integration smoke only.** These runs are 250-step CPU integration checks with a small [32, 32] policy;
they do not show that any policy learned the task.

- `smoke_r4__*`: the v2 smoke run of record, `python -m rl.smoke --out artifacts/rl_v2_smoke_r4`, run on commit
  `cfe0305` (training sources identical to `8131761`, the code used for all validation runs). All seven
  conditions, 250 simulator steps / 230 SAC updates each, checkpoint reload with replay restoration and exact
  inference probes, scripted real-simulator neutral/progress/retreat feedback check, periodic-snapshot continuation
  80 -> 120 steps. `smoke_r4__manifest.json` lists every file and SHA-256.
- `superseded_rl_v2_smoke__*`, `superseded_rl_v2_smoke_r2__*`, `superseded_rl_v2_smoke_r3__*`: earlier smoke runs
  (commits `bbd3782`, `3113b6b`, `50888a9`), superseded by later fixes and kept as history.

Each archive is `<run>.tar.gz` with every snapshot (`model.zip`, `replay_buffer.pkl`, `rng.pt`, `probe.npz`,
`metadata.json`), logs, configs and evaluation records. `SHA256SUMS` covers every asset in this Release.
Details: `reports/rl_v2_execution.md` on branch `rl-v2-experiments`.
