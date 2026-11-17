# Decoder-improvement study: CV fits, Stages C, D and E new fits (EEGNet learning rate x dropout; band comparison; loss comparison)

Scope: **train-only cross-validation fits** of the decoder-improvement study (branch
`decoder-improvement-v1-experiments`), 90 fits planned in `reports/decoder_improvement/plans/stage_C/D/E.json`
(only fits first run in these stages; exact duplicates reused from earlier stages are in the Release of
the stage that first ran them). All screened, losing, selected and reference fits are included; no fit failed.
Participants: the seven outer-training participants only (inner folds cv0/cv1/cv2); no outer validation or test data.

Per fit (`fit_<fit_id>.tar`): `fit.json` (complete resolved recipe, fold, seed, identity hash), `status.json` (result,
epochs, best epoch zero- and one-based, per-participant AUROC, trainable parameter names/counts, provenance: input,
initialization and training-source hashes, every file's SHA-256), `train_log.jsonl` (every epoch's metrics),
`inner_val_predictions.csv` (per-trial margins/probabilities at the selected checkpoint), `probe.npz`, and **every
checkpoint**: `best.ckpt`, `last.ckpt` and every periodic `epoch_XXXX.ckpt` (CBraMod every 5, EEGNet every 25 epochs;
never pruned), each with model, optimizer, scheduler, RNG (python/numpy/torch CPU/CUDA), sampler generator, epoch and
step counters, early-stopping state, trainability mask and module modes, class weights and input scaler.
`evidence.tar`: plans, decisions, decision log, ledger, sanitized per-fit stdout and scheduler logs.
`stage_index.json` maps fits to archives.

Archives above 52,428,800 bytes are split into ordered parts; `payload_manifest.json` lists every logical archive (size,
SHA-256, ordered parts) and every member file (size, SHA-256). `SHA256SUMS` covers all assets except itself.
Plan/decision commit: `fb51285`. Fetch, reassemble, verify and extract:
`python scripts/decoder_improvement/release_tools.py fetch --tag <this tag> --out <new dir>`.
Top-level resolver for all study Releases: `artifact_index.md` / `artifact_index.json` on the branch.
