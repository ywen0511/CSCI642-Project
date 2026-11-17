# Decoder-improvement study: smoke / preflight evidence (not a result)

Scope: **smoke and preflight evidence only** for the decoder-improvement study on branch
`decoder-improvement-v1-experiments`. Not a performance result.

- `evidence.tar`: the diagnostic gate report (`diagnostics_8122312.json`, all 8 checks passed) and its working
  directory with the small diagnostic fits (CBraMod temporal-mean head with the last two blocks unfrozen and EEGNet,
  3 epochs on participant s04; uninterrupted vs interrupted+resumed runs, every best/last/periodic checkpoint with
  full resumable state), the acquisition test report (fresh acquisition of all pinned inputs + byte-identical
  regeneration of the 1-20 Hz band), the environment lock, and the stage plans/decisions committed at packaging time.
- `stage_index.json`, `payload_manifest.json` (every logical asset with size/SHA-256 and every archive member with
  size/SHA-256), `SHA256SUMS`.

Source commit of the diagnostic run: `0869c01`. Verify with
`python scripts/decoder_improvement/release_tools.py fetch --tag <this tag> --out <new dir>`.
