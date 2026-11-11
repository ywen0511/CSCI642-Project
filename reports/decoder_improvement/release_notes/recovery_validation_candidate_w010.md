# Historical recovery: RL v2 validation candidate w010 (EEG weight 0.1; validation participants s11/s13; RL seeds 100-102)

Scope: **historical recovery** of the RL v2 suite published as [`rl-v2-validation-w010-20261110-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-validation-w010-20261110-cfe0305).
That Release holds each run's final snapshot only (by the user's instruction of 2026-11-10). This recovery Release
holds the **complete original run archives** (33 runs): every periodic (10k-step) and final snapshot with model,
optimizer/entropy state, replay buffer, RNG state, inference probe and metadata, plus all logs and evaluation records.
No training was rerun: the archives are the exact original packages (training commit `cfe0305`), each checked
against the full-archive SHA-256 in the original package manifest before splitting.

The original Release and its assets are unchanged. Audit: `reports/decoder_improvement/historical_delivery_audit.json`
on branch `decoder-improvement-v1-experiments`.

Layout: archives larger than 52,428,800 bytes are split into ordered parts `<archive>.part-NN`.
`payload_manifest.json` lists every logical archive (full size and SHA-256, ordered parts with sizes and hashes,
every member file with size and SHA-256); `original_package_manifest.json` is the manifest written when the suite
was packaged. `SHA256SUMS` covers every asset except itself. Download, reassemble, verify and extract
automatically with:

    python scripts/decoder_improvement/release_tools.py fetch --tag <this tag> --out <new dir>
