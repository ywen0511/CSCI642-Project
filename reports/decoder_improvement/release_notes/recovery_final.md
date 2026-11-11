# Historical recovery: RL v2 final comparison (frozen protocol, test participants s02/s08/s09, RL seeds 0-4)

Scope: **historical recovery** of the RL v2 suite published as [`rl-v2-final-20261110-25bb0f4`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-final-20261110-25bb0f4).
That Release holds each run's final snapshot only (by the user's instruction of 2026-11-10). This recovery Release
holds the **complete original run archives** (75 runs): every periodic (10k-step) and final snapshot with model,
optimizer/entropy state, replay buffer, RNG state, inference probe and metadata, plus all logs and evaluation records.
No training was rerun: the archives are the exact original packages (training commit `25bb0f4`), each checked
against the full-archive SHA-256 in the original package manifest before splitting.

The original Release and its assets are unchanged. Audit: `reports/decoder_improvement/historical_delivery_audit.json`
on branch `decoder-improvement-v1-experiments`.

Layout: archives larger than 52,428,800 bytes are split into ordered parts `<archive>.part-NN`.
`payload_manifest.json` lists every logical archive (full size and SHA-256, ordered parts with sizes and hashes,
every member file with size and SHA-256); `original_package_manifest.json` is the manifest written when the suite
was packaged. `SHA256SUMS` covers every asset except itself. Download, reassemble, verify and extract
automatically with:

    python scripts/decoder_improvement/release_tools.py fetch --tag <this tag> --out <new dir>
