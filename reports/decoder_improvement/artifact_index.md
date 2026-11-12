# Decoder-improvement study: artifact index

Generated 2026-11-12T04:48:57.760581+00:00 from the live paginated GitHub inventory by `scripts/decoder_improvement/artifact_index.py` (machine-readable: `artifact_index.json`, which also lists every asset URL, size and SHA-256 digest and the ordered parts of every logical archive).

| Release | target | assets | uploaded | GB | logical archives | split archives |
|---|---|---|---|---|---|---|
| [`decoder-v1-smoke-20261111-0869c01`](https://github.com/ywen0511/CSCI642-Project/releases/tag/decoder-v1-smoke-20261111-0869c01) | `0869c01` | 6 | 6 | 0.13 | 2 | 1 |
| [`decoder-v1-cv-a-20261111-6baf061`](https://github.com/ywen0511/CSCI642-Project/releases/tag/decoder-v1-cv-a-20261111-6baf061) | `6baf061` | 671 | 671 | 33.19 | 65 | 63 |
| [`decoder-v1-cv-cde-20261111-fb51285`](https://github.com/ywen0511/CSCI642-Project/releases/tag/decoder-v1-cv-cde-20261111-fb51285) | `fb51285` | 309 | 309 | 12.16 | 92 | 27 |
| [`decoder-v1-cv-f-20261111-026103a`](https://github.com/ywen0511/CSCI642-Project/releases/tag/decoder-v1-cv-f-20261111-026103a) | `026103a` | 182 | 182 | 8.69 | 20 | 12 |
| [`decoder-v1-final-20261111-f8031ec`](https://github.com/ywen0511/CSCI642-Project/releases/tag/decoder-v1-final-20261111-f8031ec) | `f8031ec` | 95 | 95 | 3.98 | 23 | 10 |
| [`rl-v2-recovery-final-20261111-25bb0f4`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-recovery-final-20261111-25bb0f4) | `25bb0f4` | 379 | 379 | 18.77 | 77 | 75 |
| [`rl-v2-recovery-w010-20261111-cfe0305`](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-recovery-w010-20261111-cfe0305) | `cfe0305` | 169 | 169 | 8.24 | 35 | 33 |

Download, reassemble, verify and extract any Release: `python scripts/decoder_improvement/release_tools.py fetch --tag <tag> --out <new dir>`.
Reassembly by hand: concatenate the parts listed for a logical archive in index order and compare the full-archive SHA-256 given in `artifact_index.json`.

