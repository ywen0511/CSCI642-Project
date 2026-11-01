# Artifact Index

- Release: https://github.com/ywen0511/CSCI642-Project/releases/tag/eeg-stage1-20261101-bccbb8a
- Tag: `eeg-stage1-20261101-bccbb8a` -> source commit `bccbb8a2918c0a08381852feb1c6b1eb81e19548`
- Repository: https://github.com/ywen0511/CSCI642-Project
- Published: 2026-11-01T20:43:56Z
- Raw EEG and the original pretrained weights are not redistributed; see `reports/data_audit.md` and `reports/protocol_decisions.md` for sources, revisions and SHA-256.

## Assets

| asset | bytes | SHA-256 | verified after copy/download |
|---|---|---|---|
| `SHA256SUMS` | 2252 | `(checksum file)` | n/a |
| `manifest.json` | 103574 | `b9803034673b5bec41c2ea7da2bf20c2bc4b07f348d2d8b0241d944f17d783b9` | yes |
| `metrics_summary.json` | 102872 | `f158e571469feef4055023cd45100b2d32d20d11085fe6c4715c988ac47cacca` | yes |
| `run_bd_ctnet_s42.tar.gz` | 6859952 | `ea0b4f63fcf6ea0c23b1cf23440fdb0b22324fed7a1c90f9bad31af17e6d33d1` | yes |
| `run_bd_eegconformer_s42.tar.gz` | 20465866 | `06a16d944e53894d2f3edf1df8d3fef16af92ddda2c64c3c187f66b1bd7be05c` | yes |
| `run_bd_eeginception_erp_s42.tar.gz` | 1420069 | `a380e6bdd9d02fdfaec6d6648d57ee361be08008cc7f7759f25d32202f31fb85` | yes |
| `run_bd_eegnex_s42.tar.gz` | 2633989 | `9e8b90ea63bdf27cb2e84e6b992d7108e2c84fddb9e9c4e3b124920502af2280` | yes |
| `run_bd_eegpt_s42.tar.gz` | 1401092393 | `4b5101d34c214e804210771e43738d32f7d2775d1645d1e65987b10d2fad550f` | yes |
| `run_bd_labram_s42.tar.gz` | 387081478 | `5af903ff8c1628f4f391ab585f392f5bfe55f3bfb419361ee6489f956d3403b4` | yes |
| `run_bd_xdawn_riemann_s42.tar.gz` | 84081 | `914d30b25e04eb9a12b0890a40f5ec9c5629ffab9f0a3076ed81a9f61c0d48d5` | yes |
| `run_cbramod_bblr1e-4_s42.tar.gz` | 616307015 | `36fc2d196c89bea7c1ea0665275317aa955725288cae50c6f4df79df624cb6f4` | yes |
| `run_cbramod_bblr1e-4_s43.tar.gz` | 616246691 | `5c77521c8736d2bde1ef82016aa89798bed9498c6595a63a0fd79b5be4e9d21c` | yes |
| `run_cbramod_bblr1e-4_s44.tar.gz` | 616188538 | `83de1e97d296b551623008ea180a48500495dec77dd6f842210a84ea388931b8` | yes |
| `run_cbramod_bblr1e-4_s45.tar.gz` | 616490817 | `87a10eb7cf71773cf8001d76c30bee99d7d946a893720d75bc0d07de8e0bd746` | yes |
| `run_cbramod_bblr1e-4_s46.tar.gz` | 616826863 | `05f936d03a8203659450d17cbee7bbcbf5efa12f3dad3c5f2a0ab710d7ff9038` | yes |
| `run_cbramod_bblr3e-5_s42.tar.gz` | 616458871 | `2397cc3dbfbfde0121b26a9ad33e0ad918fae43c7e9588a7e5bebad5d13849da` | yes |
| `run_eegnet_lr1e-3_s42.tar.gz` | 417092 | `8e795e2a43d9a8f2767775bd4e3717273e913f64f95844b50ac625b9f73a8dd5` | yes |
| `run_eegnet_lr1e-3_s43.tar.gz` | 334911 | `10aaab07143eb3fbdf6bba571f9f3dfe6e7d364de3f0e0cac8a8b0c9f6958035` | yes |
| `run_eegnet_lr1e-3_s44.tar.gz` | 407548 | `e10e3d8acf775671710dcfaf5c74654a1b37c7839e9480804b14dc8c3d8deff5` | yes |
| `run_eegnet_lr1e-3_s45.tar.gz` | 397846 | `80bbf8073c63e5c42f72933fe4fb9f93536d1da1953757c03e19c7201f84e3d8` | yes |
| `run_eegnet_lr1e-3_s46.tar.gz` | 336065 | `155cd28a81fb12dcea182c39af8eb1beade13c361f4e802b70b708f43553b1ba` | yes |
| `run_eegnet_lr3e-4_s42.tar.gz` | 161514 | `564e31e7c284d6a31a08a774acdd90cc838a3298434dde6b8da2d4ffc3795360` | yes |
| `shared_metadata.tar.gz` | 956584 | `79ce4bcdbed74013223fc2fdd77df6fda32c191b5737660983aa4a3a427d9bc9` | yes |
| `test_predictions.csv` | 434000 | `b199805ace8dd1e977f033e64f7bb9528a858bbd12c8e465a8d10dd9ba756169` | yes |
| `validation_predictions.csv` | 302783 | `db0f3a9aa35a06af16550ee8a65904e9708f8d38981450ea1fe4a3d269eda3eb` | yes |

All assets verified in a fresh directory: **True**. Prediction reproduction from the verified `best.ckpt` copies (|dp| <= 1e-06): **True**.

| checkpoint | split | n | max abs diff p_error_raw | max abs diff p_error | within tolerance |
|---|---|---|---|---|---|
| `eegnet_lr1e-3_s42:best` | validation | 959 | 5.00e-11 | 4.99e-11 | True |
| `eegnet_lr1e-3_s42:best` | test | 1429 | 5.00e-11 | 5.00e-11 | True |
| `cbramod_bblr1e-4_s42:best` | validation | 959 | 4.98e-11 | 4.98e-11 | True |
| `cbramod_bblr1e-4_s42:best` | test | 1429 | 5.00e-11 | 4.99e-11 | True |

## Training resumption from the verified `last.ckpt` copies

| source checkpoint | resumed at epoch / step | after | new epochs finite loss | original file unchanged | ok |
|---|---|---|---|---|---|
| `eegnet_lr1e-3_s42/last.ckpt` | 154 / 8316 | 156 / 8424 | True | True | True |
| `cbramod_bblr1e-4_s42/last.ckpt` | 39 / 2106 | 41 / 2214 | True | True | True |

## Commands

```bash
# download all assets of the release into a fresh directory
gh release download eeg-stage1-20261101-bccbb8a --repo ywen0511/CSCI642-Project --dir release_download
cd release_download && sha256sum -c SHA256SUMS && cd ..
# extract (one archive per run + shared metadata)
mkdir -p release_extract && for f in release_download/*.tar.gz; do tar -xzf "$f" -C release_extract; done
# rebuild the derived data (needs the raw dataset, see README) and reproduce predictions
python scripts/verify_release.py --download-dir release_download --extract-dir release_extract_check \
    --prep-dir data/derived/prep_v1 --out verify.json
# resume training from a downloaded last.ckpt (copied; the original stays untouched)
python scripts/check_resume.py --mode continue --ckpt release_extract/runs/<run_id>/checkpoints/last.ckpt \
    --prep-dir data/derived/prep_v1 --work-dir /tmp/resume_<run_id> --epochs 2 --out resume_<run_id>.json
```
