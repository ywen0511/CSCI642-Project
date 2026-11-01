#!/usr/bin/env python
"""Render reports/artifact_index.md from the GitHub Release asset list (gh api JSON), the local
SHA256SUMS, the download-verification result and the resumption-check results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-json", default=None, help="gh api repos/<o>/<r>/releases/tags/<tag>")
    ap.add_argument("--local-sizes", default=None,
                    help="'name bytes' lines of locally packaged assets (used when nothing was uploaded)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repo-url", default="https://github.com/ywen0511/CSCI642-Project")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--sha256sums", required=True)
    ap.add_argument("--verify-json", required=True)
    ap.add_argument("--resume-jsons", nargs="*", default=[])
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--out", default="reports/artifact_index.md")
    a = ap.parse_args()
    if a.release_json:
        rel = json.loads(Path(a.release_json).read_text())
        uploaded = True
    else:
        sizes = [l.split() for l in Path(a.local_sizes).read_text().splitlines() if l.strip()]
        rel = {"html_url": f"{a.repo_url}/releases/tag/{a.tag} (NOT CREATED)", "tag_name": a.tag,
               "published_at": None, "assets": [{"name": n, "size": int(b)} for n, b in sizes]}
        uploaded = False
    sums = dict(reversed(l.split()) for l in Path(a.sha256sums).read_text().splitlines())
    ver = json.loads(Path(a.verify_json).read_text())
    repo = rel["html_url"].split("/releases/")[0]
    status = [] if uploaded else [
        "**Status: NOT UPLOADED.** The GitHub Release could not be created because no available account "
        "has write access to the repository (both `zhengtaoyao` and `KZTRGG_gme` report `push: false`). "
        f"All assets below are packaged and checksummed at `{a.local_dir}` on the project filer. The "
        "verification results in this file come from a local rehearsal: the assets were copied into a fresh "
        "directory, which stood in for a GitHub download. Repeat the checks after the real upload "
        "(see Commands).\n"]
    L = ["# Artifact Index\n", *status,
         f"- Release: {rel['html_url']}",
         f"- Tag: `{rel['tag_name']}` -> source commit `{a.source_sha}`",
         f"- Repository: {repo}",
         f"- Published: {rel.get('published_at')}",
         "- Raw EEG and the original pretrained weights are not redistributed; see `reports/data_audit.md` "
         "and `reports/protocol_decisions.md` for sources, revisions and SHA-256.\n",
         "## Assets\n", "| asset | bytes | SHA-256 | verified after copy/download |", "|---|---|---|---|"]
    for asset in sorted(rel["assets"], key=lambda x: x["name"]):
        n = asset["name"]
        v = ver["assets"].get(n, {})
        okv = "yes" if v.get("sha256_ok") and v.get("size_matches_api", True) else ("n/a" if n == "SHA256SUMS" else "NO")
        L.append(f"| `{n}` | {asset['size']} | `{sums.get(n, '(checksum file)')}` | {okv} |")
    L.append("")
    L.append(f"All assets verified in a fresh directory: **{ver['all_assets_ok']}**. "
             f"Prediction reproduction from the verified `best.ckpt` copies (|dp| <= {ver['tolerance_abs_p']}): "
             f"**{ver['all_reproduction_ok']}**.\n")
    L.append("| checkpoint | split | n | max abs diff p_error_raw | max abs diff p_error | within tolerance |")
    L.append("|---|---|---|---|---|---|")
    for k, r in ver["reproduction"].items():
        for s in ["validation", "test"]:
            x = r[s]
            L.append(f"| `{k}` | {s} | {x['n']} | {x['max_abs_diff_p_error_raw']:.2e} | {x['max_abs_diff_p_error']:.2e} | "
                     f"{x['within_tolerance']} |")
    if a.resume_jsons:
        L.append("\n## Training resumption from the verified `last.ckpt` copies\n")
        L.append("| source checkpoint | resumed at epoch / step | after | new epochs finite loss | original file unchanged | ok |")
        L.append("|---|---|---|---|---|---|")
        for p in a.resume_jsons:
            r = json.loads(Path(p).read_text())
            L.append(f"| `{Path(r['source_ckpt']).parent.parent.name}/last.ckpt` | {r['resumed_epoch']} / "
                     f"{r['resumed_global_step']} | {r['final_epoch']} / {r['final_global_step']} | "
                     f"{r['all_losses_finite']} | {r['source_sha256_before'] == r['source_sha256_after']} | {r['ok']} |")
    tag = rel["tag_name"]
    L += ["\n## Commands\n", "```bash",
          f"# download all assets of the release into a fresh directory",
          f"gh release download {tag} --repo {repo.split('github.com/')[1]} --dir release_download",
          "cd release_download && sha256sum -c SHA256SUMS && cd ..",
          "# extract (one archive per run + shared metadata)",
          "mkdir -p release_extract && for f in release_download/*.tar.gz; do tar -xzf \"$f\" -C release_extract; done",
          "# rebuild the derived data (needs the raw dataset, see README) and reproduce predictions",
          "python scripts/verify_release.py --download-dir release_download --extract-dir release_extract_check \\",
          "    --prep-dir data/derived/prep_v1 --out verify.json",
          "# resume training from a downloaded last.ckpt (copied; the original stays untouched)",
          "python scripts/check_resume.py --mode continue --ckpt release_extract/runs/<run_id>/checkpoints/last.ckpt \\",
          "    --prep-dir data/derived/prep_v1 --work-dir /tmp/resume_<run_id> --epochs 2 --out resume_<run_id>.json",
          "```"]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
