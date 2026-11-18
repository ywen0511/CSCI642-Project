#!/usr/bin/env python
"""Write Release notes for one final-v3 Release from its staged package manifest(s).

Usage: python scripts/final_v3_release_notes.py --tag TAG --commit SHA --title TITLE --status STATUS \
          --stage-dir DIR [--config PATH] [--extra TEXT] --out notes.md
"""
import argparse
import json
from pathlib import Path

REPO = "https://github.com/ywen0511/CSCI642-Project"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--tag", "--commit", "--title", "--status", "--stage-dir", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--config")
    p.add_argument("--extra", default="")
    a = p.parse_args()
    stage = Path(a.stage_dir)
    manifests = sorted(stage.glob("*manifest.json"))
    lines = [f"# {a.title}", "", f"Status: **{a.status}**", "",
             f"- Execution branch: [`final-v3-experiments`]({REPO}/tree/final-v3-experiments); tag pinned to the "
             f"commit the runs were trained from: `{a.commit}`.",
             f"- Assignment: [FINAL_V3_HANDOFF.md]({REPO}/blob/final-v3-experiments/FINAL_V3_HANDOFF.md); status and "
             f"reports: [reports/final_v3/]({REPO}/tree/final-v3-experiments/reports/final_v3)."]
    if a.config:
        lines.append(f"- Protocol: [`{a.config}`]({REPO}/blob/final-v3-experiments/{a.config}).")
    lines += ["", "## Contents", ""]
    total = 0
    for m in manifests:
        d = json.loads(m.read_text())
        scope = m.name.split("__")[0]
        runs = d["runs"]
        size = sum(x["bytes"] for x in d["assets"])
        total += size
        lines.append(f"- `{scope}`: {d['scope']}; {len(runs)} runs ({sum(r['completed'] for r in runs)} completed), "
                     f"{size / 1e9:.2f} GB of archives.")
    lines += ["",
              "Each run archive holds the run directory: final snapshot (SB3 model with optimizer and entropy state, "
              "replay buffer, RNG state, inference probe, metadata), every log, the evaluation record and history, "
              "the transition log and the run config. Periodic snapshots (`checkpoints/step_*`) are not in the "
              "archives; each is listed with its SHA-256 in `<scope>__manifest.json` (`in_archive: false`) and kept "
              "on the project filer. `<scope>__shared_evidence.tar.gz` holds the suite plan, summaries and job logs.",
              "", "## Verify", "",
              "1. Download every asset; `sha256sum -c SHA256SUMS`.",
              "2. Reassemble split archives: `cat <name>.part-* > <name>`, then check the full-archive SHA-256 listed "
              "in the manifest's `assets`.",
              "3. Extract, then `python -m rl.verify_checkpoint <run>/checkpoints/final_*` (hashes and inference "
              "probe on CPU), or `python scripts/rl_verify_release.py` for the whole Release.", ""]
    if a.extra:
        lines += [a.extra, ""]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(a.out, f"{total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
