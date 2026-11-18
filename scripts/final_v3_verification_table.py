#!/usr/bin/env python
"""Markdown table of fresh-download verification reports (scripts/rl_verify_release.py output), one row per Release,
plus a continuation-check table. Used for reports/final_project_report.md Section 8 and reports/final_v3/execution.md.

Usage: python scripts/final_v3_verification_table.py reports/final_v3/release_verification --out table.md
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = Path(a.dir)
    lines = ["| Release | assets | GB | inventory + every SHA-256 | split archives / full-archive hashes checked | "
             "snapshots hash-validated | inference verified (final / periodic) | max action diff | ok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "tag" not in r or "n_assets_api" not in r:
            continue
        inv = bool(r["inventory_ok"] and r["sha256sums_cover_all_assets"] and not r["sha256_mismatches"])
        lines.append(
            f"| [`{r['tag']}`]({r['release_url']}) | {r['n_assets_api']} | {r['bytes_api'] / 1e9:.2f} | {inv} | "
            f"{r['n_split_archives']} / {r['full_archive_hash_checked']} | {r['n_snapshots_hash_validated']} | "
            f"{r['n_final_snapshots_verified']} / {r['n_periodic_snapshots_verified']} | "
            f"{r['max_action_difference']:.1e} | **{r['ok']}** |")
    cont = ["| continuation report | downloaded snapshot | clean checkout | steps | updates | replay size | restored transitions unchanged | ok |",
            "|---|---|---|---|---|---|---|---|"]
    for f in sorted(d.glob("continuation_*.json")):
        if "attempt" in f.name:
            continue
        r = json.loads(f.read_text())
        v = r.get("verification", {})
        snap = "/".join(Path(r["snapshot"]).parts[-4:])
        cont.append(f"| [`{f.name}`]({f.name}) | `{snap}` | `{r['checkout_git_sha'][:7]}` | "
                    f"{r['snapshot_steps']} -> {v.get('steps')} | {r['snapshot_updates']} -> {v.get('updates')} | "
                    f"{r['snapshot_replay_size']} -> {v.get('replay_size')} | {r['restored_transitions_unchanged']} | **{r['ok']}** |")
    Path(a.out).write_text("\n".join(lines) + "\n\n" + "\n".join(cont) + "\n")
    print("\n".join(lines + [""] + cont))


if __name__ == "__main__":
    main()
