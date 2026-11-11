#!/usr/bin/env python
"""Top-level resolver for every decoder-study and recovery Release (from the live, paginated remote inventory).

Usage: python scripts/decoder_improvement/artifact_index.py --out-json artifact_index.json --out-md artifact_index.md \
          --tags TAG [TAG ...] [--payload-manifests DIR]

For each tag: Release URL, target commit, draft state; every asset with scope, URL, size, GitHub SHA-256 digest and
upload state; from the Release's payload_manifest.json, every logical archive with its full size/SHA-256 and the
ordered list of parts (asset URLs) that reassemble it. Fails if any asset is not 'uploaded'.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_inventory import inventory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="ywen0511/CSCI642-Project")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    inv = inventory(a.repo)
    rel = {r["tag"]: r for r in inv["releases"]}
    out = {"repository": a.repo, "generated_utc": inv["generated_utc"], "releases": []}
    md = ["# Decoder-improvement study: artifact index", "",
          f"Generated {inv['generated_utc']} from the live paginated GitHub inventory by "
          "`scripts/decoder_improvement/artifact_index.py` (machine-readable: `artifact_index.json`, which also lists "
          "every asset URL, size and SHA-256 digest and the ordered parts of every logical archive).", "",
          "| Release | target | assets | uploaded | GB | logical archives | split archives |", "|---|---|---|---|---|---|---|"]
    problems = []
    for t in a.tags:
        r = rel.get(t)
        if r is None:
            problems.append(f"missing release {t}")
            continue
        assets = {x["name"]: x for x in r["assets"]}
        bad = [n for n, x in assets.items() if x["state"] != "uploaded" or x["size"] == 0]
        problems += [f"{t}: asset not uploaded {n}" for n in bad]
        logical = []
        if "payload_manifest.json" in assets:
            with urllib.request.urlopen(urllib.request.Request(assets["payload_manifest.json"]["url"],
                                                               headers={"User-Agent": "csci642-index"}), timeout=120) as f:
                pm = json.loads(f.read())
            for la in pm["logical_assets"]:
                parts = [{"name": p["name"], "url": assets[p["name"]]["url"], "bytes": p["bytes"], "sha256": p["sha256"]}
                         for p in la.get("parts", [])]
                logical.append({"name": la["name"], "bytes": la["bytes"], "sha256": la["sha256"],
                                "url": None if parts else assets[la["name"]]["url"], "parts": parts,
                                "n_members": len(la.get("members", []))})
        out["releases"].append({"tag": t, "url": r["html_url"], "target_commitish": r["target_commitish"],
                                "draft": r["draft"], "published_at": r["published_at"],
                                "assets": [{"name": n, "url": x["url"], "bytes": x["size"], "sha256": (x["digest"] or "").replace("sha256:", ""),
                                            "state": x["state"]} for n, x in sorted(assets.items())],
                                "logical_archives": logical})
        md.append(f"| [`{t}`]({r['html_url']}) | `{r['target_commitish'][:7]}` | {len(assets)} | {len(assets) - len(bad)} | "
                  f"{sum(x['size'] for x in assets.values()) / 1e9:.2f} | {len(logical)} | {sum(1 for l in logical if l['parts'])} |")
    md += ["", "Download, reassemble, verify and extract any Release: "
               "`python scripts/decoder_improvement/release_tools.py fetch --tag <tag> --out <new dir>`.",
           "Reassembly by hand: concatenate the parts listed for a logical archive in index order and compare the "
           "full-archive SHA-256 given in `artifact_index.json`.", ""]
    out["problems"] = problems
    Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md[4:4 + len(a.tags) + 1]))
    if problems:
        print(problems[:10])
        sys.exit(1)


if __name__ == "__main__":
    main()
