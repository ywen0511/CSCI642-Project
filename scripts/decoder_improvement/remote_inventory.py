#!/usr/bin/env python
"""Paginated inventory of every Release and every asset (all upload states) of the repository.

Uses the public GitHub REST API (no token needed for this public repository). Writes JSON with, per Release:
tag, id, draft/prerelease, target, created/published, and per asset: name, state, size, digest (when the API
provides one), download URL.

Usage: python scripts/decoder_improvement/remote_inventory.py --out remote_inventory.json [--repo OWNER/NAME]
"""
import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone


def get(url, token=None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "csci642-audit"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def paginate(url, token=None):
    out, page = [], 1
    while True:
        batch = get(f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}", token)
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def inventory(repo, token=None):
    releases = paginate(f"https://api.github.com/repos/{repo}/releases", token)
    out = []
    for r in releases:
        assets = paginate(f"https://api.github.com/repos/{repo}/releases/{r['id']}/assets", token)
        out.append({"tag": r["tag_name"], "id": r["id"], "name": r["name"], "draft": r["draft"],
                    "prerelease": r["prerelease"], "target_commitish": r["target_commitish"],
                    "created_at": r["created_at"], "published_at": r["published_at"], "html_url": r["html_url"],
                    "n_assets": len(assets),
                    "n_uploaded": sum(a["state"] == "uploaded" for a in assets),
                    "assets": [{"name": a["name"], "state": a["state"], "size": a["size"], "digest": a.get("digest"),
                                "url": a["browser_download_url"], "id": a["id"]} for a in assets]})
    return {"repository": repo, "generated_utc": datetime.now(timezone.utc).isoformat(), "releases": out}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="ywen0511/CSCI642-Project")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    inv = inventory(a.repo)
    with open(a.out, "w") as f:
        json.dump(inv, f, indent=1)
    for r in inv["releases"]:
        states = {}
        for x in r["assets"]:
            states[x["state"]] = states.get(x["state"], 0) + 1
        print(f"{r['tag']:42s} draft={r['draft']!s:5s} assets={r['n_assets']:4d} states={states} "
              f"bytes={sum(x['size'] for x in r['assets'])}")
