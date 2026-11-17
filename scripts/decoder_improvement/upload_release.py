#!/usr/bin/env python
"""Idempotent, ledgered upload of a payload directory to a DRAFT GitHub Release, digest round-trip, then publish.

Usage (laptop, authenticated gh CLI):
  python scripts/decoder_improvement/upload_release.py --payload DIR --tag TAG --target SHA --title T --notes FILE
         --ledger LEDGER.jsonl [--publish]

1. Verifies the local payload against its SHA256SUMS.
2. Creates the Release as a draft (if it does not exist; existing published Releases are never modified).
3. Uploads every asset that is missing, not 'uploaded', or size-mismatched (bounded retries with backoff; failed
   'starter' entries of THIS draft are deleted before re-upload; nothing else is ever deleted).
4. Authenticated round trip: every remote asset must be 'uploaded' with the local size and a GitHub sha256 digest
   equal to the local SHA-256; no unexpected extra asset.
5. --publish: publishes the draft only after step 4 passes.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = "ywen0511/CSCI642-Project"


def sh(*args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check)


def api(path, method="GET"):
    r = sh("gh", "api", "-X", method, path, check=False)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:300])
    return json.loads(r.stdout) if r.stdout.strip() else None


def find_release(tag):
    page = 1
    while True:
        rels = api(f"repos/{REPO}/releases?per_page=100&page={page}")
        for r in rels:
            if r["tag_name"] == tag:
                return r
        if len(rels) < 100:
            return None
        page += 1


def assets(rid):
    out, page = [], 1
    while True:
        b = api(f"repos/{REPO}/releases/{rid}/assets?per_page=100&page={page}")
        out += b
        if len(b) < 100:
            return out
        page += 1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(ledger, **rec):
    with open(ledger, "a") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **rec}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    d = Path(a.payload)
    sums = dict(reversed(l.split()) for l in (d / "SHA256SUMS").read_text().splitlines() if l.strip())
    local = {p.name: p for p in d.iterdir() if p.is_file()}
    if set(local) != set(sums) | {"SHA256SUMS"}:
        raise SystemExit(f"local payload inventory != SHA256SUMS: {sorted(set(local) ^ (set(sums) | {'SHA256SUMS'}))}")
    bad = [n for n, h in sums.items() if sha(local[n]) != h]
    if bad:
        raise SystemExit(f"local checksum mismatch: {bad[:5]}")
    want = {n: {"size": p.stat().st_size, "sha256": sums.get(n) or sha(p)} for n, p in local.items()}
    rel = find_release(a.tag)
    if rel is None:
        sh("gh", "release", "create", a.tag, "--repo", REPO, "--draft", "--target", a.target, "--title", a.title,
           "--notes-file", a.notes)
        for _ in range(20):  # the release list is eventually consistent right after creation
            rel = find_release(a.tag)
            if rel:
                break
            time.sleep(15)
        if rel is None:
            raise SystemExit(f"draft {a.tag} created but not listed")
        log(a.ledger, tag=a.tag, event="draft_created", release_id=rel["id"])
    elif not rel["draft"]:
        remote = {x["name"]: x for x in assets(rel["id"])}
        if all(n in remote and remote[n]["state"] == "uploaded" and remote[n]["size"] == w["size"] for n, w in want.items()):
            print("already published and complete:", a.tag)
            return
        raise SystemExit(f"{a.tag} is already published but incomplete; published assets are never modified")
    for rnd in range(1, 9):
        remote = {x["name"]: x for x in assets(rel["id"])}
        todo = [n for n, w in want.items()
                if n not in remote or remote[n]["state"] != "uploaded" or remote[n]["size"] != w["size"]]
        print(f"round {rnd}: {len(todo)} to upload", flush=True)
        if not todo:
            break
        for n in sorted(todo, key=lambda n: want[n]["size"]):
            if n in remote:  # failed entry of this draft only
                api(f"repos/{REPO}/releases/assets/{remote[n]['id']}", "DELETE")
                log(a.ledger, tag=a.tag, event="deleted_failed_draft_entry", asset=n, state=remote[n]["state"])
            r = sh("timeout", "3600", "gh", "release", "upload", a.tag, str(local[n]), "--repo", REPO, check=False)
            log(a.ledger, tag=a.tag, event="upload", asset=n, rc=r.returncode, round=rnd,
                err=r.stderr.strip()[-200:] if r.returncode else "")
            if r.returncode:
                time.sleep(min(30 * rnd, 300))
    remote = {x["name"]: x for x in assets(rel["id"])}
    problems = []
    for n, w in want.items():
        x = remote.get(n)
        if x is None or x["state"] != "uploaded" or x["size"] != w["size"]:
            problems.append(f"{n}: {None if x is None else (x['state'], x['size'])}")
        elif x.get("digest") and x["digest"] != f"sha256:{w['sha256']}":
            problems.append(f"{n}: digest {x['digest']} != local")
    extra = sorted(set(remote) - set(want))
    problems += [f"unexpected remote asset {n}" for n in extra]
    report = {"tag": a.tag, "release_id": rel["id"], "n_assets": len(want), "digest_checked": sum(
        1 for n in want if remote.get(n, {}).get("digest")), "problems": problems}
    log(a.ledger, event="roundtrip", **report)
    print(json.dumps(report)[:1500])
    if problems:
        raise SystemExit(1)
    if a.publish:
        sh("gh", "release", "edit", a.tag, "--repo", REPO, "--draft=false")
        log(a.ledger, tag=a.tag, event="published")
        print("PUBLISHED", a.tag)


if __name__ == "__main__":
    main()
