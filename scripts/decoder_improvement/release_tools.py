#!/usr/bin/env python
"""Release packaging and fresh-download verification with 50 MiB parts and member-level manifests.

  pack     --src DIR --out NEW_DIR --scope TEXT [--member-manifest JSON ...]
           Every file in DIR becomes a Release asset; files larger than 52,428,800 bytes are split into ordered
           parts <name>.part-00, -01, ... Writes payload_manifest.json (logical asset, full size/hash, ordered parts
           with sizes/hashes, archive member lists with size/hash) and SHA256SUMS over every payload asset
           (manifest included, SHA256SUMS itself excluded).
  tar      --src DIR --out FILE.tar   (deterministic, uncompressed, sorted members, mtime 0, uid/gid 0)
  fetch    --tag TAG --out NEW_DIR [--repo R] [--token-env VAR]
           Paginated asset listing (all states), download every asset, check API sizes, SHA256SUMS, reassemble
           parts per payload_manifest.json, check full hashes, extract archives safely (no absolute/traversal paths)
           and check every member against the manifest. Writes NEW_DIR/fetch_report.json; exit 1 on any mismatch.
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

PART = 52_428_800


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def member_list(tar_path):
    out = []
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.isfile():
                h = hashlib.sha256()
                f = tf.extractfile(m)
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
                out.append({"path": m.name, "bytes": m.size, "sha256": h.hexdigest()})
    return out


def deterministic_tar(src, out):
    src = Path(src)
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(src.rglob("*")):
            if p.is_dir():
                continue
            ti = tf.gettarinfo(str(p), arcname=f"{src.name}/{p.relative_to(src)}")
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            with open(p, "rb") as f:
                tf.addfile(ti, f)


def pack(src, out, scope, extra_manifests):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    assets, logical = [], []
    for p in sorted(x for x in src.iterdir() if x.is_file()):
        full = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if p.name.endswith((".tar", ".tar.gz", ".tgz")):
            full["members"] = member_list(p)
        if full["bytes"] > PART:
            parts = []
            with open(p, "rb") as f:
                i = 0
                while True:
                    b = f.read(PART)
                    if not b:
                        break
                    pn = f"{p.name}.part-{i:02d}"
                    (out / pn).write_bytes(b)
                    parts.append({"name": pn, "index": i, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
                    i += 1
            full["parts"] = parts
            full["part_bytes"] = PART
            assets += [x["name"] for x in parts]
        else:
            os.link(p, out / p.name) if os.stat(src).st_dev == os.stat(out).st_dev else (out / p.name).write_bytes(p.read_bytes())
            assets.append(p.name)
        logical.append(full)
    manifest = {"schema": "csci642-release-payload/v1", "scope": scope, "part_bytes": PART, "logical_assets": logical,
                "extra": [json.loads(Path(m).read_text()) for m in extra_manifests],
                "reassemble": "concatenate parts in index order; compare full sha256; or run release_tools.py fetch"}
    (out / "payload_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    assets.append("payload_manifest.json")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(out / a)}  {a}\n" for a in sorted(assets)))
    print(json.dumps({"assets": len(assets) + 1, "logical": len(logical),
                      "bytes": sum((out / a).stat().st_size for a in assets)}))


def api(url, token=None):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "csci642-fetch"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if i == 5:
                raise
            time.sleep(10 * (i + 1))


def download(asset, dest, token=None):
    h = {"User-Agent": "csci642-fetch", "Accept": "application/octet-stream"}
    url = asset["browser_download_url"]
    if token:  # draft releases need the authenticated API asset URL
        url, h["Authorization"] = asset["url"], f"Bearer {token}"
    for i in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600) as r, open(dest, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
            if dest.stat().st_size == asset["size"]:
                return {"name": asset["name"], "status": 200, "bytes": asset["size"], "attempts": i + 1,
                        "url": asset["browser_download_url"]}
        except Exception:
            pass
        time.sleep(15 * (i + 1))
    raise RuntimeError(f"download failed: {asset['name']}")


def safe_extract(tar_path, dest):
    dest = Path(dest).resolve()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if m.name.startswith("/") or ".." in Path(m.name).parts or not str(target).startswith(str(dest)):
                raise ValueError(f"unsafe member path: {m.name}")
            if m.issym() or m.islnk():
                raise ValueError(f"link member rejected: {m.name}")
        tf.extractall(dest, filter="data")


def fetch(tag, out, repo, token_env):
    token = os.environ.get(token_env) if token_env else None
    out = Path(out)
    dl, ex = out / "download", out / "extract"
    dl.mkdir(parents=True, exist_ok=False)
    ex.mkdir()
    rel = None
    for r in [api(f"https://api.github.com/repos/{repo}/releases?per_page=100&page={p}", token) for p in range(1, 4)]:
        rel = rel or next((x for x in r if x["tag_name"] == tag), None)
    if rel is None:
        raise SystemExit(f"release {tag} not found")
    assets, page = [], 1
    while True:
        b = api(f"https://api.github.com/repos/{repo}/releases/{rel['id']}/assets?per_page=100&page={page}", token)
        assets += b
        if len(b) < 100:
            break
        page += 1
    rep = {"tag": tag, "release_url": rel["html_url"], "draft": rel["draft"], "target": rel["target_commitish"],
           "n_assets": len(assets), "states": {}, "problems": []}
    for a in assets:
        rep["states"][a["state"]] = rep["states"].get(a["state"], 0) + 1
        if a["state"] != "uploaded" or a["size"] == 0:
            rep["problems"].append(f"asset not uploaded or empty: {a['name']} ({a['state']}, {a['size']})")
    names = [a["name"] for a in assets]
    if len(set(names)) != len(names):
        rep["problems"].append("duplicate asset names")
    rep["downloads"] = [download(a, dl / a["name"], token) for a in sorted(assets, key=lambda a: a["name"])
                        if a["state"] == "uploaded"]
    sums = dict(reversed(l.split()) for l in (dl / "SHA256SUMS").read_text().splitlines() if l.strip())
    on_disk = sorted(p.name for p in dl.iterdir())
    if sorted(set(sums) | {"SHA256SUMS"}) != on_disk:
        rep["problems"].append(f"SHA256SUMS inventory != downloaded files: {sorted(set(on_disk) ^ (set(sums) | {'SHA256SUMS'}))}")
    rep["sha256_mismatch"] = [n for n, h in sums.items() if (dl / n).exists() and sha256_file(dl / n) != h]
    rep["problems"] += [f"sha256 mismatch {n}" for n in rep["sha256_mismatch"]]
    man = json.loads((dl / "payload_manifest.json").read_text())
    n_members = 0
    for la in man["logical_assets"]:
        path = dl / la["name"]
        if "parts" in la:
            path = out / "reassembled" / la["name"]
            path.parent.mkdir(exist_ok=True)
            with open(path, "wb") as w:
                for prt in sorted(la["parts"], key=lambda x: x["index"]):
                    w.write((dl / prt["name"]).read_bytes())
        if path.stat().st_size != la["bytes"] or sha256_file(path) != la["sha256"]:
            rep["problems"].append(f"full-file mismatch {la['name']}")
            continue
        if "members" in la:
            safe_extract(path, ex / la["name"].split(".tar")[0])
            for m in la["members"]:
                p = ex / la["name"].split(".tar")[0] / m["path"]
                n_members += 1
                if not p.is_file() or p.stat().st_size != m["bytes"] or sha256_file(p) != m["sha256"]:
                    rep["problems"].append(f"member mismatch {la['name']}:{m['path']}")
        if "parts" in la:
            path.unlink()
    rep.update(n_logical=len(man["logical_assets"]), n_members_verified=n_members, ok=not rep["problems"])
    (out / "fetch_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("tag", "n_assets", "states", "n_logical", "n_members_verified", "ok")}))
    if rep["problems"]:
        print("\n".join(rep["problems"][:20]))
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scope", required=True)
    p.add_argument("--member-manifest", nargs="*", default=[])
    t = sub.add_parser("tar")
    t.add_argument("--src", required=True)
    t.add_argument("--out", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--tag", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--repo", default="ywen0511/CSCI642-Project")
    f.add_argument("--token-env", default=None, help="env var holding a token (only for draft releases)")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.src, a.out, a.scope, a.member_manifest)
    elif a.cmd == "tar":
        deterministic_tar(a.src, a.out)
    else:
        fetch(a.tag, a.out, a.repo, a.token_env)
