#!/usr/bin/env python
"""Package decoder-study runs for a scoped Release (then split by release_tools.py pack).

Usage: python scripts/decoder_improvement/package_study.py --fits FIT_ID [...] | --plans PLAN.json [...]
          --runs-root RUNS --logs-root PROJECT --evidence FILE_OR_DIR [...] --out NEW_STAGE_DIR

Writes one deterministic uncompressed tar per fit directory (fit_<fit_id>.tar: every file of the run, including all
best/last/final/periodic checkpoints, logs, predictions, probes, status) and evidence.tar (plans, decisions,
ledger, per-fit stdout logs, scheduler logs, given evidence files). Text logs outside the fit directories are
sanitized: the private project path is replaced by <PROJECT_ROOT> and other filer paths by <FILER>; fit
directories contain no private paths (checked) and are archived byte-for-byte. Writes stage_index.json mapping
each fit to its archive, status, epochs and checkpoint list.
"""
import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "decoder_improvement"))
from release_tools import deterministic_tar  # noqa: E402

PROJECT = "/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-decoder"
FILER = re.compile(r"/data/rtxw01/nbak/robot_intel[^\s\"',:]*")


def sanitize(text):
    return FILER.sub("<FILER>", text.replace(PROJECT, "<PROJECT_ROOT>"))


def add_text(tf, arcname, text):
    import io
    data = text.encode()
    ti = tarfile.TarInfo(arcname)
    ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
    tf.addfile(ti, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="*", default=[])
    ap.add_argument("--plans", nargs="*", default=[])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fits = list(dict.fromkeys(a.fits + [f["fit_id"] for p in a.plans for f in json.loads(Path(p).read_text())["fits"]]))
    index = {"fits": {}}
    for fid in fits:
        d = runs / fid
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".log", ".txt"} and PROJECT in p.read_text(errors="ignore"):
                raise ValueError(f"private path inside fit directory: {p}")
        st = json.loads((d / "status.json").read_text())
        deterministic_tar(d, out / f"fit_{fid}.tar")
        index["fits"][fid] = {"archive": f"fit_{fid}.tar", "state": st.get("state"), "mode": st.get("mode"),
                              "epochs_completed": st.get("epochs_completed"),
                              "best_epoch_one_based": st.get("best_epoch_one_based"),
                              "checkpoints": sorted(p.name for p in (d / "checkpoints").glob("*.ckpt"))}
    logs = Path(a.logs_root)
    with tarfile.open(out / "evidence.tar", "w", format=tarfile.PAX_FORMAT) as tf:
        for fid in fits:
            p = logs / "runs_logs" / f"{fid}.log"
            if p.exists():
                add_text(tf, f"evidence/runs_logs/{fid}.log", sanitize(p.read_text(errors="replace")))
        for p in sorted((logs / "logs").glob("stage_*.log")) + sorted((logs / "logs").glob("diag_*.log")):
            add_text(tf, f"evidence/scheduler_logs/{p.name}", sanitize(p.read_text(errors="replace")))
        if (logs / "ledger" / "jobs.jsonl").exists():
            add_text(tf, "evidence/ledger/jobs.jsonl", sanitize((logs / "ledger" / "jobs.jsonl").read_text()))
        for e in a.evidence + a.plans:
            e = Path(e)
            for p in ([e] if e.is_file() else sorted(x for x in e.rglob("*") if x.is_file())):
                rel = p.name if e.is_file() else f"{e.name}/{p.relative_to(e)}"
                if p.suffix in {".json", ".jsonl", ".csv", ".log", ".txt", ".md"}:
                    add_text(tf, f"evidence/files/{rel}", sanitize(p.read_text(errors="replace")))
                else:
                    ti = tf.gettarinfo(str(p), arcname=f"evidence/files/{rel}")
                    ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
                    with open(p, "rb") as f:
                        tf.addfile(ti, f)
    index["sanitization"] = {"<PROJECT_ROOT>": "private project directory on the cluster filer",
                             "<FILER>": "other private filer paths", "applied_to": "evidence logs only"}
    (out / "stage_index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps({"fits": len(fits), "out": str(out)}))


if __name__ == "__main__":
    main()
