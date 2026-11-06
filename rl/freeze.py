"""Freeze a chosen validation protocol before enabling test-participant feedback."""
import argparse
import json
from pathlib import Path

from .feedback import sha256
from .summarize import summarize


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-suite", required=True)
    p.add_argument("--rationale", required=True, help="Document validation-only choice and all candidate suites")
    p.add_argument("--out", required=True, help="New directory for the frozen protocol and evidence")
    a = p.parse_args()
    result = summarize(a.validation_suite)
    if result["split"] != "validation":
        raise ValueError("Cannot freeze from test results")
    cfg = result["config"].copy()
    if cfg["status"] != "pilot":
        raise ValueError("Expected a validation candidate protocol")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    result["selection_rationale"] = a.rationale
    evidence = out / "validation_evidence.json"
    evidence.write_text(json.dumps(result, indent=2) + "\n")
    cfg.update(status="frozen", validation_evidence_file=evidence.name,
               validation_evidence_sha256=sha256(evidence))
    (out / "protocol.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(out / "protocol.json")
