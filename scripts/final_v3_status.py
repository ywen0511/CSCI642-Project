#!/usr/bin/env python
"""Update reports/final_v3/status.json: dotted-key assignments with JSON values, plus an updated_utc stamp.

Usage: python scripts/final_v3_status.py phase='"stage1"' suites.sensitivity_w010.completed=75 blockers='[]'
"""
import datetime
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "reports/final_v3/status.json"

if __name__ == "__main__":
    status = json.loads(PATH.read_text()) if PATH.exists() else {}
    for arg in sys.argv[1:]:
        key, value = arg.split("=", 1)
        node = status
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = json.loads(value)
    status["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PATH.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
