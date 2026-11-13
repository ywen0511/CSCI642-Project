#!/usr/bin/env python
"""ErrP-HRI robot condition -> ErrP-Bench (dataset_id errphri_robot). Usage: --raw <dataset-ErrP-HRI> --out <dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from errphri import main  # noqa: E402

if __name__ == "__main__":
    main(["--config", str(Path(__file__).resolve().parent / "errphri_robot.yaml")] + sys.argv[1:])
