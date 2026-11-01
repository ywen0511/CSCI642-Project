"""Shared helpers: seeding, atomic writes, RNG state capture, hashing, environment info."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_states(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if st.get("torch_cuda") is not None and torch.cuda.is_available():
        n = min(len(st["torch_cuda"]), torch.cuda.device_count())
        torch.cuda.set_rng_state_all(st["torch_cuda"][:n])


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(obj: Any, path: str | Path) -> None:
    atomic_write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", path)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_sha(root: Path = REPO_ROOT) -> str:
    """Source commit. Falls back to the SOURCE_SHA env var / file when .git is absent
    (the GPU nodes run a copy of the repository without .git)."""
    if os.environ.get("SOURCE_SHA"):
        return os.environ["SOURCE_SHA"]
    f = root / "SOURCE_SHA"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
    }
    import importlib.metadata as md
    for mod, dist in [("mne", "mne"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("pandas", "pandas"),
                      ("braindecode", "braindecode"), ("einops", "einops"), ("pyriemann", "pyriemann")]:
        try:
            info[mod] = md.version(dist)  # no import side effects
        except Exception:
            info[mod] = None
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_mem_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                              for i in range(torch.cuda.device_count())]
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().split()[0]
        except Exception:
            info["driver"] = None
    return info


def load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
