"""Verify snapshot hashes, saved deterministic inference, and replay restoration."""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .checkpoint import validate_snapshot


def verify(path, tolerance=1e-6):
    path = Path(path)
    meta = validate_snapshot(path)
    if not (path / "probe.npz").is_file():
        raise ValueError("No inference probe stored")
    data = np.load(path / "probe.npz", allow_pickle=False)
    obs = {k: data[k] for k in data.files if k != "expected_actions"}
    model = SAC.load(path / "model.zip", device="cpu")
    model.load_replay_buffer(path / "replay_buffer.pkl")
    actual, _ = model.predict(obs, deterministic=True)
    error = float(np.max(np.abs(actual - data["expected_actions"])))
    ok = (np.isfinite(error) and error <= tolerance and model.replay_buffer.size() == meta["replay_size"]
          and model.num_timesteps == meta["num_timesteps"] and model._n_updates == meta["n_updates"])
    if not ok:
        raise ValueError(f"Snapshot verification failed: max action difference {error}")
    return {"ok": True, "max_action_difference": error, "tolerance": tolerance,
            "replay_size": model.replay_buffer.size(), "steps": model.num_timesteps,
            "updates": model._n_updates, "scope": "hashes, inference, model/replay loading; continuation is a separate run"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        result = verify(a.checkpoint)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
