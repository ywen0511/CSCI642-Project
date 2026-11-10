"""Atomic SAC snapshots including replay and RNG; restart at a fresh episode."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .feedback import sha256


def save_snapshot(model, wrapper, destination, run_spec):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tmp = destination.with_name(destination.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=False)
    model.save(tmp / "model.zip")
    model.save_replay_buffer(tmp / "replay_buffer.pkl")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(), "action_space": model.action_space.np_random.bit_generator.state}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, tmp / "rng.pt")
    # Save fixed inference probes from actual replay observations when available.
    replay = model.replay_buffer
    n = min(4, replay.size())
    if n:
        obs = {k: np.array(v[:n, 0], copy=True) for k, v in replay.observations.items()}
        actions, _ = model.predict(obs, deterministic=True)
        np.savez(tmp / "probe.npz", **obs, expected_actions=actions)
    metadata = {
        "schema": "csci642-sac-snapshot/v1", "run_spec": run_spec,
        "num_timesteps": model.num_timesteps, "n_updates": model._n_updates,
        "replay_size": replay.size(), "wrapper_counters": wrapper.counters(),
        "resume_semantics": "model/optimizers/replay/RNG restored; new simulator episode; not bitwise trajectory continuation",
        "phase": "final" if destination.name.startswith("final") else "after rollout storage, before next gradient update",
        "sha256": {f.name: sha256(f) for f in sorted(tmp.iterdir()) if f.is_file()},
    }
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tmp.rename(destination)
    if wrapper.handle:
        wrapper.handle.flush()
    return destination


def validate_snapshot(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    required = {"model.zip", "replay_buffer.pkl", "rng.pt"}
    if not required.issubset(meta["sha256"]):
        raise ValueError("Incomplete checkpoint inventory")
    for name, digest in meta["sha256"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return meta


def restore_snapshot(path, env, wrapper, run_spec, device):
    path = Path(path)
    meta = validate_snapshot(path)
    # Training budget and output paths are outside run_spec. Protocol cannot drift.
    if meta["run_spec"] != run_spec:
        raise ValueError("Resume protocol, feedback identity or run identity differs")
    model = SAC.load(path / "model.zip", env=env, device=device, force_reset=True)
    model.load_replay_buffer(path / "replay_buffer.pkl")
    if model.replay_buffer.size() != meta["replay_size"]:
        raise ValueError("Replay buffer size mismatch")
    wrapper.restore_counters(meta["wrapper_counters"])
    model.get_env().seed(wrapper.next_episode_seed())
    # Load only snapshots produced by this project or another trusted source.
    rng = torch.load(path / "rng.pt", weights_only=False, map_location="cpu")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    model.action_space.np_random.bit_generator.state = rng["action_space"]
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return model, meta
