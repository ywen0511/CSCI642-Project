"""Supplementary baselines from braindecode (reported separately from the main comparison).

Config fields:
  model.name        'bd_<tag>'
  model.bd_class    braindecode.models class name (e.g. EEGConformer, Labram, EEGPT)
  model.hf_repo     optional Hugging Face checkpoint (pretrained foundation models)
  model.head_prefix parameter-name prefix of the newly initialised classification head
  model.kwargs      extra constructor kwargs
The input view must provide (chans, times); sampling rate comes from model.sfreq.
For pretrained models the architecture is always rebuilt through ``from_pretrained`` (also
when reloading our fine-tuned checkpoint) so the module layout matches the hub config.
"""
from __future__ import annotations

import mne
import torch.nn as nn

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4",
            "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def chs_info(names, sfreq):
    info = mne.create_info(list(names), sfreq, "eeg")
    info.set_montage("standard_1020", on_missing="raise")
    return info["chs"]


class BDWrapper(nn.Module):
    def __init__(self, net: nn.Module, head_prefixes: list[str], forward_kwargs: dict | None = None):
        super().__init__()
        self.net = net
        self.head_prefixes = tuple(head_prefixes)
        self.forward_kwargs = forward_kwargs or {}

    def _split(self, head: bool):
        for n, p in self.net.named_parameters():
            if n.startswith(self.head_prefixes) == head:
                yield p

    def head_parameters(self):
        return self._split(True)

    def backbone_parameters(self):
        return self._split(False)

    def forward(self, x):
        out = self.net(x, **self.forward_kwargs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def build_braindecode(mcfg: dict, input_shape: tuple, load_pretrained: bool = True) -> nn.Module:
    import braindecode.models as M
    cls = getattr(M, mcfg["bd_class"])
    n_chans, n_times = input_shape
    kw = dict(n_chans=n_chans, n_outputs=2, n_times=n_times, sfreq=mcfg["sfreq"], **mcfg.get("kwargs", {}))
    names = list(mcfg.get("channels") or CHANNELS)  # ErrP-Bench: channel list of the prepared dataset
    if mcfg.get("needs_chs_info", False):
        assert n_chans == len(names)
        kw["chs_info"] = chs_info(names, mcfg["sfreq"])
    net = cls(**kw)
    report = None
    if mcfg.get("hf_repo") and load_pretrained:
        report = load_pretrained_partial(net, mcfg["hf_repo"], mcfg.get("hf_revision"),
                                         keep_model_keys=mcfg.get("keep_model_keys", []))
    fkw = {"ch_names": names} if mcfg.get("forward_ch_names", False) else {}
    w = BDWrapper(net, mcfg.get("head_prefix", ["final_layer"]), fkw)
    w.pretrained_load_report = report
    return w


def load_pretrained_partial(net: nn.Module, repo: str, revision: str | None, keep_model_keys=()) -> dict:
    """Load every pretrained tensor whose shape matches. Leading-length mismatches on positional
    tables (e.g. LaBraM temporal_embedding, 16 -> n_patches+1) are truncated to the first rows,
    which is exactly the slice the forward pass uses. Keys in ``keep_model_keys`` (buffers derived
    from our own channel list, e.g. EEGPT chans_id) are never overwritten. Everything not loaded
    is listed in the returned report."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path = hf_hub_download(repo, "model.safetensors", revision=revision)
    sd = load_file(path)
    own = net.state_dict()
    loaded, truncated, skipped_shape, skipped_kept, unused = [], [], [], [], []
    new = {}
    for k, v in sd.items():
        if k not in own:
            unused.append(k)
            continue
        if k in keep_model_keys:
            skipped_kept.append(k)
            continue
        t = own[k]
        if t.shape == v.shape:
            new[k] = v
            loaded.append(k)
        elif (t.dim() == v.dim() and t.shape[0] == v.shape[0] and t.dim() >= 2
              and t.shape[2:] == v.shape[2:] and t.shape[1] < v.shape[1]):
            new[k] = v[:, :t.shape[1]].clone()
            truncated.append(f"{k}: {tuple(v.shape)} -> {tuple(t.shape)}")
        else:
            skipped_shape.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(t.shape)}")
    missing = [k for k in own if k not in new]
    net.load_state_dict({**own, **new}, strict=True)
    return {"repo": repo, "revision": revision, "file": "model.safetensors", "n_loaded": len(loaded),
            "truncated": truncated, "skipped_shape_mismatch": skipped_shape,
            "kept_model_values": skipped_kept, "ckpt_keys_not_in_model": unused,
            "model_keys_not_from_ckpt": missing}
