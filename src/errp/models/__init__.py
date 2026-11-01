"""Model registry.

Main comparison (required): ``eegnet`` (from scratch) and ``cbramod`` (official pretrained
backbone, full fine-tuning). Supplementary baselines (reported separately) are built from
braindecode implementations; pretrained foundation-model baselines load the braindecode
Hugging Face checkpoints.
"""
from __future__ import annotations

import os

import torch.nn as nn


def build_model(mcfg: dict, input_shape: tuple, load_pretrained: bool = True) -> nn.Module:
    name = mcfg["name"]
    kw = dict(mcfg.get("kwargs", {}))
    if name == "eegnet":
        from .eegnet import EEGNet
        n_chans, n_times = input_shape
        return EEGNet(n_chans=n_chans, n_times=n_times, n_classes=2, **kw)
    if name == "cbramod":
        from .cbramod import CBraModClassifier
        n_chans, n_patches, patch = input_shape
        assert patch == 200, "CBraMod expects 200-sample (1 s at 200 Hz) patches"
        return CBraModClassifier(n_chans=n_chans, n_patches=n_patches, n_classes=2,
                                 pretrained_path=os.path.expandvars(mcfg["pretrained_path"]) if load_pretrained else None,
                                 **kw)
    if name.startswith("bd_"):
        from .zoo import build_braindecode
        return build_braindecode(mcfg, input_shape, load_pretrained=load_pretrained)
    raise ValueError(f"unknown model {name}")


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Optimizer parameter groups. A pretrained backbone may use its own learning rate
    (official CBraMod fine-tuning uses a separate, larger head learning rate)."""
    tcfg = cfg["train"]
    if hasattr(model, "backbone_parameters") and tcfg.get("backbone_lr") is not None:
        bb, hd = list(model.backbone_parameters()), list(model.head_parameters())
        assert bb and hd, f"empty parameter group (backbone={len(bb)}, head={len(hd)})"
        return [{"params": bb, "lr": tcfg["backbone_lr"], "name": "backbone"},
                {"params": hd, "lr": tcfg["lr"], "name": "head"}]
    return [{"params": list(model.parameters()), "lr": tcfg["lr"], "name": "all"}]
