"""Decoder variants for the improvement study.

CBraMod: official backbone (vendored, pinned weights) with
  - historical_flatten_mlp: Flatten(27*2*200) -> Linear(10800,400) -> ELU -> Dropout -> Linear(400,200) -> ELU
    -> Dropout -> Linear(200,2)  (4,401,002 parameters; identical to the historical 'all_patch_reps' head)
  - temporal_mean_linear: mean over the two time patches only (channels kept), Flatten(27*200) -> Linear(5400,2)
    (10,802 parameters). Distinct from the inherited 'avgpool' head (channel+time pooling).
Adaptation modes (head always trainable):
  - frozen: no backbone parameter trainable; the whole backbone kept in eval mode;
  - last2:  only backbone.encoder.layers.10 and .11 trainable (patch embedding, positional/spectral encodings and
            layers 0-9 frozen, kept in eval mode; the encoder has no final norm);
  - full:   every backbone parameter that is trainable in the official model (mask_encoding is a fixed buffer-like
            parameter with requires_grad=False in the official code and stays fixed).
EEGNet: historical Keras-port architecture with a configurable dropout rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..models.cbramod_official import CBraMod
from ..models.eegnet import EEGNet

N_LAYERS = 12
LAST2 = ("encoder.layers.10.", "encoder.layers.11.")


class TemporalMeanLinear(nn.Module):
    def __init__(self, n_chans: int, d: int = 200, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(n_chans * d, n_classes)

    def forward(self, feats):  # (B, C, S, d)
        return self.linear(feats.mean(dim=2).flatten(1))


class CBraModVariant(nn.Module):
    def __init__(self, n_chans: int, n_patches: int, head: str, adaptation: str, dropout: float = 0.1,
                 pretrained_path: str | None = None):
        super().__init__()
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30,
                                n_layer=N_LAYERS, nhead=8)
        self.load_report = None
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            own = self.backbone.state_dict()
            shape_mismatch = [k for k in sd if k in own and tuple(sd[k].shape) != tuple(own[k].shape)]
            res = self.backbone.load_state_dict(sd, strict=True)
            self.load_report = {"n_tensors": len(sd), "missing": list(res.missing_keys),
                                "unexpected": list(res.unexpected_keys), "shape_mismatch": shape_mismatch,
                                "reconstruction_head_keys_removed": sorted(k for k in own if k.startswith("proj_out."))}
        self.backbone.proj_out = nn.Identity()  # pretraining reconstruction head removed
        d = n_chans * n_patches * 200
        if head == "historical_flatten_mlp":
            self.classifier = nn.Sequential(
                nn.Flatten(1), nn.Linear(d, n_patches * 200), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(n_patches * 200, 200), nn.ELU(), nn.Dropout(dropout), nn.Linear(200, 2))
        elif head == "temporal_mean_linear":
            self.classifier = TemporalMeanLinear(n_chans)
        else:
            raise ValueError(head)
        self.head, self.adaptation = head, adaptation
        self._apply_trainability()

    # ------------------------------------------------------------- trainability
    def _backbone_trainable(self, name: str, param: nn.Parameter, official_trainable: bool) -> bool:
        if self.adaptation == "frozen":
            return False
        if self.adaptation == "full":
            return official_trainable
        if self.adaptation == "last2":
            return official_trainable and name.startswith(LAST2)
        raise ValueError(self.adaptation)

    def _apply_trainability(self):
        for n, p in self.backbone.named_parameters():
            p.requires_grad_(self._backbone_trainable(n, p, p.requires_grad))
        for p in self.classifier.parameters():
            p.requires_grad_(True)

    def trainable_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def frozen_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if not p.requires_grad]

    def train(self, mode: bool = True):
        """Frozen modules stay in eval mode (no dropout/normalisation-statistics changes) whenever the parent is
        put in training mode; unfrozen parts follow the requested mode."""
        super().train(mode)
        if mode:
            if self.adaptation == "frozen":
                self.backbone.eval()
            elif self.adaptation == "last2":
                self.backbone.eval()
                for i in (N_LAYERS - 2, N_LAYERS - 1):
                    self.backbone.encoder.layers[i].train(True)
        return self

    def module_modes(self) -> dict:
        return {n: m.training for n, m in self.named_modules() if n}

    def param_groups(self, head_lr: float, backbone_lr: float | None) -> list[dict]:
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = [p for p in self.classifier.parameters() if p.requires_grad]
        groups = [{"params": hd, "lr": head_lr, "name": "head"}]
        if bb:
            assert backbone_lr is not None
            groups.append({"params": bb, "lr": backbone_lr, "name": "backbone"})
        return groups

    def forward(self, x):  # x: (B, C, S, 200) in uV/100
        return self.classifier(self.backbone(x))


class EEGNetVariant(EEGNet):
    adaptation = "full"

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def frozen_names(self):
        return [n for n, p in self.named_parameters() if not p.requires_grad]

    def module_modes(self):
        return {n: m.training for n, m in self.named_modules() if n}

    def param_groups(self, lr, _unused=None):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr, "name": "all"}]


def build(recipe: dict, input_shape: tuple, pretrained_path: str | None = None):
    if recipe["arch"] == "cbramod":
        n_chans, n_patches, patch = input_shape
        assert patch == 200
        return CBraModVariant(n_chans, n_patches, recipe["head"], recipe["adaptation"], recipe["dropout"],
                              pretrained_path)
    if recipe["arch"] == "eegnet":
        n_chans, n_times = input_shape
        m = recipe["model"]
        return EEGNetVariant(n_chans=n_chans, n_times=n_times, n_classes=2, F1=m["F1"], D=m["D"], F2=m["F2"],
                             kern_length=m["kern_length"], sep_kern_length=m["sep_kern_length"],
                             dropout=recipe["dropout"], norm_rate=m["norm_rate"], pool1=m["pool1"], pool2=m["pool2"])
    raise ValueError(recipe["arch"])


def count(model, trainable_only=False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
