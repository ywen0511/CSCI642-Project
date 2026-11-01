"""CBraMod backbone (official code + official pretrained weights) with a binary head.

Input contract of the official checkpoint (verified against the official repository):
- sampling rate 200 Hz, patches of 1 s = 200 samples, tensor (batch, chans, n_patches, 200);
- amplitude in microvolts divided by 100 (official downstream datasets return ``data / 100``);
- pretraining used TUEG resampled to 200 Hz and band-pass filtered 0.3-75 Hz.

Classification head: the official ``all_patch_reps`` head (used by the official
fine-tuning scripts), sized for this dataset: flatten (C x S x 200) -> 4*200 -> 200 -> 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cbramod_official import CBraMod


class CBraModClassifier(nn.Module):
    def __init__(self, n_chans: int, n_patches: int, n_classes: int = 2, dropout: float = 0.1,
                 pretrained_path: str | None = None, head: str = "all_patch_reps"):
        super().__init__()
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                                seq_len=30, n_layer=12, nhead=8)
        self.pretrained_loaded = False
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=True), None
            self.pretrained_loaded = True
        self.backbone.proj_out = nn.Identity()
        d = n_chans * n_patches * 200
        if head == "all_patch_reps":
            self.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(d, n_patches * 200), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(n_patches * 200, 200), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(200, n_classes),
            )
        elif head == "avgpool":
            self.classifier = nn.Sequential(nn.Flatten(1, 2), nn.Linear(200, n_classes))
        else:
            raise ValueError(head)
        self.head_type = head

    def head_parameters(self):
        return self.classifier.parameters()

    def backbone_parameters(self):
        return self.backbone.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, chans, n_patches, 200), already in uV / 100
        feats = self.backbone(x)  # (batch, chans, n_patches, 200)
        if self.head_type == "avgpool":
            feats = feats.mean(dim=(1, 2), keepdim=True)
        return self.classifier(feats)
