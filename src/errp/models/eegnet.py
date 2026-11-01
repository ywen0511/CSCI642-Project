"""PyTorch port of EEGNet (Lawhern et al., 2018).

Reference implementation: https://github.com/vlawhern/arl-eegmodels (EEGModels.py,
function ``EEGNet``; commit recorded in reports/protocol_decisions.md). This port is
written for this project and keeps the reference layer order and settings:

    Conv2D(F1, (1, kernLength), padding='same', no bias) -> BN
    DepthwiseConv2D((Chans, 1), depth_multiplier=D, no bias, max_norm(1.0)) -> BN -> ELU
    AveragePooling2D((1, 4)) -> Dropout
    SeparableConv2D(F2, (1, 16), padding='same', no bias) -> BN -> ELU
    AveragePooling2D((1, 8)) -> Dropout
    Flatten -> Dense(nb_classes, max_norm(norm_rate)) -> softmax

Differences that are unavoidable in a port:
- Keras BatchNormalization defaults (momentum=0.99, epsilon=1e-3) are mapped to
  PyTorch (momentum=0.01, eps=1e-3).
- Keras 'same' padding for even kernels pads (k-1)//2 on the left and k//2 on the
  right; we reproduce that asymmetry explicitly.
- max_norm constraints are applied after every optimizer step via ``apply_constraints``
  (Keras applies them after each update as well). Keras max_norm uses axis=0 on its
  kernel layout, which corresponds to the norm over all weights feeding one output unit.
- The model returns logits; softmax is applied by the loss / at inference. Class order
  is index 0 = non-error, index 1 = error, so P(error) = softmax(logits)[:, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_1d(k: int) -> tuple[int, int]:
    # TensorFlow/Keras 'same' padding for stride 1.
    total = k - 1
    return total // 2, total - total // 2


def _max_norm_(w: torch.Tensor, max_value: float, dims: tuple[int, ...]) -> None:
    with torch.no_grad():
        norms = w.norm(p=2, dim=dims, keepdim=True)
        desired = norms.clamp(max=max_value)
        w.mul_(desired / (norms + 1e-7))


class EEGNet(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2, F1: int = 8, D: int = 2,
                 F2: int | None = None, kern_length: int = 64, sep_kern_length: int = 16,
                 dropout: float = 0.5, norm_rate: float = 0.25, pool1: int = 4, pool2: int = 8):
        super().__init__()
        F2 = F1 * D if F2 is None else F2
        self.n_chans, self.n_times = n_chans, n_times
        bn = dict(momentum=0.01, eps=1e-3)
        self.pad1 = _same_pad_1d(kern_length)
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, **bn)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, **bn)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.pad2 = _same_pad_1d(sep_kern_length)
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, sep_kern_length), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2, **bn)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        n_out = F2 * ((n_times // pool1) // pool2)
        self.dense = nn.Linear(n_out, n_classes)
        self.norm_rate = norm_rate
        self._init_keras_like()

    def _init_keras_like(self):
        # Keras defaults: glorot_uniform kernels, zero bias.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def apply_constraints(self) -> None:
        # depthwise kernel max_norm(1.0): norm over the (Chans x 1) spatial filter of each output map
        _max_norm_(self.depthwise.weight, 1.0, dims=(1, 2, 3))
        # dense max_norm(norm_rate): norm over the inputs of each output unit
        _max_norm_(self.dense.weight, self.norm_rate, dims=(1,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, chans, times) -> (batch, 1, chans, times)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.pad(x, (self.pad1[0], self.pad1[1], 0, 0))
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = F.pad(x, (self.pad2[0], self.pad2[1], 0, 0))
        x = self.sep_point(self.sep_depth(x))
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
        return self.dense(x.flatten(1))
