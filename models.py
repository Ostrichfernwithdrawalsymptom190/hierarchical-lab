"""A small CNN backbone + three heads, one per stage of the pathway.

We deliberately swap the repo's ResNet50+CBAM for a compact CNN so an epoch is
seconds on MPS and you can iterate on the *ideas*. The head shapes are what
carry the lessons:

  FlatModel        backbone -> 100                      (no hierarchy at all)
  TwoHeadModel     backbone -> {20, 100}                (deep supervision / lloss)
  CoarseToFineModel backbone -> 20, then cat(20,100)->100  (the repo's conditioning)

Every model exposes `.features(x)` returning the pooled penultimate vector, so
Stage 5 can watch the representation organize by superclass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hierarchy import NUM_COARSE, NUM_FINE

FEATURE_DIM = 256


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Backbone(nn.Module):
    """32x32x3 -> FEATURE_DIM vector. Shared across all three heads."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _block(3, 64),    # 32 -> 16
            _block(64, 128),  # 16 -> 8
            _block(128, FEATURE_DIM),  # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return x


class FlatModel(nn.Module):
    """Stage 1: plain 100-way classifier. No idea the hierarchy exists."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.fine = nn.Linear(FEATURE_DIM, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fine(self.backbone(x))


class TwoHeadModel(nn.Module):
    """Stage 2: coarse + fine heads on a shared trunk (deep supervision)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.coarse = nn.Linear(FEATURE_DIM, NUM_COARSE)
        self.fine = nn.Linear(FEATURE_DIM, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.coarse(z), self.fine(z)


class CoarseToFineModel(nn.Module):
    """Stage 3+: the repo's move — coarse logits are concatenated into the fine head.

    Faithful to model/resnet50.py:
        level_1 = softmax_reg1(linear_lvl1(z))          # feat->20->20
        level_2 = softmax_reg2(cat(level_1, linear_lvl2(z)))  # cat(20,100)->100
    We return raw logits (softmax lives in the loss), matching their pre-softmax cat.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.linear_lvl1 = nn.Linear(FEATURE_DIM, NUM_COARSE)
        self.softmax_reg1 = nn.Linear(NUM_COARSE, NUM_COARSE)
        self.linear_lvl2 = nn.Linear(FEATURE_DIM, NUM_FINE)
        self.softmax_reg2 = nn.Linear(NUM_COARSE + NUM_FINE, NUM_FINE)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        level_1 = self.softmax_reg1(self.linear_lvl1(z))
        level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_lvl2(z)), dim=1))
        return level_1, level_2
