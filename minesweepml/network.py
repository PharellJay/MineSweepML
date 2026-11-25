"""The network: a fully convolutional residual CNN that outputs P(mine) for every cell.

It has no dense layers, so one set of weights plays every board size. Each
residual block widens the receptive field by 4 cells; 6 blocks see 27 x 27
cells around a cell, more than any local Minesweeper deduction needs.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import COVERED
from .encoding import NUM_PLANES, crop, encode


@dataclass
class NetConfig:
    blocks: int = 6
    channels: int = 32


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class MineNet(nn.Module):
    """Input (B, 12, H, W) -> mine logits (B, H, W)."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(self.cfg.blocks)])
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Evaluator:
    """Batched inference on visible boards of one size: returns P(mine) per cell.

    The model runs in the channels-last memory layout, which is faster for
    small CPU convolutions. Weights loaded into it later keep that layout.
    """

    def __init__(self, model: MineNet):
        self.model = model.eval().to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def __call__(self, visible: np.ndarray, mines: int) -> np.ndarray:
        x = encode(visible, mines).contiguous(memory_format=torch.channels_last)
        return torch.sigmoid(crop(self.model(x))).numpy()

    def probabilities(self, visible: np.ndarray, mines: int) -> np.ndarray:
        """P(mine) for one board; NaN on revealed cells."""
        p = self(visible[None], mines)[0].astype(np.float64)
        p[visible != COVERED] = np.nan
        return p


def save_checkpoint(path: str, model: MineNet, optimizer=None, **meta) -> None:
    """Atomically write a checkpoint (model weights, optional optimizer, metadata)."""
    data = {"model": model.state_dict(), "net": asdict(model.cfg), **meta}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_checkpoint(path_or_data) -> MineNet:
    data = load_checkpoint(path_or_data) if isinstance(path_or_data, (str, os.PathLike)) else path_or_data
    model = MineNet(NetConfig(**data["net"]))
    model.load_state_dict(data["model"])
    return model.eval()
