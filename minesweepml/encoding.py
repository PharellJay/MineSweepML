"""Visible boards -> network input.

Each cell is one-hot encoded over 11 states: off-board, covered, and the
numbers 0..8. A twelfth plane holds the mine density of the covered cells
(mines / covered cells), the one global fact a player knows. Boards get a
one-cell ring of off-board cells, so every edge looks the same to the
network whatever the board size: input (B, 12, rows + 2, cols + 2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .board import COVERED

OFF_BOARD = -2
NUM_STATES = 11  # off-board, covered, 0..8
NUM_PLANES = NUM_STATES + 1


def pad_visible(visible: np.ndarray) -> np.ndarray:
    """(..., R, C) visible boards -> (..., R + 2, C + 2) with an off-board ring."""
    pad = [(0, 0)] * (visible.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(visible, pad, constant_values=OFF_BOARD)


def mine_density(visible: np.ndarray, mines: int) -> np.ndarray:
    """Mines per covered cell, per board of a (B, R, C) batch."""
    covered = (visible == COVERED).reshape(len(visible), -1).sum(1)
    return mines / np.maximum(covered, 1)


def encode(visible: np.ndarray, mines: int) -> torch.Tensor:
    """(B, R, C) int8 visible boards of one size -> float input (B, 12, R + 2, C + 2)."""
    padded = pad_visible(visible).astype(np.int64) - OFF_BOARD  # 0 = off-board, 1 = covered, 2..10 = numbers
    padded = np.clip(padded, 0, NUM_STATES - 1)  # an exploded mine never reaches the network; keep it in range
    x = F.one_hot(torch.from_numpy(padded), NUM_STATES).permute(0, 3, 1, 2).float()
    density = torch.from_numpy(mine_density(visible, mines).astype(np.float32))
    plane = density.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
    return torch.cat([x, plane], dim=1)


def crop(logits: torch.Tensor) -> torch.Tensor:
    """(B, R + 2, C + 2) network output -> (B, R, C) for the board cells."""
    return logits[:, 1:-1, 1:-1]
