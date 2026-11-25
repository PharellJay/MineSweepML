import numpy as np
import pytest

from minesweepml.board import DIFFICULTIES, Board
from minesweepml.encoding import NUM_PLANES, NUM_STATES, crop, encode, mine_density, pad_visible


@pytest.mark.parametrize("name", list(DIFFICULTIES))
def test_shapes_and_one_hot(name):
    spec = DIFFICULTIES[name]
    boards = []
    for seed in range(3):
        b = Board(spec, seed)
        b.reveal(*spec.first_click())
        boards.append(b.visible())
    vis = np.stack(boards)
    x = encode(vis, spec.mines).numpy()
    assert x.shape == (3, NUM_PLANES, spec.rows + 2, spec.cols + 2)
    assert np.all(x[:, :NUM_STATES].sum(1) == 1)  # exactly one state per cell
    ring = np.ones((spec.rows + 2, spec.cols + 2), bool)
    ring[1:-1, 1:-1] = False
    assert np.all(x[:, 0][:, ring] == 1) and np.all(x[:, 0][:, ~ring] == 0)  # off-board plane = the ring
    covered = (vis == -1)
    assert np.array_equal(x[:, 1, 1:-1, 1:-1] == 1, covered)
    for k in range(3):
        assert np.allclose(x[k, -1], spec.mines / covered[k].sum())


def test_pad_and_crop_are_inverse():
    import torch

    vis = np.random.default_rng(0).integers(-1, 9, size=(2, 9, 9)).astype(np.int8)
    padded = pad_visible(vis)
    assert padded.shape == (2, 11, 11)
    assert np.array_equal(crop(torch.from_numpy(padded)).numpy(), vis)


def test_density_of_untouched_board():
    vis = np.full((1, 16, 30), -1, dtype=np.int8)
    assert mine_density(vis, 99)[0] == pytest.approx(99 / 480)
