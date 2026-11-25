from itertools import combinations

import numpy as np
import pytest

from minesweepml.board import COVERED, Board, get_spec, neighbours
from minesweepml.solver import analyse


def brute_force(visible: np.ndarray, mines: int) -> np.ndarray:
    """P(mine) per covered cell by enumerating every consistent layout."""
    rows, cols = visible.shape
    covered = [tuple(rc) for rc in np.argwhere(visible == COVERED)]
    numbers = [(r, c, int(visible[r, c])) for r, c in np.argwhere(visible >= 0)]
    hits = np.zeros((rows, cols))
    total = 0
    for layout in combinations(range(len(covered)), mines):
        mine_set = {covered[i] for i in layout}
        if all(sum((nr, nc) in mine_set for nr, nc in neighbours(r, c, rows, cols)) == v for r, c, v in numbers):
            total += 1
            for cell in mine_set:
                hits[cell] += 1
    prob = np.full((rows, cols), np.nan)
    for cell in covered:
        prob[cell] = hits[cell] / total
    return prob


def random_position(spec, seed, extra_moves):
    """A real game position: the opening plus a few random safe reveals."""
    rng = np.random.default_rng(seed)
    b = Board(spec, seed, safe_neighbours=False)
    b.reveal(*spec.first_click())
    for _ in range(extra_moves):
        if b.done:
            break
        safe = np.argwhere(~b.mines & ~b.revealed)
        r, c = safe[rng.integers(len(safe))]
        b.reveal(int(r), int(c))
    return b


@pytest.mark.parametrize("seed", range(40))
def test_probabilities_match_brute_force(seed):
    spec = get_spec("4x6/5")
    b = random_position(spec, seed, extra_moves=seed % 3)
    if b.done:
        return
    vis = b.visible()
    a = analyse(vis, spec.mines)
    assert a.exact
    expected = brute_force(vis, spec.mines)
    covered = vis == COVERED
    assert np.allclose(a.prob[covered], expected[covered], atol=1e-12)
    assert set(a.safe) == {tuple(rc) for rc in np.argwhere(covered & (expected == 0))}
    assert set(a.mines) == {tuple(rc) for rc in np.argwhere(covered & (expected == 1))}


def test_certain_cells_are_true_on_real_boards():
    for seed in range(60):
        b = random_position(get_spec("expert"), seed, extra_moves=3)
        if b.done:
            continue
        a = analyse(b.visible(), b.spec.mines)
        assert all(not b.mines[c] for c in a.safe)
        assert all(b.mines[c] for c in a.mines)
        covered = b.visible() == COVERED
        # probabilities over all covered cells add up to the mine count
        assert a.prob[covered].sum() == pytest.approx(b.spec.mines, abs=1e-6 if a.exact else 5)


def test_subset_rule():
    # Mines at (0,0) and (0,3): every number in row 1 is a 1. No single number settles a cell,
    # but {00,01} is a subset of {00,01,02} with the same count, so (0,2) is safe.
    vis = np.array([
        [COVERED] * 5,
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)
    a = analyse(vis, 2)
    assert a.safe == [(0, 2)] and a.mines == []
    assert np.allclose(a.prob[0, [0, 1, 3, 4]], 0.5)
