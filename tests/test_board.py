import numpy as np
import pytest

from minesweepml.board import COVERED, DIFFICULTIES, EXPLODED, Board, Spec, get_spec, neighbour_counts, neighbours


def test_get_spec_names_and_custom():
    assert get_spec("Expert") == DIFFICULTIES["expert"]
    assert get_spec("16x30/99") is DIFFICULTIES["expert"]
    s = get_spec("8x10/12")
    assert (s.rows, s.cols, s.mines, s.key) == (8, 10, 12, "8x10/12")
    for bad in ("huge", "17x30/10", "5x5/25", "5x5/0"):
        with pytest.raises(ValueError):
            get_spec(bad)


def test_neighbour_counts_match_definition():
    rng = np.random.default_rng(0)
    mines = rng.random((7, 9)) < 0.3
    counts = neighbour_counts(mines)
    for r in range(7):
        for c in range(9):
            assert counts[r, c] == sum(mines[nr, nc] for nr, nc in neighbours(r, c, 7, 9))


@pytest.mark.parametrize("name", list(DIFFICULTIES))
def test_first_click_and_neighbours_are_safe(name):
    spec = DIFFICULTIES[name]
    for seed in range(30):
        b = Board(spec, seed)
        r, c = spec.first_click()
        opened = b.reveal(r, c)
        assert b.mines.sum() == spec.mines
        assert not b.mines[r, c]
        assert all(not b.mines[nr, nc] for nr, nc in neighbours(r, c, spec.rows, spec.cols))
        assert b.counts[r, c] == 0 and len(opened) >= 9  # the opening always cascades


def test_safe_cell_only_rule():
    spec = get_spec("5x5/20")
    b = Board(spec, 3, safe_neighbours=True)  # too dense for a safe neighbourhood: falls back to the cell
    b.reveal(2, 2)
    assert not b.mines[2, 2] and b.mines.sum() == 20


def test_seeded_boards_are_reproducible():
    a, b = Board("expert", 42), Board("expert", 42)
    a.reveal(8, 15)
    b.reveal(8, 15)
    assert np.array_equal(a.mines, b.mines)
    c = Board("expert", 43)
    c.reveal(8, 15)
    assert not np.array_equal(a.mines, c.mines)


def test_flood_fill_reveals_zero_region_and_its_border():
    b = Board("intermediate", 7)
    b.reveal(8, 8)
    vis = b.visible()
    rows, cols = vis.shape
    for r in range(rows):
        for c in range(cols):
            if vis[r, c] == 0:  # every neighbour of a revealed zero is revealed
                assert all(vis[nr, nc] != COVERED for nr, nc in neighbours(r, c, rows, cols))
    assert not (b.revealed & b.mines).any()


def test_win_and_loss():
    spec = Spec(3, 3, 1)
    b = Board(spec, 0, safe_neighbours=False)
    b.reveal(0, 0)
    mine = tuple(np.argwhere(b.mines)[0])
    for r, c in np.argwhere(~b.mines & ~b.revealed):
        if not b.done:
            b.reveal(int(r), int(c))
    assert b.state == "won" and b.cleared() == 1.0

    b = Board(spec, 0, safe_neighbours=False)
    b.reveal(0, 0)
    if not b.done:
        b.reveal(*mine)
        assert b.state == "lost"
        assert b.visible()[mine] == EXPLODED
        with pytest.raises(RuntimeError):
            b.reveal(0, 1)


def test_illegal_reveals():
    b = Board("beginner", 1)
    b.reveal(4, 4)
    with pytest.raises(ValueError):
        b.reveal(4, 4)
    with pytest.raises(ValueError):
        b.reveal(9, 0)
