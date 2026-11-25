"""Minesweeper rules: board specs, seeded mine placement, reveal with flood fill.

A board is fully determined by its spec, a seed and the first click, so games
can be replayed and benchmarks repeated exactly. Mines are placed when the
first cell is revealed: by default that cell and its neighbours stay
mine-free, so the first click always opens an area (the rule of modern
clients). With `safe_neighbours=False` only the clicked cell is safe.

Players never see `mines` or `counts`; they get `visible()`: -1 for a
covered cell, 0..8 for a revealed number, and EXPLODED for the mine that
ended a lost game.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

import numpy as np

COVERED = -1
EXPLODED = 9

MAX_ROWS, MAX_COLS = 16, 30  # the largest standard board; the network pads every board to this


@dataclass(frozen=True)
class Spec:
    rows: int
    cols: int
    mines: int
    name: str = "custom"

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def key(self) -> str:
        """The name of a standard difficulty, else 'RxC/M'."""
        return self.name if self.name in DIFFICULTIES else f"{self.rows}x{self.cols}/{self.mines}"

    @property
    def label(self) -> str:
        if self.name in DIFFICULTIES:
            return f"{self.name.capitalize()} ({self.cols}x{self.rows}, {self.mines} mines)"
        return f"Custom ({self.cols}x{self.rows}, {self.mines} mines)"

    def first_click(self) -> tuple[int, int]:
        """Every game opens on the centre cell, so all players start from the same opening."""
        return self.rows // 2, self.cols // 2


DIFFICULTIES = {
    "beginner": Spec(9, 9, 10, "beginner"),
    "intermediate": Spec(16, 16, 40, "intermediate"),
    "expert": Spec(16, 30, 99, "expert"),
}
DIFFICULTY_NAMES = tuple(DIFFICULTIES)

_CUSTOM = re.compile(r"(\d+)\s*x\s*(\d+)\s*/\s*(\d+)")


def get_spec(spec: str | Spec) -> Spec:
    """'beginner' / 'intermediate' / 'expert', or 'RxC/M' (rows x cols / mines)."""
    if isinstance(spec, Spec):
        return spec
    text = spec.strip().lower()
    if text in DIFFICULTIES:
        return DIFFICULTIES[text]
    m = _CUSTOM.fullmatch(text)
    if not m:
        raise ValueError(f"unknown difficulty {spec!r}: use {', '.join(DIFFICULTY_NAMES)} or ROWSxCOLS/MINES")
    rows, cols, mines = map(int, m.groups())
    if not (2 <= rows <= MAX_ROWS and 2 <= cols <= MAX_COLS):
        raise ValueError(f"custom boards are 2-{MAX_ROWS} rows by 2-{MAX_COLS} columns")
    if not 1 <= mines <= rows * cols - 1:
        raise ValueError("a custom board needs between 1 and cells-1 mines")
    for name, std in DIFFICULTIES.items():
        if (std.rows, std.cols, std.mines) == (rows, cols, mines):
            return std
    return Spec(rows, cols, mines)


def neighbours(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    yield rr, cc


def neighbour_counts(mines: np.ndarray) -> np.ndarray:
    """Number of mines among the 8 neighbours of every cell."""
    padded = np.pad(mines.astype(np.int8), 1)
    rows, cols = mines.shape
    out = np.zeros((rows, cols), dtype=np.int8)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr != 1 or dc != 1:
                out += padded[dr:dr + rows, dc:dc + cols]
    return out


def place_mines(spec: Spec, seed: int, first_click: tuple[int, int], safe_neighbours: bool = True) -> np.ndarray:
    """Seeded mine layout that keeps the first click (and by default its neighbours) free."""
    r0, c0 = first_click
    safe = {(r0, c0)}
    if safe_neighbours:
        safe.update(neighbours(r0, c0, spec.rows, spec.cols))
        if spec.cells - len(safe) < spec.mines:  # a dense custom board: fall back to a safe cell only
            safe = {(r0, c0)}
    allowed = np.array([r * spec.cols + c for r in range(spec.rows) for c in range(spec.cols) if (r, c) not in safe])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(allowed, size=spec.mines, replace=False)
    mines = np.zeros(spec.cells, dtype=bool)
    mines[chosen] = True
    return mines.reshape(spec.rows, spec.cols)


_TEXT = ".012345678*"  # index = visible value + 1


def to_text(visible: np.ndarray) -> str:
    """Compact row-major text of a visible board for the web UI: '.' covered, digits, '*' exploded."""
    return "".join(_TEXT[v + 1] for v in visible.ravel().tolist())


class Board:
    """One game. `reveal` returns the cells it uncovered, in flood-fill order."""

    def __init__(self, spec: str | Spec, seed: int, safe_neighbours: bool = True):
        self.spec = get_spec(spec)
        self.seed = int(seed)
        self.safe_neighbours = safe_neighbours
        self.mines: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.revealed = np.zeros((self.spec.rows, self.spec.cols), dtype=bool)
        self.exploded: tuple[int, int] | None = None
        self.state = "playing"  # playing / won / lost
        self.moves: list[tuple[int, int]] = []
        self.safe_left = self.spec.cells - self.spec.mines

    @property
    def done(self) -> bool:
        return self.state != "playing"

    def reveal(self, r: int, c: int) -> list[tuple[int, int]]:
        if self.done:
            raise RuntimeError("the game is over")
        rows, cols = self.spec.rows, self.spec.cols
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"cell ({r}, {c}) is off the board")
        if self.revealed[r, c]:
            raise ValueError(f"cell ({r}, {c}) is already revealed")
        if self.mines is None:
            self.mines = place_mines(self.spec, self.seed, (r, c), self.safe_neighbours)
            self.counts = neighbour_counts(self.mines)
        self.moves.append((r, c))
        if self.mines[r, c]:
            self.exploded = (r, c)
            self.state = "lost"
            return [(r, c)]
        opened = []
        queue = deque([(r, c)])
        self.revealed[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            opened.append((cr, cc))
            if self.counts[cr, cc] == 0:
                for nr, nc in neighbours(cr, cc, rows, cols):
                    if not self.revealed[nr, nc]:
                        self.revealed[nr, nc] = True
                        queue.append((nr, nc))
        self.safe_left -= len(opened)
        if self.safe_left == 0:
            self.state = "won"
        return opened

    def visible(self) -> np.ndarray:
        """What a player sees: -1 covered, 0..8 revealed numbers, 9 the exploded mine."""
        out = np.full((self.spec.rows, self.spec.cols), COVERED, dtype=np.int8)
        if self.counts is not None:
            out[self.revealed] = self.counts[self.revealed]
        if self.exploded is not None:
            out[self.exploded] = EXPLODED
        return out

    def cleared(self) -> float:
        """Share of the safe cells revealed so far."""
        safe = self.spec.cells - self.spec.mines
        return (safe - self.safe_left) / safe
