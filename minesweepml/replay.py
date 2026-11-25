"""Game records: a game is its board (spec, seed, opening rule) plus its moves.

Boards are deterministic, so a record of a few hundred bytes rebuilds every
frame of a game exactly. The web viewer, the GIF renderer and the benchmark
all work from records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .board import Board, get_spec
from .solver import analyse


@dataclass
class GameRecord:
    spec: str  # difficulty key: 'beginner' / 'intermediate' / 'expert' / 'RxC/M'
    seed: int
    moves: list[tuple[int, int]] = field(default_factory=list)
    result: str = "playing"  # won / lost / playing
    cleared: float = 0.0  # share of safe cells revealed
    player: str = ""
    safe_neighbours: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["moves"] = [list(m) for m in self.moves]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GameRecord":
        d = dict(d)
        d["moves"] = [tuple(m) for m in d.get("moves", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_board(cls, board: Board, player: str = "") -> "GameRecord":
        return cls(spec=board.spec.key, seed=board.seed, moves=list(board.moves), result=board.state,
                   cleared=board.cleared(), player=player, safe_neighbours=board.safe_neighbours)

    def new_board(self) -> Board:
        return Board(get_spec(self.spec), self.seed, self.safe_neighbours)


def frames(record: GameRecord):
    """Yield (board, opened cells) before the first move and after every move.

    The same Board object is yielded each time and changes in place; copy
    `board.visible()` if you keep frames.
    """
    board = record.new_board()
    yield board, []
    for r, c in record.moves:
        opened = board.reveal(r, c)
        yield board, opened


def final_board(record: GameRecord) -> Board:
    board = record.new_board()
    for r, c in record.moves:
        board.reveal(r, c)
    return board


def move_quality(record: GameRecord) -> dict:
    """Judge every move after the opening against the solver.

    forced guesses: moves made when no covered cell was provably safe.
    blunders: mines hit while a provably safe cell existed.
    """
    board = record.new_board()
    spec = board.spec
    guesses = blunders = 0
    for n, (r, c) in enumerate(record.moves):
        if n > 0:
            a = analyse(board.visible(), spec.mines)
            if not a.safe:
                guesses += 1
            elif board.mines[r, c]:
                blunders += 1
        board.reveal(r, c)
    return {"guesses": guesses, "blunders": blunders, "moves": len(record.moves)}


def visible_stack(record: GameRecord) -> np.ndarray:
    """Every frame's visible board, shape (moves + 1, rows, cols)."""
    return np.stack([board.visible() for board, _ in frames(record)])
