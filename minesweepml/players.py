"""Players and the game loop shared by benchmarks, simulations and GIFs.

A player sees only `View`s: the visible board, the spec (so the mine count)
and a per-game memo it may use for its own bookkeeping. `choose` takes a list
of views so that the network can score many games in one batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import COVERED, Board, Spec, get_spec
from .replay import GameRecord
from .solver import analyse


@dataclass
class View:
    visible: np.ndarray
    spec: Spec
    memo: dict = field(default_factory=dict)


class RandomPlayer:
    """Clicks a uniformly random covered cell."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def choose(self, views: list[View]) -> list[tuple[int, int]]:
        out = []
        for v in views:
            cells = np.argwhere(v.visible == COVERED)
            r, c = cells[self.rng.integers(len(cells))]
            out.append((int(r), int(c)))
        return out


class SolverPlayer:
    """Plays every provably safe cell, otherwise the covered cell with the lowest P(mine)."""

    name = "solver"

    def choose(self, views: list[View]) -> list[tuple[int, int]]:
        return [self._choose_one(v) for v in views]

    @staticmethod
    def _choose_one(v: View) -> tuple[int, int]:
        queue = v.memo.setdefault("safe", [])
        while queue:  # cells proven safe earlier stay safe
            r, c = queue.pop()
            if v.visible[r, c] == COVERED:
                return r, c
        a = analyse(v.visible, v.spec.mines)
        if a.safe:
            queue.extend(reversed(a.safe))
            return queue.pop()
        prob = np.where(v.visible == COVERED, a.prob, np.inf)
        r, c = np.unravel_index(int(np.argmin(prob)), prob.shape)
        return int(r), int(c)


class NetPlayer:
    """Clicks the covered cell with the lowest P(mine) according to the network.

    Games of the same size share one batched forward pass.
    """

    def __init__(self, source, name: str | None = None):
        from pathlib import Path

        from .network import Evaluator, MineNet, model_from_checkpoint

        if isinstance(source, MineNet):
            model = source
        else:
            model = model_from_checkpoint(str(source))
            name = name or Path(source).stem
        self.evaluator = Evaluator(model)
        self.name = name or "net"

    def choose(self, views: list[View]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = [(0, 0)] * len(views)
        groups: dict[Spec, list[int]] = {}
        for i, v in enumerate(views):
            groups.setdefault(v.spec, []).append(i)
        for spec, idxs in groups.items():
            vis = np.stack([views[i].visible for i in idxs])
            p = self.evaluator(vis, spec.mines)
            p[vis != COVERED] = np.inf
            flat = p.reshape(len(idxs), -1).argmin(1)
            for i, f in zip(idxs, flat):
                out[i] = (int(f // spec.cols), int(f % spec.cols))
        return out

    def probabilities(self, visible: np.ndarray, spec: Spec) -> np.ndarray:
        return self.evaluator.probabilities(visible, spec.mines)


BASELINES = ("random", "solver")


def make_baseline(name: str, seed: int | None = None):
    if name == "random":
        return RandomPlayer(seed)
    if name == "solver":
        return SolverPlayer()
    raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")


def make_player(spec: str, seed: int | None = None):
    """'random', 'solver', or a bot name / checkpoint file / checkpoint label."""
    if spec in BASELINES:
        return make_baseline(spec, seed)
    from .runs import checkpoint_label, resolve_checkpoint

    path = resolve_checkpoint(spec)
    return NetPlayer(path, name=checkpoint_label(path))


def play_games(player, spec: str | Spec, seeds, safe_neighbours: bool = True, parallel: int = 32,
               on_move=None) -> list[GameRecord]:
    """Play one game per seed, `parallel` at a time, and return their records in seed order.

    Every game opens on the centre cell, so all players start from the same
    opening on the same seed. `on_move(board)` is called after every move.
    """
    spec = get_spec(spec)
    seeds = list(seeds)
    records: list[GameRecord | None] = [None] * len(seeds)
    label = getattr(player, "name", type(player).__name__)
    pending = list(range(len(seeds)))[::-1]
    active: list[tuple[int, Board, View]] = []
    while pending or active:
        while pending and len(active) < parallel:
            idx = pending.pop()
            board = Board(spec, seeds[idx], safe_neighbours)
            board.reveal(*spec.first_click())
            if board.done:  # a board can be won by its opening alone
                records[idx] = GameRecord.from_board(board, label)
                continue
            active.append((idx, board, View(board.visible(), spec)))
        if not active:
            continue
        moves = player.choose([v for _, _, v in active])
        still = []
        for (idx, board, view), (r, c) in zip(active, moves):
            board.reveal(r, c)
            if on_move:
                on_move(board)
            if board.done:
                records[idx] = GameRecord.from_board(board, label)
            else:
                view.visible = board.visible()
                still.append((idx, board, view))
        active = still
    return records


def play_game(player, spec: str | Spec, seed: int, safe_neighbours: bool = True) -> GameRecord:
    return play_games(player, spec, [seed], safe_neighbours, parallel=1)[0]
