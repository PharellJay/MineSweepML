"""Worker processes: play training games with the current network.

Each worker keeps `parallel_games` games going at once. For every move, the
games of each board size share one batched forward pass, and every game
clicks the covered cell with the lowest P(mine), or with probability epsilon
a random covered cell. A share `record_prob` of the positions before a move
is kept; a finished game sends them, with its true mine layout as the
target, to the trainer.

With `forgive_share` > 0, that share of games carries on after hitting a
mine: the move is replaced by a random safe cell, so a weak network also
reaches late-game positions. The game's result still counts as lost at the
first mine, so the statistics stay honest.
"""

from __future__ import annotations

import queue
import signal
import time

import numpy as np

from .board import COVERED, DIFFICULTIES, Board, to_text

LIVE_GRID_GAMES = 8  # thumbnails per worker in the live view
SEED_LIMIT = 2**32  # training seeds; benchmark seeds start at 2**40, so the two never meet


class _Game:
    __slots__ = ("board", "spec", "states", "forgive", "real")

    def __init__(self, board: Board, forgive: bool):
        self.board = board
        self.spec = board.spec
        self.states: list[np.ndarray] = []
        self.forgive = forgive
        self.real: tuple[str, float, int] | None = None  # (result, cleared, moves) at the first mine


def _ignore_sigint_and_nice(nice: int) -> None:
    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent stops workers gracefully
    apply_niceness(nice)


def worker_main(worker_id, cfg_dict, weights_path, weights_version, epsilon, counters, game_queue, live_queue,
                focus, pause_event, throttle_event, stop_event, seed):
    """Play games until `stop_event` is set; unfinished games are discarded.

    `pause_event` is the user's pause; `throttle_event` is set by the trainer
    while it is behind on the positions already sent.
    """
    import torch

    from .config import TrainConfig
    from .network import Evaluator, MineNet, NetConfig, load_checkpoint

    cfg = TrainConfig.from_dict(cfg_dict)
    _ignore_sigint_and_nice(cfg.nice)
    live_queue.cancel_join_thread()  # live snapshots are disposable; never block exit on them
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MineNet(NetConfig(cfg.blocks, cfg.channels))
    evaluator = Evaluator(model)
    loaded_version = -1
    mix = cfg.mix()
    names, shares = list(mix), np.array(list(mix.values()))

    def new_game() -> _Game:
        while True:
            spec = DIFFICULTIES[names[rng.choice(len(names), p=shares)]]
            board = Board(spec, int(rng.integers(SEED_LIMIT)), cfg.safe_neighbours)
            board.reveal(*spec.first_click())
            if not board.done:  # a board won by its opening alone teaches nothing
                return _Game(board, rng.random() < cfg.forgive_share)

    def finish(slot: int, game: _Game) -> None:
        board = game.board
        result, cleared, moves = game.real or (board.state, board.cleared(), len(board.moves))
        states = (np.stack(game.states) if game.states
                  else np.zeros((0, game.spec.rows, game.spec.cols), dtype=np.int8))
        game_queue.put({
            "worker": worker_id, "difficulty": game.spec.name, "visible": states, "mines": board.mines.copy(),
            "result": result, "cleared": cleared, "moves": moves, "eps": float(epsilon.value),
        })
        games[slot] = new_game()

    games = [new_game() for _ in range(cfg.parallel_games)]
    last_live = last_grid = 0.0

    while not stop_event.is_set():
        if pause_event.is_set() or throttle_event.is_set():
            time.sleep(0.05)
            continue
        if weights_version.value != loaded_version:
            loaded_version = weights_version.value
            model.load_state_dict(load_checkpoint(weights_path)["model"])
            model.eval()

        eps = float(epsilon.value)
        visible = [g.board.visible() for g in games]
        probs: list[np.ndarray | None] = [None] * len(games)
        groups: dict[str, list[int]] = {}
        for slot, g in enumerate(games):
            groups.setdefault(g.spec.name, []).append(slot)
        for name, slots in groups.items():
            p = evaluator(np.stack([visible[s] for s in slots]), DIFFICULTIES[name].mines)
            for s, ps in zip(slots, p):
                probs[s] = ps
        counters[worker_id] += len(games)

        moves = []
        for slot, g in enumerate(games):
            covered = visible[slot] == COVERED
            if rng.random() < eps:
                cells = np.flatnonzero(covered)
                flat = int(cells[rng.integers(len(cells))])
            else:
                flat = int(np.where(covered, probs[slot], np.inf).argmin())
            moves.append(divmod(flat, g.spec.cols))

        now = time.time()
        if focus[0] == worker_id and now - last_live > 0.25:
            last_live = now
            slot = min(max(focus[1], 0), len(games) - 1)
            _send_live(live_queue, games[slot], visible[slot], probs[slot], moves[slot], worker_id, slot)
        if now - last_grid > 1.0:
            last_grid = now
            _send_grid(live_queue, games, visible, worker_id)

        for slot, (g, (r, c)) in enumerate(zip(games, moves)):
            if stop_event.is_set():
                break
            board = g.board
            if rng.random() < cfg.record_prob:
                g.states.append(visible[slot])
            if board.mines[r, c] and g.forgive:
                if g.real is None:
                    g.real = ("lost", board.cleared(), len(board.moves) + 1)
                safe = np.argwhere(~board.mines & ~board.revealed)
                r, c = (int(x) for x in safe[rng.integers(len(safe))])
            board.reveal(r, c)
            if board.done:
                finish(slot, g)


def _send_grid(live_queue, games: list[_Game], visible: list[np.ndarray], worker_id: int) -> None:
    """A small snapshot of this worker's first games: enough to draw thumbnails."""
    items = [{"worker": worker_id, "slot": slot, "difficulty": g.spec.name, "rows": g.spec.rows, "cols": g.spec.cols,
              "cells": to_text(visible[slot]), "moves": len(g.board.moves)}
             for slot, g in enumerate(games[:LIVE_GRID_GAMES])]
    try:
        live_queue.put_nowait({"type": "grid", "worker": worker_id, "games": items})
    except queue.Full:
        pass


def _send_live(live_queue, game: _Game, visible: np.ndarray, probs: np.ndarray, move, worker_id: int,
               slot: int) -> None:
    """The followed game: the position, the network's P(mine) per cell (in %) and the move it is about to play."""
    covered = visible == COVERED
    pct = np.where(covered, np.round(probs * 100), -1).astype(int).ravel().tolist()
    msg = {"type": "live", "worker": worker_id, "slot": slot, "difficulty": game.spec.name,
           "rows": game.spec.rows, "cols": game.spec.cols, "cells": to_text(visible), "probs": pct,
           "next": list(move), "moves": len(game.board.moves)}
    try:
        live_queue.put_nowait(msg)
    except queue.Full:
        pass
