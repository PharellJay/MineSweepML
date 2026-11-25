"""Benchmark: win rate on a fixed set of seeded boards, with a 95% interval.

Every player plays the same boards: seed BENCHMARK_BASE + i for board i,
opened on the centre cell. Training seeds are below 2**32, so training never
sees a benchmark board. Results are kept in runs/benchmarks.json.

Per difficulty: wins with a Wilson 95% interval, the average share of safe
cells cleared and moves per game, and with `quality` on, forced guesses
(moves with no provably safe cell) and blunders (mines hit while a provably
safe cell existed) per game, judged by the solver.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import sqrt
from pathlib import Path

from .board import DIFFICULTY_NAMES, get_spec

BENCHMARK_BASE = 2**40
CHUNK = 50  # boards per task


def benchmark_seeds(n: int) -> list[int]:
    return [BENCHMARK_BASE + i for i in range(n)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["won"] for r in rows)
    lo, hi = wilson(wins, n)
    out = {"boards": n, "wins": wins, "win_rate": wins / n if n else 0.0, "ci_low": lo, "ci_high": hi,
           "cleared": sum(r["cleared"] for r in rows) / n if n else 0.0,
           "moves": sum(r["moves"] for r in rows) / n if n else 0.0}
    if rows and "guesses" in rows[0]:
        out["guesses"] = sum(r["guesses"] for r in rows) / n
        out["blunders"] = sum(r["blunders"] for r in rows) / n
    return out


def player_label(player: str) -> str:
    if player in ("random", "solver"):
        return player
    from .runs import checkpoint_label

    return checkpoint_label(player)


def _init_worker(nice: int) -> None:
    import signal

    from .config import apply_niceness

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_niceness(nice)
    import torch

    torch.set_num_threads(1)


def _play_chunk(player: str, difficulty: str, seeds: list[int], safe_neighbours: bool, quality: bool) -> list[dict]:
    from .players import make_player, play_games
    from .replay import move_quality

    p = make_player(player, seed=seeds[0])
    rows = []
    for rec in play_games(p, difficulty, seeds, safe_neighbours):
        row = {"seed": rec.seed, "won": rec.result == "won", "cleared": rec.cleared, "moves": len(rec.moves)}
        if quality:
            q = move_quality(rec)
            row.update(guesses=q["guesses"], blunders=q["blunders"])
        rows.append(row)
    return rows


def run_benchmark(player: str, difficulties=DIFFICULTY_NAMES, boards: int = 1000, processes: int = 2,
                  quality: bool = False, safe_neighbours: bool = True, nice: int = 10, progress=None,
                  cancel=None) -> dict:
    """Play `boards` benchmark boards per difficulty with `player` ('random', 'solver' or a checkpoint path)."""
    if player not in ("random", "solver"):
        player = str(Path(player).resolve())
    difficulties = [get_spec(d).key for d in difficulties]
    seeds = benchmark_seeds(boards)
    tasks = [(d, seeds[i:i + CHUNK]) for d in difficulties for i in range(0, boards, CHUNK)]
    rows: dict[str, list[dict]] = {d: [] for d in difficulties}
    start = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max(1, processes), mp_context=ctx, initializer=_init_worker, initargs=(nice,)) as ex:
        futures = {ex.submit(_play_chunk, player, d, s, safe_neighbours, quality): d for d, s in tasks}
        done = 0
        for fut in as_completed(futures):
            if cancel is not None and cancel():
                for f in futures:
                    f.cancel()
                raise InterruptedError("benchmark cancelled")
            rows[futures[fut]].extend(fut.result())
            done += 1
            if progress:
                progress(done, len(tasks), {d: summarize(r) for d, r in rows.items() if r})
    return {"player": player_label(player), "path": player if player not in ("random", "solver") else None,
            "boards": boards, "safe_neighbours": safe_neighbours, "quality": quality, "created": time.time(),
            "seconds": time.time() - start,
            "difficulties": {d: summarize(sorted(r, key=lambda x: x["seed"])) for d, r in rows.items()}}


def results_path() -> Path:
    from .runs import runs_dir

    return runs_dir() / "benchmarks.json"


def load_results() -> list[dict]:
    from .runs import read_json

    return read_json(results_path(), []) or []


def save_result(result: dict) -> None:
    from .runs import write_json

    results = load_results()
    results.append(result)
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, results)


def baseline_results(boards: int) -> dict[str, dict]:
    """The newest saved random and solver results with this many boards."""
    out = {}
    for r in load_results():
        if r["player"] in ("random", "solver") and r["boards"] == boards:
            out[r["player"]] = r
    return out


def eval_process_main(checkpoint: str, difficulties, boards: int, nice: int, out_queue, safe_neighbours: bool,
                      tag: str) -> None:
    """Periodic evaluation during training, in its own niced process."""
    _init_worker(nice)
    try:
        from .players import NetPlayer, play_games

        player = NetPlayer(checkpoint)
        seeds = benchmark_seeds(boards)
        result = {}
        for d in difficulties:
            recs = play_games(player, d, seeds, safe_neighbours)
            result[d] = summarize([{"won": r.result == "won", "cleared": r.cleared, "moves": len(r.moves)}
                                   for r in recs])
        out_queue.put({"tag": tag, "difficulties": result})
    except Exception as exc:  # reported to the trainer, which keeps training
        out_queue.put({"tag": tag, "error": repr(exc)})
