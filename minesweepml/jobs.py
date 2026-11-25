"""Background jobs for the web interface: benchmarks and batches of simulated games.

One job runs at a time in its own process (which starts a pool of game
processes), so the web server stays responsive. Progress goes to the server
through an event queue; `stop_event` cancels within a chunk of games.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

CHUNK = 25  # games per simulation task


def _sim_chunk(player: str, difficulty: str, seeds: list[int], safe_neighbours: bool) -> list[dict]:
    from .players import make_player, play_games

    p = make_player(player, seed=seeds[0])
    return [r.to_dict() for r in play_games(p, difficulty, seeds, safe_neighbours)]


def _simulate(params: dict, emit, stop_event) -> dict:
    from .benchmark import _init_worker, player_label, summarize

    seeds = [params["seed"] + i for i in range(params["games"])]
    chunks = [seeds[i:i + CHUNK] for i in range(0, len(seeds), CHUNK)]
    records = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(params["processes"], mp_context=ctx, initializer=_init_worker,
                             initargs=(params.get("nice", 10),)) as ex:
        futures = [ex.submit(_sim_chunk, params["player"], params["difficulty"], c, params["safe_neighbours"])
                   for c in chunks]
        for done, fut in enumerate(as_completed(futures), 1):
            if stop_event.is_set():
                for f in futures:
                    f.cancel()
                raise InterruptedError
            records.extend(fut.result())
            emit({"type": "job", "event": "progress", "done": len(records), "total": len(seeds)})
    records.sort(key=lambda r: r["seed"])
    summary = summarize([{"won": r["result"] == "won", "cleared": r["cleared"], "moves": len(r["moves"])}
                         for r in records])
    return {"player": player_label(params["player"]), "path": params["player"], "difficulty": params["difficulty"],
            "summary": summary, "records": records}


def _benchmark(params: dict, emit, stop_event) -> dict:
    from .benchmark import CHUNK as BENCH_CHUNK
    from .benchmark import run_benchmark, save_result

    total_chunks = len(params["difficulties"]) * -(-params["boards"] // BENCH_CHUNK)

    def progress(done, total, partial):
        emit({"type": "job", "event": "progress", "done": done, "total": total_chunks, "partial": partial})

    result = run_benchmark(params["player"], params["difficulties"], params["boards"], params["processes"],
                           quality=params["quality"], safe_neighbours=params["safe_neighbours"],
                           nice=params.get("nice", 10), progress=progress, cancel=stop_event.is_set)
    save_result(result)
    return result


def job_main(kind: str, params: dict, events, stop_event) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the server cancels jobs itself

    def emit(event: dict) -> None:
        events.put({**event, "kind": kind})

    emit({"type": "job", "event": "started", "started": time.time()})
    try:
        result = (_benchmark if kind == "benchmark" else _simulate)(params, emit, stop_event)
    except InterruptedError:
        emit({"type": "job", "event": "cancelled"})
        return
    except Exception as exc:
        emit({"type": "job", "event": "error", "message": repr(exc)})
        return
    emit({"type": "job", "event": "completed", "result": result})
