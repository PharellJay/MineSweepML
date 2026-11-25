"""Web interface: FastAPI app with REST endpoints and a WebSocket for live updates.

Training runs in a separate process (trainer.training_process_main) whose
events are read by a background thread and pushed to every connected
browser. Benchmarks and simulation batches run as one job at a time in
their own process (jobs.job_main). The server process itself is not niced,
so it stays responsive while the niced processes use the remaining CPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import queue
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .board import DIFFICULTIES, get_spec
from .config import BOT_FIELDS, CPU_COUNT, TrainConfig, all_presets, parse_duration
from .metrics import read_history
from .runs import (bot_settings, create_bot, delete_bot, list_bots, list_checkpoints, rename_bot, resolve_bot,
                   resolve_checkpoint, update_bot)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TrainingManager:
    """Owns the training process and fans its events out to WebSocket clients."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.control = None
        self.events = None
        self.bot: str | None = None
        self.status: dict = {"type": "status", "state": "idle"}
        self.live: dict | None = None
        self.grid: dict | None = None
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, cfg: TrainConfig, bot: str | None, fork: str | None) -> None:
        from .trainer import Control, training_process_main

        with self.lock:
            if self.running:
                raise RuntimeError("training is already running")
            self.control = Control(self.ctx)
            self.events = self.ctx.Queue()
            self.live = self.grid = None
            self.bot = bot
            self.status = {"type": "status", "state": "starting", "preset": cfg.preset, "bot": bot}
            self.proc = self.ctx.Process(target=training_process_main, name="minesweepml-trainer",
                                         args=(cfg.to_dict(), bot, fork, {}, self.control, self.events))
            self.proc.start()
            threading.Thread(target=self._read_events, args=(self.proc, self.events), daemon=True).start()
        self.broadcast(self.status)

    def pause(self) -> None:
        if self.running:
            self.control.pause.set()

    def resume(self) -> None:
        if self.running:
            self.control.pause.clear()

    def focus(self, worker: int, slot: int) -> None:
        workers = self.status.get("workers") or 0
        slots = self.status.get("parallel_games") or 0
        if not (0 <= worker < workers and 0 <= slot < slots):
            raise ValueError(f"no game {worker}/{slot} (training has {workers} workers x {slots} games)")
        if self.running:
            self.control.focus[0], self.control.focus[1] = worker, slot

    def stop(self) -> None:
        if self.running:
            self.control.stop.set()

    def training_bot(self) -> str | None:
        """Name of the bot being trained, if any."""
        if not self.running:
            return None
        return self.status.get("bot") or self.bot

    def shutdown(self, timeout: float = 120.0) -> None:
        """Stop training gracefully (checkpoint included) when the server exits."""
        if self.running:
            print("Stopping training gracefully before exit...", flush=True)
            self.control.stop.set()
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.terminate()

    def _read_events(self, proc, events) -> None:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    break
                continue
            kind = event.get("type")
            if kind == "status":
                self.status = event
            elif kind == "live":
                self.live = event
            elif kind == "grid":
                self.grid = event
            self.broadcast(event)
        if self.status.get("state") not in ("stopped", "completed", "error"):
            self.status = {**self.status, "state": "error" if proc.exitcode else "stopped"}
            self.broadcast(self.status)

    def broadcast(self, event: dict) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._fan_out, event)

    def _fan_out(self, event: dict) -> None:
        for q in list(self.clients):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # a slow client misses an update rather than stalling the others


manager = TrainingManager()


class JobManager:
    """Runs one benchmark or simulation batch at a time in a child process."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.stop_event = None
        self.job: dict | None = None
        self.last_simulation: dict | None = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, kind: str, params: dict) -> dict:
        from .jobs import job_main

        with self.lock:
            if self.running:
                raise RuntimeError(f"a {self.job['kind']} job is already running")
            self.stop_event = self.ctx.Event()
            events = self.ctx.Queue()
            self.job = {"kind": kind, "state": "starting", "params": params, "done": 0, "total": 0,
                        "started": time.time()}
            # not a daemon: the job starts a pool of game processes itself
            self.proc = self.ctx.Process(target=job_main, name=f"minesweepml-{kind}",
                                         args=(kind, params, events, self.stop_event))
            self.proc.start()
            threading.Thread(target=self._read_events, args=(self.proc, events), daemon=True).start()
        manager.broadcast({"type": "job", "event": "starting", "kind": kind, "job": self.public()})
        return self.public()

    def public(self) -> dict | None:
        """The job without bulky results, for status calls and broadcasts."""
        if self.job is None:
            return None
        job = {k: v for k, v in self.job.items() if k != "result"}
        result = self.job.get("result")
        if result is not None:
            job["result"] = {k: v for k, v in result.items() if k != "records"}
        return job

    def uses_bot(self, name: str) -> bool:
        if not self.running:
            return False
        player = str(self.job["params"].get("player", ""))
        return f"/{name}/checkpoints/" in player

    def cancel(self) -> None:
        if self.running:
            self.stop_event.set()

    def shutdown(self, timeout: float = 20.0) -> None:
        if self.running:
            self.stop_event.set()
            self.proc.join(timeout)
            if self.proc.is_alive():
                self.proc.terminate()

    def _read_events(self, proc, events) -> None:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    break
                continue
            self._apply(event)
            out = dict(event)
            if out.get("result") and "records" in out["result"]:
                out["result"] = {k: v for k, v in out["result"].items() if k != "records"}
            manager.broadcast(out)
        if self.job and self.job["state"] in ("starting", "running"):
            self.job["state"] = "error"
            self.job["message"] = f"job process exited with code {proc.exitcode}"
            manager.broadcast({"type": "job", "event": "error", "kind": self.job["kind"],
                               "message": self.job["message"]})

    def _apply(self, event: dict) -> None:
        job = self.job
        kind = event["event"]
        if kind == "started":
            job["state"] = "running"
        elif kind == "progress":
            job.update(done=event["done"], total=event["total"], partial=event.get("partial"))
        elif kind in ("completed", "cancelled", "error"):
            job["state"] = kind
            job["result"] = event.get("result")
            job["message"] = event.get("message")
            job["finished"] = time.time()
            if kind == "completed" and job["kind"] == "simulate":
                self.last_simulation = event["result"]


jobs = JobManager()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    yield
    await run_in_threadpool(jobs.shutdown)
    await run_in_threadpool(manager.shutdown)


app = FastAPI(title="MineSweepML", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Browsers revalidate the page and its scripts (a cheap ETag check), so an update shows without a hard refresh."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _bad(exc: Exception, code: int = 400):
    raise HTTPException(code, str(exc))


# -- info and training ----------------------------------------------------------


@app.get("/api/info")
def info():
    return {"cpu_count": CPU_COUNT, "presets": all_presets(), "defaults": TrainConfig().to_dict(),
            "bot_fields": list(BOT_FIELDS),
            "difficulties": {k: {"rows": s.rows, "cols": s.cols, "mines": s.mines, "label": s.label}
                             for k, s in DIFFICULTIES.items()},
            "status": manager.status, "live": manager.live, "grid": manager.grid, "job": jobs.public()}


class StartRequest(BaseModel):
    bot: str | None = None  # existing bot to train, or the name of a new one
    fork: str | None = None  # checkpoint: start a new bot from a copy of it
    preset: str = "balanced"
    workers: int | None = None
    threads: int | None = None
    batch_size: int | None = None
    nice: int | None = None
    buffer_size: int | None = None
    parallel_games: int | None = None
    duration: str = "stop"  # "games", "time" or "stop"
    games: int | None = None
    time: str | None = None
    eval_minutes: float | None = None
    eval_boards: int | None = None
    checkpoint_minutes: float | None = None
    reset_lr: bool | None = None


@app.post("/api/training/start")
def start_training(req: StartRequest):
    try:
        max_games = max_seconds = None
        if req.duration == "games":
            if not req.games or req.games < 1:
                raise ValueError("number of games must be at least 1")
            max_games = req.games
        elif req.duration == "time":
            max_seconds = parse_duration(req.time or "")
        cfg = TrainConfig.from_preset(
            req.preset, workers=req.workers, threads=req.threads, batch_size=req.batch_size, nice=req.nice,
            buffer_size=req.buffer_size, parallel_games=req.parallel_games, max_games=max_games,
            max_seconds=max_seconds, eval_minutes=req.eval_minutes, eval_boards=req.eval_boards,
            checkpoint_minutes=req.checkpoint_minutes, reset_lr=req.reset_lr or None,
        )
        for field in ("workers", "threads", "batch_size", "parallel_games", "eval_boards"):
            if getattr(cfg, field) < 1:
                raise ValueError(f"{field.replace('_', ' ')} must be at least 1")
        if not 5_000 <= cfg.buffer_size <= 5_000_000:
            raise ValueError("the replay buffer must hold between 5,000 and 5,000,000 positions")
        if not 0 <= cfg.nice <= 19:
            raise ValueError("niceness must be between 0 and 19")
        bot = (req.bot or "").strip() or None
        fork = str(resolve_checkpoint(req.fork)) if req.fork else None
        if bot and not fork:
            try:
                bot = resolve_bot(bot).name
            except FileNotFoundError:
                pass  # a new bot with this name
        manager.start(cfg, bot, fork)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        _bad(exc)
    except RuntimeError as exc:
        _bad(exc, 409)
    return manager.status


@app.post("/api/training/pause")
def pause_training():
    manager.pause()
    return {"ok": manager.running}


@app.post("/api/training/resume")
def resume_training():
    manager.resume()
    return {"ok": manager.running}


@app.post("/api/training/stop")
def stop_training():
    manager.stop()
    return {"ok": manager.running}


@app.get("/api/training/status")
def training_status():
    return {"status": manager.status, "live": manager.live, "grid": manager.grid, "running": manager.running}


class FocusRequest(BaseModel):
    worker: int
    slot: int


@app.post("/api/training/focus")
def focus_game(req: FocusRequest):
    try:
        manager.focus(req.worker, req.slot)
    except ValueError as exc:
        _bad(exc)
    return {"ok": manager.running}


# -- bots -------------------------------------------------------------------------


def _latest_benchmarks() -> dict[str, dict]:
    from .benchmark import load_results

    latest = {}
    for result in load_results():  # oldest first, so the newest wins
        if result.get("path"):
            latest[str(Path(result["path"]).resolve())] = result
    return latest


@app.get("/api/bots")
def bots():
    latest = _latest_benchmarks()
    training = manager.training_bot()
    out = []
    for b in list_bots():
        ckpts = [c for c in list_checkpoints() if c["bot"] == b["name"]]
        bench = next((latest[str(Path(c["path"]).resolve())] for c in ckpts
                      if str(Path(c["path"]).resolve()) in latest), None)
        out.append({**b, "training": b["name"] == training,
                    "checkpoint_list": [{"label": c["label"], "path": c["path"],
                                         "benchmark": latest.get(str(Path(c["path"]).resolve()))} for c in ckpts],
                    "benchmark": bench})
    return out


class BotRequest(BaseModel):
    name: str
    checkpoint: str | None = None  # start from a copy of this checkpoint; empty = untrained
    settings: dict = {}


@app.post("/api/bots")
def new_bot(req: BotRequest):
    try:
        source = resolve_checkpoint(req.checkpoint) if req.checkpoint else None
        path = create_bot(req.name, settings=req.settings, from_checkpoint=source)
    except FileExistsError as exc:
        _bad(exc, 409)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        _bad(exc)
    return {"name": path.name, "settings": bot_settings(path)}


class BotUpdate(BaseModel):
    name: str | None = None  # rename
    settings: dict = {}


def _busy(name: str) -> str | None:
    if manager.training_bot() == name:
        return f"bot {name!r} is training; stop it first"
    if jobs.uses_bot(name):
        return f"a {jobs.job['kind']} job is using bot {name!r}; wait for it or cancel it"
    return None


@app.patch("/api/bots/{name}")
def edit_bot(name: str, req: BotUpdate):
    try:
        path = resolve_bot(name)
        if req.settings:
            update_bot(path, req.settings)
        if req.name and req.name.strip() != path.name:
            if busy := _busy(path.name):
                _bad(RuntimeError(busy), 409)
            path = rename_bot(path, req.name)
    except FileNotFoundError as exc:
        _bad(exc, 404)
    except FileExistsError as exc:
        _bad(exc, 409)
    except (ValueError, TypeError) as exc:
        _bad(exc)
    return {"name": path.name, "settings": bot_settings(path)}


@app.delete("/api/bots/{name}")
def remove_bot(name: str):
    try:
        path = resolve_bot(name)
    except FileNotFoundError as exc:
        _bad(exc, 404)
    if busy := _busy(path.name):
        _bad(RuntimeError(busy), 409)
    delete_bot(path)
    return {"deleted": path.name}


@app.get("/api/bots/{name}/history")
def bot_history(name: str, limit: int = 2000):
    try:
        return read_history(resolve_bot(name), limit)
    except FileNotFoundError as exc:
        _bad(exc, 404)


@app.get("/api/checkpoints")
def checkpoints():
    latest = _latest_benchmarks()
    return [{**c, "benchmark": latest.get(str(Path(c["path"]).resolve()))} for c in list_checkpoints()]


# -- simulation and replays ---------------------------------------------------------


def _player_spec(player: str) -> str:
    return player if player in ("random", "solver") else str(resolve_checkpoint(player))


@lru_cache(maxsize=4)
def _net_player(path: str, mtime: float):
    from .players import NetPlayer

    return NetPlayer(path)


_play_lock = threading.Lock()


def _player_object(spec: str, seed: int):
    from .players import make_baseline

    if spec in ("random", "solver"):
        return make_baseline(spec, seed)
    return _net_player(spec, Path(spec).stat().st_mtime)


class GameRequest(BaseModel):
    player: str
    difficulty: str = "expert"
    seed: int | None = None
    safe_neighbours: bool = True


def _play_one(req: GameRequest) -> dict:
    import random

    import torch

    from .players import play_games
    from .replay import replay_payload
    from .runs import checkpoint_label

    spec = get_spec(req.difficulty)
    seed = req.seed if req.seed is not None else random.randrange(2**31)
    player = _player_spec(req.player)
    with _play_lock:  # one game at a time keeps the server's CPU use bounded
        torch.set_num_threads(2)
        obj = _player_object(player, seed)
        record = play_games(obj, spec, [seed], req.safe_neighbours, parallel=1)[0]
    record.player = player if player in ("random", "solver") else checkpoint_label(player)
    return {"record": {**record.to_dict(), "path": player}, "replay": replay_payload(record)}


@app.post("/api/simulate/game")
async def simulate_game(req: GameRequest):
    try:
        return await run_in_threadpool(_play_one, req)
    except (ValueError, FileNotFoundError) as exc:
        _bad(exc)


class BatchRequest(BaseModel):
    player: str
    difficulty: str = "expert"
    games: int = 100
    seed: int = 0
    processes: int | None = None
    safe_neighbours: bool = True


def _default_processes() -> int:
    free = CPU_COUNT - 2 if not manager.running else CPU_COUNT // 3
    return max(1, free)


def _check_processes(processes: int | None) -> int:
    processes = processes or _default_processes()
    if not 1 <= processes <= CPU_COUNT:
        raise ValueError(f"processes must be between 1 and {CPU_COUNT}")
    return processes


@app.post("/api/simulate/batch")
def simulate_batch(req: BatchRequest):
    try:
        if not 1 <= req.games <= 10_000:
            raise ValueError("games must be between 1 and 10,000")
        get_spec(req.difficulty)
        params = {"player": _player_spec(req.player), "difficulty": req.difficulty, "games": req.games,
                  "seed": req.seed, "processes": _check_processes(req.processes),
                  "safe_neighbours": req.safe_neighbours, "nice": 10}
        return jobs.start("simulate", params)
    except (ValueError, FileNotFoundError) as exc:
        _bad(exc)
    except RuntimeError as exc:
        _bad(exc, 409)


@app.get("/api/simulate/last")
def last_simulation():
    """The records of the last finished simulation batch (they are not broadcast)."""
    return jobs.last_simulation


class ReplayRequest(BaseModel):
    record: dict
    player: str | None = None  # whose P(mine) to show: a checkpoint path/label or 'solver'


def _record(data: dict):
    from .replay import GameRecord

    rec = GameRecord.from_dict(data)
    get_spec(rec.spec)
    return rec


@app.post("/api/replay")
def replay(req: ReplayRequest):
    from .replay import replay_payload

    try:
        return replay_payload(_record(req.record))
    except (ValueError, TypeError, KeyError) as exc:
        _bad(exc)


def _probabilities(req: ReplayRequest) -> dict:
    import torch

    from .replay import frame_probabilities

    rec = _record(req.record)
    spec = _player_spec(req.player or "solver")
    if spec == "random":
        raise ValueError("the random player has no probabilities; pick a network or the solver")
    with _play_lock:
        torch.set_num_threads(2)
        player = "solver" if spec == "solver" else _player_object(spec, 0)
        return {"player": req.player, "probs": frame_probabilities(rec, player)}


@app.post("/api/replay/probabilities")
async def replay_probabilities(req: ReplayRequest):
    try:
        return await run_in_threadpool(_probabilities, req)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        _bad(exc)


class GifRequest(BaseModel):
    record: dict
    player: str | None = None  # whose P(mine) the overlay shows
    overlay: bool = True
    fps: float = 8.0
    title: str | None = None


def _gif(req: GifRequest) -> bytes:
    from .gif import render_record

    rec = _record(req.record)
    spec = None if not req.overlay else _player_spec(req.player or "solver")
    if spec == "random":
        spec = None
    with _play_lock:
        prob_player = None if spec is None else ("solver" if spec == "solver" else _player_object(spec, 0))
        return render_record(rec, prob_player=prob_player, title=req.title, fps=req.fps)


@app.post("/api/replay/gif")
async def replay_gif(req: GifRequest):
    try:
        data = await run_in_threadpool(_gif, req)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        _bad(exc)
    return Response(content=data, media_type="image/gif",
                    headers={"Content-Disposition": f'attachment; filename="minesweepml-{req.record.get("spec", "game")}'
                                                    f'-{req.record.get("seed", 0)}.gif"'.replace("/", "-")})


# -- benchmarks -------------------------------------------------------------------


@app.get("/api/benchmarks")
def benchmarks():
    from .benchmark import load_results

    return {"results": load_results(), "job": jobs.public(), "running": jobs.running,
            "training": manager.running, "cpu_count": CPU_COUNT, "default_processes": _default_processes()}


class BenchmarkRequest(BaseModel):
    player: str
    difficulties: list[str] = ["beginner", "intermediate", "expert"]
    boards: int = 500
    processes: int | None = None
    quality: bool = False
    safe_neighbours: bool = True


@app.post("/api/benchmarks")
def start_benchmark(req: BenchmarkRequest):
    try:
        if not 10 <= req.boards <= 10_000:
            raise ValueError("boards per difficulty must be between 10 and 10,000")
        if not req.difficulties:
            raise ValueError("pick at least one difficulty")
        difficulties = [get_spec(d).key for d in req.difficulties]
        params = {"player": _player_spec(req.player), "difficulties": difficulties, "boards": req.boards,
                  "processes": _check_processes(req.processes), "quality": req.quality,
                  "safe_neighbours": req.safe_neighbours, "nice": 10}
        return jobs.start("benchmark", params)
    except (ValueError, FileNotFoundError) as exc:
        _bad(exc)
    except RuntimeError as exc:
        _bad(exc, 409)


@app.post("/api/jobs/cancel")
def cancel_job():
    jobs.cancel()
    return {"ok": jobs.running}


@app.get("/api/jobs")
def job_state():
    return {"job": jobs.public(), "running": jobs.running}


# -- live updates -------------------------------------------------------------------


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    manager.clients.add(q)
    try:
        await ws.send_json({"type": "hello", "status": manager.status, "live": manager.live, "grid": manager.grid,
                            "job": jobs.public()})
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.clients.discard(q)


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}/"
    print(f"MineSweepML web interface: {url}  (Ctrl+C stops training gracefully and exits)", flush=True)
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=3)
