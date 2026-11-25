"""Training orchestration: workers, replay buffer, optimisation, checkpoints,
periodic evaluation and graceful stop.

Processes (all started with the "spawn" method):
    trainer       this module; owns the network, optimiser and replay buffer
    workers x N   play games with the current network (workers.worker_main), one core each
    evaluator     occasional benchmark of the newest checkpoint
The controller (CLI or web server) talks to the trainer through `Control`
(pause/stop events) and receives events through the `emit` callback.

The network learns to predict the mine layout: for every stored position,
the target is the true mine map of its board, and the loss is the binary
cross-entropy over the covered cells. Its minimum is P(mine | what is
visible), so a lower loss means safer clicks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from .benchmark import eval_process_main
from .board import COVERED, DIFFICULTIES
from .config import TrainConfig, apply_niceness
from .encoding import crop, encode
from .metrics import MetricsDB
from .network import MineNet, NetConfig, load_checkpoint, save_checkpoint
from .runs import (bot_checkpoints, bot_settings, create_bot, latest_checkpoint, new_bot_name, read_json,
                   resolve_bot, update_bot, write_json)
from .workers import worker_main

SUSPEND_GAP = 60.0  # seconds between two loop passes (normally about one) that mean the computer slept


class Control:
    """Pause/stop flags shared by the controller and the training processes,
    plus the game the live view follows in detail."""

    def __init__(self, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.pause = ctx.Event()
        self.stop = ctx.Event()
        self.focus = ctx.Array("i", [0, 0])  # (worker, game slot)


class _Part:
    """Ring buffer of positions of one difficulty: visible board and true mine map."""

    def __init__(self, capacity: int, rows: int, cols: int):
        self.capacity = capacity
        self.visible = np.zeros((capacity, rows, cols), dtype=np.int8)
        self.mines = np.zeros((capacity, rows, cols), dtype=bool)
        self.size = 0
        self.pos = 0

    def add(self, visible: np.ndarray, mines: np.ndarray) -> None:
        for v in visible:
            self.visible[self.pos] = v
            self.mines[self.pos] = mines
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def chronological(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return (np.arange(self.capacity) + self.pos) % self.capacity


class ReplayBuffer:
    """One ring buffer per difficulty, sized by its share of the training mix."""

    def __init__(self, capacity: int, mix: dict[str, float]):
        self.mix = mix
        self.parts = {}
        for name, spec in DIFFICULTIES.items():
            share = mix.get(name, 0.0)
            self.parts[name] = _Part(max(2_000, int(capacity * share)), spec.rows, spec.cols)

    @property
    def size(self) -> int:
        return sum(p.size for p in self.parts.values())

    def add(self, difficulty: str, visible: np.ndarray, mines: np.ndarray) -> None:
        self.parts[difficulty].add(visible, mines)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[str, np.ndarray, np.ndarray]:
        """A batch from one difficulty, picked by its share of the mix among the parts holding data."""
        names = [k for k, p in self.parts.items() if p.size > 0 and self.mix.get(k, 0) > 0]
        if not names:
            names = [k for k, p in self.parts.items() if p.size > 0]
            weights = np.array([self.parts[k].size for k in names], dtype=float)
        else:
            weights = np.array([self.mix[k] for k in names])
        name = names[rng.choice(len(names), p=weights / weights.sum())]
        part = self.parts[name]
        ids = rng.integers(0, part.size, size=n)
        return name, part.visible[ids], part.mines[ids]

    def save(self, path: Path) -> None:
        arrays = {}
        for name, part in self.parts.items():
            order = part.chronological()
            if len(order):
                arrays[f"{name}_visible"] = part.visible[order]
                arrays[f"{name}_mines"] = part.mines[order]
        if not arrays:
            return
        tmp = Path(str(path) + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        with np.load(path) as data:
            for name, part in self.parts.items():
                if f"{name}_visible" in data:
                    vis, mines = data[f"{name}_visible"], data[f"{name}_mines"]
                    keep = slice(max(0, len(vis) - part.capacity), None)
                    for v, m in zip(vis[keep], mines[keep]):
                        part.add(v[None], m)


def augment(visible: np.ndarray, mines: np.ndarray, rng: np.random.Generator):
    """Random flips (and a transpose on square boards); Minesweeper is symmetric under all of them."""
    if rng.random() < 0.5:
        visible, mines = visible[:, ::-1], mines[:, ::-1]
    if rng.random() < 0.5:
        visible, mines = visible[:, :, ::-1], mines[:, :, ::-1]
    if visible.shape[1] == visible.shape[2] and rng.random() < 0.5:
        visible, mines = visible.transpose(0, 2, 1), mines.transpose(0, 2, 1)
    return np.ascontiguousarray(visible), np.ascontiguousarray(mines)


def masked_loss(model: MineNet, visible: np.ndarray, mines: np.ndarray, n_mines: int) -> torch.Tensor:
    """Binary cross-entropy between the predicted and the true mine map, over covered cells."""
    logits = crop(model(encode(visible, n_mines)))
    mask = torch.from_numpy(visible == COVERED)
    target = torch.from_numpy(mines).float()
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


class Trainer:
    def __init__(self, cfg: TrainConfig, control: Control, emit=None, bot: str | None = None,
                 fork: str | None = None, bot_overrides: dict | None = None):
        """Train `bot` (created if it does not exist), or a new bot forked from checkpoint `fork`."""
        self.cfg = cfg
        self.control = control
        self.emit = emit or (lambda event: None)
        self.bot_name = bot
        self.fork = fork
        self.bot_overrides = bot_overrides or {}
        self.rng = np.random.default_rng(cfg.seed)

    # -- setup ------------------------------------------------------------

    def _setup(self) -> None:
        cfg = self.cfg
        if self.fork:
            source = Path(self.fork).resolve()
            parent = source.parent.parent.name if source.parent.name == "checkpoints" else "model"
            self.bot_dir = create_bot(self.bot_name or f"{parent}-from-{source.stem}", from_checkpoint=source)
            if self.bot_overrides:
                update_bot(self.bot_dir, self.bot_overrides)
        else:
            try:
                self.bot_dir = resolve_bot(self.bot_name) if self.bot_name else None
            except FileNotFoundError:
                self.bot_dir = None
            if self.bot_dir is None:
                settings = {**cfg.bot_settings(), **self.bot_overrides}
                self.bot_dir = create_bot(self.bot_name or new_bot_name(), settings=settings)
            elif self.bot_overrides:
                update_bot(self.bot_dir, self.bot_overrides)
        for key, value in bot_settings(self.bot_dir).items():  # the bot's own settings win
            setattr(cfg, key, value)

        ckpt_path = latest_checkpoint(self.bot_dir)
        data = load_checkpoint(str(ckpt_path)) if ckpt_path else None
        if data:
            cfg.blocks, cfg.channels = data["net"]["blocks"], data["net"]["channels"]
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.mix())
        if (self.bot_dir / "buffer.npz").exists():
            self.buffer.load(self.bot_dir / "buffer.npz")
        self.run_extra = {k: v for k, v in (read_json(self.bot_dir / "state.json", {}) or {}).items() if k == "origin"}

        self.model = MineNet(NetConfig(cfg.blocks, cfg.channels))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.lr_scale = 1.0
        self.lr_since = 0
        self.games = self.steps = self.positions = self.moves = 0
        self.elapsed_base = 0.0
        self.active_seconds = 0.0
        self.checkpoint = None
        if data:
            self.model.load_state_dict(data["model"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
                for group in self.optimizer.param_groups:
                    group["weight_decay"] = cfg.weight_decay
            self.lr_scale, self.lr_since = data.get("lr_scale", 1.0), data.get("lr_since", 0)
            self.games = data.get("games", 0)
            self.steps = data.get("steps", 0)
            self.positions = data.get("positions", 0)
            self.moves = data.get("moves", 0)
            self.elapsed_base = data.get("elapsed", 0.0)
            self.checkpoint = str(ckpt_path)
        self.lr_note = None
        if data and cfg.reset_lr and (self.lr_scale != 1.0 or self.lr_since):
            self.lr_note = f"Learning rate reset to {cfg.lr:.2g} (was {self._lr():.2g})"
            self.lr_scale, self.lr_since = 1.0, self.games
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr()
        self.db = MetricsDB(self.bot_dir / "metrics.db")
        write_json(self.bot_dir / "config.json", cfg.to_dict())
        from .report import machine_info

        write_json(self.bot_dir / "hardware.json", machine_info())
        self.last_eval = self._latest_eval()

    # -- main loop --------------------------------------------------------

    def run(self) -> str:
        """Train until a limit is reached or stop is requested. Returns the final state."""
        cfg = self.cfg
        self._setup()
        torch.set_num_threads(cfg.threads)
        apply_niceness(cfg.nice)
        psutil.cpu_percent(percpu=True)  # prime the CPU counters

        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.weights_path = self.bot_dir / "weights.pt"
        save_checkpoint(str(self.weights_path), self.model)
        self.version = ctx.Value("i", 0)
        self.epsilon = ctx.Value("d", cfg.epsilon(self.games))
        self.counters = ctx.Array("q", cfg.workers)
        self.game_queue = ctx.Queue()
        self.live_queue = ctx.Queue(maxsize=8 + 2 * cfg.workers)
        self.eval_queue = ctx.Queue()
        self.worker_stop = ctx.Event()
        self.throttle = ctx.Event()
        self.eval_proc = None
        self.eval_meta = None
        self.eval_suspended = False

        if self.games == 0 and not bot_checkpoints(self.bot_dir):
            self._save_checkpoint()  # the untrained starting point, kept for comparisons

        seed = cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(4), "little")
        self.workers = [
            ctx.Process(
                target=worker_main, daemon=True, name=f"minesweepml-worker-{i}",
                args=(i, cfg.to_dict(), str(self.weights_path), self.version, self.epsilon, self.counters,
                      self.game_queue, self.live_queue, self.control.focus, self.control.pause, self.throttle,
                      self.worker_stop, seed + i),
            )
            for i in range(cfg.workers)
        ]
        for w in self.workers:
            w.start()

        self.live_grid: dict[int, list] = {}
        self.last_grid_emit = 0.0
        self.session_games = 0
        self.session_positions = 0
        self.trained_samples = 0
        self.recent_games: deque = deque()
        self.loss_acc = [0.0, 0]
        self.last_rates = (time.time(), 0, self.positions)
        self.rates = {"games_per_sec": 0.0, "moves_per_sec": 0.0, "samples_per_sec": 0.0}
        self.db.insert("sessions", {"elapsed": self._elapsed(), "event": "start",
                                    "detail": f"preset={cfg.preset} workers={cfg.workers} {cfg.duration_text()}"})
        self.emit({"type": "log", "message": f"Bot {self.bot_dir.name} started: {cfg.preset} preset, "
                                            f"{cfg.workers} workers, {cfg.duration_text()}, {cfg.mix_text()}"})
        if self.lr_note:
            self.emit({"type": "log", "message": self.lr_note})

        now = time.time()
        last_tick = last_status = last_ckpt = last_eval = now
        reason = "error"
        state = "running"
        try:
            while True:
                now = time.time()
                paused = self.control.pause.is_set()
                step = now - last_tick
                if step > SUSPEND_GAP:  # the computer slept (or the process was stopped): no training happened
                    self._note_suspend(step)
                    last_ckpt += step
                    last_eval += step
                    step = 0.0
                if not paused:
                    self.active_seconds += step
                last_tick = now
                if paused != (state == "paused"):
                    state = "paused" if paused else "running"
                    self._suspend_eval(paused)
                    self.emit({"type": "log", "message": "Training paused" if paused else "Training resumed"})
                    last_status = 0
                if self.control.stop.is_set():
                    reason = "stopped"
                    break
                if cfg.max_games and self.session_games >= cfg.max_games:
                    reason = "completed"
                    break
                if cfg.max_seconds and self.active_seconds >= cfg.max_seconds:
                    reason = "completed"
                    break

                got = self._drain_games()
                self._drain_live()
                self._poll_eval()
                trained = 0 if paused else self._train_some(budget=0.5)

                if now - last_status >= 1.0:
                    self._system_metrics(now)
                    self._emit_status(state)
                    last_status = now
                if not paused and now - last_ckpt >= cfg.checkpoint_minutes * 60:
                    self._save_checkpoint()
                    last_ckpt = now
                if (not paused and cfg.eval_minutes > 0 and self.eval_proc is None
                        and now - last_eval >= cfg.eval_minutes * 60):
                    self._start_eval()
                    last_eval = now
                if not got and not trained:
                    time.sleep(0.05)
        finally:
            self._shutdown(reason)
        return reason

    def _note_suspend(self, seconds: float) -> None:
        from .config import format_duration

        self.db.insert("sessions", {"elapsed": self._elapsed(), "event": "suspended",
                                    "detail": f"{seconds:.0f}s not counted as training time"})
        self.emit({"type": "log", "message": f"The computer was suspended for {format_duration(seconds)}; "
                                            f"that time does not count as training time"})

    # -- games ------------------------------------------------------------

    def _drain_games(self, limit: int = 256) -> int:
        n, rows = 0, []
        while n < limit:
            try:
                msg = self.game_queue.get_nowait()
            except queue.Empty:
                break
            n += 1
            if self.cfg.max_games and self.session_games >= self.cfg.max_games:
                continue  # finished after the limit was reached: not counted
            rows.append(self._add_game(msg))
        if rows:
            self.db.insert_many("games", rows)
        return n

    def _add_game(self, msg: dict) -> dict:
        self.buffer.add(msg["difficulty"], msg["visible"], msg["mines"])
        n = len(msg["visible"])
        self.games += 1
        self.session_games += 1
        self.positions += n
        self.session_positions += n
        self.recent_games.append(time.time())
        self.epsilon.value = self.cfg.epsilon(self.games)
        row = {"elapsed": self._elapsed(), "game": self.games, "difficulty": msg["difficulty"],
               "result": msg["result"], "moves": msg["moves"], "cleared": msg["cleared"], "worker": msg["worker"],
               "eps": msg["eps"]}
        self.emit({"type": "game", **row})
        return row

    def _drain_live(self) -> None:
        """Forward the newest detailed snapshot, and all games merged into one grid event per second."""
        latest = None
        while True:
            try:
                msg = self.live_queue.get_nowait()
            except queue.Empty:
                break
            if msg["type"] == "grid":
                self.live_grid[msg["worker"]] = msg["games"]
            else:
                latest = msg
        if latest is not None:
            self.emit(latest)
        if self.live_grid and time.time() - self.last_grid_emit >= 1.0:
            self.last_grid_emit = time.time()
            self.emit({"type": "grid", "focus": list(self.control.focus[:]),
                       "games": [g for w in sorted(self.live_grid) for g in self.live_grid[w]]})

    # -- optimisation -----------------------------------------------------

    def _train_some(self, budget: float) -> int:
        cfg = self.cfg
        if self.buffer.size < min(cfg.min_buffer, cfg.buffer_size):
            return 0
        steps, start = 0, time.time()
        while self._lag() > 0 and time.time() - start < budget:
            self._train_step()
            self.trained_samples += cfg.batch_size
            steps += 1
        self._update_throttle()
        return steps

    def _lag(self) -> float:
        """Training samples still owed to the positions stored this session."""
        return self.session_positions * self.cfg.sample_reuse - self.trained_samples

    def _update_throttle(self) -> None:
        """Pause the workers while training is more than `max_lag` batches behind, resume below a quarter of it."""
        lag = self._lag() / self.cfg.batch_size
        if lag > self.cfg.max_lag:
            self.throttle.set()
        elif lag < self.cfg.max_lag / 4:
            self.throttle.clear()

    def _train_step(self) -> None:
        name, visible, mines = self.buffer.sample(self.cfg.batch_size, self.rng)
        visible, mines = augment(visible, mines, self.rng)
        self.model.train()
        loss = masked_loss(self.model, visible, mines, DIFFICULTIES[name].mines)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        self.loss_acc[0] += loss.item()
        self.loss_acc[1] += 1
        if self.steps % 10 == 0:
            self._log_losses()
        if self.steps % self.cfg.publish_every == 0:
            self._publish_weights()

    def _log_losses(self) -> None:
        total, n = self.loss_acc
        if not n:
            return
        row = {"elapsed": self._elapsed(), "step": self.steps, "games": self.games, "loss": total / n,
               "lr": self.optimizer.param_groups[0]["lr"]}
        self.db.insert("train", row)
        self.emit({"type": "train", **row})
        self.loss_acc = [0.0, 0]

    def _publish_weights(self) -> None:
        self.model.eval()
        save_checkpoint(str(self.weights_path), self.model)
        with self.version.get_lock():
            self.version.value += 1

    # -- checkpoints and evaluation ----------------------------------------

    def _elapsed(self) -> float:
        return self.elapsed_base + self.active_seconds

    def _save_checkpoint(self) -> Path:
        path = self.bot_dir / "checkpoints" / f"ckpt_{self.games:07d}g.pt"
        self.model.eval()
        save_checkpoint(str(path), self.model, self.optimizer, games=self.games, steps=self.steps,
                        positions=self.positions, moves=self.moves, elapsed=self._elapsed(), bot=self.bot_dir.name,
                        config=self.cfg.to_dict(), created=time.time(), lr_scale=self.lr_scale,
                        lr_since=self.lr_since)
        self.checkpoint = str(path)
        self._write_state("running")
        self.emit({"type": "checkpoint", "path": str(path), "name": path.name, "games": self.games})
        return path

    def _write_state(self, status: str) -> None:
        write_json(self.bot_dir / "state.json", {
            "status": status, "games": self.games, "steps": self.steps, "positions": self.positions,
            "moves": self.moves, "elapsed": self._elapsed(), "checkpoint": self.checkpoint,
            "preset": self.cfg.preset, "updated": time.time(), **self.run_extra,
        })

    def _eval_difficulties(self) -> list[str]:
        return list(self.cfg.mix())

    def _start_eval(self) -> None:
        cfg = self.cfg
        ckpt = self._save_checkpoint()
        self.eval_meta = {"checkpoint": ckpt.name, "games": self.games, "steps": self.steps,
                          "elapsed": self._elapsed()}
        self.eval_proc = self.ctx.Process(
            target=eval_process_main, daemon=True, name="minesweepml-eval",
            args=(str(ckpt), self._eval_difficulties(), cfg.eval_boards, cfg.nice, self.eval_queue,
                  cfg.safe_neighbours, ckpt.name),
        )
        self.eval_proc.start()
        self.emit({"type": "log", "message": f"Evaluating {ckpt.name} on {cfg.eval_boards} benchmark boards per "
                                            f"difficulty"})

    def _suspend_eval(self, suspend: bool) -> None:
        if self.eval_proc is None or not self.eval_proc.is_alive() or suspend == self.eval_suspended:
            return
        try:
            proc = psutil.Process(self.eval_proc.pid)
            proc.suspend() if suspend else proc.resume()
            self.eval_suspended = suspend
        except psutil.Error:
            pass

    def _poll_eval(self) -> None:
        if self.eval_proc is None:
            return
        try:
            result = self.eval_queue.get_nowait()
        except queue.Empty:
            if not self.eval_proc.is_alive() and self.eval_proc.exitcode not in (None, 0):
                self.emit({"type": "log", "message": f"Evaluation process exited with code {self.eval_proc.exitcode}"})
                self.eval_proc = None
            return
        self.eval_proc.join(timeout=5)
        self.eval_proc = None
        meta = self.eval_meta
        if "error" in result:
            self.emit({"type": "log", "message": f"Evaluation failed: {result['error']}"})
            return
        rows = []
        for difficulty, s in result["difficulties"].items():
            rows.append({"elapsed": meta["elapsed"], "games": meta["games"], "step": meta["steps"],
                         "checkpoint": meta["checkpoint"], "difficulty": difficulty, "boards": s["boards"],
                         "wins": s["wins"], "cleared": s["cleared"], "moves": s["moves"]})
        self.db.insert_many("evals", rows)
        self.last_eval = {"checkpoint": meta["checkpoint"], "games": meta["games"],
                          "difficulties": result["difficulties"]}
        self.emit({"type": "eval", **self.last_eval, "elapsed": meta["elapsed"], "step": meta["steps"]})
        summary = ", ".join(f"{d} {100 * s['win_rate']:.1f}% won ({100 * s['cleared']:.0f}% cleared)"
                            for d, s in result["difficulties"].items())
        self.emit({"type": "log", "message": f"Eval {meta['checkpoint']}: {summary}"})
        self._update_lr()

    def _latest_eval(self) -> dict | None:
        evals = self.db.fetch("evals")
        if not evals:
            return None
        last = evals[-1]["checkpoint"]
        from .benchmark import summarize

        diffs = {}
        for e in evals:
            if e["checkpoint"] == last:
                rows = [{"won": i < e["wins"], "cleared": e["cleared"], "moves": e["moves"]}
                        for i in range(e["boards"])]
                diffs[e["difficulty"]] = summarize(rows)
        return {"checkpoint": last, "games": evals[-1]["games"], "difficulties": diffs}

    def eval_scores(self) -> list[tuple[int, float]]:
        """(games, mix-weighted win rate) of every evaluation since the last learning-rate change."""
        mix = self.cfg.mix()
        by_ckpt: dict[str, dict] = {}
        for e in self.db.fetch("evals"):
            item = by_ckpt.setdefault(e["checkpoint"], {"games": e["games"], "score": 0.0, "weight": 0.0})
            w = mix.get(e["difficulty"], 0.0)
            item["score"] += w * e["wins"] / max(1, e["boards"])
            item["weight"] += w
        return [(v["games"], v["score"] / v["weight"]) for v in by_ckpt.values()
                if v["weight"] > 0 and v["games"] >= self.lr_since]

    def _lr(self) -> float:
        return max(self.cfg.min_lr, self.cfg.lr * self.lr_scale)

    def _update_lr(self) -> None:
        """Lower the learning rate when the averaged benchmark score stops improving."""
        cfg = self.cfg
        if cfg.lr_patience <= 0 or self._lr() <= cfg.min_lr:
            return
        if not lr_should_drop([s for _, s in self.eval_scores()], cfg.lr_patience, cfg.lr_window):
            return
        self.lr_scale *= cfg.lr_drop
        self.lr_since = self.games
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr()
        self.emit({"type": "log", "message": f"Learning rate lowered to {self._lr():.2g}: the benchmark score "
                                            f"averaged over {cfg.lr_window} evaluations has not improved in the "
                                            f"last {cfg.lr_patience}"})

    # -- status -----------------------------------------------------------

    def _system_metrics(self, now: float) -> None:
        t0, moves0, pos0 = self.last_rates
        moves = sum(self.counters[:])
        dt = max(1e-6, now - t0)
        while self.recent_games and now - self.recent_games[0] > 60:
            self.recent_games.popleft()
        window = min(60.0, max(1.0, self.active_seconds))
        self.rates = {
            "games_per_sec": len(self.recent_games) / window,
            "moves_per_sec": (moves - moves0) / dt,
            "samples_per_sec": (self.positions - pos0) / dt,
        }
        self.moves += moves - moves0
        self.last_rates = (now, moves, self.positions)
        per_core = psutil.cpu_percent(percpu=True)
        row = {"elapsed": self._elapsed(), "games": self.games, **self.rates,
               "cpu_total": sum(per_core) / max(1, len(per_core)), "cpu_per_core": per_core,
               "mem_percent": psutil.virtual_memory().percent}
        self.db.insert("system", row)
        self.emit({"type": "system", **row})

    def _eta(self) -> float | None:
        cfg = self.cfg
        if cfg.max_seconds:
            return max(0.0, cfg.max_seconds - self.active_seconds)
        if cfg.max_games:
            rate = self.rates["games_per_sec"] or (self.session_games / max(1.0, self.active_seconds))
            return (cfg.max_games - self.session_games) / rate if rate > 0 else None
        return None

    def _emit_status(self, state: str) -> None:
        cfg = self.cfg
        self.emit({
            "type": "status", "state": state, "bot": self.bot_dir.name, "preset": cfg.preset,
            "workers": cfg.workers, "parallel_games": cfg.parallel_games, "threads": cfg.threads,
            "batch_size": cfg.batch_size, "nice": cfg.nice, "duration": cfg.duration_text(),
            "max_games": cfg.max_games, "max_seconds": cfg.max_seconds, "elapsed": self._elapsed(),
            "session_elapsed": self.active_seconds, "eta": self._eta(), "games": self.games,
            "session_games": self.session_games, "positions": self.positions, "moves": self.moves,
            "steps": self.steps, "buffer": self.buffer.size,
            "checkpoint": Path(self.checkpoint).name if self.checkpoint else None,
            "evaluating": self.eval_proc is not None, "lr": self.optimizer.param_groups[0]["lr"],
            "eps": self.epsilon.value, "net": f"{cfg.blocks}x{cfg.channels}", "mix": cfg.mix_text(),
            "throttled": self.throttle.is_set(),
            "last_eval": self.last_eval, **self.rates,
        })

    # -- shutdown ---------------------------------------------------------

    def _shutdown(self, reason: str) -> None:
        self._emit_status("stopping")
        self.emit({"type": "log", "message": "Stopping: discarding unfinished games and saving a checkpoint"})
        self.worker_stop.set()
        self.throttle.clear()
        self.control.pause.clear()
        if self.eval_proc is not None:
            self._suspend_eval(False)
            self.eval_proc.terminate()
            self.eval_proc.join(timeout=5)
            self.eval_proc = None
        deadline = time.time() + 20
        while any(w.is_alive() for w in self.workers) and time.time() < deadline:
            self._drain_games()  # games completed before the stop still count
            self._drain_live()
            time.sleep(0.05)
        for w in self.workers:
            if w.is_alive():
                w.terminate()
            w.join(timeout=5)
        self._drain_games()
        if reason == "completed":  # train on the last games before the final checkpoint
            self._train_some(budget=30.0)
        self._log_losses()
        self._save_checkpoint()
        self.buffer.save(self.bot_dir / "buffer.npz")
        final = "completed" if reason == "completed" else "stopped"
        self._write_state(final)
        self.db.insert("sessions", {"elapsed": self._elapsed(), "event": final,
                                    "detail": f"games={self.games} steps={self.steps}"})
        self.db.close()
        self._emit_status(final)
        self.emit({"type": "log", "message": f"Bot {self.bot_dir.name} {final}: {self.games:,} games, "
                                            f"{self.steps:,} training steps, checkpoint {Path(self.checkpoint).name}"})


def lr_should_drop(scores: list[float], patience: int, window: int) -> bool:
    """True when the best window-average score is older than the last `patience` evaluations."""
    if len(scores) < window + patience:
        return False
    averages = [sum(scores[i - window + 1:i + 1]) / window for i in range(window - 1, len(scores))]
    best_at = max(range(len(averages)), key=lambda i: (averages[i], -i))
    return len(averages) - 1 - best_at >= patience


def training_process_main(cfg_dict: dict, bot, fork, bot_overrides, control: Control, event_queue) -> None:
    """Entry point when training runs in a child process of the web server."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the server stops us gracefully
    trainer = Trainer(TrainConfig.from_dict(cfg_dict), control, event_queue.put, bot, fork, bot_overrides)
    try:
        trainer.run()
    except Exception as exc:
        event_queue.put({"type": "log", "message": f"Training failed: {exc!r}"})
        event_queue.put({"type": "status", "state": "error", "error": repr(exc)})
        raise
