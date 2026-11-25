"""Per-bot metrics in SQLite, so charts survive restarts.

`elapsed` is the bot's cumulative training time in seconds (paused time
excluded), so curves from later sessions continue where they left off.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS train (
    ts REAL, elapsed REAL, step INTEGER, games INTEGER, loss REAL, lr REAL);
CREATE TABLE IF NOT EXISTS games (
    ts REAL, elapsed REAL, game INTEGER, difficulty TEXT, result TEXT, moves INTEGER,
    cleared REAL, worker INTEGER, eps REAL);
CREATE TABLE IF NOT EXISTS system (
    ts REAL, elapsed REAL, games INTEGER, games_per_sec REAL, moves_per_sec REAL,
    samples_per_sec REAL, cpu_total REAL, cpu_per_core TEXT, mem_percent REAL);
CREATE TABLE IF NOT EXISTS evals (
    ts REAL, elapsed REAL, games INTEGER, step INTEGER, checkpoint TEXT, difficulty TEXT,
    boards INTEGER, wins INTEGER, cleared REAL, moves REAL);
CREATE TABLE IF NOT EXISTS sessions (
    ts REAL, elapsed REAL, event TEXT, detail TEXT);
"""

TABLE_COLUMNS = {
    "train": ("ts", "elapsed", "step", "games", "loss", "lr"),
    "games": ("ts", "elapsed", "game", "difficulty", "result", "moves", "cleared", "worker", "eps"),
    "system": ("ts", "elapsed", "games", "games_per_sec", "moves_per_sec", "samples_per_sec",
               "cpu_total", "cpu_per_core", "mem_percent"),
    "evals": ("ts", "elapsed", "games", "step", "checkpoint", "difficulty", "boards", "wins", "cleared", "moves"),
    "sessions": ("ts", "elapsed", "event", "detail"),
}


class MetricsDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, table: str, row: dict, commit: bool = True) -> None:
        cols = TABLE_COLUMNS[table]
        row = dict(row)
        row.setdefault("ts", time.time())
        if table == "system" and not isinstance(row.get("cpu_per_core"), str):
            row["cpu_per_core"] = json.dumps(row.get("cpu_per_core") or [])
        self.conn.execute(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [row.get(c) for c in cols]
        )
        if commit:
            self.conn.commit()

    def insert_many(self, table: str, rows: list[dict]) -> None:
        for row in rows:
            self.insert(table, row, commit=False)
        self.conn.commit()

    def fetch(self, table: str, limit: int | None = None) -> list[dict]:
        cols = TABLE_COLUMNS[table]
        rows = [dict(zip(cols, r)) for r in self.conn.execute(f"SELECT {','.join(cols)} FROM {table} ORDER BY rowid")]
        if table == "system":
            for r in rows:
                r["cpu_per_core"] = json.loads(r["cpu_per_core"] or "[]")
        if limit and len(rows) > limit:  # thin long histories evenly, keep the last row
            stride = len(rows) / limit
            rows = [rows[int(i * stride)] for i in range(limit - 1)] + [rows[-1]]
        return rows

    def game_summary(self, window: int = 100, points: int = 400) -> list[dict]:
        """Rolling win rate and cleared share per difficulty over the last `window` games of it.

        Computed over all games with running sums, then thinned to about `points` rows.
        """
        rows = self.conn.execute("SELECT elapsed, game, difficulty, result, cleared, moves FROM games ORDER BY rowid")
        recent: dict[str, deque] = {}
        sums: dict[str, list[float]] = {}
        out = []
        for elapsed, game, difficulty, result, cleared, moves in rows:
            q = recent.setdefault(difficulty, deque())
            s = sums.setdefault(difficulty, [0.0, 0.0, 0.0])
            item = (1.0 if result == "won" else 0.0, cleared or 0.0, moves or 0)
            q.append(item)
            for i in range(3):
                s[i] += item[i]
            if len(q) > window:
                old = q.popleft()
                for i in range(3):
                    s[i] -= old[i]
            n = len(q)
            out.append({"elapsed": elapsed, "game": game, "difficulty": difficulty,
                        "win_rate": s[0] / n, "cleared": s[1] / n, "moves": s[2] / n, "n": n})
        if len(out) > points:
            stride = len(out) / points
            out = [out[int(i * stride)] for i in range(points - 1)] + [out[-1]]
        return out

    def history(self, limit: int = 2000) -> dict:
        h = {t: self.fetch(t, None if t in ("evals", "sessions") else limit) for t in TABLE_COLUMNS if t != "games"}
        h["games"] = self.game_summary()
        return h

    def close(self) -> None:
        self.conn.close()


def read_history(bot_dir: str | Path, limit: int = 2000) -> dict:
    path = Path(bot_dir) / "metrics.db"
    if not path.exists():
        return {t: [] for t in TABLE_COLUMNS}
    db = MetricsDB(path)
    try:
        return db.history(limit)
    finally:
        db.close()
