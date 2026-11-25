"""Bot directories and checkpoint discovery.

runs/<bot>/
    bot.json           the bot's own settings (network size, difficulty mix, exploration, learning rate)
    config.json        full training configuration of the latest session
    state.json         status, counters and the latest checkpoint
    metrics.db         SQLite metrics (losses, games, system, evaluations)
    weights.pt         newest weights, read by the workers
    buffer.npz         replay buffer saved on stop (used when training continues)
    checkpoints/       ckpt_<games>g.pt files
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from .config import BOT_FIELDS, TrainConfig, validate_bot_settings

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR.parent / "models"

BOT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,59}")
MAX_BLOCKS, MAX_CHANNELS = 20, 256


def runs_dir() -> Path:
    return Path(os.environ.get("MINESWEEPML_RUNS", "runs")).resolve()


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def check_name(name: str) -> str:
    name = name.strip()
    if not BOT_NAME.fullmatch(name):
        raise ValueError("use 1-60 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit")
    return name


def create_bot(name: str, settings: dict | None = None, from_checkpoint: str | Path | None = None) -> Path:
    """Create a bot, untrained or starting from a copy of a checkpoint.

    A copied checkpoint becomes the bot's first checkpoint and fixes its
    network size; training then continues from it.
    """
    name = check_name(name)
    path = runs_dir() / name
    if path.exists():
        raise FileExistsError(f"a bot named {name!r} already exists")
    defaults = TrainConfig().bot_settings()
    settings = {**defaults, **validate_bot_settings(settings or {})}
    blocks, channels = int(settings["blocks"]), int(settings["channels"])
    data = None
    if from_checkpoint:
        from .network import load_checkpoint

        data = load_checkpoint(str(from_checkpoint))
        blocks, channels = data["net"]["blocks"], data["net"]["channels"]
        saved = (data.get("config") or {})
        settings.update({k: saved[k] for k in BOT_FIELDS if k in saved and k not in ("blocks", "channels")})
    if not (1 <= blocks <= MAX_BLOCKS and 8 <= channels <= MAX_CHANNELS):
        raise ValueError(f"network size must be 1-{MAX_BLOCKS} blocks of 8-{MAX_CHANNELS} channels")
    settings.update(blocks=blocks, channels=channels)
    (path / "checkpoints").mkdir(parents=True)
    state = {"status": "new", "games": 0, "steps": 0, "positions": 0, "elapsed": 0.0, "checkpoint": None,
             "updated": time.time()}
    if data is not None:
        source = Path(from_checkpoint)
        games = data.get("games") or 0
        dest = path / "checkpoints" / f"ckpt_{games:07d}g.pt"
        shutil.copyfile(source, dest)
        state.update(games=games, steps=data.get("steps") or 0, positions=data.get("positions") or 0,
                     elapsed=data.get("elapsed") or 0.0, checkpoint=str(dest), origin=str(source.resolve()))
    write_json(path / "bot.json", {**settings, "created": time.time()})
    write_json(path / "state.json", state)
    return path


def bot_settings(bot: Path) -> dict:
    data = read_json(Path(bot) / "bot.json", {}) or {}
    return {k: data[k] for k in BOT_FIELDS if k in data}


def update_bot(bot: Path, settings: dict) -> dict:
    """Change a bot's training settings. The network size is fixed once a bot exists."""
    bot = Path(bot)
    settings = validate_bot_settings(settings)
    current = read_json(bot / "bot.json", {}) or {}
    for key in ("blocks", "channels"):
        if key in settings and current.get(key) is not None and int(settings[key]) != int(current[key]):
            raise ValueError("the network size of an existing bot cannot change; create a new bot instead")
    current.update(settings)
    write_json(bot / "bot.json", current)
    return bot_settings(bot)


def rename_bot(bot: Path, new_name: str) -> Path:
    new_name = check_name(new_name)
    target = runs_dir() / new_name
    if target.exists():
        raise FileExistsError(f"a bot named {new_name!r} already exists")
    bot = Path(bot)
    bot.rename(target)
    # checkpoint paths in state.json point into the old directory
    state = read_json(target / "state.json", {}) or {}
    if state.get("checkpoint"):
        state["checkpoint"] = str(target / "checkpoints" / Path(state["checkpoint"]).name)
        write_json(target / "state.json", state)
    return target


def delete_bot(bot: Path) -> None:
    bot = Path(bot).resolve()
    if bot.parent != runs_dir() or not (bot / "checkpoints").is_dir():
        raise ValueError(f"{bot} is not a bot directory")
    shutil.rmtree(bot)


def new_bot_name() -> str:
    return time.strftime("bot-%Y%m%d-%H%M%S")


def resolve_bot(name_or_path: str) -> Path:
    """Find a bot directory by name or path."""
    path = Path(name_or_path)
    for candidate in (path, runs_dir() / name_or_path):
        if candidate.is_dir() and (candidate / "checkpoints").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"no bot named {name_or_path!r} in {runs_dir()}")


def bot_checkpoints(bot: Path) -> list[Path]:
    return sorted(Path(bot, "checkpoints").glob("ckpt_*.pt"))


def latest_checkpoint(bot: Path) -> Path | None:
    state = read_json(Path(bot) / "state.json", {}) or {}
    if state.get("checkpoint") and Path(state["checkpoint"]).exists():
        return Path(state["checkpoint"])
    ckpts = bot_checkpoints(bot)
    return ckpts[-1] if ckpts else None


def list_bots() -> list[dict]:
    base = runs_dir()
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.iterdir()):
        if not (path / "checkpoints").is_dir():
            continue
        state = read_json(path / "state.json", {}) or {}
        settings = bot_settings(path)
        out.append({
            "name": path.name,
            "path": str(path),
            "status": state.get("status", "unknown"),
            "games": state.get("games", 0),
            "steps": state.get("steps", 0),
            "elapsed": state.get("elapsed", 0.0),
            "preset": state.get("preset"),
            "checkpoints": len(bot_checkpoints(path)),
            "origin": state.get("origin"),
            "net": {"blocks": settings.get("blocks"), "channels": settings.get("channels")},
            "settings": settings,
            "updated": path.stat().st_mtime,
        })
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out


def checkpoint_label(path: str | Path) -> str:
    path = Path(path).resolve()
    if path.parent == MODELS_DIR.resolve():
        return f"models/{path.name}"
    if path.parent.name == "checkpoints":
        return f"{path.parent.parent.name}/{path.name}"
    return path.name


def list_checkpoints() -> list[dict]:
    """Every checkpoint a game can be played with: bundled models first, then bots (newest first)."""
    out = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            out.append({"label": f"models/{p.name}", "path": str(p), "bot": None})
    for bot in list_bots():
        for p in reversed(bot_checkpoints(Path(bot["path"]))):
            out.append({"label": f"{bot['name']}/{p.name}", "path": str(p), "bot": bot["name"]})
    return out


def resolve_checkpoint(spec: str) -> Path:
    """A checkpoint file, a bot name/dir (-> its latest checkpoint) or a label from list_checkpoints."""
    path = Path(spec)
    if path.is_file():
        return path.resolve()
    for item in list_checkpoints():
        if item["label"] == spec:
            return Path(item["path"])
    bot = resolve_bot(spec)
    ckpt = latest_checkpoint(bot)
    if ckpt is None:
        raise FileNotFoundError(f"bot {spec!r} has no checkpoints yet")
    return ckpt
