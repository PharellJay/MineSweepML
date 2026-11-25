"""Training configuration, intensity presets and duration parsing."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields

CPU_COUNT = os.cpu_count() or 4

PRESET_NAMES = ("light", "balanced", "heavy", "hardcore")

# Settings that belong to a bot rather than to one training session. They are saved in the
# bot's bot.json, edited on the Bots tab, and applied whenever that bot trains.
BOT_FIELDS = ("blocks", "channels", "mix_beginner", "mix_intermediate", "mix_expert", "safe_neighbours",
              "eps_start", "eps_end", "eps_games", "record_prob", "forgive_share", "lr")


def preset_values(name: str, cores: int = CPU_COUNT) -> dict:
    """Resolve an intensity preset for a machine with `cores` logical CPUs.

    Workers play games and use one core each; the trainer uses `threads`
    cores. Unlike a tree search, a Minesweeper move costs one forward pass,
    so training, not playing, is the bottleneck: a training position costs
    about three forward passes, several times over. The trainer therefore
    gets the larger share (its convolutions stop scaling beyond about 4
    threads), and workers pause whenever training falls behind. Workers and
    trainer run niced so the web server always gets CPU first. A stored
    position takes about 1 KB at Expert size, less on the smaller boards.
    """
    name = name.lower()
    threads = max(1, min(4, cores // 3))
    if name == "light":
        return dict(workers=max(1, cores // 12), threads=max(1, min(2, cores // 6)), batch_size=128, nice=19,
                    buffer_size=100_000)
    if name == "balanced":
        return dict(workers=max(1, cores // 6), threads=max(1, min(4, cores // 4)), batch_size=256, nice=10,
                    buffer_size=200_000)
    if name == "heavy":
        return dict(workers=max(1, cores // 4), threads=threads, batch_size=256, nice=8, buffer_size=300_000)
    if name == "hardcore":
        return dict(workers=max(1, cores // 2 - 1), threads=threads, batch_size=256, nice=5, buffer_size=400_000)
    raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESET_NAMES)}")


def apply_niceness(target: int) -> None:
    """Set this process's niceness to `target` (it can only be raised)."""
    try:
        current = os.nice(0)
        if target > current:
            os.nice(target - current)
    except (OSError, AttributeError):
        pass


def all_presets(cores: int = CPU_COUNT) -> dict:
    return {name: preset_values(name, cores) for name in PRESET_NAMES}


_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms]?)", re.I)


def parse_duration(text: str) -> float:
    """Parse '2h', '90m', '1h30m', '45s' or a bare number of minutes into seconds."""
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    total, pos = 0.0, 0
    for m in _DURATION_RE.finditer(text):
        if text[pos:m.start()].strip():
            raise ValueError(f"cannot parse duration {text!r}")
        value, unit = float(m.group(1)), m.group(2) or "m"
        total += value * {"h": 3600, "m": 60, "s": 1}[unit]
        pos = m.end()
    if text[pos:].strip() or total <= 0:
        raise ValueError(f"cannot parse duration {text!r}")
    return total


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


@dataclass
class TrainConfig:
    # intensity
    preset: str = "balanced"
    workers: int = 2
    threads: int = 2
    batch_size: int = 256
    nice: int = 10
    parallel_games: int = 32  # games played together per worker (one network batch per move)

    # duration: at most one of games / seconds; neither means "until stopped"
    max_games: int | None = None
    max_seconds: float | None = None

    # network (fixed when a bot is created)
    blocks: int = 6
    channels: int = 32

    # games: share of each difficulty in training
    mix_beginner: float = 0.2
    mix_intermediate: float = 0.3
    mix_expert: float = 0.5
    safe_neighbours: bool = True  # the first click opens an area

    # exploration: a random covered cell instead of the network's choice, decaying over games
    eps_start: float = 0.2
    eps_end: float = 0.01
    eps_games: int = 20_000
    record_prob: float = 0.25  # share of positions stored for training (neighbouring positions are similar)
    forgive_share: float = 0.0  # share of games that continue after a mine (data only; the result counts as lost)

    # optimisation
    lr: float = 1e-3
    lr_patience: int = 3  # evaluations without a new best score before the learning rate drops (0 = never)
    lr_window: int = 3  # evaluations averaged per score in that rule
    lr_drop: float = 0.3
    min_lr: float = 1e-4
    reset_lr: bool = False  # this session only: undo earlier drops
    weight_decay: float = 1e-4
    buffer_size: int = 200_000
    min_buffer: int = 5_000
    sample_reuse: float = 2.0  # times each stored position is (on average) used for training
    max_lag: int = 40  # batches training may fall behind the stored positions before workers pause
    publish_every: int = 25  # training steps between weight updates sent to workers

    # bookkeeping
    checkpoint_minutes: float = 10.0
    eval_minutes: float = 10.0  # 0 disables periodic evaluation
    eval_boards: int = 200  # benchmark boards per difficulty in a periodic evaluation
    seed: int | None = None

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "TrainConfig":
        values = preset_values(preset)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(preset=preset.lower(), **values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def bot_settings(self) -> dict:
        return {k: getattr(self, k) for k in BOT_FIELDS}

    def mix(self) -> dict[str, float]:
        """Normalised share of each difficulty; zero shares are left out."""
        raw = {"beginner": self.mix_beginner, "intermediate": self.mix_intermediate, "expert": self.mix_expert}
        total = sum(max(0.0, v) for v in raw.values())
        if total <= 0:
            raise ValueError("the difficulty mix needs at least one positive share")
        return {k: v / total for k, v in raw.items() if v > 0}

    def epsilon(self, games: int) -> float:
        if self.eps_games <= 0:
            return self.eps_end
        frac = min(1.0, games / self.eps_games)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def duration_text(self) -> str:
        if self.max_games:
            return f"{self.max_games} games"
        if self.max_seconds:
            return f"{format_duration(self.max_seconds)} wall-clock"
        return "until stopped"

    def mix_text(self) -> str:
        return ", ".join(f"{name} {100 * share:.0f}%" for name, share in self.mix().items())


def validate_bot_settings(settings: dict) -> dict:
    """Check user-edited bot settings; returns the cleaned values."""
    out = {}
    for key, value in settings.items():
        if key not in BOT_FIELDS:
            raise ValueError(f"unknown bot setting {key!r}")
        out[key] = value
    for key in ("mix_beginner", "mix_intermediate", "mix_expert"):
        if key in out and not 0 <= float(out[key]) <= 1:
            raise ValueError("difficulty shares must be between 0 and 1")
    if any(k.startswith("mix_") for k in out):
        cfg = TrainConfig(**{k: v for k, v in out.items() if k.startswith("mix_")})
        cfg.mix()  # raises if all are zero
    for key in ("eps_start", "eps_end", "record_prob", "forgive_share"):
        if key in out and not 0 <= float(out[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if "eps_games" in out and int(out["eps_games"]) < 0:
        raise ValueError("eps_games cannot be negative")
    if "lr" in out and not 1e-6 <= float(out["lr"]) <= 1e-1:
        raise ValueError("the learning rate must be between 1e-6 and 0.1")
    return out
