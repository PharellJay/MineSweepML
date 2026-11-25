import numpy as np
import pytest
import torch

from minesweepml.benchmark import benchmark_seeds, run_benchmark, wilson
from minesweepml.board import DIFFICULTIES, Board, neighbour_counts
from minesweepml.config import TrainConfig
from minesweepml.metrics import read_history
from minesweepml.network import MineNet, NetConfig
from minesweepml.players import SolverPlayer, play_games
from minesweepml.runs import bot_checkpoints, list_bots, read_json
from minesweepml.trainer import Control, ReplayBuffer, Trainer, augment, lr_should_drop, masked_loss


def positions(name: str, n: int, seed: int = 0):
    spec = DIFFICULTIES[name]
    vis, mines = [], []
    for s in range(seed, seed + n):
        b = Board(spec, s)
        b.reveal(*spec.first_click())
        vis.append(b.visible())
        mines.append(b.mines)
    return np.stack(vis), np.stack(mines)


def test_replay_buffer_round_trip(tmp_path):
    buf = ReplayBuffer(10_000, {"beginner": 0.5, "expert": 0.5})
    vis, mines = positions("expert", 3)
    for v, m in zip(vis, mines):
        buf.add("expert", np.stack([v, v]), m)
    assert buf.size == 6
    name, v, m = buf.sample(4, np.random.default_rng(0))
    assert name == "expert" and v.shape == (4, 16, 30) and m.dtype == bool
    buf.save(tmp_path / "b.npz")
    other = ReplayBuffer(10_000, {"beginner": 0.5, "expert": 0.5})
    other.load(tmp_path / "b.npz")
    assert other.size == 6
    assert np.array_equal(other.parts["expert"].visible[:6], buf.parts["expert"].visible[:6])


@pytest.mark.parametrize("name", ["beginner", "expert"])
def test_augmentation_keeps_boards_consistent(name):
    vis, mines = positions(name, 8)
    rng = np.random.default_rng(1)
    for _ in range(6):
        v, m = augment(vis, mines, rng)
        for vk, mk in zip(v, m):
            counts = neighbour_counts(mk)
            shown = vk >= 0
            assert np.array_equal(vk[shown], counts[shown])  # numbers still match their mines
            assert not mk[shown].any()


def test_loss_learns_a_fixed_batch():
    torch.manual_seed(0)
    vis, mines = positions("beginner", 64)
    model = MineNet(NetConfig(2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = masked_loss(model, vis, mines, 10)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.8


def test_lr_plateau_rule():
    assert not lr_should_drop([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], patience=3, window=3)
    assert lr_should_drop([0.1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2], patience=3, window=3)
    assert not lr_should_drop([0.5, 0.5], patience=3, window=3)


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_benchmark_matches_direct_play():
    result = run_benchmark("solver", ["beginner"], boards=60, processes=2, nice=0)
    direct = play_games(SolverPlayer(), "beginner", benchmark_seeds(60))
    s = result["difficulties"]["beginner"]
    assert s["boards"] == 60 and s["wins"] == sum(r.result == "won" for r in direct)
    assert benchmark_seeds(1)[0] >= 2**40  # never a training seed


def test_short_training_session():
    cfg = TrainConfig(workers=1, threads=1, parallel_games=8, batch_size=32, min_buffer=200, max_games=150,
                      mix_beginner=1.0, mix_intermediate=0.0, mix_expert=0.0, blocks=2, channels=8,
                      eval_minutes=0, checkpoint_minutes=60, nice=0, buffer_size=5_000, record_prob=1.0, seed=3)
    trainer = Trainer(cfg, Control(), bot="tiny")
    assert trainer.run() == "completed"
    bots = list_bots()
    assert [b["name"] for b in bots] == ["tiny"] and bots[0]["status"] == "completed"
    assert bots[0]["games"] >= 150
    assert len(bot_checkpoints(trainer.bot_dir)) == 2  # the untrained start and the final checkpoint
    h = read_history(trainer.bot_dir)
    assert h["train"] and h["games"]
    assert (trainer.bot_dir / "buffer.npz").exists()
    state = read_json(trainer.bot_dir / "state.json")
    assert state["steps"] > 0

    # continuing the bot picks up its counters and buffer
    cfg2 = TrainConfig(**{**cfg.to_dict(), "max_games": 20})
    again = Trainer(cfg2, Control(), bot="tiny")
    assert again.run() == "completed"
    assert read_json(trainer.bot_dir / "state.json")["games"] >= state["games"] + 20


def test_suspended_time_is_not_training_time(monkeypatch):
    """A jump of the wall clock (the laptop slept) must not count against a time limit."""
    import threading
    import time as real_time

    import minesweepml.trainer as trainer_mod

    class Clock:
        offset = 0.0

        @staticmethod
        def time():
            return real_time.time() + Clock.offset

        sleep = staticmethod(real_time.sleep)

    monkeypatch.setattr(trainer_mod, "time", Clock)
    threading.Timer(2.0, lambda: setattr(Clock, "offset", 1000.0)).start()
    cfg = TrainConfig(workers=1, threads=1, parallel_games=4, batch_size=16, min_buffer=100, max_seconds=6,
                      mix_beginner=1.0, mix_intermediate=0.0, mix_expert=0.0, blocks=1, channels=8,
                      eval_minutes=0, checkpoint_minutes=60, nice=0, buffer_size=5_000)
    trainer = Trainer(cfg, Control(), bot="sleepy")
    start = real_time.time()
    assert trainer.run() == "completed"
    assert real_time.time() - start >= 5.5  # it trained for the full limit, not 2 s
    state = read_json(trainer.bot_dir / "state.json")
    assert state["elapsed"] < 60
    events = [s["event"] for s in read_history(trainer.bot_dir)["sessions"]]
    assert "suspended" in events
