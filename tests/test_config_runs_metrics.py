import pytest

from minesweepml.config import PRESET_NAMES, TrainConfig, parse_duration, preset_values, validate_bot_settings
from minesweepml.metrics import MetricsDB, read_history
from minesweepml.network import MineNet, NetConfig, save_checkpoint
from minesweepml.runs import (bot_settings, create_bot, delete_bot, latest_checkpoint, list_bots, list_checkpoints,
                              rename_bot, resolve_bot, resolve_checkpoint, update_bot)


def test_presets_fit_the_machine():
    for cores in (2, 4, 12, 32):
        for name in PRESET_NAMES:
            v = preset_values(name, cores)
            assert v["workers"] >= 1 and 1 <= v["threads"] <= 4


def test_durations():
    assert parse_duration("1h30m") == 5400
    assert parse_duration("45s") == 45
    assert parse_duration("90") == 5400
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_mix_and_epsilon():
    cfg = TrainConfig(mix_beginner=1, mix_intermediate=0, mix_expert=3, eps_start=0.2, eps_end=0.0, eps_games=100)
    assert cfg.mix() == {"beginner": 0.25, "expert": 0.75}
    assert cfg.epsilon(0) == 0.2 and cfg.epsilon(50) == pytest.approx(0.1) and cfg.epsilon(500) == 0.0
    with pytest.raises(ValueError):
        TrainConfig(mix_beginner=0, mix_intermediate=0, mix_expert=0).mix()
    with pytest.raises(ValueError):
        validate_bot_settings({"eps_start": 2})
    with pytest.raises(ValueError):
        validate_bot_settings({"preset": "x"})


def test_bot_lifecycle():
    bot = create_bot("Alpha", {"mix_expert": 1.0, "blocks": 2, "channels": 8})
    assert resolve_bot("Alpha") == bot
    assert bot_settings(bot)["blocks"] == 2
    with pytest.raises(FileExistsError):
        create_bot("Alpha")
    with pytest.raises(ValueError):
        create_bot("../evil")
    update_bot(bot, {"lr": 5e-4, "mix_beginner": 0.5})
    assert bot_settings(bot)["lr"] == 5e-4
    with pytest.raises(ValueError):
        update_bot(bot, {"channels": 64})  # the network size is fixed

    ckpt = bot / "checkpoints" / "ckpt_0000000g.pt"
    save_checkpoint(str(ckpt), MineNet(NetConfig(2, 8)), games=0)
    renamed = rename_bot(bot, "Beta")
    assert renamed.name == "Beta" and not bot.exists()
    assert latest_checkpoint(renamed).parent.parent == renamed
    assert resolve_checkpoint("Beta") == latest_checkpoint(renamed)
    assert [c["label"] for c in list_checkpoints()] == ["Beta/ckpt_0000000g.pt"]

    fork = create_bot("Gamma", from_checkpoint=resolve_checkpoint("Beta/ckpt_0000000g.pt"))
    assert bot_settings(fork)["channels"] == 8 and latest_checkpoint(fork) is not None
    assert {b["name"] for b in list_bots()} == {"Beta", "Gamma"}

    delete_bot(renamed)
    assert {b["name"] for b in list_bots()} == {"Gamma"}


def test_metrics_round_trip(tmp_path):
    db = MetricsDB(tmp_path / "metrics.db")
    for i in range(250):
        db.insert("games", {"elapsed": i, "game": i + 1, "difficulty": "expert" if i % 2 else "beginner",
                            "result": "won" if i % 4 == 1 else "lost", "moves": 10, "cleared": 0.5, "worker": 0,
                            "eps": 0.1}, commit=False)
    db.insert("train", {"elapsed": 1, "step": 10, "games": 5, "loss": 0.3, "lr": 1e-3})
    summary = db.game_summary(window=100)
    last_expert = [r for r in summary if r["difficulty"] == "expert"][-1]
    assert last_expert["n"] == 100 and last_expert["win_rate"] == pytest.approx(0.5)
    db.close()
    h = read_history(tmp_path)
    assert h["train"][0]["loss"] == 0.3 and h["games"]
