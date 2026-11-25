import numpy as np
import torch

from minesweepml.board import COVERED, DIFFICULTIES, Board
from minesweepml.network import Evaluator, MineNet, NetConfig, model_from_checkpoint, save_checkpoint
from minesweepml.players import NetPlayer, play_games


def test_one_model_every_board_size():
    model = MineNet(NetConfig(2, 8))
    ev = Evaluator(model)
    for spec in DIFFICULTIES.values():
        b = Board(spec, 0)
        b.reveal(*spec.first_click())
        p = ev(b.visible()[None], spec.mines)
        assert p.shape == (1, spec.rows, spec.cols)
        assert np.all((p >= 0) & (p <= 1))
        single = ev.probabilities(b.visible(), spec.mines)
        assert np.all(np.isnan(single[b.visible() != COVERED]))


def test_checkpoint_round_trip(tmp_path):
    model = MineNet(NetConfig(3, 16))
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), model, games=7)
    loaded = model_from_checkpoint(str(path))
    assert loaded.cfg == NetConfig(3, 16)
    x = torch.randn(2, 12, 11, 11)
    model.eval()
    assert torch.allclose(model(x), loaded(x))


def test_net_player_plays_legal_games(tmp_path):
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), MineNet(NetConfig(2, 8)))
    player = NetPlayer(str(path))
    for difficulty in ("beginner", "expert"):
        for rec in play_games(player, difficulty, range(5)):
            assert rec.result in ("won", "lost")
            assert len(set(rec.moves)) == len(rec.moves)
