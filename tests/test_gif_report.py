from PIL import Image

from minesweepml.benchmark import save_result, summarize
from minesweepml.gif import render_gif, render_record
from minesweepml.metrics import MetricsDB
from minesweepml.network import MineNet, NetConfig, save_checkpoint
from minesweepml.players import SolverPlayer, play_game
from minesweepml.report import START, END, generate_report
from minesweepml.runs import create_bot, write_json


def test_gif_has_a_frame_per_move_and_holds_the_end(tmp_path):
    rec = play_game(SolverPlayer(), "intermediate", 4)
    data = render_record(rec, prob_player="solver", fps=10, hold_ms=2500)
    path = tmp_path / "g.gif"
    path.write_bytes(data)
    img = Image.open(path)
    assert img.n_frames == len(rec.moves) + 1
    img.seek(img.n_frames - 1)
    assert img.info["duration"] == 2500
    assert len(data) < 5_000_000


def test_render_gif_from_a_bot(tmp_path):
    bot = create_bot("Giffy", {"blocks": 2, "channels": 8})
    save_checkpoint(str(bot / "checkpoints" / "ckpt_0000000g.pt"), MineNet(NetConfig(2, 8)), games=0)
    res = render_gif("Giffy", tmp_path / "x.gif", "beginner", seed=1, fps=20)
    assert res["path"].exists() and res["frames"] == res["moves"] + 1


def test_report_writes_the_results_section(tmp_path):
    bot = create_bot("Rep", {"blocks": 2, "channels": 8})
    ckpt = bot / "checkpoints" / "ckpt_0000100g.pt"
    save_checkpoint(str(ckpt), MineNet(NetConfig(2, 8)), games=100)
    write_json(bot / "config.json", {"preset": "light", "blocks": 2, "channels": 8, "mix_beginner": 1.0,
                                     "mix_intermediate": 0.0, "mix_expert": 0.0, "batch_size": 32})
    write_json(bot / "state.json", {"games": 100, "steps": 10, "positions": 500, "elapsed": 60.0,
                                    "checkpoint": str(ckpt)})
    db = MetricsDB(bot / "metrics.db")
    db.insert("train", {"elapsed": 30, "step": 10, "games": 50, "loss": 0.5, "lr": 1e-3})
    db.insert("evals", {"elapsed": 60, "games": 100, "step": 10, "checkpoint": ckpt.name, "difficulty": "beginner",
                        "boards": 20, "wins": 5, "cleared": 0.6, "moves": 12})
    db.close()
    rows = [{"won": i < 7, "cleared": 0.9, "moves": 15} for i in range(20)]
    save_result({"player": "Rep/ckpt_0000100g.pt", "path": str(ckpt), "boards": 20, "created": 1.0,
                 "difficulties": {"beginner": summarize(rows)}})
    save_result({"player": "solver", "path": None, "boards": 20, "created": 1.0,
                 "difficulties": {"beginner": summarize([{"won": True, "cleared": 1.0, "moves": 20}] * 20)}})
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nintro\n\n{START}\nold\n{END}\n\nfooter\n")
    generate_report("Rep", readme, tmp_path / "docs" / "results", export_best=False)
    text = readme.read_text()
    assert "intro" in text and "footer" in text and "old" not in text
    assert "35.0%" in text  # the bot's benchmark: 7 of 20
    assert "rule-based solver | 100.0%" in text
    assert "25.0%" in text  # the evaluation during training: 5 of 20
    assert (tmp_path / "docs" / "results" / "loss.png").exists()
