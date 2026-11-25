import numpy as np

from minesweepml.players import RandomPlayer, SolverPlayer, play_game, play_games
from minesweepml.replay import GameRecord, final_board, frames, move_quality, visible_stack


def test_records_replay_exactly():
    records = play_games(RandomPlayer(0), "intermediate", range(20))
    for rec in records:
        again = GameRecord.from_dict(rec.to_dict())
        board = final_board(again)
        assert board.state == rec.result
        assert board.cleared() == rec.cleared
        assert visible_stack(again).shape[0] == len(rec.moves) + 1


def test_batched_play_equals_one_at_a_time():
    batched = play_games(SolverPlayer(), "beginner", range(12), parallel=5)
    single = [play_game(SolverPlayer(), "beginner", s) for s in range(12)]
    assert [r.moves for r in batched] == [r.moves for r in single]


def test_every_move_is_legal():
    for rec in play_games(RandomPlayer(1), "expert", range(10)):
        for n, (board, _) in enumerate(frames(rec)):
            if n < len(rec.moves):
                r, c = rec.moves[n]
                assert not board.revealed[r, c]


def test_solver_beats_random_on_beginner():
    seeds = range(200)
    solver = play_games(SolverPlayer(), "beginner", seeds)
    random = play_games(RandomPlayer(2), "beginner", seeds)
    solver_wins = np.mean([r.result == "won" for r in solver])
    random_wins = np.mean([r.result == "won" for r in random])
    assert solver_wins > 0.8
    assert random_wins < 0.2


def test_solver_never_blunders():
    for rec in play_games(SolverPlayer(), "intermediate", range(20)):
        assert move_quality(rec)["blunders"] == 0
