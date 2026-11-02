import random

import chess
import pandas as pd

from chesslm.data import (
    base_time_seconds,
    filter_games,
    normalize_fen,
    puzzle_to_example,
    rating_band,
    remove_overlap,
    sample_positions,
    split_by_game,
)


def test_puzzle_applies_opponent_move_first():
    # Lichess puzzle 00008: the stored FEN has Black to move. Black's f2g3 (bishop
    # takes rook) is the opponent's move; White's e6e7 is the answer.
    fen, answer = puzzle_to_example(
        "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
    )
    board = chess.Board(fen)
    assert board.turn == chess.WHITE
    assert board.piece_at(chess.G3) == chess.Piece(chess.BISHOP, chess.BLACK)
    assert answer == "e6e7"
    assert chess.Move.from_uci(answer) in board.legal_moves


def test_puzzle_with_black_to_solve():
    # Lichess puzzle 0000D: White's d3d6 is the opponent's move; Black solves with f8d8.
    fen, answer = puzzle_to_example(
        "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27",
        "d3d6 f8d8 d6d8 f6d8",
    )
    board = chess.Board(fen)
    assert board.turn == chess.BLACK
    assert answer == "f8d8"
    assert chess.Move.from_uci(answer) in board.legal_moves


def test_rating_band():
    edges = [1200, 1800, 2400]
    assert rating_band(800, edges) == "<1200"
    assert rating_band(1200, edges) == "1200-1800"
    assert rating_band(2399, edges) == "1800-2400"
    assert rating_band(2400, edges) == "2400+"


def test_normalize_fen_ignores_move_counters():
    a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 7 30"
    assert normalize_fen(a) == normalize_fen(b)


def test_base_time_seconds():
    assert base_time_seconds("600+8") == 600
    assert base_time_seconds("60+0") == 60
    assert base_time_seconds("-") is None


def test_filter_games():
    cfg = {"min_elo": 2000, "exclude_bots": True, "min_base_time_seconds": 180, "termination": "Normal"}
    games = pd.DataFrame({
        "name":        ["keep", "low rating", "bot", "bullet", "time forfeit"],
        "WhiteElo":    [2100, 1900, 2100, 2100, 2100],
        "BlackElo":    [2200, 2200, 2200, 2200, 2200],
        "WhiteTitle":  [None, None, "BOT", None, None],
        "BlackTitle":  [None, None, None, None, None],
        "TimeControl": ["600+5", "600+5", "600+5", "60+0", "600+5"],
        "Termination": ["Normal", "Normal", "Normal", "Normal", "Time forfeit"],
    })
    kept, funnel = filter_games(games, cfg)
    assert list(kept.name) == ["keep"]
    assert list(funnel.values()) == [5, 4, 3, 2, 1]


def test_sample_positions_records_the_move_played_from_each_position():
    moves = [chess.Move.from_uci(m) for m in
             "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6".split()]
    rows = sample_positions("game1", moves, per_game=3, skip_plies=10, rng=random.Random(0))
    assert len(rows) == 3
    for row in rows:
        assert row["ply"] >= 10  # opening moves skipped
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"])
        assert move in board.legal_moves
        assert board.san(move) == row["move_san"]
        assert len(row["history_uci"].split()) == row["ply"]


def test_remove_overlap():
    start = chess.STARTING_FEN
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    positions = pd.DataFrame({"fen": [start, after_e4]})
    # Same position as after_e4, but with different move counters: must still count as overlap.
    kept, removed = remove_overlap(positions, [after_e4.replace("0 1", "3 9")])
    assert removed == 1
    assert list(kept.fen) == [start]


def test_split_by_game_keeps_each_game_on_one_side():
    positions = pd.DataFrame({"game_id": [g for g in "abcdefghij" for _ in range(3)], "fen": range(30)})
    train, val = split_by_game(positions, val_fraction=0.2, seed=42)
    assert set(train.game_id).isdisjoint(val.game_id)
    assert len(train) + len(val) == len(positions)
    assert val.game_id.nunique() == 2
