import chess
import numpy as np
import pandas as pd
import pytest

from chesslm.analysis import chance_correct, illegal_reason, random_move_losses, sign_test, wilson_interval

# Back-rank position: both Ra8# and Rb8# are checkmate.
BACK_RANK = "6k1/5ppp/8/8/8/8/8/RR4K1 w - - 0 1"
# White's knight on e2 is pinned to its king by the rook on e8.
PINNED = "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1"


class FakeScorer:
    def centipawn_loss(self, fen, move):
        return 7


def test_wilson_interval():
    low, high = wilson_interval(78, 1000)
    assert low < 0.078 < high and high - low == pytest.approx(0.033, abs=0.002)
    assert wilson_interval(0, 250)[0] == pytest.approx(0, abs=1e-12)  # never below 0%


def test_sign_test():
    assert sign_test(5, 5) == 1.0
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 2**10)
    assert sign_test(22, 41) == sign_test(41, 22) < 0.05
    assert sign_test(np.int64(17), np.int64(62)) == sign_test(17, 62) < 1e-6  # counts from pandas


def test_chance_correct_counts_every_mate():
    board = chess.Board(BACK_RANK)
    n_legal = board.legal_moves.count()
    puzzle = {"fen": BACK_RANK, "answer_uci": "a1a8", "mate_in_1": True}
    assert chance_correct(puzzle) == pytest.approx(2 / n_legal)
    assert chance_correct({**puzzle, "mate_in_1": False}) == pytest.approx(1 / n_legal)


def test_random_move_losses():
    puzzles = pd.DataFrame([{"puzzle_id": "x", "fen": BACK_RANK}])
    losses = random_move_losses(puzzles, FakeScorer(), moves_per_puzzle=3, seed=0)
    assert list(losses.columns) == ["puzzle_id", "cp_loss"] and len(losses) == 3


@pytest.mark.parametrize("reply, reason", [
    ("a8a1", "no own piece there"),       # a8 is empty
    ("g8f8", "no own piece there"),       # Black's king, but it is White's turn
    ("a1h8", "piece can't move there"),   # a rook can't move diagonally
    ("Qd3", "no own piece there"),        # White has no queen
    ("Rc3", "piece can't move there"),
])
def test_illegal_reason(reply, reason):
    assert illegal_reason(BACK_RANK, reply) == reason


def test_illegal_reason_pinned_piece():
    assert illegal_reason(PINNED, "e2c3") == "leaves king in check"
    assert illegal_reason(PINNED, "Nc3") == "leaves king in check"
