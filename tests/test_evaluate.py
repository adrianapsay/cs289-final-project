import chess
import pandas as pd

from chesslm.evaluate import grade, is_correct, summarize

# Back-rank position: both Ra8# and Rb8# are checkmate.
BACK_RANK = "6k1/5ppp/8/8/8/8/8/RR4K1 w - - 0 1"


class FakeScorer:
    """Stands in for Stockfish, so these tests run without the engine."""

    def centipawn_loss(self, fen, move):
        return 0


def puzzle(answer_uci, mate_in_1):
    return {"puzzle_id": "x", "band": "<1200", "fen": BACK_RANK, "answer_uci": answer_uci, "mate_in_1": mate_in_1}


def test_any_mate_counts_in_mate_in_1():
    board = chess.Board(BACK_RANK)
    other_mate = chess.Move.from_uci("b1b8")
    assert is_correct(board, other_mate, puzzle("a1a8", mate_in_1=True))
    assert not is_correct(board, other_mate, puzzle("a1a8", mate_in_1=False))
    assert board.fen() == BACK_RANK  # checking for mate must not change the board


def test_grade_reply():
    row = grade(puzzle("a1a8", mate_in_1=True), "I play Ra8#", FakeScorer())
    assert row["move"] == "a1a8" and row["outcome"] == "legal" and row["correct"]

    row = grade(puzzle("a1a8", mate_in_1=True), "Qd8", FakeScorer())
    assert row["outcome"] == "illegal" and not row["correct"] and row["cp_loss"] is None


def test_summarize():
    graded = pd.DataFrame({
        "band": ["<1200", "<1200", "2400+", "2400+"],
        "outcome": ["legal", "legal", "illegal", "unparseable"],
        "correct": [True, False, False, False],
        "cp_loss": [0, 300, None, None],
        "seconds": [0.5] * 4,
        "n_tokens": [4] * 4,
    })
    s = summarize(graded)
    assert s["puzzle_accuracy"] == 0.25
    assert s["accuracy_by_band"] == {"2400+": 0.0, "<1200": 0.5}
    assert s["legal_rate"] == 0.5 and s["illegal_rate"] == 0.25 and s["unparseable_rate"] == 0.25
    assert s["acpl"] == 150  # legal moves only
