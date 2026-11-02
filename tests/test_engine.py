import os
import shutil

import chess
import pytest

from chesslm.engine import MoveScorer

pytestmark = pytest.mark.skipif(
    not (os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")), reason="Stockfish not installed"
)
CFG = {"path": None, "depth": 10, "cp_loss_cap": 1000}

# After 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6: Qxf7# wins at once; Qh3 throws the win away.
SCHOLARS_MATE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_best_move_has_zero_loss():
    with MoveScorer(CFG) as scorer:
        best, _ = scorer.best_move(SCHOLARS_MATE)
        assert best.uci() == "h5f7"
        assert scorer.centipawn_loss(SCHOLARS_MATE, best) == 0


def test_missing_a_mate_costs_the_capped_maximum():
    with MoveScorer(CFG) as scorer:
        loss = scorer.centipawn_loss(SCHOLARS_MATE, chess.Move.from_uci("h5h3"))
        assert loss == CFG["cp_loss_cap"]


def test_results_are_cached():
    with MoveScorer(CFG) as scorer:
        move = chess.Move.from_uci("h5h3")
        scorer.centipawn_loss(SCHOLARS_MATE, move)
        assert (SCHOLARS_MATE, "h5h3") in scorer.loss_cache
