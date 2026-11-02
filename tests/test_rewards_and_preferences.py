import os
import random
import shutil

import pandas as pd
import pytest

from chesslm.engine import MoveScorer
from chesslm.preferences import build_pairs, rejected_move, to_dpo_dataset
from chesslm.rewards import make_reward_function, move_reward

pytestmark = pytest.mark.skipif(
    not (os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")), reason="Stockfish not installed"
)
STOCKFISH = {"path": None, "depth": 10, "cp_loss_cap": 1000}
DENSE = {"mode": "dense", "invalid": -1.0, "legal_bonus": 0.1, "stockfish_depth": 10}
SPARSE = {**DENSE, "mode": "sparse"}
PROMPT = {"notation": "uci", "history_moves": 0, "show_legal_moves": False}

# After 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6: Qxf7# (h5f7) mates; Qh3 throws the win away; Nf6 is illegal for White.
SCHOLARS_MATE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_dense_reward():
    with MoveScorer(STOCKFISH) as scorer:
        assert move_reward("h5f7", SCHOLARS_MATE, scorer, DENSE) == pytest.approx(1.1)  # best move
        assert move_reward("Qh3", SCHOLARS_MATE, scorer, DENSE) == pytest.approx(0.1)   # legal, but loses the cap
        assert move_reward("Nf6", SCHOLARS_MATE, scorer, DENSE) == -1.0                 # illegal
        assert move_reward("no idea", SCHOLARS_MATE, scorer, DENSE) == -1.0             # unparseable


def test_sparse_reward():
    with MoveScorer(STOCKFISH) as scorer:
        assert move_reward("Qxf7#", SCHOLARS_MATE, scorer, SPARSE) == 1.0
        assert move_reward("Bxf7+", SCHOLARS_MATE, scorer, SPARSE) == 0.0
        assert move_reward("Nf6", SCHOLARS_MATE, scorer, SPARSE) == -1.0


def test_reward_function_accepts_grpo_inputs():
    reward = make_reward_function(DENSE, STOCKFISH)
    completions = [[{"role": "assistant", "content": "h5f7"}], [{"role": "assistant", "content": "junk"}]]
    rewards = reward(completions, fen=[SCHOLARS_MATE, SCHOLARS_MATE], trainer_state=None)
    reward.scorer.engine.quit()
    assert rewards == pytest.approx([1.1, -1.0])


def test_rejected_move_is_clearly_worse():
    with MoveScorer(STOCKFISH) as scorer:
        move, loss = rejected_move(SCHOLARS_MATE, scorer, 200, random.Random(0))
        assert loss >= 200 and move.uci() != "h5f7"
        # No move can lose more than the cap, so a higher threshold finds nothing.
        assert rejected_move(SCHOLARS_MATE, scorer, 2000, random.Random(0)) == (None, None)


def test_build_pairs():
    positions = pd.DataFrame([{"fen": SCHOLARS_MATE, "history_uci": "", "history_san": ""}])
    dpo_cfg = {"rejected_min_cp_loss": 200, "seed": 42}
    with MoveScorer(STOCKFISH) as scorer:
        pairs = build_pairs(positions, PROMPT, dpo_cfg, scorer)
    dataset = to_dpo_dataset(pairs)
    assert dataset.column_names == ["prompt", "chosen", "rejected"]
    assert dataset[0]["chosen"] == [{"role": "assistant", "content": "h5f7"}]
    assert dataset[0]["rejected"][0]["content"] != "h5f7"
    assert SCHOLARS_MATE in dataset[0]["prompt"][0]["content"]
