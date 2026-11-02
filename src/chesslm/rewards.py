"""RL reward: score a model's answer with Stockfish (used with TRL's GRPO trainer).

- An illegal or unparseable answer gets `invalid` (-1).
- dense mode: a legal move gets `legal_bonus` plus up to 1 more, depending on how close
  it is to Stockfish's best move (1 for the best move, 0 for a loss at the cap or worse).
- sparse mode: 1 for Stockfish's best move, 0 for any other legal move.
"""

import chess
from datasets import Dataset

from chesslm.engine import MoveScorer
from chesslm.prompts import build_messages, parse_move


def move_reward(text, fen, scorer, reward_cfg):
    """The reward for one answer to one position."""
    move, _ = parse_move(text, chess.Board(fen))
    if move is None:
        return reward_cfg["invalid"]
    if reward_cfg["mode"] == "sparse":
        best, _ = scorer.best_move(fen)
        return 1.0 if move == best else 0.0
    return reward_cfg["legal_bonus"] + 1 - scorer.centipawn_loss(fen, move) / scorer.cap


def make_reward_function(reward_cfg, stockfish_cfg):
    """A reward function in the form TRL's GRPOTrainer expects.

    GRPO passes the model's answers as `completions` and each dataset column as a
    keyword argument, so the dataset needs a `fen` column (see `to_rl_dataset`).
    """
    scorer = MoveScorer({**stockfish_cfg, "depth": reward_cfg["stockfish_depth"]})

    def chess_reward(completions, fen, **kwargs):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        return [move_reward(text, f, scorer, reward_cfg) for text, f in zip(texts, fen)]

    chess_reward.scorer = scorer  # call chess_reward.scorer.engine.quit() when training is done
    return chess_reward


def to_rl_dataset(positions, prompt_cfg):
    """Prompts for RL (the same as in SFT and evaluation), plus each position's FEN for the reward."""
    return Dataset.from_list([
        {"prompt": build_messages(row, prompt_cfg), "fen": row["fen"]}
        for row in positions.to_dict("records")
    ])
