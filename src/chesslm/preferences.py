"""DPO preference pairs: for each position, a better move ("chosen") and a worse one ("rejected").

The chosen move is Stockfish's best move; the rejected move is a legal move that loses at
least `rejected_min_cp_loss` centipawns. Prompts are the same as in SFT and evaluation.
"""

import random

import chess
import pandas as pd
from datasets import Dataset

from chesslm.prompts import build_messages, move_to_text


def rejected_move(fen, scorer, min_cp_loss, rng):
    """A random legal move that loses at least min_cp_loss centipawns, or None if there is none."""
    moves = list(chess.Board(fen).legal_moves)
    rng.shuffle(moves)
    for move in moves:
        loss = scorer.centipawn_loss(fen, move)
        if loss >= min_cp_loss:
            return move, loss
    return None, None


def build_pairs(positions, prompt_cfg, dpo_cfg, scorer):
    """Build one preference pair per position. Positions with no clearly worse move are skipped."""
    rng = random.Random(dpo_cfg["seed"])
    notation = prompt_cfg["notation"]
    rows = []
    for row in positions.to_dict("records"):
        board = chess.Board(row["fen"])
        best, _ = scorer.best_move(row["fen"])
        worse, loss = rejected_move(row["fen"], scorer, dpo_cfg["rejected_min_cp_loss"], rng)
        if worse is None:
            continue
        rows.append({
            "prompt": build_messages(row, prompt_cfg),
            "chosen": [{"role": "assistant", "content": move_to_text(board, best, notation)}],
            "rejected": [{"role": "assistant", "content": move_to_text(board, worse, notation)}],
            "fen": row["fen"],
            "rejected_cp_loss": loss,
        })
    return pd.DataFrame(rows)


def to_dpo_dataset(pairs):
    """The three columns TRL's DPOTrainer expects: prompt, chosen, rejected."""
    return Dataset.from_pandas(pairs[["prompt", "chosen", "rejected"]], preserve_index=False)
