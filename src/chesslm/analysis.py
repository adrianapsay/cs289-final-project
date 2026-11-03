"""Tools for interpreting graded runs: uncertainty, a random-move baseline, and the kinds of mistakes made."""

import math
import random

import chess
import pandas as pd

from chesslm.evaluate import is_correct
from chesslm.prompts import SAN_PATTERN, UCI_PATTERN, clean_token

PIECE_LETTERS = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


# ---------------------------------------------------------------- how sure are we?

def wilson_interval(successes, n, z=1.96):
    """95% interval for a proportion (e.g. accuracy on n puzzles). Stays inside [0, 1] even near 0%."""
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def sign_test(gained, lost):
    """Two-sided p-value for a paired comparison of two models on the same puzzles.

    Only puzzles where the models disagree count: `gained` are solved by the new model only,
    `lost` by the old model only. If neither model were better, each disagreement would go
    either way with probability 1/2. (This is McNemar's exact test.)
    """
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- random-move baseline

def chance_correct(puzzle):
    """Probability that a uniformly random legal move solves the puzzle."""
    board = chess.Board(puzzle["fen"])
    legal = list(board.legal_moves)
    return sum(is_correct(board, move, puzzle) for move in legal) / len(legal)


def random_move_losses(puzzles, scorer, moves_per_puzzle, seed):
    """Centipawn loss of random legal moves: `moves_per_puzzle` per puzzle, drawn without replacement."""
    rng = random.Random(seed)
    rows = []
    for puzzle in puzzles.to_dict("records"):
        legal = list(chess.Board(puzzle["fen"]).legal_moves)
        for move in rng.sample(legal, min(moves_per_puzzle, len(legal))):
            rows.append({"puzzle_id": puzzle["puzzle_id"], "cp_loss": scorer.centipawn_loss(puzzle["fen"], move)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- kinds of illegal moves

def illegal_reason(fen, reply):
    """Why an answer graded "illegal" is not allowed in this position.

    Returns "no own piece there", "piece can't move there", "leaves king in check", or
    "ambiguous" (SAN that fits more than one piece). Reads the first move in the reply,
    as the grader does.
    """
    board = chess.Board(fen)
    for token in reply.split():
        token = clean_token(token)
        if UCI_PATTERN.fullmatch(token.lower()):
            move = chess.Move.from_uci(token.lower())
            piece = board.piece_at(move.from_square)
            if piece is None or piece.color != board.turn:
                return "no own piece there"
            return "leaves king in check" if board.is_pseudo_legal(move) else "piece can't move there"
        if SAN_PATTERN.fullmatch(token):
            return _san_reason(board, token)
    return None


def _san_reason(board, san):
    if san.startswith("O-O"):
        return "piece can't move there"
    piece_type = PIECE_LETTERS.get(san[0], chess.PAWN)
    target = chess.parse_square(SAN_PATTERN.fullmatch(san).group(1).rstrip("QRBN=")[-2:])
    if not board.pieces(piece_type, board.turn):
        return "no own piece there"
    candidates = [m for m in board.pseudo_legal_moves
                  if m.to_square == target and board.piece_type_at(m.from_square) == piece_type]
    if not candidates:
        return "piece can't move there"
    if not any(board.is_legal(m) for m in candidates):
        return "leaves king in check"
    return "ambiguous"
