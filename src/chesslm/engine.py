"""Stockfish wrapper: find the engine, start it, and score moves."""

import os
import shutil

import chess
import chess.engine

# A forced mate is scored as a very large advantage. Losses are capped (cp_loss_cap) anyway.
MATE_SCORE = 10_000


def find_stockfish(configured_path=None):
    """Return the Stockfish binary path.

    Checks, in order: the path from the config, the STOCKFISH_PATH environment
    variable, then `stockfish` on the system PATH. Nothing is hard-coded, so the
    same code works on a Mac, Colab, and Windows.
    """
    path = configured_path or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "Stockfish not found. Install it (brew/apt install stockfish) "
            "or set STOCKFISH_PATH to the binary."
        )
    return path


def open_engine(configured_path=None):
    """Start Stockfish and return a python-chess engine handle. Call .quit() when done."""
    return chess.engine.SimpleEngine.popen_uci(find_stockfish(configured_path))


class MoveScorer:
    """Scores moves with Stockfish at a fixed depth, caching results by position and move.

    Use it in a `with` block so the engine is shut down afterwards:
        with MoveScorer(load_config("eval")["stockfish"]) as scorer:
            scorer.centipawn_loss(fen, move)
    """

    def __init__(self, stockfish_cfg):
        self.engine = open_engine(stockfish_cfg["path"])
        self.limit = chess.engine.Limit(depth=stockfish_cfg["depth"])
        self.cap = stockfish_cfg["cp_loss_cap"]
        self.best_cache = {}  # fen -> (best move, score)
        self.loss_cache = {}  # (fen, move) -> centipawn loss

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.engine.quit()

    def score(self, board, pov):
        """Stockfish's evaluation of `board` in centipawns, from the point of view of `pov` (a color)."""
        info = self.engine.analyse(board, self.limit)
        return info["score"].pov(pov).score(mate_score=MATE_SCORE)

    def best_move(self, fen):
        """Stockfish's best move in the position, and its score for the side to move."""
        if fen not in self.best_cache:
            board = chess.Board(fen)
            info = self.engine.analyse(board, self.limit)
            score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            self.best_cache[fen] = (info["pv"][0], score)
        return self.best_cache[fen]

    def centipawn_loss(self, fen, move):
        """How much worse `move` is than Stockfish's best move, from the mover's point of view.

        0 means the move is as good as the best move. Capped at cp_loss_cap.
        """
        key = (fen, move.uci())
        if key not in self.loss_cache:
            best, best_score = self.best_move(fen)
            if move == best:
                loss = 0
            else:
                board = chess.Board(fen)
                mover = board.turn
                board.push(move)
                loss = best_score - self.score(board, mover)
            self.loss_cache[key] = min(max(loss, 0), self.cap)
        return self.loss_cache[key]
