"""Stockfish wrapper: find the engine and start it.

Move scoring and caching are added in Step 4 (evaluation harness).
"""

import os
import shutil

import chess.engine


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
