"""Data pipeline: Lichess puzzles -> evaluation sets, Lichess games -> training examples."""

import io
import random

import chess
import chess.pgn
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from chesslm.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"              # downloads (not committed)
PROCESSED_DIR = REPO_ROOT / "data" / "processed"  # training data (not committed)
EVAL_DIR = REPO_ROOT / "data" / "eval"            # test/dev puzzles (committed, so everyone grades on the same set)


def download(repo, filename, revision):
    """Download one dataset file into data/raw/. Skipped if it's already there."""
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=RAW_DIR)


def normalize_fen(fen):
    """Keep only the first four FEN fields (pieces, side to move, castling, en passant).

    The last two fields are move counters. They don't change the position, so two
    FENs that differ only there describe the same position.
    """
    return " ".join(fen.split()[:4])


# ---------------------------------------------------------------- puzzles

def puzzle_to_example(fen, moves):
    """Return (position the solver sees, first solution move in UCI) for a Lichess puzzle.

    Lichess stores the position *before* the opponent's move. The first move in
    `moves` is that opponent move; the second is the first move of the solution.
    """
    board = chess.Board(fen)
    opponent_move, solution_move = moves.split()[:2]
    board.push_uci(opponent_move)
    return board.fen(), solution_move


def rating_band(rating, edges):
    """Label a puzzle rating, e.g. with edges [1200, 1800, 2400]: '<1200', '1200-1800', ..., '2400+'."""
    if rating < edges[0]:
        return f"<{edges[0]}"
    for low, high in zip(edges, edges[1:]):
        if rating < high:
            return f"{low}-{high}"
    return f"{edges[-1]}+"


def load_puzzles(cfg):
    """Download the puzzle files and return one table of all puzzles with their rating band."""
    columns = ["PuzzleId", "FEN", "Moves", "Rating", "Themes"]
    paths = [download(cfg["repo"], f, cfg["revision"]) for f in cfg["files"]]
    puzzles = pd.concat([pd.read_parquet(p, columns=columns) for p in paths], ignore_index=True)
    puzzles["band"] = puzzles["Rating"].apply(rating_band, edges=cfg["rating_band_edges"])
    return puzzles


def sample_eval_sets(puzzles, cfg, seed):
    """Draw disjoint test and dev sets with the same number of puzzles in every rating band."""
    n_bands = len(cfg["rating_band_edges"]) + 1
    test_per_band = cfg["test_size"] // n_bands
    dev_per_band = cfg["dev_size"] // n_bands

    test_parts, dev_parts = [], []
    for _, band in puzzles.groupby("band"):
        picked = band.sample(n=test_per_band + dev_per_band, random_state=seed)
        test_parts.append(picked.iloc[:test_per_band])
        dev_parts.append(picked.iloc[test_per_band:])
    return to_eval_rows(pd.concat(test_parts)), to_eval_rows(pd.concat(dev_parts))


def to_eval_rows(puzzles):
    """Convert sampled puzzles into the solver's position, the answer in both notations, and a mate-in-1 flag."""
    rows = []
    for p in puzzles.itertuples():
        fen, answer_uci = puzzle_to_example(p.FEN, p.Moves)
        board = chess.Board(fen)
        opponent_move = chess.Move.from_uci(p.Moves.split()[0])
        rows.append({
            "puzzle_id": p.PuzzleId,
            "fen": fen,
            # Puzzles only know the opponent's last move, not the whole game.
            "history_uci": opponent_move.uci(),
            "history_san": chess.Board(p.FEN).san(opponent_move),
            "answer_uci": answer_uci,
            "answer_san": board.san(chess.Move.from_uci(answer_uci)),
            "rating": p.Rating,
            "band": p.band,
            # In mate-in-1 puzzles any checkmating move counts as correct.
            "mate_in_1": "mateIn1" in list(p.Themes),
            "themes": " ".join(p.Themes),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- games

def base_time_seconds(time_control):
    """'600+8' -> 600 (starting clock in seconds). Correspondence games ('-') -> None."""
    base = time_control.split("+")[0]
    return int(base) if base.isdigit() else None


def filter_games(games, cfg):
    """Apply the rating, bot, time-control, and termination filters. Returns (kept games, funnel counts)."""
    funnel = {"games read": len(games)}
    games = games[(games.WhiteElo >= cfg["min_elo"]) & (games.BlackElo >= cfg["min_elo"])]
    funnel[f"both players rated {cfg['min_elo']}+"] = len(games)
    if cfg["exclude_bots"]:
        games = games[(games.WhiteTitle != "BOT") & (games.BlackTitle != "BOT")]
        funnel["no bot accounts"] = len(games)
    base_time = games.TimeControl.map(base_time_seconds)
    games = games[base_time >= cfg["min_base_time_seconds"]]
    funnel[f"not bullet (base time >= {cfg['min_base_time_seconds']}s)"] = len(games)
    games = games[games.Termination == cfg["termination"]]
    funnel[f"termination = {cfg['termination']}"] = len(games)
    return games, funnel


def load_games(cfg):
    """Download the game files and return the games that pass the filters, plus funnel counts.

    Files are read in chunks so a 1 GB file never has to fit in memory at once.
    """
    columns = ["Site", "WhiteElo", "BlackElo", "WhiteTitle", "BlackTitle", "TimeControl", "Termination", "movetext"]
    kept, funnel = [], {}
    for filename in cfg["files"]:
        path = download(cfg["repo"], filename, cfg["revision"])
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=100_000):
            games, counts = filter_games(batch.to_pandas(), cfg)
            kept.append(games)
            funnel = {k: funnel.get(k, 0) + v for k, v in counts.items()}
    games = pd.concat(kept, ignore_index=True)
    games["game_id"] = games.Site.str.rsplit("/", n=1).str[-1]
    return games, funnel


def parse_moves(movetext):
    """Return the list of moves in a game's PGN move text (clock comments are ignored)."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    return list(game.mainline_moves())


def sample_positions(game_id, moves, per_game, skip_plies, rng):
    """Pick a few positions from one game, each paired with the move actually played there."""
    candidates = range(skip_plies, len(moves))
    chosen = set(rng.sample(candidates, min(per_game, len(candidates))))
    board = chess.Board()
    sans = []  # every move so far in SAN, which needs the board as it was before each move
    rows = []
    for ply, move in enumerate(moves):
        if ply in chosen:
            rows.append({
                "game_id": game_id,
                "ply": ply,
                "fen": board.fen(),
                # Every earlier move, in both notations, so prompts can include recent moves.
                "history_uci": " ".join(m.uci() for m in moves[:ply]),
                "history_san": " ".join(sans),
                "move_uci": move.uci(),
                "move_san": board.san(move),
            })
        sans.append(board.san(move))
        board.push(move)
    return rows


def build_positions(games, cfg_games, cfg_positions, seed):
    """Parse games, drop short ones, stop at max_games, and sample positions from each game."""
    rng = random.Random(seed)
    min_plies = 2 * cfg_games["min_moves"]
    skip_plies = 2 * cfg_positions["skip_opening_moves"]
    rows, n_games = [], 0
    for game in games.itertuples():
        moves = parse_moves(game.movetext)
        if len(moves) < min_plies:
            continue
        for row in sample_positions(game.game_id, moves, cfg_positions["per_game"], skip_plies, rng):
            row["white_elo"], row["black_elo"] = game.WhiteElo, game.BlackElo
            rows.append(row)
        n_games += 1
        if n_games == cfg_games["max_games"]:
            break
    return pd.DataFrame(rows), n_games


# ---------------------------------------------------------------- leakage and splits

def remove_overlap(positions, eval_fens):
    """Drop training positions that match any evaluation position. Returns (kept, number removed)."""
    eval_keys = {normalize_fen(f) for f in eval_fens}
    overlap = positions.fen.map(normalize_fen).isin(eval_keys)
    return positions[~overlap].reset_index(drop=True), int(overlap.sum())


def split_by_game(positions, val_fraction, seed):
    """Split into train/validation by game, so no game contributes to both."""
    game_ids = sorted(positions.game_id.unique())
    random.Random(seed).shuffle(game_ids)
    val_games = set(game_ids[: int(len(game_ids) * val_fraction)])
    is_val = positions.game_id.isin(val_games)
    return positions[~is_val].reset_index(drop=True), positions[is_val].reset_index(drop=True)
