"""Turn a chess position into a chat prompt, and read a move back out of the model's reply.

Training (SFT, DPO, RL) and evaluation all use these functions, so every model is
asked the same question in the same way. Settings come from configs/prompt.yaml.
"""

import re

import chess

NOTATION_NAMES = {"uci": "UCI notation", "san": "standard algebraic notation (SAN)"}
NOTATION_EXAMPLES = {"uci": "e2e4", "san": "Nf3"}

# What a move can look like. UCI: from-square + to-square (+ promotion piece).
# SAN: castling, a piece move (Nf3, Rxe7), or a pawn move (e4, exd5, e8=Q), plus an optional + or #.
UCI_PATTERN = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")
SAN_PATTERN = re.compile(r"(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](x[a-h])?[1-8](=?[QRBN])?)[+#]?")


# ---------------------------------------------------------------- building prompts

def move_to_text(board, move, notation):
    """Write a move in the chosen notation. `board` is the position before the move."""
    return move.uci() if notation == "uci" else board.san(move)


def recent_moves(row, prompt_cfg):
    """The last N moves of a training position or puzzle, in the chosen notation."""
    n = prompt_cfg["history_moves"]
    if n == 0:
        return []
    return row[f"history_{prompt_cfg['notation']}"].split()[-n:]


def user_message(fen, moves_before, prompt_cfg):
    """The text of the question we ask the model about one position."""
    board = chess.Board(fen)
    notation = prompt_cfg["notation"]
    lines = [
        f"Chess position (FEN): {fen}",
        f"{'White' if board.turn == chess.WHITE else 'Black'} to move.",
    ]
    if moves_before:
        lines.append(f"Recent moves: {' '.join(moves_before)}")
    if prompt_cfg["show_legal_moves"]:
        legal = sorted(move_to_text(board, m, notation) for m in board.legal_moves)
        lines.append(f"Legal moves: {' '.join(legal)}")
    lines.append(
        f"Reply with the best move in {NOTATION_NAMES[notation]} "
        f"(for example {NOTATION_EXAMPLES[notation]}) and nothing else."
    )
    return "\n".join(lines)


def build_messages(row, prompt_cfg):
    """Chat messages for one position (a training row or a puzzle row): a single user turn."""
    return [{"role": "user", "content": user_message(row["fen"], recent_moves(row, prompt_cfg), prompt_cfg)}]


def build_prompt(tokenizer, row, prompt_cfg):
    """The exact text the model receives, including the chat template's turn markers."""
    return tokenizer.apply_chat_template(
        build_messages(row, prompt_cfg), add_generation_prompt=True, tokenize=False
    )


# ---------------------------------------------------------------- reading the answer

def clean_token(token):
    """Strip punctuation and move numbers around a word: '**Nf3**' -> 'Nf3', '1.e4' -> 'e4', '0-0' -> 'O-O'."""
    token = token.strip(".,;:!?\"'`*()[]{}<>")
    token = re.sub(r"^\d+\.+", "", token)  # move numbers: "1.e4", "12...Nf6"
    # Castling is sometimes written with zeros: "0-0" -> "O-O", "0-0-0" -> "O-O-O".
    return re.sub(r"^0-0(-0)?", lambda m: m.group().replace("0", "O"), token)


def parse_move(text, board):
    """Find the first move in the model's reply and check it against the position.

    Returns (move, reason): reason is "legal", "illegal" (it looks like a move but
    isn't allowed here), or "unparseable" (no move found). The move is None unless legal.
    Both notations are accepted, whichever one the model was asked for.
    """
    for token in text.split():
        token = clean_token(token)
        if UCI_PATTERN.fullmatch(token.lower()):
            move = chess.Move.from_uci(token.lower())
            return (move, "legal") if move in board.legal_moves else (None, "illegal")
        if SAN_PATTERN.fullmatch(token):
            try:
                return board.parse_san(token), "legal"
            except ValueError:  # illegal, or ambiguous (e.g. "Nd2" when two knights can go there)
                return None, "illegal"
    return None, "unparseable"
