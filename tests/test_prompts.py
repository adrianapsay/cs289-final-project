import chess
import pytest

from chesslm.prompts import build_messages, parse_move, recent_moves, user_message

UCI_CFG = {"notation": "uci", "history_moves": 0, "show_legal_moves": False}

# After 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6: White can mate with Qxf7#, or check with Bxf7+.
SCHOLARS_MATE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
PROMOTION = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"        # White pawn on e7 can promote
CASTLING = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"     # White can castle both ways
TWO_KNIGHTS = "4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1"      # knights on b1 and f1 can both reach d2


@pytest.mark.parametrize("fen, text, expected_uci, reason", [
    # plain answers and whitespace
    (chess.STARTING_FEN, "e2e4", "e2e4", "legal"),
    (chess.STARTING_FEN, "  e2e4\n", "e2e4", "legal"),
    (chess.STARTING_FEN, "E2E4", "e2e4", "legal"),
    (chess.STARTING_FEN, "e4", "e2e4", "legal"),
    # moves inside sentences, markdown, and move numbers
    (chess.STARTING_FEN, "The best move is Nf3.", "g1f3", "legal"),
    (chess.STARTING_FEN, "**Nf3**", "g1f3", "legal"),
    (chess.STARTING_FEN, "1. e4", "e2e4", "legal"),
    (chess.STARTING_FEN, "1.e4", "e2e4", "legal"),
    (chess.STARTING_FEN, "e4 or d4", "e2e4", "legal"),  # the first move counts
    # check and checkmate symbols
    (SCHOLARS_MATE, "Qxf7#", "h5f7", "legal"),
    (SCHOLARS_MATE, "Bxf7+", "c4f7", "legal"),
    (SCHOLARS_MATE, "h5f7", "h5f7", "legal"),
    # promotions
    (PROMOTION, "e7e8q", "e7e8q", "legal"),
    (PROMOTION, "e8=Q", "e7e8q", "legal"),
    (PROMOTION, "e8Q", "e7e8q", "legal"),
    (PROMOTION, "e7e8", None, "illegal"),  # promotion piece missing
    # castling
    (CASTLING, "O-O", "e1g1", "legal"),
    (CASTLING, "0-0-0", "e1c1", "legal"),
    (CASTLING, "e1g1", "e1g1", "legal"),
    # illegal moves
    (chess.STARTING_FEN, "e2e5", None, "illegal"),
    (chess.STARTING_FEN, "Nf6", None, "illegal"),
    (TWO_KNIGHTS, "Nd2", None, "illegal"),  # ambiguous: which knight?
    (chess.STARTING_FEN, "f7f7", None, "illegal"),  # looks like UCI, but the same square twice
    (TWO_KNIGHTS, "Nbd2", "b1d2", "legal"),
    # no move at all
    (chess.STARTING_FEN, "I think White should attack.", None, "unparseable"),
    (chess.STARTING_FEN, "", None, "unparseable"),
])
def test_parse_move(fen, text, expected_uci, reason):
    move, got_reason = parse_move(text, chess.Board(fen))
    assert got_reason == reason
    assert (move.uci() if move else None) == expected_uci


def test_prompt_says_whose_turn_and_notation():
    message = user_message(SCHOLARS_MATE, [], UCI_CFG)
    assert SCHOLARS_MATE in message
    assert "White to move." in message
    assert "UCI notation" in message
    assert "Recent moves" not in message and "Legal moves" not in message


def test_prompt_options():
    cfg = {"notation": "san", "history_moves": 2, "show_legal_moves": True}
    row = {"fen": chess.STARTING_FEN, "history_uci": "", "history_san": ""}
    assert recent_moves(row, cfg) == []

    row = {"fen": SCHOLARS_MATE, "history_uci": "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6",
           "history_san": "e4 e5 Bc4 Nc6 Qh5 Nf6"}
    message = build_messages(row, cfg)[0]["content"]
    assert "Recent moves: Qh5 Nf6" in message
    assert "Qxf7#" in message  # listed among the legal moves, in SAN
    assert "standard algebraic notation" in message
