import chess

from chesslm.train_sft import to_sft_example

ROW = {"fen": chess.STARTING_FEN, "history_uci": "", "history_san": "", "move_uci": "g1f3", "move_san": "Nf3"}


def test_sft_example_uses_the_evaluation_prompt_and_the_move_played():
    example = to_sft_example(ROW, {"notation": "uci", "history_moves": 0, "show_legal_moves": False})
    assert example["prompt"][0]["role"] == "user"
    assert chess.STARTING_FEN in example["prompt"][0]["content"]
    assert example["completion"] == [{"role": "assistant", "content": "g1f3"}]


def test_sft_example_answer_follows_the_notation_setting():
    example = to_sft_example(ROW, {"notation": "san", "history_moves": 0, "show_legal_moves": False})
    assert example["completion"][0]["content"] == "Nf3"
