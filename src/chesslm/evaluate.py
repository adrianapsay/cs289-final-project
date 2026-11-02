"""Evaluation harness: ask a model every puzzle, read its answers, and grade them.

Every model (base, SFT, DPO, RL) is graded by this code on the same puzzles with
greedy decoding, so differences in scores come from training alone.
"""

import json
import time

import chess
import pandas as pd
import torch

from chesslm.config import REPO_ROOT
from chesslm.model import load_model, load_tokenizer
from chesslm.prompts import build_prompt, parse_move

RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------- model

def load_eval_model(model_cfg, adapter_path=None):
    """Load the tokenizer and model for evaluation, optionally with a trained LoRA adapter on top."""
    tokenizer = load_tokenizer(model_cfg)
    # Batched generation needs padding on the left, so every answer starts right after its prompt.
    tokenizer.padding_side = "left"

    model = load_model(model_cfg)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    # Greedy decoding for every model: switch off Gemma's default random sampling.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_replies(model, tokenizer, prompts, gen_cfg):
    """Generate a reply to each prompt, in batches. Returns (replies, number of tokens in each reply)."""
    replies, n_tokens = [], []
    for start in range(0, len(prompts), gen_cfg["batch_size"]):
        batch = tokenizer(
            prompts[start : start + gen_cfg["batch_size"]],
            return_tensors="pt", padding=True, add_special_tokens=False,  # the prompt already has <bos>
        ).to(model.device)
        output = model.generate(**batch, max_new_tokens=gen_cfg["max_new_tokens"])
        for new_tokens in output[:, batch["input_ids"].shape[1] :]:
            new_tokens = new_tokens[new_tokens != tokenizer.pad_token_id]
            replies.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
            n_tokens.append(len(new_tokens))
    return replies, n_tokens


# ---------------------------------------------------------------- grading

def is_correct(board, move, puzzle):
    """The move matches the puzzle's solution. In mate-in-1 puzzles, any checkmating move counts."""
    if move is None:
        return False
    if move.uci() == puzzle["answer_uci"]:
        return True
    if puzzle["mate_in_1"]:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        return is_mate
    return False


def grade(puzzle, reply, scorer):
    """Grade one reply: what move it contains, whether it's legal and correct, and its centipawn loss."""
    board = chess.Board(puzzle["fen"])
    move, outcome = parse_move(reply, board)
    return {
        "puzzle_id": puzzle["puzzle_id"],
        "band": puzzle["band"],
        "reply": reply,
        "move": move.uci() if move else None,
        "outcome": outcome,  # legal / illegal / unparseable
        "correct": is_correct(board, move, puzzle),
        "cp_loss": scorer.centipawn_loss(puzzle["fen"], move) if move else None,
    }


def summarize(graded):
    """Summary metrics for a table of graded replies."""
    legal = graded[graded.outcome == "legal"]
    return {
        "n_puzzles": len(graded),
        "puzzle_accuracy": graded.correct.mean(),
        "accuracy_by_band": graded.groupby("band").correct.mean().to_dict(),
        "legal_rate": (graded.outcome == "legal").mean(),
        "illegal_rate": (graded.outcome == "illegal").mean(),
        "unparseable_rate": (graded.outcome == "unparseable").mean(),
        # ACPL is measured on legal moves only; it is undefined if there are none.
        "acpl": legal.cp_loss.mean() if len(legal) else None,
        "seconds_per_position": graded.seconds.mean(),
        "tokens_per_reply": graded.n_tokens.mean(),
    }


def evaluate(model, tokenizer, puzzles, prompt_cfg, gen_cfg, scorer):
    """Ask the model every puzzle and grade the replies. Returns one row per puzzle."""
    rows = puzzles.to_dict("records")
    prompts = [build_prompt(tokenizer, row, prompt_cfg) for row in rows]

    start = time.time()
    replies, n_tokens = generate_replies(model, tokenizer, prompts, gen_cfg)
    seconds_per_position = (time.time() - start) / len(rows)

    graded = pd.DataFrame([grade(row, reply, scorer) for row, reply in zip(rows, replies)])
    graded["n_tokens"] = n_tokens
    graded["seconds"] = seconds_per_position
    return graded


# ---------------------------------------------------------------- saving

def save_run(run_name, graded, summary, config):
    """Save results/<run_name>.json (config, every reply, summary) and update results/summary.csv."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{run_name}.json", "w") as f:
        json.dump({"run": run_name, "config": config, "summary": summary,
                   "rows": graded.to_dict("records")}, f, indent=2, default=str)

    row = {"run": run_name, **{k: v for k, v in summary.items() if k != "accuracy_by_band"},
           **{f"accuracy_{band}": acc for band, acc in summary["accuracy_by_band"].items()}}
    summary_path = RESULTS_DIR / "summary.csv"
    table = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if len(table):
        table = table[table.run != run_name]  # re-running a run replaces its row
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.round(4).to_csv(summary_path, index=False)
    return table
