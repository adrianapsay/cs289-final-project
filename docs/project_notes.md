# Project notes

These notes explain what we are building, why, and what the results show. They are meant to be readable without going through the code. Technical terms are defined in the glossary.

---

## Overview

### Research question

Large language models can describe chess, but they often play it poorly, frequently proposing moves that are not legal in the position. We study which post-training methods improve a small open model's chess play, and why. Our model is Google's Gemma 4 E2B. We compare three methods: supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL). We apply each separately, and later in combination, with similar data and compute budgets so the comparison is fair.

### Pipeline

1. **Training data.** We collect games from Lichess, a free online chess platform, keeping only games in which both players are rated 2000 or higher.
2. **Training examples.** From each game we sample a few positions. Each example pairs a position with the move the player actually made.
3. **Training.** We fine-tune the model on these examples (SFT). DPO and RL are applied on top of the SFT model in later stages of the project.
4. **Evaluation.** We test every model on the same 1,000 Lichess puzzles. Each puzzle has a single correct move, and none of them appear in the training data.
5. **Comparison.** We report every model's scores alongside those of the untrained base model.

The training data and the test puzzles must not overlap. If a test position appeared in training, a high score could reflect memorization rather than chess ability. We check for and remove any such overlap (see *data leakage*).

### Post-training methods

- **Supervised fine-tuning (SFT)** trains the model to reproduce reference answers. Given a board position, the model learns to assign high probability to the move a strong player made. This is the first method we apply, and the other two build on its result.
- **Direct preference optimization (DPO)** trains on pairs of answers to the same position: one preferred and one rejected. In our case, the preferred move is Stockfish's best move and the rejected move is a clearly worse legal move. The model learns to make the preferred answer more likely than the rejected one. DPO learns from a fixed dataset of such pairs.
- **Reinforcement learning (RL)** has not been covered in the course yet; we will apply it later in the project. In brief: the model generates its own answers, a *reward function* scores each one, and training adjusts the model so that high-scoring answers become more likely. Unlike SFT, RL needs no example answers, only a way to score them. Here the score will come from Stockfish, based on whether the move is legal and how close it is to the engine's best move.

### Evaluation metrics

- **Puzzle accuracy** (primary): the percentage of puzzles in which the model's move matches the solution.
- **Legal-move rate:** the percentage of the model's answers that are legal moves in the given position.
- **Average centipawn loss (ACPL):** for legal moves, how much worse the model's move is than Stockfish's best move, measured in centipawns. Lower is better.

---

## Glossary

**Models and training**
- **Language model (LLM):** a model trained on large amounts of text to predict the next token. It produces a chess move by generating text, for example `e2e4`.
- **Parameters:** the internal numerical values that training adjusts.
- **Gemma 4 E2B:** the model we use, released by Google. It has about 2.3 billion effective parameters (about 5.1 billion including its embedding tables).
- **Instruction-tuned ("-it"):** a version of the model that Google has already trained to follow instructions in a chat format. We start from `gemma-4-E2B-it`.
- **Base model:** the model as released, before any of our training. It serves as our baseline.
- **Post-training / fine-tuning:** further training of an already trained model on task-specific data.
- **LoRA (low-rank adaptation):** a fine-tuning method that keeps the original parameters frozen and trains a small number of additional parameters added to selected layers. It greatly reduces memory and compute.
- **Adapter:** the set of additional parameters trained by LoRA. It is saved separately from the base model.
- **QLoRA:** LoRA applied to a base model stored in compressed 4-bit form, which further reduces GPU memory use.
- **Token:** the unit of text a model reads and generates, often a fragment of a word. For example, the model sees `e2e4` as four tokens: `e`, `2`, `e`, `4`.
- **Tokenizer:** the component that converts text into tokens.
- **Chat template:** the fixed format that marks who is speaking in each message. An instruction-tuned model expects its input in this format.
- **Greedy decoding:** always choosing the most likely next token. This makes outputs deterministic, so every model is evaluated under the same conditions.
- **Seed:** a fixed initial value for random processes (sampling, shuffling), so that results can be reproduced exactly.
- **Reward function:** in RL, the function that assigns a score to each output the model generates.
- **GRPO (group relative policy optimization):** the RL method we plan to use. For each position it samples several answers and scores each one relative to the group's average.

**Chess**
- **Stockfish:** a free, open-source chess engine, much stronger than any human player. We use it as our reference for how good a move is.
- **Centipawn:** one hundredth of a pawn, the unit Stockfish uses to evaluate positions. A score of +100 means White is ahead by roughly one pawn; negative scores favor Black.
- **Search depth:** how many moves ahead Stockfish searches. We use a fixed depth of 12, so evaluations are comparable.
- **FEN (Forsyth–Edwards Notation):** a single line of text that encodes a full board position: piece placement, side to move, and castling rights. The starting position is `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`.
- **UCI notation:** writes a move as its origin and destination squares, for example `e2e4`.
- **SAN (standard algebraic notation):** the notation used in chess literature, for example `e4`, `Nf3`, `O-O` (castling).
- **Lichess puzzle:** a position taken from a real game in which one move is clearly best, rated by difficulty.

**Experimental design**
- **Test set:** the 1,000 puzzles used for final evaluation. They are never used for training or for tuning settings.
- **Development (dev) set:** a separate set of 200 puzzles, used for quick checks during development.
- **Data leakage:** overlap between training data and test data, which inflates test scores. We prevent it by removing any training position identical to a test or dev position.
- **Ablation:** an experiment that changes a single factor (for example, the rating threshold) while keeping everything else fixed, to measure how much that factor matters.

---

## Repository layout

| Location | Contents |
|---|---|
| `notebooks/` | The work, step by step, with results shown inline. Start here. |
| `configs/` | All experiment settings. Changing an experiment means editing these files. |
| `src/chesslm/` | Shared code used by the notebooks. |
| `results/` | Scores for each run, and figures. |
| `data/eval/` | The fixed test and dev puzzles. Every model is graded on these. |
| `docs/project_notes.md` | This document. |
| `CONTRIBUTIONS.md` | A dated record of who did what. |

---

## Progress log

### Step 1: Environment setup (Nov 1)

**What we built.**
- The repository structure.
- A Python environment with pinned package versions.
- An installation of the Stockfish engine.
- A setup notebook, [00_setup_check.ipynb](../notebooks/00_setup_check.ipynb), that verifies each component.

**Why it matters.** Every later step depends on these tools. Pinning package versions ensures the code behaves identically on each machine we use (a laptop, Google Colab, and a GPU machine). Differences in results can then be attributed to the experiments rather than to the software environment.

**How we know it works.** The setup notebook runs without errors on Adrian's Mac:
- The required libraries load.
- The Mac's GPU is detected. It is adequate for small tests but not for full training.
- Stockfish produces sensible evaluations: the starting position is close to equal, and removing Black's queen shifts the evaluation by roughly the value of a queen.
- The Gemma 4 tokenizer loads, and we can inspect the exact chat format the model receives.

**Key numbers** (from [00_setup_check.ipynb](../notebooks/00_setup_check.ipynb)):
- **Stockfish 19 at depth 12:**
  - starting position: +32 centipawns
  - position with Black's queen removed: +886 centipawns
- **Model:** `google/gemma-4-E2B-it`, with a vocabulary of 262,144 tokens.
- **Token counts:**
  - `e2e4` (UCI): 4 tokens
  - `e4` (SAN): 2 tokens
  - `Nf3` (SAN): 3 tokens
  - the starting position as a FEN: 30 tokens

**Choices we made and why.**
- **Gemma 4 E2B:** it fits on a 12 GB GPU with QLoRA, and it can be downloaded without an access request.
- **Starting SFT settings:** taken from the project plan (LoRA rank 16, learning rate 2e-4, one epoch). They are stored in `configs/sft.yaml`, and each is a candidate for ablation.
- **Move notation:** to be decided in Step 2. UCI and SAN tokenize differently, which suggests notation may affect learning. That makes it a natural ablation.

**Limitations / open questions.**
- The full model weights (~10 GB) have not been downloaded yet; this happens in Step 4.
- The machine for training is still to be decided: the GPU laptop, Colab, or Databricks.

### Step 2: Data pipeline (Nov 1)

**What we built.** A pipeline, [01_data_pipeline.ipynb](../notebooks/01_data_pipeline.ipynb) with its code in `src/chesslm/data.py`, that produces two kinds of data:
- **Evaluation sets.** A test set of 1,000 Lichess puzzles and a dev set of 200, with equal numbers in four rating bands. They are saved in `data/eval/` and committed, so every model, including Nathan's DPO and RL models, is graded on the same puzzles.
- **Training data.** 150,000 positions sampled from 50,000 games between players rated 2000 or higher. Each position is paired with the move that was played. The positions are split by game into training and validation sets.

**Why it matters.** The evaluation sets define how every model in the project is measured, so they are fixed once and never changed. The training data determines what SFT can learn: the model can only imitate the moves it sees.

**How we know it works.**
- Nine unit tests (`tests/test_data.py`) cover the puzzle format, the game filters, position sampling, leakage removal, and the train/validation split.
- The notebook shows a puzzle before and after the opponent's move and confirms that the answer is legal in the solver's position.
- No training position appears in the test or dev sets (overlap after removal: 0).
- The test and dev sets share no puzzles, and the training and validation sets share no games.

**Key numbers** (from [results/data_stats.json](../results/data_stats.json)):

| Filter step | Games remaining |
|---|---|
| Games read (one file, September 2025) | 1,404,031 |
| Both players rated 2000+ | 245,522 |
| No bot accounts | 242,573 |
| Not bullet (≥ 3 minutes per player) | 92,501 |
| Ended normally | 70,904 |
| At least 20 moves (first 50,000 kept) | 50,000 |

- **Positions:** 150,000 sampled (3 per game). Training: 142,500 positions from 47,500 games. Validation: 7,500 positions from 2,500 games. Removed for overlap with the evaluation sets: 0.
- **Test set:** 250 puzzles per band (<1200, 1200–1800, 1800–2400, 2400+). 110 are mate-in-1 puzzles.
- **Dev set:** 50 puzzles per band.
- **Puzzle database:** 6,157,341 puzzles.

**Choices we made and why.**
- **Rating threshold 2000** for both players, so the training moves are strong (ablation candidate).
- **Bot accounts excluded.** The rating histogram showed a cluster of players near 3000; 98% of those were Lichess bot accounts, which are chess engines. Since we train on human play, we exclude games involving bots. This setting can be switched off (ablation candidate).
- **Three positions per game, skipping the first five moves.** Several positions per game give variety without many near-identical examples. Opening moves are largely memorized sequences.
- **Equal puzzles per rating band**, so we can report accuracy at each difficulty level.
- **Both notations stored** (UCI and SAN). UCI is the default for prompts; SAN is available as an ablation (applied in Step 3).
- **Pinned dataset versions.** The puzzle database is updated regularly; pinning the version makes the data reproducible.

**Limitations / open questions.**
- **Puzzles and games differ.** Puzzles are tactical positions with one clearly best move; most game positions are quieter, and the move a 2000-rated player chose is not always the best one. SFT on games may therefore transfer only partly to puzzles. DPO and RL, which use Stockfish's judgment, may help here.
- **Overall test accuracy is not a natural average.** Equal bands give hard puzzles more weight than they have in the full database, so the per-band numbers are the more informative ones.
- **The leakage check catches identical positions only.** Very similar positions can still appear in both sets.
- **All games come from one month** (September 2025).
- **The first SFT run uses a subsample** of the training data (Step 5), as planned.
