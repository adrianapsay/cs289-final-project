# Project notes

What we are building, why, and what the results show, readable without the code. Technical terms are defined in the glossary.

---

## Overview

**Research question.** Language models can describe chess but often play it poorly, frequently proposing illegal moves. We study which post-training methods improve a small open model (Google's Gemma 4 E2B), and why. We compare supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL), first separately and then in combination, with similar data and compute budgets.

**Pipeline.**
1. Collect Lichess games between players rated 2000 or higher.
2. Turn each game into a few examples: a board position and the move that was played.
3. Train the model on these examples (SFT). DPO and RL are applied on top of the SFT model later.
4. Test every model on the same 1,000 Lichess puzzles, none of which appear in the training data.
5. Compare each model with the untrained base model.

**Methods.**
- **SFT** trains the model to reproduce reference answers: given a position, it learns to produce the move a strong player made.
- **DPO** trains on pairs of answers to the same position, one preferred (Stockfish's best move) and one rejected (a clearly worse move), and makes the preferred one more likely.
- **RL** (not yet covered in the course) lets the model generate its own answers, scores each one with a *reward function* (here based on Stockfish), and makes high-scoring answers more likely.

**Metrics.**
- **Puzzle accuracy** (primary): the percentage of puzzles solved.
- **Legal-move rate:** the percentage of answers that are legal moves.
- **ACPL:** how much worse a legal move is than Stockfish's best move, in centipawns. Lower is better.

---

## Glossary

- **Base model:** the model as released, before our training; our baseline.
- **Parameters:** the numerical values inside a model that training adjusts.
- **LoRA:** fine-tuning that freezes the original parameters and trains a small set of added ones (the **adapter**). **QLoRA** does the same with the base model compressed to 4 bits, to save GPU memory.
- **Token:** the unit of text a model reads and writes. The model sees `e2e4` as four tokens.
- **Chat template:** the fixed format that marks who is speaking in each message.
- **Greedy decoding:** always choosing the most likely next token, so results are reproducible.
- **Seed:** a fixed starting value for random choices, so results can be repeated exactly.
- **Stockfish:** a free chess engine, far stronger than any human; our reference for move quality.
- **Centipawn:** one hundredth of a pawn. +100 means White is about one pawn ahead.
- **FEN:** a one-line text encoding of a full board position.
- **UCI / SAN:** two ways to write a move: `e2e4` (from-square, to-square) and `e4` or `Nf3` (standard chess notation).
- **Test set / dev set:** 1,000 puzzles for final evaluation (never used for training or tuning) and 200 separate puzzles for checks during development.
- **Data leakage:** overlap between training and test data, which inflates test scores.
- **Ablation:** an experiment that changes one factor and keeps everything else fixed.

---

## Repository layout

| Location | Contents |
|---|---|
| `notebooks/` | The work, step by step, with results shown inline. Start here. |
| `configs/` | All experiment settings. |
| `src/chesslm/` | Shared code used by the notebooks. |
| `results/` | Scores for each run, and figures. |
| `data/eval/` | The fixed test and dev puzzles. |
| `CONTRIBUTIONS.md` | A dated record of who did what. |

---

## Planned

Each factor below can be changed in a config file without touching the code. An ablation changes one factor, retrains, and grades the result with the same evaluation.

| Factor | Kind | Setting (file) | Default | Alternatives |
|---|---|---|---|---|
| Move notation | Input features | `notation` (`prompt.yaml`) | UCI | SAN |
| Move history in the prompt | Input features | `history_moves` (`prompt.yaml`) | 0 (board only) | 1 (last move) |
| Legal moves listed in the prompt | Input features | `show_legal_moves` (`prompt.yaml`) | off | on (raise `max_seq_length` to 384) |
| Training data source | Training data | to be added | games | puzzles (not the test or dev puzzles); a mix of both |
| Player rating threshold | Training data | `min_elo` (`data.yaml`) | 2000 | e.g. 1600, 2200 |
| Games against bot accounts | Training data | `exclude_bots` (`data.yaml`) | excluded | included |
| Number of training examples | Training data | `train_examples` (`sft.yaml`) | 20,000 | e.g. 5,000, 50,000 |
| LoRA rank | Hyperparameters | `rank` (`sft.yaml`) | 16 | e.g. 8, 32 |
| Learning rate | Hyperparameters | `learning_rate` (`sft.yaml`) | 2e-4 | e.g. 1e-4, 5e-4 |
| Model size | Model | `id` (`sft.yaml`) | Gemma 4 E2B | Gemma 4 E4B |

DPO and RL will add their own factors, for example the DPO rejection threshold and the RL reward (sparse or dense, following Hwang et al., cited in our draft).

---

## Progress log

### Step 1: Environment setup (Nov 1)

- **What we built:** the repository, a Python environment with pinned package versions, Stockfish, and a setup notebook ([00_setup_check.ipynb](../notebooks/00_setup_check.ipynb)).
- **Why it matters:** pinned versions make the code behave identically on every machine, so differences in results come from the experiments, not the setup.
- **How we know it works:** the setup notebook runs without errors. Stockfish rates the starting position as nearly equal (+32 centipawns) and a position without Black's queen as about a queen up (+886).
- **Choices:** Gemma 4 E2B, because it fits on a 12 GB GPU with QLoRA and needs no access request.

### Step 2: Data pipeline (Nov 1)

- **What we built:** [01_data_pipeline.ipynb](../notebooks/01_data_pipeline.ipynb) builds a 1,000-puzzle test set and a 200-puzzle dev set (equal numbers in four rating bands, committed in `data/eval/`), and 150,000 training positions from 50,000 games.
- **Why it matters:** every model is graded on the same fixed puzzles, and the training data determines what SFT can learn.
- **How we know it works:** unit tests cover the puzzle format, filters, sampling, and splits. No training position appears in the test or dev sets.
- **Key numbers** ([results/data_stats.json](../results/data_stats.json)):

  | Filter step | Games remaining |
  |---|---|
  | Games read (September 2025) | 1,404,031 |
  | Both players rated 2000+ | 245,522 |
  | No bot accounts | 242,573 |
  | Not bullet (≥ 3 minutes) | 92,501 |
  | Ended normally | 70,904 |
  | At least 20 moves (first 50,000 kept) | 50,000 |

  Training: 142,500 positions (47,500 games). Validation: 7,500 positions (2,500 games).
- **Choices:**
  - Bot accounts are excluded: a cluster of players near 3000 turned out to be 98% bots, and we train on human play.
  - Three positions per game, skipping the first five moves, which are mostly memorized openings.
  - Dataset versions are pinned, so the data can be rebuilt exactly.
- **Limitations:**
  - Puzzles are sharp tactical positions, while most game positions are quiet, so SFT on games may transfer only partly to puzzles.
  - Equal rating bands give hard puzzles extra weight, so per-band accuracy is the more informative number.

### Step 3: Prompts and reading the answer (Nov 1)

- **What we built:** `src/chesslm/prompts.py` turns a position into the model's prompt (the board as a FEN, whose turn it is, and an instruction to reply with one move). It also reads the move back from the reply as *legal*, *illegal*, or *unparseable*.
- **Why it matters:** every model is asked the same question and graded by the same reader. Separating illegal from unparseable answers distinguishes misunderstanding the rules from ignoring the instruction, as in the LLM CHESS benchmark cited in our draft.
- **How we know it works:** 27 tests cover sentences, markdown, move numbers, check symbols, promotions, castling, and ambiguous moves.
- **Choices:**
  - One prompt configuration (`configs/prompt.yaml`) is shared by training and evaluation.
  - UCI is the default notation.
  - The reader accepts both notations, so correct moves are never penalized for format.
  - Move history is limited to the last move, because test puzzles include only the opponent's last move.
- **Limitations:** the reader grades the first move it finds in a reply.

### Step 4: Evaluation harness (Nov 1)

- **What we built:** `src/chesslm/evaluate.py` and [02_eval_base_model.ipynb](../notebooks/02_eval_base_model.ipynb). They grade every puzzle on accuracy (any checkmating move counts in mate-in-1 puzzles), legality, and centipawn loss (Stockfish depth 12, capped at 1000). Each run is saved to `results/`.
- **Why it matters:** all models are graded by this same code, on the same puzzles, with the same settings.
- **How we know it works:** unit tests check the grading rules and Stockfish scoring, and the pipeline ran end to end on dev puzzles.
- **Key numbers:** base-model scores will be added after the GPU run. On a few dev puzzles, the base model answered in SAN even when asked for UCI.
- **Choices:**
  - Greedy decoding, since Gemma samples randomly by default.
  - Only the language model is loaded (4.63 billion parameters, about 9.3 GB), skipping the unused image and audio encoders, so it fits on a 12 GB GPU.
  - The model runs on an RTX 5070 Ti; a 16 GB Mac needed about 54 seconds per puzzle.
- **Limitations:** ACPL covers legal moves only, so it can rest on few moves when legality is low.

### Step 5: First SFT run (Nov 1; training runs on the GPU)

- **What we built:** `src/chesslm/train_sft.py` and [03_sft_first_run.ipynb](../notebooks/03_sft_first_run.ipynb). The notebook runs a short trial (500 examples), then the real run (20,000 examples). It then grades the result and makes the comparison table and figures.
- **Why it matters:** SFT is the first post-training method and the starting point for DPO and RL.
- **How we know it works:** on a small model of the same architecture, the loss decreased and the adapter saved correctly. The loss covers only the answer, not the prompt.
- **Key numbers:** loss curves, test scores, and compute used will be added after the GPU run.
- **Choices:**
  - Settings from the project plan: LoRA rank 16 on the attention and MLP layers, learning rate 2e-4, one epoch, effective batch size 32, seed 42.
  - QLoRA (4-bit), to fit in 12 GB of GPU memory.
- **Limitations:** one run with one seed, so differences of a few percentage points may be noise.

### Step 8: Inputs for DPO and RL (Nov 1)

- **What we built:** `src/chesslm/preferences.py`, `src/chesslm/rewards.py`, and [04_dpo_rl_inputs.ipynb](../notebooks/04_dpo_rl_inputs.ipynb).
  - DPO preference pairs: Stockfish's best move is *chosen*, and a legal move losing at least 200 centipawns is *rejected*.
  - RL reward: −1 for an illegal or unparseable answer. For a legal move, the dense mode scores closeness to Stockfish's best, and the sparse mode gives 1 for the best move only.
- **Why it matters:** these are what DPO and RL train on. Both use the same prompts as SFT and evaluation, and both can be built without a GPU or the SFT model.
- **How we know it works:** unit tests check both rewards and the pair format. In the notebook, the mating move scores 1.1, a legal but losing move scores 0.1, and illegal answers score −1.
- **Key numbers:** 4,788 pairs from 5,000 training positions; 212 positions had no move losing 200+ centipawns. Rejected moves lose 620 centipawns at the median. Building the pairs took about 5 minutes on a laptop CPU.
- **Choices:**
  - Rejection threshold of 200 centipawns (about two pawns), from the project plan.
  - A shallower Stockfish search for building pairs (depth 10) and for the reward (depth 8) than for evaluation (12), for speed.
  - Dense reward by default, with sparse available as an ablation (following Hwang et al., cited in our draft).
- **Limitations:**
  - Many rejected moves are very bad (median 620 centipawns), so the pairs may be easy to tell apart. A narrower threshold range is an ablation candidate.
  - The reward is computed at a shallow depth, so its scores are approximate.
