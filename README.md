# Chess LLM post-training (CS 289A)

**Start here:** [docs/project_notes.md](docs/project_notes.md) explains what we did, notebook by notebook, and what the results show. This README covers status, setup, and how to run things.

## Project description

This project studies how post-training methods improve a small language model's chess play. We fine-tune Gemma 4 E2B with supervised fine-tuning (SFT), apply direct preference optimization (DPO) and reinforcement learning (RL) on top of it, first separately and then in combination, and evaluate every model on held-out Lichess puzzles. Language models handle many reasoning tasks well but still play chess poorly, often proposing illegal moves, and prior work has mostly studied these training methods one at a time rather than comparing them under the same conditions. Our aim is not only a stronger chess-playing model, but a clear picture of what each method contributes and whether combining them helps beyond any single method, with every method given comparable data and compute.

## How the data is used

- **Training: Lichess games.** About 142,500 positions from games between players rated 2000+, each paired with the move that was played. Games give a large and varied set of realistic positions.
- **Testing: Lichess puzzles.** 1,000 puzzles (plus 200 for quick checks), each with one correct move, so answers can be graded objectively. Puzzles are never used for training, and no training position appears among them.
- **Why?** games supply many varied, realistic positions to learn from, while puzzles each have one verified answer, which makes them a fair exam. Keeping the two separate means test scores measure how well the model handles positions it has never seen (not whether it remembers answers from training).
- **Planned:** training on puzzles (different ones from the test set), or on a mix of puzzles and games, instead of games only.

## Status

**Done**
- Data pipeline, evaluation, and SFT training code (notebooks 00–03)
- DPO preference pairs and RL reward (notebook 04)
- Base and SFT results on the GPU, then DPO and RL training code and first DPO and RL runs (notebooks 02, 03, 05, 06)
- Analysis of the results for all four models (notebook 07)
- Report outline, abstract, and related work (Overleaf, thanks Hannah!)

**Results so far** (1,000 test puzzles; details in `results/progress_report_table.md` and notebook 07)

| Model | Accuracy | Legal answers | Training compute |
|---|---|---|---|
| base | 2.8% | 41% | – |
| SFT | 7.8% | 72% | 6.6 GPU-hours* (1.9M tokens) |
| DPO (from SFT) | 5.9% | 63% | 0.33 GPU-hours (0.87M tokens) |
| RL / GRPO (from SFT) | 3.3% | 82% | 0.54 GPU-hours (1.5M tokens) |

\*SFT ran before a memory fix and was slowed by it (its trial ran about 5 times faster per step), so compare methods by tokens rather than hours.

**Still to do for the progress report (Nov 6)**
- Write the report: progress, contributions, results (Hannah)
- Agree on the updated timeline (everyone)

**Later:** ablations, combined methods (e.g. SFT → DPO → RL), games against Stockfish, ChessQA, a chessboard visualizer.

## Contributions

- **Adrian:** repository and environment, data pipeline, prompts and move parsing, evaluation, SFT training code, and DPO/RL inputs (details in `CONTRIBUTIONS.md`).
- **Hannah:** Overleaf document, report outline, abstract, and related work.
- **Nathan:** GPU runs (base, SFT, DPO, RL), DPO and RL training code, the memory fix that lets training fit in 12 GB, and the RL analysis (details in `CONTRIBUTIONS.md`).

## For the report (data and methods section)

Key facts for the Data and Methods sections. More detail is in `docs/project_notes.md`, exact numbers are in `results/data_stats.json`, and all settings are in `configs/`.

**Data**
- **Training games:** Lichess game database, September 2025 (one file of 1.4M games), filtered to 50,000 games:
  - both players rated 2000+
  - no bot accounts
  - at least 3 minutes per player
  - a normal ending
  - at least 20 moves
- **Training examples:** 3 positions per game, skipping the first 5 moves. Each example is a board position and the move played. 142,500 for training and 7,500 for validation, split by game.
- **Test puzzles:** from the Lichess puzzle database (6.2M puzzles). 1,000 test and 200 dev puzzles, with equal numbers in four rating bands (<1200, 1200–1800, 1800–2400, 2400+). Each has one correct move.
- **No leakage:** no training position appears among the test or dev puzzles.

**Method**
- **Model:** Gemma 4 E2B, instruction-tuned. Only the language part is used: 4.63B parameters, about 2.3B "effective".
- **Prompt:** the board as a FEN, whose turn it is, and "reply with the best move". It is the same for every model, in training and testing.
- **SFT:** LoRA (rank 16) on 20,000 examples, one epoch. The model learns to produce the move that was played.
- **DPO:** 4,788 pairs. The preferred move is Stockfish's best; the rejected move is a legal move losing 200+ centipawns (our metric that grades how worse the model's move is than Stockfish's best). Training continues the SFT adapter for one epoch over 95% of the pairs (5% held out), with β = 0.1 and the SFT model as the reference.
- **RL (GRPO):** the model's own answers are scored by Stockfish. Illegal answers get −1; legal moves score higher the closer they are to Stockfish's best. Training continues the SFT adapter on 2,000 training positions, with 8 sampled answers per position and no KL penalty.
- **Evaluation:**
  - puzzle accuracy (primary; overall and per rating band)
  - legal-move rate (illegal and unparseable answers counted separately)
  - average centipawn loss (ACPL) against Stockfish
- **Fair comparison:** every model gets the same prompts, puzzles, and grading. Decoding is deterministic, seeds are fixed, data and model versions are pinned, and compute is logged per run.

---

## Running on a GPU

Everything below runs on an NVIDIA GPU (12 GB is enough), in this order:

1. **Set up** (once, about 20 min). See [Setup](#setup).
2. **Base and SFT results.** Run notebooks `00` → `03`, then push the results. See [Run the notebooks](#run-the-notebooks) and [Push the results](#push-the-results).
3. **DPO and RL inputs.** Run notebook `04` (about 5 min).
4. **DPO.** Run notebook `05`: it trains from the SFT model with the pairs from `04`, and grades the result.
5. **RL.** Run notebook `06`: a GRPO run from the SFT model with the reward from `04`, graded and compared with the other three models.
6. **Analysis.** Run notebook `07`, then push.

### Setup

You need an up-to-date NVIDIA driver (supporting CUDA 13.2), Python 3.12 or 3.13, Git, and about 25 GB of free disk space. In a terminal:

```bash
git clone https://github.com/adrianapsay/cs289-final-project.git
cd cs289-final-project
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

Then install Stockfish, the chess engine we use to score moves. Download it from [stockfishchess.org/download](https://stockfishchess.org/download/), unzip it, tell the code where it is, and open a new terminal:

```bash
setx STOCKFISH_PATH "C:\path\to\stockfish.exe"
```

### Run the notebooks

Open each notebook in VS Code, choose the `.venv` kernel, and click **Run All**. Wait for each one to finish before starting the next.

| Notebook | What it does | Rough time |
|---|---|---|
| `00_setup_check` | Checks the setup. It should report your GPU (CUDA). | 1 min |
| `01_data_pipeline` | Builds the training data (downloads 1.9 GB). | 5 min |
| `02_eval_base_model` | Grades the untrained model on the puzzles (downloads the 10 GB model the first time). | up to 30 min |
| `03_sft_first_run` | Trains the SFT model, grades it, and makes the comparison table and figures. | up to 1 hour |
| `04_dpo_rl_inputs` | Builds the DPO pairs and shows the RL reward (no GPU needed). | 5 min |
| `05_dpo_first_run` | Trains the SFT model further with DPO, and grades it. | 25 min |
| `06_grpo_first_run` | Trains the SFT model further with RL (GRPO), grades it, and makes the four-model comparison table and figures. | 40 min |
| `07_understanding_results` | Explains why the models score as they do, from the saved results (no GPU needed). | 3 min |

Keep the laptop plugged in and stop it from sleeping while notebooks run.

### Push the results

**Why:** the progress report needs these results (the comparison table, the figures, and the compute used).

```bash
git add results notebooks
git commit -m "Base model and SFT results"
git push
```

Keep `checkpoints/sft_v1/`: it is the SFT model, and DPO and RL start from it. Back it up (zipped) to the team drive, since it is not on GitHub. The same goes for `checkpoints/dpo_v1/` and `checkpoints/grpo_v1/`.

### DPO and RL training

- Both start from the SFT model (`checkpoints/sft_v1/`) and keep training its LoRA adapter. Settings are in the `training` sections of `configs/dpo.yaml` and `configs/rl.yaml`; the code is in `src/chesslm/train_dpo.py` and `src/chesslm/train_rl.py`.
- For DPO, a frozen copy of the SFT adapter is the reference model, so `beta` limits how far DPO moves from the SFT model (not from the base model).
- Each model is graded like the others, with the run names `dpo_v1` and `grpo_v1`. Their scores appear in `results/summary.csv` next to base and SFT.

### If something goes wrong

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other apps that use the GPU. Then lower `batch_size` in `configs/eval.yaml` (to 4) or `configs/sft.yaml` (to 2), and run the notebook again. |
| `00_setup_check` says "CPU only" | PyTorch was installed without GPU support. Run the `pip install torch ...` line again, then restart the notebook. |
| `Stockfish not found` | Check `STOCKFISH_PATH`, then restart VS Code. |
| The model download stops partway (no network traffic, no progress) | Stop the notebook, delete the `.incomplete` file in `%USERPROFILE%\.cache\huggingface\hub\models--google--gemma-4-E2B-it\blobs\`, run `setx HF_HUB_DISABLE_XET 1`, open a new terminal, and run the notebook again. |
| `DLL load failed ... An Application Control policy has blocked this file` | Windows Smart App Control blocked a library file it does not trust (it happened with pandas 3.1.0). Install pandas 2.3.3 instead (`pip install pandas==2.3.3`), which it accepts; the project works with either. |
| Anything else | Send Adrian the error message. |

---

## Setup on a Mac (data and code work)

A Mac is fine for the data pipeline and the tests, but too slow to run the model.

```bash
brew install stockfish
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run `notebooks/00_setup_check.ipynb` and `notebooks/01_data_pipeline.ipynb`. Run the tests with `pytest`.

---

## Repo layout

| Folder | What's in it |
|---|---|
| `notebooks/` | The work, step by step. Start here. |
| `configs/` | All settings (data, prompts, evaluation, training). |
| `src/chesslm/` | Shared code used by the notebooks. |
| `tests/` | Automatic checks for the shared code. |
| `results/` | Scores for each run (`summary.csv` has one row per run), figures, and the comparison table. |
| `docs/project_notes.md` | Plain-language walkthrough: each notebook, the training methods, and the results. |
| `data/eval/` | The fixed test and dev puzzles (on GitHub). |
| `data/raw/`, `data/processed/` | Downloads and training data. Not on GitHub (too large); rebuilt by `01_data_pipeline.ipynb`. |
| `checkpoints/` | Trained adapters. Not on GitHub (too large); share them directly. |
