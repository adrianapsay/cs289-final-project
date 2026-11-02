# Chess LLM post-training (CS 289A)

## Project description

This project studies how post-training methods improve a small language model's chess play. We fine-tune Gemma 4 E2B with supervised fine-tuning (SFT), apply direct preference optimization (DPO) and reinforcement learning (RL) on top of it, first separately and then in combination, and evaluate every model on held-out Lichess puzzles. Language models handle many reasoning tasks well but still play chess poorly, often proposing illegal moves, and prior work has mostly studied these training methods one at a time rather than comparing them under the same conditions. Our aim is not only a stronger chess-playing model, but a clear picture of what each method contributes and whether combining them helps beyond any single method, with every method given comparable data and compute.

## How the data is used

- **Training: Lichess games.** About 142,500 positions from games between players rated 2000+, each paired with the move that was played. Games give a large and varied set of realistic positions.
- **Testing: Lichess puzzles.** 1,000 puzzles (plus 200 for quick checks), each with one correct move, so answers can be graded objectively. Puzzles are never used for training, and no training position appears among them.
- **Why?** games supply many varied, realistic positions to learn from, while puzzles each have one verified answer, which makes them a fair exam. Keeping the two separate means test scores measure chess skill (not memorization).
- **Planned comparison:** training on puzzles (different ones from the test set), or on a mix of puzzles and games, instead of games only.

---

## Next steps: run the model on the GPU

The code is ready. The model is too large for a laptop without an NVIDIA GPU, so these steps run on the RTX 5070 Ti. You will set things up once, run four notebooks in order, and push the results back. Most of the time is waiting.

**Before you start, make sure you have:**
- an up-to-date NVIDIA driver (it must support CUDA 13.2; update through the NVIDIA app if unsure)
- Python 3.12 or 3.13, and Git
- about 25 GB of free disk space
- write access to this GitHub repo (ask Adrian)

### 1. Get the code and install

In a terminal:

```bash
git clone https://github.com/adrianapsay/cs289-final-project.git
cd cs289-final-project
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

(On Linux, activate with `source .venv/bin/activate` instead.)

### 2. Install Stockfish

Stockfish is the chess engine we use to score moves. Download it from [stockfishchess.org/download](https://stockfishchess.org/download/) and unzip it. Then tell the code where it is, and open a new terminal afterwards:

```bash
setx STOCKFISH_PATH "C:\path\to\stockfish.exe"
```

(On Linux: `sudo apt install stockfish`. No path needed.)

### 3. Run the notebooks, in order

Open each notebook in VS Code, choose the `.venv` kernel, and click **Run All**. Wait for each one to finish before starting the next.

| Notebook | What it does | Rough time |
|---|---|---|
| `00_setup_check` | Checks the setup. It should report your GPU (CUDA). | 1 min |
| `01_data_pipeline` | Builds the training data (downloads 1.9 GB). | 5 min |
| `02_eval_base_model` | Grades the untrained model on the puzzles (downloads the 10 GB model the first time). | 30 min |
| `03_sft_first_run` | Trains our first model, grades it, and makes the comparison table and figures. | 1–2 hours |

The times are estimates. Each notebook shows its progress while it runs.

### 4. Send the results back

```bash
git add results notebooks
git commit -m "Base model and SFT results"
git push
```

Then zip the folder `checkpoints/sft_v1/` and share it with the team. It is the trained model, and DPO and RL start from it.

### If something goes wrong

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other apps that use the GPU. Then lower `batch_size` in `configs/eval.yaml` (to 4) or `configs/sft.yaml` (to 2), and run the notebook again. |
| `00_setup_check` says "CPU only" | PyTorch was installed without GPU support. Run the `pip install torch ...` line from step 1 again, then restart the notebook. |
| `Stockfish not found` | Check step 2, then restart VS Code so it picks up `STOCKFISH_PATH`. |
| Anything else | Send Adrian the error message. |

---

## DPO and RL

The inputs for DPO and RL are ready, and building them needs no GPU:
1. Run `01_data_pipeline` (if not done yet), then `04_dpo_rl_inputs` (about 5 min). This builds the DPO preference pairs and shows the RL reward.
2. The end of `04_dpo_rl_inputs` shows the few lines that plug the pairs and the reward into TRL's `DPOTrainer` and `GRPOTrainer`.
3. Both start from the SFT model (`checkpoints/sft_v1/`). Grade the results with `chesslm.evaluate`, as in notebooks 02 and 03.

Settings are in `configs/dpo.yaml` and `configs/rl.yaml`.

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
| `docs/project_notes.md` | Plain-language explanation of the project. |
| `data/eval/` | The fixed test and dev puzzles (on GitHub). |
| `data/raw/`, `data/processed/` | Downloads and training data. Not on GitHub (too large); rebuilt by `01_data_pipeline.ipynb`. |
| `checkpoints/` | Trained adapters. Not on GitHub (too large); share them directly. |
