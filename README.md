# Chess LLM post-training (CS 289A)

## Project description

_TODO: written by the team._

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
