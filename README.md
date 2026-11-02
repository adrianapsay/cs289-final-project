# Chess LLM post-training (CS 289A)

## Project description

This project studies how post-training methods improve a small language model's chess play. We fine-tune Gemma 4 E2B with supervised fine-tuning (SFT), apply direct preference optimization (DPO) and reinforcement learning (RL) on top of it, first separately and then in combination, and evaluate every model on held-out Lichess puzzles. Language models handle many reasoning tasks well but still play chess poorly, often proposing illegal moves, and prior work has mostly studied these training methods one at a time rather than comparing them under the same conditions. Our aim is not only a stronger chess-playing model, but a clear picture of what each method contributes and whether combining them helps beyond any single method, with every method given comparable data and compute.

## How the data is used

- **Training: Lichess games.** About 142,500 positions from games between players rated 2000+, each paired with the move that was played. Games give a large and varied set of realistic positions.
- **Testing: Lichess puzzles.** 1,000 puzzles (plus 200 for quick checks), each with one correct move, so answers can be graded objectively. Puzzles are never used for training, and no training position appears among them.
- **Why?** games supply many varied, realistic positions to learn from, while puzzles each have one verified answer, which makes them a fair exam. Keeping the two separate means test scores measure chess skill (not memorization).
- **Planned:** training on puzzles (different ones from the test set), or on a mix of puzzles and games, instead of games only.

## Status

**Done**
- Data pipeline, evaluation, and SFT training code (notebooks 00–03), tested on a small scale
- DPO preference pairs and RL reward (notebook 04)
- Report outline, abstract, and related work (Overleaf, thanks Hannah!)

**Still to do for the progress report (Nov 6)**
- Run notebooks 02–03 on the GPU to get the base vs. SFT results (Nathan)
- Then run DPO and a first RL run from the SFT model, for preliminary results (Nathan)
- Write the report: progress, contributions, results (Hannah)
- Agree on the updated timeline (everyone)

**Later:** ablations, combined methods (e.g. SFT → DPO → RL), games against Stockfish, ChessQA, a chessboard visualizer.

## Contributions

- **Adrian:** repository and environment, data pipeline, prompts and move parsing, evaluation, SFT training code, and DPO/RL inputs (details in `CONTRIBUTIONS.md`).
- **Hannah:** Overleaf document, report outline, abstract, and related work.
- **Nathan:** to be added.

---

## Next steps (Nathan)

Everything below runs on your GPU, in this order. Results pushed by Wed Nov 4.

1. **Set up** (once, about 20 min). See [Setup](#setup).
2. **Base and SFT results.** Run notebooks `00` → `03`, then push the results. See [Run the notebooks](#run-the-notebooks) and [Push the results](#push-the-results).
3. **DPO and RL inputs.** Run notebook `04` (about 5 min).
4. **DPO.** Train from the SFT model with the pairs from `04`, grade, and push. See [DPO and RL training](#dpo-and-rl-training).
5. **RL.** Do a first GRPO run from the SFT model with the reward from `04`, grade, and push.

Tip: while `03` is training, write the DPO and RL training code, so it is ready when the SFT model is.

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

Keep the laptop plugged in and stop it from sleeping while notebooks run.

### Push the results

**Why:** the progress report needs these results (the base vs. SFT table, the figures, and the compute used).

```bash
git add results notebooks
git commit -m "Base model and SFT results"
git push
```

Keep `checkpoints/sft_v1/`: it is the SFT model, and DPO and RL start from it. Back it up (zipped) to the team drive, since it is not on GitHub.

### DPO and RL training

- The end of `04_dpo_rl_inputs` shows the few lines that plug the pairs and the reward into TRL's `DPOTrainer` and `GRPOTrainer`.
- Start both from the SFT model (`checkpoints/sft_v1/`). Settings are in `configs/dpo.yaml` and `configs/rl.yaml`.
- Grade each model like notebooks 02 and 03 do: `evaluate`, then `save_run` with the run names `dpo_v1` and `grpo_v1`. Their scores then appear in `results/summary.csv` next to base and SFT, and the comparison table and figures can include them.
- Push the results the same way as above.

### If something goes wrong

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other apps that use the GPU. Then lower `batch_size` in `configs/eval.yaml` (to 4) or `configs/sft.yaml` (to 2), and run the notebook again. |
| `00_setup_check` says "CPU only" | PyTorch was installed without GPU support. Run the `pip install torch ...` line again, then restart the notebook. |
| `Stockfish not found` | Check `STOCKFISH_PATH`, then restart VS Code. |
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
