# Chess LLM post-training (CS 289A)

## Project description

_TODO: written by the team._

## Setup

**1. Install Stockfish** (the chess engine we use to score moves):
- macOS: `brew install stockfish`
- Linux / Colab: `apt install stockfish`
- Windows: download it from stockfishchess.org, then set the `STOCKFISH_PATH` environment variable to the file.

**2. Install the Python packages** (from the repo folder; Python 3.12 or 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On an NVIDIA GPU machine, first install PyTorch 2.15.0 for CUDA 12.8 or newer, following pytorch.org. Then run the commands above.

**3. Check that it all works:** open `notebooks/00_setup_check.ipynb`, choose the `.venv` kernel, and click **Run All**. Every section should print a result with no errors.

## Repo layout

| Folder | What's in it |
|---|---|
| `notebooks/` | The work, step by step. Start here. |
| `configs/` | All settings (data, evaluation, training). |
| `src/chesslm/` | Shared code used by the notebooks. |
| `tests/` | Automatic checks for the shared code. |
| `results/` | Scores for each run, plus figures. |
| `docs/project_notes.md` | Plain-language explanation of the project. |
| `data/` | Downloaded data. Not on GitHub (too large). |
