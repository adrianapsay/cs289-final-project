# Contributions

Dated, factual log of who built what.

## 2026-11-01

- **Adrian:** set up the repository structure (`configs/`, `src/chesslm/`, `notebooks/`, `tests/`, `docs/`, `results/`).
- **Adrian:** set up the pinned Python 3.12 environment (`requirements.txt`, `pyproject.toml`).
- **Adrian:** added Stockfish setup and a helper that finds the engine from config, an environment variable, or PATH (`src/chesslm/engine.py`).
- **Adrian:** built the setup-check notebook (`notebooks/00_setup_check.ipynb`): library imports, hardware, Stockfish evaluation, Gemma 4 tokenizer and how it splits chess moves.
- **Adrian:** chose the model ID `google/gemma-4-E2B-it` and recorded the starting SFT settings (`configs/sft.yaml`).
- **Adrian:** built the data pipeline (`notebooks/01_data_pipeline.ipynb`, `src/chesslm/data.py`, `configs/data.yaml`): 1,000-puzzle test set and 200-puzzle dev set (equal rating bands, committed in `data/eval/`), and 150,000 training positions from 50,000 filtered Lichess games, split by game.
- **Adrian:** added the leakage check (0 overlapping positions) and 9 unit tests (`tests/test_data.py`), including the puzzle-format check.
- **Adrian:** saved dataset statistics to `results/data_stats.json`.
