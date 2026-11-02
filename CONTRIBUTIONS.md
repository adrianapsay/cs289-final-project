# Contributions

Dated, factual log of who built what.

## 2026-11-01

- **Adrian:** set up the repository structure (`configs/`, `src/chesslm/`, `scripts/`, `notebooks/`, `tests/`, `docs/`, `results/`).
- **Adrian:** set up the pinned Python 3.12 environment (`requirements.txt`, `pyproject.toml`).
- **Adrian:** added Stockfish setup and a helper that finds the engine from config, an environment variable, or PATH (`src/chesslm/engine.py`).
- **Adrian:** built the setup-check notebook (`notebooks/00_setup_check.ipynb`): library imports, hardware, Stockfish evaluation, Gemma 4 tokenizer and how it splits chess moves.
- **Adrian:** chose the model ID `google/gemma-4-E2B-it` and recorded the starting SFT settings (`configs/sft.yaml`).
