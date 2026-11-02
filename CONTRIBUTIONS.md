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
- **Adrian:** built the shared prompt builder and move reader (`src/chesslm/prompts.py`, `configs/prompt.yaml`): three prompt options (board only, + last move, + legal moves) in UCI or SAN; answers classified as legal, illegal, or unparseable. 27 unit tests (`tests/test_prompts.py`).
- **Adrian:** added the opponent's last move to the evaluation puzzles and SAN move history to the training data, so the move-history prompt option is comparable between training and testing.
- **Adrian:** built the evaluation harness (`src/chesslm/evaluate.py`, `src/chesslm/engine.py`, `notebooks/02_eval_base_model.ipynb`): puzzle accuracy (any mate counts in mate-in-1), legal/illegal/unparseable rates, Stockfish centipawn loss with caching, greedy decoding, results saved per run.
- **Adrian:** built text-only model loading (`src/chesslm/model.py`), saving ~0.95 GB of GPU memory.
- **Adrian:** built the SFT training code and notebook (`src/chesslm/train_sft.py`, `notebooks/03_sft_first_run.ipynb`): LoRA/QLoRA with TRL, loss on the answer only, compute logging, comparison table and figures (`src/chesslm/plots.py`).
- **Adrian:** wrote GPU setup and run instructions for the team (README).
