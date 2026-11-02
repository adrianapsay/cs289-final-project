# Contributions

Contributions so far.

## Adrian (2026-11-01)

- **Repository and environment:** project structure, pinned dependencies, Stockfish setup, and a setup-check notebook.
- **Data pipeline:** 1,000 test and 200 dev puzzles (equal rating bands), and 142,500 training positions from 50,000 filtered Lichess games, with a leakage check.
- **Prompts and move parsing:** a shared prompt format with ablation options, and a reader that classifies answers as legal, illegal, or unparseable.
- **Evaluation:** puzzle accuracy, legal-move rate, and Stockfish centipawn loss, with greedy decoding.
- **SFT training:** LoRA/QLoRA fine-tuning, comparison table, and figures.
- **DPO and RL inputs:** 4,788 DPO preference pairs (Stockfish's best move vs. a move losing 200+ centipawns) and a Stockfish-based RL reward with dense and sparse modes.
- **Tests and documentation:** 49 unit tests, README with GPU run instructions, and project notes.
