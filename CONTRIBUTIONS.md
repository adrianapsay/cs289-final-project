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

## Adrian (2026-11-03)

- **Results analysis:** notebook 07, which explains why the models score as they do (confidence intervals, paired tests, a random-move baseline, reasons for illegal moves, and the DPO training diagnostics), with 10 unit tests.
- **Project notes:** `docs/project_notes.md`, a walkthrough of every notebook, each training method, and the results.

## Nathan (2026-11-03)

- **GPU runs:** ran notebooks 00–07 on an RTX 5070 Ti Laptop GPU: base model and SFT results, the DPO pairs, and the first DPO and RL runs, with every model graded on the 1,000 test puzzles.
- **DPO and RL training:** `train_dpo.py` and `train_rl.py` (TRL's DPOTrainer and GRPOTrainer, continuing the SFT adapter, with the SFT model as DPO's reference), their settings in `configs/dpo.yaml` and `configs/rl.yaml`, and notebooks 05 and 06.
- **Memory fix:** Gemma's per-layer embedding table (4.7 GB) stays on the CPU during 4-bit training, so training fits in 12 GB without spilling into slow shared memory.
- **Fixes:** a sampled answer like `f7f7` (same square twice) crashed the move reader during RL; it now counts as illegal. Troubleshooting notes for a stalled model download and a Windows security block.
- **Analysis:** RL notes in notebook 07 and the project notes (RL gains legal moves but loses accuracy by switching to safe pawn moves), and a four-model comparison table and figures.
- **Tests:** 4 more unit tests (CPU embedding table, DPO validation split, same-square moves).
