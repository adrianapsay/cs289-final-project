# Contributions

Contributions so far.

## Adrian (2026-11-01)

- Set up the repository, with pinned dependencies, Stockfish, and a notebook that checks the setup.
- Built the data pipeline. It produces 1,000 test and 200 dev puzzles split evenly across rating bands, and 142,500 training positions from 50,000 filtered Lichess games, and it checks that no training position appears in the puzzles.
- Wrote the prompt format that every model shares, with options for ablations, and the move reader that sorts answers into legal, illegal, and unparseable.
- Built the evaluation, which grades each model on puzzle accuracy, legal-move rate, and Stockfish centipawn loss.
- Wrote the SFT training code (LoRA and QLoRA), along with the comparison table and figures.
- Built the inputs for DPO and RL. These are 4,788 preference pairs, each pairing Stockfish's best move with a move that loses 200+ centipawns, and a Stockfish-based RL reward with dense and sparse versions.
- Wrote 49 unit tests, the README with instructions for running on a GPU, and the project notes.

## Adrian (2026-11-03)

- Analysed the results in notebook 07, which explains why the models score as they do using confidence intervals, paired tests, a random-move baseline, the reasons behind illegal moves, and DPO's training logs. Added 10 unit tests for it.
- Wrote the project notes (`docs/project_notes.md`), which walk through every notebook, each training method, and the results.

## Nathan (2026-11-03)

- Ran notebooks 00–07 on an RTX 5070 Ti Laptop GPU, which produced the base and SFT results, the DPO pairs, and the first DPO and RL runs, with every model graded on the 1,000 test puzzles.
- Wrote the DPO and RL training code (`train_dpo.py` and `train_rl.py`) using TRL's DPOTrainer and GRPOTrainer, along with their settings in `configs/dpo.yaml` and `configs/rl.yaml` and notebooks 05 and 06. Both continue training the SFT adapter, and DPO uses the SFT model as its reference.
- Kept Gemma's per-layer embedding table (4.7 GB) on the CPU during 4-bit training, so that training fits in 12 GB of GPU memory without slowing down.
- Fixed a crash in the move reader during RL, so an answer like `f7f7` (the same square twice) now counts as illegal, and added troubleshooting notes for a stalled model download and a Windows security block.
- Analysed the RL results in notebook 07 and the project notes, finding that RL gains legal moves but loses accuracy by switching to safe pawn moves, and made the four-model comparison table and figures.
- Added 4 unit tests, covering the CPU embedding table, DPO's validation split, and same-square moves.
