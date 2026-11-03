| Model | Accuracy | <1200 | 1200-1800 | 1800-2400 | 2400+ | Legal | Illegal | Unparseable | ACPL |
|---|---|---|---|---|---|---|---|---|---|
| base | 2.8 | 2.4 | 4.8 | 2.4 | 1.6 | 41.2 | 45.8 | 13.0 | 828.0 |
| sft_v1 | 7.8 | 8.4 | 8.4 | 10.8 | 3.6 | 72.2 | 27.8 | 0.0 | 793.0 |
| dpo_v1 | 5.9 | 5.2 | 6.0 | 8.0 | 4.4 | 63.3 | 36.7 | 0.0 | 820.0 |
| grpo_v1 | 3.3 | 3.2 | 0.8 | 6.0 | 3.2 | 81.6 | 18.4 | 0.0 | 836.0 |

Test set: 1,000 Lichess puzzles (250 per rating band). Accuracy, legal, illegal, unparseable in %; ACPL in centipawns on legal moves.
