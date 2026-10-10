# What the write-up still needs from the agent loop

Everything below fills a `TBD` in `docs/note.md` or `docs/pitch.md`. Definitions of the arms,
settings and log stages are already agreed and written into the note.

## Numbers

1. Full grid at 2048: received and written into the note.
2. **Per arm (random, raw profile, template, checker), from `results/summary.json`:** runs completed; median, minimum and maximum final time; runs ending within 2% of grid best; median attempts to get within 2%; median time after 3 attempts; attempts lost to `compile`, `unreadable` and `duplicate`.
3. **Adapter shape** (K=256, M=4096, N=12288): the full grid of 96 (best, worst, share within 2% of best, count failing to compile), the exact assertion text for AWS's 16/2/8, the starts used, and per-arm numbers.

## Files

4. `runs/<arm>_2048_v2.jsonl`, `results/curve.csv` and `results/curve.png` on `main`, so the note can point at them.

## Excerpts for the pitch

5. One good and one bad exchange: received and written into the note and pitch.

## To confirm

6. Which pitch opening the results support: A (speed-up, random matches it), B (checker clearly ahead) or C (no speed-up).
