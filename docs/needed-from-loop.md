# What the write-up still needs from the agent loop

Everything below fills a `TBD` in `docs/note.md` or `docs/pitch.md`. Definitions of the arms,
settings and log stages are already agreed and written into the note.

## Numbers

1. Full grid at 2048: received and written into the note.
2. 2048 arms: final, received and written into the note.
3. **Adapter shape** (K=256, M=4096, N=12288): grid received. Still needed: per-arm numbers (including runs within 10% of best) and the exact assertion text for AWS's 16/2/8.

## Files

4. `runs/<arm>_2048_v2.jsonl`, `results/curve.csv` and `results/curve.png` on `main`, so the note can point at them.

## Excerpts for the pitch

5. One good and one bad exchange: received and written into the note and pitch.

## To confirm

6. Which pitch opening the results support: A (speed-up, random matches it), B (checker clearly ahead) or C (no speed-up).
7. **Per-layer table** (note section 5.5): measured results for the three unmeasured rows, and how many of the five shapes the loop tuned.
