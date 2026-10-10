# What the write-up still needs from the agent loop

Everything below fills a `TBD` in `docs/note.md` or `docs/pitch.md`. Definitions of the arms,
settings and log stages are already agreed and written into the note.

## Numbers

1. **Full grid at 2048** (`results/grid_2048.jsonl`): the best setting and its time, the share of the 75 settings within 2% of the best, and how many fail to compile.
2. **Per arm (random, raw profile, template, checker), from `results/summary.json`:** runs completed; median, minimum and maximum final time; runs ending within 2% of grid best; median attempts to get within 2%; median time after 3 attempts; attempts lost to `compile`, `unreadable` and `duplicate`.
3. **Size 4096**, if it lands: the same two items, plus the number of legal settings.

## Files

4. `runs/<arm>.jsonl`, `results/curve.csv` and `results/curve.png` on `main`, so the note can point at them.

## Excerpts for the pitch

5. **One good exchange:** the raw profile line, the checker model's instruction, the settings the optimiser chose, and the before and after times.
6. **One bad exchange:** an instruction that sent the optimiser the wrong way, or a compile failure and what the optimiser did next.

## To confirm

7. Which pitch opening the results support: A (speed-up, random matches it), B (checker clearly ahead) or C (no speed-up).
