# What the write-up still needs from the agent loop

Everything below fills a `TBD` in `docs/note.md` or `docs/pitch.md`.

## Numbers

1. **Per arm (random choice, raw profile, checker's instruction):** number of runs; attempts per run; how many runs ended faster than the 1258.6 µs start; median, minimum and maximum final time in µs.
2. **Run settings:** temperature, whether seeds differ between runs, thinking mode on or off, context length.
3. **Rejections:** count of attempts rejected at each checker stage (rules, numerics, input mutation, compile or runtime error), per arm.
4. **Final-time measurement:** each run's final kernel timed with three captures at LNC=1, the same way as the sweep, so it is comparable with the 905.8 µs yardstick.

## Files

5. **Attempt log**, one JSON line per attempt: `arm`, `run`, `attempt`, `stage_reached`, `correct`, `time_us`, `te_busy`, `dma_busy`, `bytes`, `transfers`, `instruction` (the text the optimiser saw), `kernel_path`. Tell me the path in the repo.
6. **Chart data** as CSV: `arm, run, attempt, best_time_us_so_far`. I can draw the chart from that, or send a PNG with the 905.8 µs line marked.

## Excerpts for the pitch

7. **One good instruction:** a profile, the sentence the checker model produced from it, and the kernel diff that followed, with before and after times.
8. **One bad instruction or failed attempt:** what the checker said, what the optimiser did, and why it was rejected.

## Definitions I need confirmed

9. **Random choice:** random over which set of named changes? If the optimiser model still writes the code in this arm, say so.
10. **Raw profile:** exactly which fields the optimiser sees.
11. **Starting kernel:** the naive tiled tutorial kernel, or something else?

## Worth having if there is time

12. **Code-only instruction arm:** the `verdict()` sentences from `hwcheck.py` with no checker model. A judge may ask whether the checker model adds anything over a template.
13. **A second shape**, even one, to say something about whether the result carries over.
