# Pitch (3 minutes)

`TBD` marks a number that does not exist until the agent loop has run. Do not say a number that is
not in `docs/note.md`.

## 0:00 — The number

> "We took a working matmul kernel on Trainium2 and made it **TBDx faster on the real chip**, with
> no human touching the code. The only models in the loop are 8 billion parameters, small enough to
> run on the chip itself."

Then, in one breath: the best hand-tuned AWS version is 1.39x faster than the starting kernel, so
that is the ceiling. We closed **TBD%** of that gap.

*If the loop produced no speed-up:* lead with section "What the checker caught" instead, and say
plainly that the small-model loop did not get there and where it stalled.

## 0:30 — The method

Three roles, as our mentor asked for.

1. **The workload.** An NKI matmul kernel, 2048 × 2048, running on a Trainium2 core.
2. **The checker.** It runs the kernel on the chip and first asks: is the answer still right? Then it reads the hardware profile (time, how busy the tensor engine was, how much data moved) and a small model turns that into **one instruction**.
3. **The optimiser.** A second small model edits the kernel to follow that instruction. Then we measure again.

One line to land: *"The chip already knows why a kernel is slow. The checker's job is to say it in
a sentence a small model can act on."*

## 1:15 — The chart

Show the attempt chart: best time so far against attempt number, one line per arm, with the AWS
fully optimised kernel as a dashed line at 905.8 µs.

| Arm | What it tells you |
|---|---|
| Random choice | What you get with no feedback |
| Raw profile numbers | The small model on its own |
| Checker's instruction | Our contribution |

> "Same model, same starting kernel, same number of attempts. The only thing we changed is the
> feedback. With raw numbers it got to **TBD**; with the checker's instruction it got to **TBD**,
> across **TBD** runs each."

This answers "why a model at all?" before a judge asks it.

## 2:00 — What the checker caught

Two findings from building the on-chip checker, both measured.

- **The simulator and the chip disagree.** On two of four small shapes the simulator counts wasted re-reads (3.7 MB and 7.3 MB) where the chip moved the minimum (2.4 MB and 3.7 MB). The hackathon's own optimisation levels are scored on the simulator, so an agent could lower that score without changing anything on hardware. At size 2048 the waste is real, which is why we optimise there.
- **The pod's default core setting doubles every profile count.** Our first profile said the kernel moved twice the data it needed. It did not; it ran on two physical cores. We measure on one.

> "A checker that is confidently wrong sends the agent in the wrong direction. Ours was, once, and
> the chip is how we found out."

## 2:40 — The honest claim, and close

- We did not beat AWS's engineers. Their kernel is our ceiling.
- It is one operation at one shape. We have not shown it carries over to others.
- What we showed: small models, given the right sentence, produce a measured speed-up on real silicon.

> "Every model nobody has time to hand-tune could have this loop running next to it."

## Likely questions

| Question | Answer |
|---|---|
| Why not grid search? | The random arm is our stand-in. The instruction arm reached **TBD** in **TBD** attempts against **TBD** for random. |
| Did the model just remember the AWS tutorial? | Possibly; the tutorial is public. All three arms use the same model, so the gain from the instruction is not explained by memory. |
| Is the checker a model or code? | Both. Code measures and rejects wrong kernels; an 8B model writes the instruction. |
| How noisy are the timings? | Under 1% run to run on three captures. We treat anything smaller as no change. |
| Energy? | Not measured. The pod exposes no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
