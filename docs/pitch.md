# Pitch (3 minutes)

`TBD` marks a number that does not exist until the agent loop has finished. Do not say a number
that is not in `docs/note.md`.

## 0:00 — The opening (pick one once results are in)

**A. The loop sped the kernel up, and random search did about as well.** Expected at size 2048.

> "We built a loop of two 8-billion-parameter models that tunes a matmul kernel on Trainium2 with
> no human involved. It took the kernel from about 1,209 microseconds to **TBD**, **TBDx faster**,
> measured on the real chip. And building the checker for it turned up two things about measuring
> on this chip that we did not expect."

Then go to the method, then "What the checker caught", then the arm comparison as an honest null.

**B. The checker's instruction clearly beat the other arms.** Possible at size 4096.

> "Same small model, same kernel, same number of attempts. With raw profile numbers it reached
> **TBD**. With one sentence from our checker it reached **TBD**. The feedback is what made the
> difference."

Then the method, then the chart, then "What the checker caught".

**C. The loop produced no speed-up.**

Lead with "What the checker caught", and say plainly that the small-model loop did not get there
and where it stalled.

## 0:30 — The method

Three roles, as our mentor asked for.

1. **The workload.** AWS's blocked matmul kernel, 2048 × 2048, on a Trainium2 core. Three settings control how much data it loads per transfer and how much it keeps on the chip.
2. **The checker.** Code runs the kernel on the chip and asks first: is the answer still right? Then a small model reads the hardware profile (time, how busy the compute engine was, how much data moved) and writes **one instruction**.
3. **The optimiser.** A second small model reads that instruction and picks new settings. Then we measure again.

Say this plainly: the models choose three numbers; they do not write kernel code.

One line to land: *"The chip already knows why a kernel is slow. The checker's job is to say it in
a sentence a small model can act on."*

## 1:15 — What the checker caught

Two findings from building the on-chip measurement, both measured.

- **The simulator and the chip disagree.** On two of four small shapes the simulator counts wasted re-reads (3.7 MB and 7.3 MB) where the chip moved the minimum (2.4 MB and 3.7 MB). The hackathon's own optimisation levels are scored on the simulator, so an agent could lower that score without changing anything on hardware. At size 2048 the waste is real, which is why we optimise there.
- **The pod's default core setting doubles every profile count.** Our first profile said the kernel moved twice the data it needed. It did not; it ran on two physical cores. We measure on one.

> "A checker that is confidently wrong sends the agent in the wrong direction. Ours was, once, and
> the chip is how we found out."

## 2:00 — The comparison

Show the chart (`results/curve.png`): best time so far against attempt number, one line per arm,
with the grid best as a dashed line.

| Arm | What it tells you |
|---|---|
| Random | What you get with no model |
| Raw profile | The small model reading numbers on its own |
| Template | A fixed sentence written by code |
| Checker | The checker model's instruction |

*If the arms do not separate (opening A):*

> "At this size, about a third of all settings are within 2% of the best, so random search finds a
> good one in a few tries. The models matched it; they did not beat it. Where they differ is
> **TBD** (attempts to get within 2%, result after three attempts, attempts lost to compile
> failures). At 4096, where the space is harder, **TBD**."

*If they do separate (opening B):* give the two numbers and the number of runs.

## 2:40 — The honest claim, and close

- We did not beat AWS's engineers. The kernel is theirs; the loop finds good settings for it.
- One operation, **TBD** shape(s). We have not shown it carries over to others.
- What we showed: two small models, with the chip as judge, produce a measured speed-up on real silicon with nobody in the loop.

> "The same loop could sit next to any kernel nobody has time to hand-tune. Real-time video, where
> every microsecond per frame counts, is where we would point it next."

## Likely questions

| Question | Answer |
|---|---|
| Why not grid search or random search? | At 2048 you could: the grid is 75 settings and random does well. We ran it as an arm and report it. The loop's case is larger spaces; our 4096 result is **TBD**. |
| How is this different from `neuron-explorer recommend`? | We ran it in print-only mode. It is Bedrock-powered, sends a 155,604-character prompt, and is told not to give code. It advises a person. Ours is an 8B model with a prompt under 1,500 characters that changes the kernel and measures again. |
| Do the models write the kernel? | No. They choose three block-size settings; code applies them. |
| Did the model just remember the AWS tutorial? | Possibly; the tutorial is public. Every model arm uses the same model, so differences between arms are not explained by memory. |
| Is the checker a model or code? | Both. Code measures and rejects wrong kernels; an 8B model writes the instruction. The template arm shows what code alone does. |
| How noisy are the timings? | Under 1% run to run on three captures. We treat anything smaller as no change. |
| Energy? | Not measured. The pod exposes no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
