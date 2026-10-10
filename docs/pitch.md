# Pitch (3 minutes)

`TBD` marks a number that does not exist until the agent loop has finished. Do not say a number
that is not in `docs/note.md`.

## 0:00 — The opening (pick one once results are in)

Size 2048 is final and supports A. The adapter-shape arms are still running; if the checker arm is clearly ahead there, use B and keep the 2048 result as the honest null.

**A. The loop sped the kernel up, and random search did about as well.** Expected at size 2048.

> "We built a loop of two 8-billion-parameter models that tunes a matmul kernel on Trainium2 with
> no human involved. From the slowest settings, up to 1,209 microseconds, it reached 905 in 9 runs
> out of 10, where 905 is the best possible, measured on the real chip. And building the checker for it turned up
> things about measuring on this chip that we did not expect."

Then go to the method, then "What the checker caught", then the arm comparison as an honest null.

**B. The checker's instruction clearly beat the other arms.** Possible on the second shape.

> "AWS ships good settings for this kernel at one shape. Move to a shape from a real model's
> adapter layer and those settings do not even run, and a wrong choice costs up to 7x. Same
> small model, same number of attempts: with raw profile numbers it reached **TBD**. With one
> sentence from our checker it reached **TBD**."

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

A real exchange to read out (settings are M/N/K):

| Checker said | Optimiser chose | Time |
|---|---|---|
| (start) | 1/1/1 | 1,209 µs |
| "Change M to 2 ... the current setting results in 1152 transfers" | 2/1/1 | 969 µs |
| "Change N to 2 ... This reduces data movement busy time" | 2/2/2 | 944 µs |
| "Increase N to 4" | 2/4/2 | 917 µs |

One line to land: *"The chip already knows why a kernel is slow. The checker's job is to say it in
a sentence a small model can act on."*

## 1:15 — What the checker caught

Two findings from building the on-chip measurement, both measured.

- **The simulator and the chip disagree.** On two of four small shapes the simulator counts wasted re-reads (3.7 MB and 7.3 MB) where the chip moved the minimum (2.4 MB and 3.7 MB). The hackathon's own optimisation levels are scored on the simulator, so an agent could lower that score without changing anything on hardware. At size 2048 the waste is real, which is why we optimise there.
- **The pod's default core setting doubles every profile count.** Our first profile said the kernel moved twice the data it needed. It did not; it ran on two physical cores. We measure on one.

- **The small checker model also gets things wrong, and the chip catches it.** On attempt 7 it said "Increase N to 8", which is not a legal value. The compile step rejected it.

> "A checker that is confidently wrong sends the agent in the wrong direction. Ours was, more than
> once, and the chip is how we found out."

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

> "We measured all 75 settings. 39 of them are within 2% of the best, and AWS's own default is one
> of those. So random search finds a good one in a few tries. The models matched it; they did not
> beat it: every arm ends within 0.2% of the others. The checker's one measurable effect was
> modest. It cut repeated proposals from 50 to 65 in 100 attempts down to 35. On the adapter
> shape, where only 4 of 96 settings are good, **TBD**."

*If they do separate (opening B):* give the two numbers and the number of runs.

Either way, show the adapter-shape contrast (both grids fully measured on the chip):

| | Size 2048 | Adapter shape (K=256, M=4096, N=12288) |
|---|---|---|
| AWS's default settings | 905.5 µs, at the best | Do not run: illegal at this shape |
| Spread between settings | 1.34x across all 75 | 7.14x across all 96 |
| Settings within 2% of the best | 39 of 75 | 4 of 96 |
| Do bigger blocks help? | Yes | No: large N blocks are the slowest |

> "The rule that worked at one shape is wrong at the next. That is why this needs a loop that
> measures, and not a table of good settings."

## 2:25 — From a kernel to a model

One slide: the per-layer table from `docs/note.md` section 5.5.

> "A model is mostly matrix multiplies. Qwen3-8B's config gives five distinct shapes. At 512
> tokens, AWS's default settings run on none of them. The loop tuned **TBD** of the five on the
> chip."

Say that the legality column is computed from the config, and that the model is not running on
these kernels yet.

## 2:40 — The honest claim, and close

- We did not beat AWS's engineers. The kernel is theirs; the loop finds good settings for it.
- At 2048 the 1.34x is from the slowest setting, and AWS's default is already at the best. The adapter shape is where no default exists.
- One operation, **TBD** shape(s). We have not shown it carries over to others.
- What we showed: two small models, with the chip as judge, produce a measured speed-up on real silicon with nobody in the loop.

> "The same loop could sit next to any kernel nobody has time to hand-tune. Real-time video, where
> every microsecond per frame counts, is where we would point it next."

## Likely questions

| Question | Answer |
|---|---|
| Why not grid search or random search? | At 2048 you could: we measured all 75 settings and random matched the models. We ran it as an arm and report it. At the adapter shape only 4 of 96 settings are good; our result there is **TBD**. |
| Do the raw and template arms agree because they are right? | No. They made the same first move in all 10 runs and got stuck on the same start. The template sentence barely changed what the optimiser did. |
| Is 1.34x a gain over what AWS ships? | No. At 2048 it is from the slowest setting; AWS's default measures 905.5 µs against a best of 904.7. At the adapter shape AWS's default does not run at all, and that is where tuning is needed. |
| Why do the runs start from different settings? | The served model answers the same prompt almost identically each time, so repeats from one start are not independent. Each run starts from a different one of the 10 slowest settings. |
| How is this different from `neuron-explorer recommend`? | We ran it in print-only mode. It is Bedrock-powered, sends a 155,604-character prompt, and is told not to give code. It advises a person. Ours is an 8B model with a prompt of about 1,200 to 3,000 characters that changes the kernel and measures again. |
| Do the models write the kernel? | No. They choose three block-size settings; code applies them. |
| Did you make the model itself faster? | Not measured. We tuned standalone kernels at the model's layer shapes. Summed over its 252 projections at 512 tokens, kernel time goes from **TBD** to **TBD**; that is a sum of standalone times, not a model run. A tokens-per-second number is the next step. |
| Does your tuned kernel beat AWS's compiler? | *If the compiler wins or ties (expected):* No. A plain matmul from the compiler takes **TBD** against **TBD** for our tuned kernel. What the loop does is bring a hand-written kernel to compiler-level speed with nobody involved, and keep it away from settings that are up to 7x slower. *If ours wins:* give both numbers and say it is one shape. |
| Did the model just remember the AWS tutorial? | Possibly; the tutorial is public. Every model arm uses the same model, so differences between arms are not explained by memory. |
| Is the checker a model or code? | Both. Code measures and rejects wrong kernels; an 8B model writes the instruction. The template arm shows what code alone does. |
| How noisy are the timings? | Under 1% run to run on three captures. We treat anything smaller as no change. |
| Energy? | Not measured. The pod exposes no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
