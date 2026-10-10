# Pitch (3 minutes)

`TBD` marks a number that does not exist until the agent loop has finished. Do not say a number
that is not in `docs/note.md`.

## 0:00 — The opening (two drafts; choose when checker v2 reports)

What is settled: at size 2048 every arm, including random, reaches the best. At the adapter shape
the first version of the loop lost to random choice. The redesigned checker (v2) is still running.

**Draft 1: v2 beats random. "We read the log and changed the message."**

> "We built a loop of two 8-billion-parameter models that tunes a kernel on Trainium2 with nobody
> involved. On an easy shape it worked. On a hard one it lost to random guessing. So we read its
> log. The checker was telling the optimiser 'increase N' while its own measurements said that made
> things 2.6x slower. We changed one thing, the feedback, not the model. Now it gets within 2% of
> the best in **TBD** of 10 runs, against one in three for random."

Then: the method, the before and after of the checker's message, the chart, what the chip taught us.

**Draft 2: v2 does not beat random. "For this problem, the checker matters and the model does not."**

> "We set out to show that small models, given the right feedback, can tune a kernel on Trainium2.
> We measured every setting on the real chip, 75 at one shape and 96 at another, so we know the
> right answers. Then we ran the models against random guessing. Random won. Here is exactly why,
> and here is what was worth building anyway: a checker that measures on the chip, and what it
> caught."

Then: the method, the arm table, why the models lost, what the checker caught, the in-model
inference numbers.

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

Show the chart: best time so far against attempt number, one line per arm, with the grid best as a
dashed line.

| | Size 2048 | Adapter shape |
|---|---|---|
| Settings within 2% of the best | 39 of 75 | 4 of 96 |
| Spread between settings | 1.34x | 7.14x |
| AWS's default settings | At the best | Do not run: illegal at this shape |
| Random choice, runs within 2% | 200 of 200 | 69 of 200 |
| Model arms (raw, template, checker v1), runs within 2% | 9 of 10 each | 0 of 10 each |
| Checker v2 | **TBD** | **TBD** |

> "On the easy shape everything works, including guessing. On the hard shape our models lost to
> guessing. The checker model kept saying 'increase N to 24' after the chip had shown that was 2.6x
> slower. Two thirds of its attempts were repeats or illegal values."

*Draft 1 continues:* "So we stopped showing it raw numbers and showed it comparisons: 'raising N
from 2 to 24: 1,572 to 4,112 microseconds, 2.62x slower'. Same model. Result: **TBD**."

*Draft 2 continues:* "We changed the feedback and it still did not beat random (**TBD**). For a
space of about a hundred settings, sampling against a real-chip checker is the right tool."

## 2:25 — From a kernel to a model

Two slides' worth, pick by time.

**The model's layers.** Qwen3-8B's config gives four distinct matmul shapes. At 512 tokens AWS's
default settings run on none of them. One loop run per shape improved all four, by 1.14x to 1.31x.
Summed over the model's 252 projections: 434.1 ms to 370.9 ms, 14.6% less. Say that this is a sum of
standalone kernel times, not a model run.

**Real inference time.** We put the kernel inside a traced model on the chip and timed it.

| 36-layer stack, 2048 wide | Inference time |
|---|---|
| Our kernel, unblocked | 35.56 ms |
| Our kernel, the loop's setting | 33.01 ms (7.2% faster) |
| AWS's compiler, no hand-written kernel | 33.00 ms |

> "Tuning takes inference from 35.6 to 33.0 milliseconds, which is exactly where AWS's own compiler
> lands. On a layer where settings matter more, a good setting is 4.3x faster than a bad one, and
> the compiler is still 28% ahead, because it can fuse across operations and a hand-written kernel
> cannot. We tie the compiler at best. We do not beat it."

## 2:40 — The honest claim, and close

- We did not beat AWS's engineers. The kernel is theirs; the loop finds good settings for it.
- At 2048 the 1.34x is from the slowest setting, and AWS's default is already at the best. The adapter shape is where no default exists.
- One operation, **TBD** shape(s). We have not shown it carries over to others.
- At the hard shape, the first version of the loop lost to random choice. We say so.
- What we showed: a checker that measures on the real chip, full ground truth at two shapes, and tuned settings that bring a hand-written kernel to compiler-level inference time with nobody in the loop.

> "The same loop could sit next to any kernel nobody has time to hand-tune. Real-time video, where
> every microsecond per frame counts, is where we would point it next."

## Likely questions

| Question | Answer |
|---|---|
| Why not grid search or random search? | For spaces this small you should. We measured every setting at both shapes and ran random as an arm: it matched the models at 2048 and beat the first version of our loop at the adapter shape. Checker v2: **TBD**. |
| Do the raw and template arms agree because they are right? | No. They made the same first move in all 10 runs and got stuck on the same start. The template sentence barely changed what the optimiser did. |
| Is 1.34x a gain over what AWS ships? | No. At 2048 it is from the slowest setting; AWS's default measures 905.5 µs against a best of 904.7. At the adapter shape AWS's default does not run at all, and that is where tuning is needed. |
| Why do the runs start from different settings? | The served model answers the same prompt almost identically each time, so repeats from one start are not independent. Each run starts from a different one of the 10 slowest settings. |
| How is this different from `neuron-explorer recommend`? | We ran it in print-only mode. It is Bedrock-powered, sends a 155,604-character prompt, and is told not to give code. It advises a person. Ours is an 8B model with a prompt of about 1,200 to 3,000 characters that changes the kernel and measures again. |
| Do the models write the kernel? | No. They choose three block-size settings; code applies them. |
| Did you make the model itself faster? | Not Qwen itself: its matmuls come from AWS's compiler, so our settings have nothing to plug into. In a traced 36-layer model that does use our kernel, tuning took inference from 35.56 to 33.01 ms. Summed over Qwen3-8B's 252 projections as standalone kernels, 434.1 to 370.9 ms. |
| Does your tuned kernel beat AWS's compiler? | No. It ties it. Standalone at the adapter shape: 1,360.9 µs for ours against 1,355.1 for the compiler. In the 36-layer model: 33.01 ms against 33.00. Where the compiler can fuse the matmul with the next operation it is 28% ahead. What the loop does is bring a hand-written kernel to compiler-level speed with nobody involved. |
| So why use a hand-written kernel at all? | Only where the compiler has no good answer: a new operation, or a fused pattern it does not find. We did not test such a case; matmul was the kernel we could measure completely in a day. |
| Why did the models lose to random at the adapter shape? | The checker model reasoned from a rule ("bigger blocks cut data movement") against its own measurements, and the optimiser followed it. 66 of 100 attempts were repeats or illegal values. Random wastes none. |
| Did the model just remember the AWS tutorial? | Possibly; the tutorial is public. Every model arm uses the same model, so differences between arms are not explained by memory. |
| Is the checker a model or code? | Both. Code measures and rejects wrong kernels; an 8B model writes the instruction. The template arm shows what code alone does. |
| How noisy are the timings? | Under 1% run to run on three captures. We treat anything smaller as no change. |
| Energy? | Not measured. The pod exposes no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
