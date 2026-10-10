# Hand-in note: small models that tune a kernel on the chip

Hack the Chip, NYU × Annapurna Labs. Track: kernel optimisation. Seats 205–209.

> Status: all four original arms are final at both shapes, and the per-layer runs and in-model
> inference times are measured. A redesigned checker (v2) is still running. Every `TBD` is a
> placeholder.

## 1. What we built

A loop of three roles, as the mentor specified. Both agents are Qwen3-8B (8B parameters); no larger
model is in the automated loop. Claude was used to write the harness only.

| Role | What it is | Where it runs |
|---|---|---|
| Workload | AWS's blocked tutorial matmul kernel, 2048 × 2048, float32, with three block-size settings | Trainium2 chip, seats 206–209 |
| Checker | Code measures the kernel on the chip; Qwen3-8B reads that profile and writes one instruction | Measurement on the kernel's seat; model on seat 205 |
| Optimiser | Qwen3-8B reads the instruction and picks new values for the three settings | Seat 205 |

**This is settings tuning, not free-form code editing.** The optimiser outputs three numbers
(`TILES_IN_BLOCK_M`, `_N`, `_K`) and code writes them into the kernel. The kernel's structure is
AWS's and does not change. At size 2048 there are 75 legal settings.

The model server takes the whole chip, so the model and the kernels run on separate seats.

## 2. The checker and its reasoning

Every attempt is compiled, run on a real NeuronCore and profiled. Nothing is scored on the
simulator. An attempt stops at the first stage it fails, and only an attempt that is correct on the
chip gets a time.

| Stage in the log | What it means | Why it is there |
|---|---|---|
| `unreadable` | The optimiser's reply did not contain three usable settings | A small model does not always answer in the format asked |
| `duplicate` | The setting was already tried in this run | A repeat teaches nothing and wastes an attempt |
| `compile` | The kernel did not compile, most often because the blocks need more on-chip buffer than the chip has | The feedback is shortened to the one reason a model can act on |
| `numerics` | Output differs from the NumPy reference by more than 2e-2 | A wrong kernel does not crash; it returns plausible numbers |
| `timed` | Correct on the chip; median time of three profile captures, plus engine busy, bytes and transfers | This is the score |

After a timed attempt, the checker model sees one line of raw numbers (time, compute engine busy,
data movement busy, bytes moved against the minimum, transfer count, average transfer size) and
writes one instruction. Its prompt is about 1,200 characters on the first attempt and about 3,000
on the tenth, because it includes every earlier attempt's profile line. The optimiser's prompt is
similar or smaller.

All chip measurements are at LNC=1 (one physical core). At the pod's default LNC=2 the kernel runs
on both physical cores and every count doubles; our first profile was wrong for this reason.

`hwcheck.py` is the same on-chip measurement packaged for the hackathon's own ladder. It adds the
`nkibench.py` rules scan (no host-side `@`, `.T` or `np.mean`; tiles within 128 partitions) and a
check that the kernel did not write into its inputs.

## 3. Headroom: is there anything to win on the chip?

Measured before building the loop. AWS's four tutorial matmul versions, K=M=N=2048, float32, real
chip (seat 206), three captures each, all correct. Time is the median.

| Version | Time (µs) | Tensor engine busy | Data movement busy | Bytes moved | Transfers |
|---|---|---|---|---|---|
| Tiled (naive) | 1258.6 | 80.6% | 90.7% | 318,767,104 | 1600 |
| Loads hoisted | 1093.2 | 87.3% | 92.4% | 301,989,888 | 1344 |
| Blocked | 1019.6 | 90.6% | 62.3% | 167,772,160 | 832 |
| Fully optimised | 905.8 | 97.5% | 21.7% | 67,108,864 | 96 |

- Naive to best is 1.39x faster with 4.75x fewer bytes. The minimum possible traffic is 50,331,648 bytes.
- Run-to-run spread is under 1% (naive: 1254.6 to 1265.7 µs).

Raw data: `results/tutorial_matmul_sweep_2048.json`. Script: `sweep_tutorial.py`.

## 4. What the checker caught

**The simulator and the chip disagree at small shapes.** Level 4 reference kernel, bytes moved:

| Shape | Simulator | Chip | Minimum |
|---|---|---|---|
| K=128 M=128 N=512 | 589,824 | 589,824 | 589,824 |
| K=256 M=256 N=1024 | 3,670,016 | 2,359,296 | 2,359,296 |
| K=512 M=128 N=512 | 1,572,864 | 1,572,864 | 1,572,864 |
| K=256 M=512 N=1024 | 7,340,032 | 3,670,016 | 3,670,016 |

On two shapes the simulator counts re-reads that the chip does not perform. We measured this; we
have not confirmed the mechanism. At size 2048 the waste is real on the chip (section 3). So the
hackathon's simulator-only score for levels 5–7 can reward changes that make no difference on
hardware at toy sizes.

**Three things in the pod differ from the docs.**
- LNC=2 doubles every profile count (above).
- The README says two cores stay free beside the model server. On seat 205 with the server fully up, every core request was refused with "cores busy", at LNC=2 and at LNC=1.
- The AWS toolkit's profiling recipe needs `torch_neuronx`, which the pod lacks. NKI standalone mode with NumPy inputs works instead.

**How this differs from AWS's own recommendation tool.** `neuron-explorer recommend` exists on the
pod. In print-only mode on our kernel's profile, its banner says "AWS Bedrock-powered", its prompt
is 155,604 characters (it includes the whole NKI performance guide), and it asks for two or three
written recommendations with the instruction "Please do not give code or pseudocode". That is a
hosted large model advising a person. Ours is an 8B model with a prompt of about 1,200 to 3,000
characters that changes the kernel and measures again. We did not run theirs end to end, because it needs a Bedrock
call.

## 5. The agent loop: results

### 5.1 The full grid at 2048 (measured)

All 75 legal settings, each measured on the chip with three captures.
Data: `results/grid_2048.jsonl`.

| Fact | Value |
|---|---|
| Settings that ran correctly | 74 of 75 (one, 16/4/16, overflowed the on-chip buffer at compile) |
| Slowest setting, 1/1/1 | 1,209.3 µs |
| Best setting, 4/2/4 | 904.7 µs, 1.34x faster than the slowest |
| Median setting | 919.7 µs |
| AWS's tutorial default, 16/2/8 | 905.5 µs |
| Settings within 1% / 2% / 5% of the best | 26 / 39 / 58 of 75 |

**Read this before the arm results.** The landscape is very forgiving. More than half of all
settings are within 2% of the best, and AWS's default is already there. So the 1.34x is the gain
from the worst setting, not from what AWS ships, and random choice is expected to find a good
setting in a few attempts at this shape.

### 5.2 The arms

| Arm | What chooses the next setting |
|---|---|
| Random | No model: an untried legal setting drawn uniformly |
| Raw profile | Optimiser model, shown the raw profile line of every attempt so far |
| Template | Optimiser model plus a fixed, code-written sentence built from the profile |
| Checker | Optimiser model plus the checker model's instruction |

- **10 runs per arm, 10 attempts per run.** Each run starts from a different one of the 10 slowest settings on the grid (1,209.3 down to 968.1 µs). Every arm uses the same 10 starts.
- **Why different starts.** Repeating a run from the same start is not independent: the served model answers the same prompt almost identically each time. Two checker runs from 1/1/1 matched for seven attempts.
- **Settings:** Qwen3-8B for both agents, temperature 0.6, top_p 0.95, thinking off, 8,192-token server context. Chip measurements are cached per setting (three captures, median).

| Arm | Runs | Median final time (µs) | Range (µs) | Runs within 2% of grid best | Median attempts to get within 2% | Median time after 3 attempts (µs) | Rejected: duplicate | Rejected: compile |
|---|---|---|---|---|---|---|---|---|
| Random | 10 | 906.8 | 904.8 to 911.7 | 10 of 10 | 2 | 913.7 | 0 | 2 |
| Raw profile | 10 | 905.2 | 904.7 to 938.8 | 9 of 10 | 1 | 905.2 | 50 | 7 |
| Template | 10 | 905.2 | 904.7 to 938.8 | 9 of 10 | 1 | 905.2 | 65 | 0 |
| Checker | 10 | 905.3 | 905.1 to 947.7 | 9 of 10 | 2 | 916.1 | 35 | 5 |

Grid best is 904.7 µs, so "within 2%" means 922.8 µs or faster. Each arm made 100 attempts (10 runs
of 10). Every run in every arm ended faster than its start. Source: `results/summary_2048.json`.

**Reading.**
- **No arm is ahead.** Final medians are within 0.2% of each other, and random choice is the only arm with all 10 runs within 2% of the best. At this shape the speed-up does not come from the models' guidance.
- **The checker did not make the loop faster.** The raw-profile arm got within 2% in a median of 1 attempt and the checker arm in 2, the same as random.
- **The checker modestly reduced repeated proposals:** 35 of 100 attempts were settings already tried, against 50 (raw) and 65 (template).
- **Raw and template are not independent confirmations of each other.** Their first move is the same setting in all 10 runs, and the one run that ended at 938.8 µs is the same start in both, where both got stuck re-proposing tried settings. At this shape the template sentence barely changes what the optimiser does.
- **Each model arm has one run that got stuck** outside the 2% bar; random has none.

Attempt logs, every attempt with its stage and score: `runs/<arm>_2048_v2.jsonl`. Summary and
chart: `results/summary.json`, `results/curve.csv`, `results/curve.png`.

### 5.3 One real exchange

From the first pass, starting at 1/1/1, real chip numbers. Settings are M/N/K.

| Attempt | Checker's instruction | Optimiser chose | Result |
|---|---|---|---|
| 1 | "Change M to 2 ... the current setting results in 1152 transfers" | 2/1/1 | 969.1 µs |
| 2 | "Change N to 2, increasing the number of 512-wide tiles loaded per transfer. This reduces data movement busy time" | 2/2/2 | 943.8 µs |
| 3 | "Increase N to 4" | 2/4/2 | 916.6 µs |
| 6 | (not recorded here) | 2/4/4 | 906.6 µs |
| 7 | "Increase N to 8, since the current N=4 is already dividing 4 (as required)" | 2/8/8 | Rejected at compile: N=8 is not a legal value |

In attempt 7 the checker model ignored a limit it had been told. The chip-side check caught it.

### 5.4 Second shape: one AWS's defaults do not cover

K=256, M=4096, N=12288, float32: a rank-256 adapter-style matmul onto Qwen3-8B's 12288-wide
feed-forward dimension (4096 and 12288 are read from the model's config on the pod). 96 legal
settings.

**AWS's tutorial default (16/2/8) does not run here.** It is illegal by the kernel's own
assertions: K=256 is 2 tiles and the default needs K divisible by 1024. So there is no shipped
setting to fall back on at this shape.

**Full grid** on the chip: all 96 settings, two captures each, all correct, none failed to compile.
Data: `results/grid_K256_M4096_N12288.jsonl`.

| Fact | Value |
|---|---|
| Best setting, 4/1/2 | 1,360.9 µs (tensor engine 97.2% busy) |
| Unblocked, 1/1/1 | 1,629.2 µs; the best is 1.20x faster |
| Median setting | 2,147.6 µs |
| Worst setting, 4/24/1 | 9,716.6 µs (tensor engine 23.4% busy), 7.14x slower than the best |
| Settings within 2% / 10% / 50% of the best | 4 / 11 / 44 of 96 |

- A wrong setting costs up to 7.14x here. At 2048 the whole grid spans 1.34x.
- Only 4 of 96 settings are within 2% of the best, against 39 of 75 at 2048. Random choice has roughly a 1-in-3 chance of hitting one in 10 draws, so this shape can separate the arms.
- "Bigger blocks are better" held at 2048 and is wrong here: the slowest settings are the ones with large N blocks.

**Arm design at this shape.** 10 runs per arm, 10 attempts each, same starts for every arm. The
starts are not the 10 slowest, because that would begin every run in the worst corner. They are
spread evenly by rank through the slowest 90% of the grid, from 4/24/1 (9,717 µs) to 16/2/2
(1,496 µs). Headline metrics: final time, and runs ending within 2% and within 10% of the best.

| Arm | Runs | Median final time (µs) | Range (µs) | Runs within 2% of best | Rejected: duplicate | Rejected: compile |
|---|---|---|---|---|---|---|
| Random, 200 runs (the baseline to quote) | 200 | 1,460.8 | worst 1,937.6 | 69 of 200 (34.5%) | 0 | 0 |
| Random, first 10 runs (a lucky sample) | 10 | 1,370.5 | 1,360.9 to 1,559.8 | 7 of 10 | 0 | 0 |
| Raw profile | 10 | 1,906.9 | 1,495.6 to 2,535.2 | 0 of 10 | 51 | 13 |
| Template | 10 | 1,906.9 | 1,495.6 to 2,958.4 | 0 of 10 | 77 | 0 |
| Checker (v1) | 10 | 2,284.8 | 1,495.6 to 3,114.8 | 0 of 10 | 43 | 23 |
| Checker v2 | TBD | TBD | TBD | TBD | TBD | TBD |

Grid best is 1,360.9 µs, so "within 2%" means 1,388.1 µs or faster. The random arm was rerun 200
times over the same starts using the cached grid, because the first 10 runs happened to do better
than random usually does.

**Reading: at this shape the three model arms lost to random choice.**
- Every model run ended faster than its own start, but none got within 2% of the best. Random does so about one time in three, and its median final time is 1,460.8 µs against 1,906.9 to 2,284.8 for the model arms.
- **The checker model argued from a rule of thumb against its own measurements.** In one run the start was 1/2/1 at 1,572 µs. The checker said "Change N to 24 ... reduces data movement busy time". The result was 4,112 µs. It then proposed N=12 (3,462 µs), N=18 and N=16 (both illegal), then N=24 again, and never went back below N=2. Another run proposed N=24 seven times in a row.
- 66 of the checker arm's 100 attempts produced no measurement: 43 repeats and 23 illegal values.
- For comparison, 200 random runs at size 2048: median 905.2 µs, worst 913.5, 200 of 200 within 2%.

**Checker v2 (running).** Same Qwen3-8B. Instead of raw profile lines it is shown comparisons worked
out in code from the attempts so far, for example "raising N from 2 to 24 (with M=1, K=1): 1572 ->
4112 microseconds, 2.62x SLOWER", plus the best so far, the legal values for each setting, the list
already tried, and the sentence "Trust the measured lines over any general rule". This is the
workshop's own lesson applied to our loop: change the feedback, not the model. Result: `TBD`.

**Other shapes tried.**
- *Qwen3-8B's feed-forward projection* (K=4096, M=512, N=12288): mostly flat. In a partial grid (41 of 144 settings measured), 34 ran and 7 overflowed the on-chip buffer. Of the 34, the best is 2,663.6 µs and 28 are within 2% of it, with one bad outlier at 5,626.3 µs. AWS's default is illegal here too (M=512 is 4 tiles; it needs M divisible by 2048). Not used for the arms.
- *Square 4096:* a three-setting probe gave 10,092.8 µs at 1/1/1 and 7,062.0 µs at 4/2/4, about 1.43x, but the full grid was too slow to finish in the day.

### 5.5 From a kernel to a model

A linear layer `y = x W` maps onto this kernel as K = input width, M = token count, N = output
width. So a model's config gives the list of matmul shapes the loop would have to tune. For
Qwen3-8B (hidden 4096, intermediate 12288, 32 heads and 8 key-value heads of width 128, 36 layers),
at 512 tokens:

| Layer | Shape (K / M / N) | Per layer × layers | Legal settings | AWS default 16/2/8 | Loop result, from 1/1/1 |
|---|---|---|---|---|---|
| Attention query and output projections | 4096 / 512 / 4096 | 2 × 36 | 72 | Illegal | 1,075.3 → 906.2 µs at 4/4/8 (1.19x) |
| Attention key and value projections | 4096 / 512 / 1024 | 2 × 36 | 36 | Illegal | 321.4 → 245.0 µs at 2/2/16 (1.31x) |
| Feed-forward gate and up projections | 4096 / 512 / 12288 | 2 × 36 | 144 | Illegal | 3,117.5 → 2,667.2 µs at 1/8/2 (1.17x) |
| Feed-forward down projection | 12288 / 512 / 4096 | 1 × 36 | 144 | Illegal | 3,029.1 → 2,664.7 µs at 1/4/1 (1.14x) |
| Rank-256 adapter weight product | 256 / 4096 / 12288 | not in the base model | 96 | Illegal | See section 5.4: the model arms lost to random here |

The loop column is one checker (v1) run of 10 attempts per shape, measured live on the chip with
three captures each (`runs/layer_*.jsonl`). The loop improved all four base-model shapes. Only the
gate/up shape has a grid to compare against (partial, best 2,663.6 µs, so 2,667.2 is within 0.2%);
for the other three we do not know how close to the best these are.

The shape, settings-count and legality columns are **computed from the config and the kernel's own
rules, not measured**. What they show: at 512 tokens, AWS's tutorial default runs on none of this
model's linear layers, so every one needs its own settings.

Limits of this table:
- "Illegal" is specific to 512 tokens. The default needs M divisible by 2048, so it becomes legal for the base-model rows at 2048 tokens or a multiple.
- The output head (151,936 wide) is not a multiple of 512, so this kernel cannot run it at all. It is excluded.
- These are float32 standalone kernels at the model's shapes. The model itself is not running on them, so there is no tokens-per-second number.
- The adapter row's M=4096 is a weight-product dimension, not a token count.

### 5.6 Towards the model's own speed

No tokens-per-second number for Qwen3-8B is possible today: the served model's matmuls are
generated by AWS's compiler from the whole graph, so the settings this loop tunes have nothing to
plug into without replacing layers and recompiling the model. Three measurements get as close as we
could.

**(a) Sum over the model's projections.** Qwen3-8B has 252 linear projections (query, key, value,
output, gate, up and down, in each of 36 layers). Adding up the standalone kernel time for each at
512 tokens, from the per-layer runs above:

| | Sum over 252 projections |
|---|---|
| Unblocked settings (1/1/1) | 434.1 ms |
| The loop's settings | 370.9 ms (1.17x, 14.6% less) |

This is a sum of float32 standalone kernel times, not a measured forward pass.

**(b) The kernel inside a traced model, timed as inference.** We put the kernel into small models
traced with `torch_neuronx` and ran them on one NeuronCore. "Inference time" is wall clock per call
(median of 50, after 5 warm-ups); "on chip" is the hardware profile (median of 3 captures). All
outputs were checked against CPU (relative error under 3e-6). Scripts and raw results:
`inference/`.

*A 36-layer stack of 2048 × 2048 matmuls, then a mean.* Data stays on the chip between layers.

| Matmul in each layer | Inference time (ms) | On chip (µs) |
|---|---|---|
| Our kernel, unblocked 1/1/1 | 35.556 | profile capture failed |
| Our kernel, the loop's setting 4/2/4 | 33.011 | 31,724.6 |
| Our kernel, AWS tutorial default 16/2/8 | 33.025 | 31,735.6 |
| Compiler default (no hand-written kernel) | 33.004 | 31,726.4 |

The loop's setting cuts inference time by 7.2% against the unblocked kernel, and lands on the
compiler's own time to within 0.02%.

*One adapter-shape layer (K=256, M=4096, N=12288), then a mean.*

| Matmul | Inference time (ms) | On chip (µs) |
|---|---|---|
| Our kernel, worst setting 4/24/1 | 10.627 | 10,281.3 |
| Our kernel, unblocked 1/1/1 | 2.627 | 2,428.3 |
| Our kernel, best setting 4/1/2 | 2.471 | 2,116.1 |
| Compiler default | 1.773 | 1,402.4 |

A good setting makes this model 4.3x faster than the worst one. The compiler default is a further
28% faster than our best. The profile shows why: the compiler fuses the matmul with the mean and
moves 16.9 MB, while our kernel writes its full result out and the mean reads it back, moving
516 MB. A hand-written kernel is a boundary the compiler cannot fuse across.

**(c) The same layer alone, against the compiler.** Adapter shape, no pooling, on-chip time:
compiler default 1,355.1 µs; our kernel at 4/1/2 called inside a traced model 1,460.3 µs; our
kernel at 4/1/2 standalone (grid best) 1,360.9 µs. So standalone, the tuned kernel ties the
compiler (0.4%); embedded in a traced model it is about 7% slower.

**What this adds up to.** Tuning the settings does reduce inference time in a model that uses this
kernel, by 7% to 4.3x depending on how bad the starting setting is. It does not beat what AWS's
compiler produces on its own for a plain matmul: it ties it at best. The loop's value is bringing a
hand-written kernel to compiler-level speed without a person, for the cases where a hand-written
kernel is needed at all.

Note on where (b) and (c) were measured: seat 170, in a separate Python environment with
`torch_neuronx` 2.9 (the pod's own environment does not have it). Three fixes were needed and are
recorded in `inference/README.md`.

### 5.7 What went wrong along the way

- The first pass of arm runs was cut short when the model server died, and is not quoted in the results table. The cause was ours: a request that included a `seed` parameter. The server was restarted and the loop does not send one.

## 6. Which numbers are which

| Number | Source |
|---|---|
| Times, engine busy, bytes, transfers in sections 3 and 5 | Real chip, `neuron-explorer` profile, LNC=1 |
| "Simulator" column in section 4 | CPU simulator (`nki.simulate`) |
| Compute and memory-bandwidth utilisation | The profiling tool's own estimates |
| Energy | Not available; the pod exposes no power reading, so we report none |

## 7. What we do not claim

- We do not beat AWS's engineers or AWS's compiler. The kernel is theirs; the loop finds good settings for it, and at best that ties the compiler's plain matmul.
- At the adapter shape, the first version of our model-driven loop did worse than random choice.
- The models do not write kernel code. They choose three numbers.
- One operation and one dtype, at two shapes. Nothing here shows the loop transfers to other kernels.
- At 2048, the 1.34x is measured from the slowest setting. AWS's tutorial default is already within 0.1% of the best there.
- The tutorial kernel is public, so Qwen3-8B may have seen its default settings in training. The comparison between arms is still fair, because every model arm uses the same model.
