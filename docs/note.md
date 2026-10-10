# Hand-in note: small models that tune a kernel on the chip

Hack the Chip, NYU × Annapurna Labs. Track: kernel optimisation. Seats 205–209.

> Status: sections 1–4 are measured. Section 5 (the agent loop) is running; every `TBD` is a
> placeholder, not a result.

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
writes one instruction. Its prompt and the optimiser's are each under 1,500 characters.

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
- The README says two cores stay free beside the model server; the runtime reports "cores busy".
- The AWS toolkit's profiling recipe needs `torch_neuronx`, which the pod lacks. NKI standalone mode with NumPy inputs works instead.

**How this differs from AWS's own recommendation tool.** `neuron-explorer recommend` exists on the
pod. In print-only mode on our kernel's profile, its banner says "AWS Bedrock-powered", its prompt
is 155,604 characters (it includes the whole NKI performance guide), and it asks for two or three
written recommendations with the instruction "Please do not give code or pseudocode". That is a
hosted large model advising a person. Ours is an 8B model with a prompt under 1,500 characters that
changes the kernel and measures again. We did not run theirs end to end, because it needs a Bedrock
call.

## 5. The agent loop: results

**Start:** the blocked kernel with all three settings at 1, about 1,209 µs (1,207.5 and 1,209.3 in
two measurements). **Yardstick:** the best setting in the full grid of 75, about 905 µs
(`TBD final`), which is 1.34x faster (`TBD final`).

Every arm starts from the same setting and gets 10 attempts per run. 10 runs per arm are planned.

| Arm | What chooses the next setting |
|---|---|
| Random | No model: an untried legal setting drawn uniformly |
| Raw profile | Optimiser model, shown the raw profile line of every attempt so far |
| Template | Optimiser model plus a fixed, code-written sentence built from the profile |
| Checker | Optimiser model plus the checker model's instruction |

| Arm | Runs | Median final time (µs) | Range (µs) | Median speed-up | Runs within 2% of grid best | Median attempts to get within 2% | Median time after 3 attempts (µs) | Attempts lost to compile failures |
|---|---|---|---|---|---|---|---|---|
| Random | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Raw profile | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Template | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Checker | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- **What to expect at this size.** In a partial grid (26 of 75 settings), about a third of settings were within 2% of the best. If that holds for the full grid (`TBD`), random choice will also land near the best within a few attempts, and the arms may not separate on final time. We report that as it comes out.
- **Second shape, 4096 × 4096:** `TBD`. The space is larger and more settings overflow the on-chip buffer, so it is the better test of whether the checker's instruction matters.
- **Settings:** Qwen3-8B for both agents, temperature 0.6, top_p 0.95, thinking off, 8,192-token server context. Chip measurements are cached per setting (three captures, median).
- **Attempt logs,** every attempt with its stage and score: `runs/<arm>.jsonl`. Grid: `results/grid_2048.jsonl`. Summary and chart: `results/summary.json`, `results/curve.csv`, `results/curve.png`.

## 6. Which numbers are which

| Number | Source |
|---|---|
| Times, engine busy, bytes, transfers in sections 3 and 5 | Real chip, `neuron-explorer` profile, LNC=1 |
| "Simulator" column in section 4 | CPU simulator (`nki.simulate`) |
| Compute and memory-bandwidth utilisation | The profiling tool's own estimates |
| Energy | Not available; the pod exposes no power reading, so we report none |

## 7. What we do not claim

- We do not beat AWS's engineers. The kernel is theirs; the loop finds good settings for it.
- The models do not write kernel code. They choose three numbers.
- One operation and one dtype, at one shape so far (`TBD` if 4096 lands). Nothing here shows the loop transfers to other kernels.
- The tutorial kernel is public, so Qwen3-8B may have seen its default settings in training. The comparison between arms is still fair, because every model arm uses the same model.
