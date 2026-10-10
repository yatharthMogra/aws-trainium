# Hand-in note: small models that speed up a kernel on the chip

Hack the Chip, NYU × Annapurna Labs. Track: kernel optimisation. Seats 205–209.

> Status: sections 1–4 are measured. Section 5 (the agent loop) has not run yet; every `TBD` is a
> placeholder, not a result.

## 1. What we built

A loop of three roles, as the mentor specified. Both models are Qwen3-8B (8B parameters); no larger
model is in the automated loop. Claude was used to write the harness only.

| Role | What it is | Where it runs |
|---|---|---|
| Workload | An NKI matmul kernel, K=M=N=2048, float32 | Trainium2 chip, seats 206–209 |
| Checker | Code that measures the kernel on the chip, then Qwen3-8B turns the profile into one instruction | Measurement on the kernel's seat; model on seat 205 |
| Optimiser | Qwen3-8B edits the kernel source following that instruction | Seat 205 |

The model server takes the whole chip, so the model and the kernels run on separate seats.

## 2. The checker and its reasoning

A kernel is rejected at the first stage it fails. Only a kernel that is correct on the chip gets a
speed score, so fast and wrong scores zero.

| Stage | Accepts | Rejects | Why |
|---|---|---|---|
| Rules scan (`nkibench.py`) | Source that does the work in NKI | Host-side shortcuts (`@`, `.T`, `np.mean`), tiles over 128 partitions, missing `@nki.jit`, wrong entry-point name | Otherwise the "kernel" hands the operation to NumPy and passes |
| Numerics on the chip | Output within 2e-2 of the NumPy reference | Any mismatch, reported with the element that differs | A wrong kernel does not crash; it returns plausible numbers |
| Input mutation | Inputs unchanged after the call | A kernel that writes into its input | It passes a standalone test by luck and corrupts the caller's buffer in a real graph |
| On-chip profile (`hwcheck.py`) | n/a (this stage scores, it does not reject) | n/a | Gives time on chip, tensor-engine busy, data-movement busy, bytes and transfer count |
| Instruction (Qwen3-8B) | n/a | n/a | Turns the profile into one named change for the optimiser; raw numbers alone are the control arm |

All chip measurements are at LNC=1 (one physical core). At the pod's default LNC=2 the kernel runs
on both physical cores and every count doubles; our first profile was wrong for this reason.

## 3. Headroom: is there anything for an agent to win?

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
- These four kernels are the yardstick only. The models are never shown them.

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

## 5. The agent loop: results

Start: the naive tiled kernel (1258.6 µs). Each arm gets the same starting kernel, the same number
of attempts and the same optimiser model. Only the feedback differs.

| Arm | Feedback the optimiser sees | Runs | Runs ending faster than start | Median final time (µs) | Range (µs) | Median speed-up | Gap closed |
|---|---|---|---|---|---|---|---|
| Random choice | none; a random named change | TBD | TBD | TBD | TBD | TBD | TBD |
| Raw profile | the profile numbers as measured | TBD | TBD | TBD | TBD | TBD | TBD |
| Checker's instruction | one named change from the checker model | TBD | TBD | TBD | TBD | TBD | TBD |

"Gap closed" is (1258.6 − final) / (1258.6 − 905.8), so 0% is the naive kernel and 100% matches the
AWS fully optimised one. Differences under 1% are noise.

- Attempts per run: TBD. Temperature and seed policy: TBD.
- Attempts rejected, by checker stage: TBD.
- Full attempt log, every attempt with its score: `TBD path`.

## 6. Which numbers are which

| Number | Source |
|---|---|
| Times, engine busy, bytes, transfers in sections 3 and 5 | Real chip, `neuron-explorer` profile, LNC=1 |
| "Simulator" column in section 4 | CPU simulator (`nki.simulate`) |
| Compute and memory-bandwidth utilisation | The profiling tool's own estimates |
| Energy | Not available; the pod exposes no power reading, so we report none |

## 7. What we do not claim

- We do not beat AWS's engineers; their fully optimised kernel is our ceiling.
- One operation, one shape, one dtype. Nothing here shows the loop transfers to other kernels.
- The tutorial kernels are public, so Qwen3-8B may have seen them in training. The comparison between arms is still fair, because all three arms use the same model.
