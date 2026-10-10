"""kopt.py -- the kernel optimisation loop. Small models only; the chip is the judge.

Three roles, as the track asks:

  workload    kernels/matmul_blocked.py, a matmul kernel with three block-size lines
  checker     reads the chip's profile of the latest attempt and writes ONE instruction
  optimiser   reads that instruction and rewrites the three lines

Both agents are the same <=8B model behind an OpenAI-compatible endpoint. Every attempt is compiled,
run on a NeuronCore, checked against NumPy and profiled (chip.py); nothing is scored on the simulator.

Arms, so the result is a comparison and not an anecdote:

  random     no model: an untried setting drawn uniformly from the legal grid
  raw        optimiser only, shown the raw profile numbers of every attempt so far
  template   optimiser + a fixed code-written sentence built from the profile (no checker model)
  checker    optimiser + the checker model's instruction

    python kopt.py --arm checker --runs 10 --attempts 10 --out runs/checker.jsonl

Measurements are cached by setting (results/grid_<size>.jsonl): the chip repeats to under 1%, so a
setting measured once is not measured again. Anything not in the cache is measured live.
"""

import argparse
import json
import os
import random
import re
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chip  # noqa: E402

KNOBS = chip.KNOBS

# What every model arm is told about the kernel. Deliberately says what each line DOES and not
# what value is good: the direction has to come from the measurement.
CARD = """The kernel multiplies a {M}x{K} matrix by a {K}x{N} matrix on an AI accelerator. It works in blocks. Three lines set the block sizes:

TILES_IN_BLOCK_M=<int>   how many 128-wide tiles of the left matrix are loaded per transfer
TILES_IN_BLOCK_N=<int>   how many 512-wide tiles of the right matrix are loaded per transfer
TILES_IN_BLOCK_K=<int>   how many 128-deep tiles of the shared dimension are kept on-chip and reused

Larger blocks mean fewer, bigger transfers and more reuse of data already on the chip, but they need more on-chip buffer, which is limited. M must divide {m_tiles}, N must divide {n_tiles}, K must divide {k_tiles}."""


# ---------------------------------------------------------------- measurement with a cache

class Bench:
    def __init__(self, size, cache_path, live=True):
        self.shape = chip.parse_shape(size)
        self.size, self.live, self.cache_path, self.cache = chip.shape_tag(self.shape), live, cache_path, {}
        self.tiles = (self.shape[1] // 128, self.shape[2] // 512, self.shape[0] // 128)   # M, N, K
        self.base = open(os.path.join(HERE, "kernels", "matmul_blocked.py")).read()
        if os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    r = json.loads(line)
                    self.cache[tuple(r["knobs"])] = r

    def legal(self):
        d = lambda n: [v for v in range(1, n + 1) if n % v == 0]
        return [(m, n, k) for m in d(self.tiles[0]) for n in d(self.tiles[1]) for k in d(self.tiles[2])]

    def measure(self, knobs):
        knobs = tuple(knobs)
        if knobs not in self.cache:
            if not self.live:
                raise KeyError(f"{knobs} is not in the cache and live measurement is off")
            rec = chip.measure(chip.set_knobs(self.base, *knobs), self.shape, reps=3)
            rec["knobs"] = list(knobs)
            self.cache[knobs] = rec
            open(self.cache_path, "a").write(json.dumps(rec) + "\n")
        return self.cache[knobs]


def stage_of(rec):
    if rec.get("time_us"):
        return "timed"
    return "numerics" if "WRONG RESULT" in (rec.get("error") or "") else "compile"


def short_error(rec):
    """The part of a failure a model can act on. The compiler's message is long; keep its reason."""
    e = rec.get("error") or ""
    if ("State buffer allocation failed" in e or "exceeds state buffer capacity" in e
            or "Allocated memory out of bound" in e):
        return "FAILED TO COMPILE: the blocks need more on-chip buffer than the chip has."
    m = re.search(r"assertion failed: (.*)", e) or re.search(r"AssertionError: (?!error)(.*)", e)
    if m:
        return f"FAILED TO COMPILE: {m.group(1)[:160]}"
    return ("FAILED: " + e.replace("\n", " "))[:220]


def profile_line(rec):
    """The raw numbers, one line. This is everything the `raw` arm and the checker model see."""
    if not rec.get("time_us"):
        return short_error(rec)
    return (f"time {rec['time_us']:.1f} microseconds; compute engine busy {100 * rec['te_busy']:.1f}%; "
            f"data movement busy {100 * rec['dma_busy']:.1f}%; {rec['bytes']:,} bytes moved "
            f"(minimum possible {rec['floor_bytes']:,}) in {rec['transfers']} transfers "
            f"averaging {rec['avg_transfer_bytes']:,.0f} bytes")


def knob_text(knobs):
    return "\n".join(f"{n}={v}" for n, v in zip(KNOBS, knobs))


def history_text(hist):
    return "\n".join(f"attempt {i + 1}: M={h['knobs'][0]} N={h['knobs'][1]} K={h['knobs'][2]} -> {profile_line(h['rec'])}"
                     for i, h in enumerate(hist))


# ---------------------------------------------------------------- the two agents

def ask(a, prompt, max_tokens):
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                temperature=a.temperature, top_p=0.95, chat_template_kwargs={"enable_thinking": False})
    for attempt in range(3):
        try:
            r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=180)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # the endpoint is shared by parallel runs; one retry is usually enough
            err = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"model endpoint failed three times: {err}")


def template_instruction(hist):
    """Code-written feedback from the latest profile. The ablation for the checker MODEL: same
    numbers, fixed sentences, no model."""
    rec = hist[-1]["rec"]
    if not rec.get("time_us"):
        return short_error(rec) + " Use smaller blocks than the last attempt."
    out = []
    if rec["bytes"] > 1.05 * rec["floor_bytes"]:
        out.append(f"The kernel moves {rec['bytes'] / rec['floor_bytes']:.2f}x the minimum bytes, so data is "
                   f"being loaded more than once. Keep more of it on the chip between uses.")
    if rec["dma_busy"] > rec["te_busy"]:
        out.append("Data movement is busier than the compute engine, so the engine waits for data.")
    if rec["transfers"] > 200:
        out.append(f"There are {rec['transfers']} separate transfers. Fewer, larger transfers cost less.")
    return " ".join(out) or "The compute engine is busy almost all the time. Little is left to gain."


def checker_instruction(a, hist, card):
    """The checker agent: profile in, one instruction out."""
    prompt = (f"{card}\n\nYou are the CHECKER. You do not write code. You read measurements taken on the real "
              f"chip and tell the engineer what to change next.\n\nAttempts so far (best time wins):\n"
              f"{history_text(hist)}\n\nThe latest attempt is the last line. Reply with ONE instruction of at "
              f"most two sentences: which of M, N or K to change, in which direction, and the reason taken from "
              f"the numbers. Do not repeat a setting that was already tried.")
    return ask(a, prompt, 120)


def optimiser(a, card, current, instruction=None, hist=None):
    """The optimiser agent: rewrites the three lines. Sees the instruction, or the raw history."""
    parts = [card, "\nYou are the OPTIMISER. The three lines are currently:\n\n" + knob_text(current)]
    if hist is not None:
        parts.append("\nMeasurements on the real chip so far (lower time is better):\n" + history_text(hist))
    if instruction:
        parts.append("\nSettings already tried: " + "; ".join(
            f"M={h['knobs'][0]} N={h['knobs'][1]} K={h['knobs'][2]}" for h in instruction["hist"]))
        parts.append("\nInstruction from the checker:\n" + instruction["text"])
    parts.append("\nReply with the three lines only, with new values that have not been tried, to make the kernel faster.")
    reply = ask(a, "\n".join(parts), 80)
    vals = [re.search(rf"{k}\s*=\s*(\d+)", reply) for k in KNOBS]
    return (tuple(int(v.group(1)) for v in vals) if all(vals) else None), reply


# ---------------------------------------------------------------- one run

def run_once(a, bench, arm, run_id, log, start=(1, 1, 1)):
    rng = random.Random(a.seed * 1000 + run_id)
    card = CARD.format(K=bench.shape[0], M=bench.shape[1], N=bench.shape[2], m_tiles=bench.tiles[0],
                       n_tiles=bench.tiles[1], k_tiles=bench.tiles[2])
    hist = [dict(knobs=start, rec=bench.measure(start))]
    best = hist[0]
    pool = [k for k in bench.legal() if k != start]
    rng.shuffle(pool)

    def emit(attempt, knobs, stage, rec=None, instruction="", reply=""):
        row = dict(arm=arm, run=run_id, size=bench.size, start=list(start), attempt=attempt, knobs=list(knobs) if knobs else None,
                   stage_reached=stage, correct=bool(rec and rec.get("correct")),
                   time_us=rec.get("time_us") if rec else None, te_busy=rec.get("te_busy") if rec else None,
                   dma_busy=rec.get("dma_busy") if rec else None, bytes=rec.get("bytes") if rec else None,
                   transfers=rec.get("transfers") if rec else None, error=short_error(rec) if rec and not rec.get("time_us") else None,
                   instruction=instruction, optimiser_reply=reply, best_time_us_so_far=best["rec"]["time_us"])
        log.write(json.dumps(row) + "\n")
        log.flush()

    emit(0, start, "timed", hist[0]["rec"])
    for attempt in range(1, a.attempts + 1):
        instruction, reply = "", ""
        if arm == "random":
            knobs = pool.pop()
        else:
            if arm == "raw":
                knobs, reply = optimiser(a, card, best["knobs"], hist=hist)
            else:
                instruction = (checker_instruction(a, hist, card) if arm == "checker" else template_instruction(hist))
                knobs, reply = optimiser(a, card, best["knobs"], instruction=dict(text=instruction, hist=hist))
        if knobs is None:
            emit(attempt, None, "unreadable", instruction=instruction, reply=reply)
            continue
        if any(tuple(h["knobs"]) == knobs for h in hist):
            emit(attempt, knobs, "duplicate", instruction=instruction, reply=reply)
            continue
        rec = bench.measure(knobs)
        hist.append(dict(knobs=knobs, rec=rec))
        if rec.get("time_us") and rec["time_us"] < best["rec"]["time_us"]:
            best = hist[-1]
        emit(attempt, knobs, stage_of(rec), rec, instruction, reply)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("random", "raw", "template", "checker"))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=10, help="chip attempts per run, after the starting point")
    ap.add_argument("--size", default="2048", help="2048 for square, or K4096_M512_N12288")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--model", default=os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("KOPT_BASE_URL", os.environ.get("KERNEL_AGENT_BASE_URL", "")))
    ap.add_argument("--cache", help="measurement cache (default results/grid_<size>.jsonl)")
    ap.add_argument("--no-live", action="store_true", help="fail instead of measuring a setting missing from the cache")
    ap.add_argument("--starts", default="worst", choices=("worst", "spread", "unblocked"),
                    help="worst: run i starts from the i-th slowest measured setting; unblocked: always 1,1,1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    bench = Bench(a.size, a.cache or os.path.join(HERE, "results", f"grid_{a.size}.jsonl"), live=not a.no_live)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # Measured: the served model answers the same prompt almost identically each time (two runs from
    # 1,1,1 matched for seven attempts), so repeats from one start are not independent. Runs differ
    # by where they START instead: the slowest settings on the grid.
    slow = sorted((r for r in bench.cache.values() if r.get("time_us")), key=lambda r: -r["time_us"])
    if a.starts == "worst":
        starts = [tuple(r["knobs"]) for r in slow[:a.runs]]
    elif a.starts == "spread":   # evenly through the slowest 90% of the grid, by rank
        starts = [tuple(slow[int(i * (len(slow) - 1) * 0.9 / max(a.runs - 1, 1))]["knobs"]) for i in range(a.runs)]
    else:
        starts = [(1, 1, 1)]
    finals = []
    with open(a.out, "a") as log:
        for run_id in range(a.runs):
            best = run_once(a, bench, a.arm, run_id, log, start=starts[run_id % len(starts)])
            finals.append(best["rec"]["time_us"])
            print(f"{a.arm} run {run_id}: best {best['rec']['time_us']:.1f} us at M,N,K={best['knobs']}", flush=True)
    finals.sort()
    print(f"{a.arm}: {len(finals)} runs, median {finals[len(finals) // 2]:.1f}, best {finals[0]:.1f}, worst {finals[-1]:.1f} us")


if __name__ == "__main__":
    main()
