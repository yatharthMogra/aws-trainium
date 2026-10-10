"""model_infer.py -- inference time of a small model whose matmuls are either the compiler's or ours.

Two models, each traced with torch_neuronx and run on one NeuronCore:

  stack:S:L   L matmul layers at size S x S, each h <- h.T @ W_i, then a mean over rows.
              The data stays on the chip between layers, so inference time is the layers' time.
  pool:TAG    one layer at shape TAG (e.g. K256_M4096_N12288), then a mean over rows.

Variant `compiler` lets neuronx-cc choose everything. Variant `nki:M,N,K` puts the team's blocked
NKI kernel in every layer at those block settings. Weights and inputs are identical across variants.

    python model_infer.py stack:2048:36 compiler
    python model_infer.py stack:2048:36 nki:4,2,4
"""
import glob, json, os, shutil, statistics, sys, time

os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import numpy as np
import torch
import torch_neuronx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

spec, variant = sys.argv[1], sys.argv[2]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 50
kind = spec.split(":")[0]
if kind == "stack":
    S, L = int(spec.split(":")[1]), int(spec.split(":")[2])
    K, M, N = S, S, S
else:
    d = dict((p[0], int(p[1:])) for p in spec.split(":")[1].split("_"))
    K, M, N, L = d["K"], d["M"], d["N"], 1

r = np.random.default_rng(0)
# Scaled so activations keep the same size from layer to layer.
Ws = [torch.from_numpy((r.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)) for _ in range(L)]
lhsT = torch.from_numpy(r.standard_normal((K, M)).astype(np.float32))
knobs = None if variant == "compiler" else tuple(int(v) for v in variant.split(":")[1].split(","))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Ws = torch.nn.ParameterList([torch.nn.Parameter(w.clone(), requires_grad=False) for w in Ws])
        if knobs:
            from matmul_blocked import nki_matmul_
            self.kernel = nki_matmul_

    def forward(self, h):
        for W in self.Ws:
            if knobs:
                m, n, k = knobs
                h = self.kernel(h, W, TILES_IN_BLOCK_M=m, TILES_IN_BLOCK_N=n, TILES_IN_BLOCK_K=k)
            else:
                h = h.t() @ W
        return h.mean(dim=0)


def reference(h):
    for W in Ws:
        h = h.t() @ W
    return h.mean(dim=0)


out = dict(model=spec, variant=variant, layers=L, calls=calls)
work = os.path.join(HERE, "work", f"{spec.replace(':', '_')}__{variant.replace(':', '_').replace(',', '-')}")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
try:
    t = time.time()
    traced = torch_neuronx.trace(Model().eval(), (lhsT,), compiler_workdir=work, compiler_args=["--lnc=1"])
    out["compile_s"] = round(time.time() - t, 1)
    want = reference(lhsT).numpy()
    got = traced(lhsT).numpy()
    # Relative to the size of the answer: the pooled outputs are small numbers.
    out["rel_err"] = float(np.abs(got - want).max() / np.abs(want).max())
    out["correct"] = bool(got.shape == want.shape and out["rel_err"] < 2e-2)
    for _ in range(5):
        traced(lhsT)
    ts = []
    for _ in range(calls):
        t = time.perf_counter()
        traced(lhsT)
        ts.append((time.perf_counter() - t) * 1e3)
    out.update(wall_ms_median=round(statistics.median(ts), 3), wall_ms_min=round(min(ts), 3),
               wall_ms_max=round(max(ts), 3))
    out["neff"] = glob.glob(os.path.join(work, "**", "*.neff"), recursive=True)[0]
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[-600:]}"
print(json.dumps(out), flush=True)
