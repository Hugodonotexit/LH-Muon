"""
bench_optimizers.py -- optimizer-only cost of AdamW, Lion, Muon and LH-Muon on the example GPT.

For each optimizer, on fp16 weights with synthetic gradients, reports the optimizer state (bytes and
bytes/param), the transient memory the step allocates on top of that, and the median / max step
time over --steps steps. All four run through LHMuon, so they share the same fp32 update path and
stochastic rounding. AdamW and Lion keep fp32 state here; Muon and LH-Muon keep int8.

    python scripts/bench_optimizers.py --device cuda:0
    python scripts/bench_optimizers.py --device cuda:0 --sizes 512,8,8,1408 768,12,12,2048
"""

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
from lhmuon import LHMuon, build_param_groups  # noqa: E402
from train_lm import GPT  # noqa: E402

OPTIMIZERS = [
    ("AdamW", "adamw", dict(alpha=0.0)),
    ("Lion", "lion", dict(alpha=0.0)),
    ("Muon", None, dict(alpha=0.0)),
    ("LH-Muon (alpha 0.25)", None, dict(alpha=0.25, slow_horizon=300)),
]


def bench(model, dev, steps):
    n = sum(p.numel() for p in model.parameters())
    rows = []
    for label, kind, kw in OPTIMIZERS:
        groups = build_param_groups(model, weight_decay=0.1)
        if kind:
            for g in groups:
                g["kind"] = kind
        opt = LHMuon(groups, lr=1e-3, total_steps=10000, **kw)
        for p in model.parameters():
            p.grad = (torch.randn_like(p, dtype=torch.float32) * 1e-3).to(p.dtype)
        opt.step()                                               # allocates the state
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        ts = []
        for _ in range(steps):
            torch.cuda.synchronize(dev)
            t = time.time()
            opt.step()
            torch.cuda.synchronize(dev)
            ts.append(time.time() - t)
        ts.sort()
        state = sum(v for (_, where), v in opt.state_bytes().items() if where in ("device", "host"))
        rows.append((label, state, state / n, torch.cuda.max_memory_allocated(dev) - base,
                     ts[len(ts) // 2], ts[-1]))
        del opt
        torch.cuda.empty_cache()
    return n, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--sizes", nargs="+", default=["512,8,8,1408", "768,12,12,2048"],
                    help="d,layers,heads,ffn per model")
    a = ap.parse_args()
    dev = torch.device(a.device)
    print(f"device: {torch.cuda.get_device_name(dev)}")
    for spec in a.sizes:
        d, layers, heads, ffn = map(int, spec.split(","))
        torch.manual_seed(0)
        model = GPT(42000, d, layers, heads, ffn).to(dev).half()
        n, rows = bench(model, dev, a.steps)
        print(f"\n{n / 1e6:.1f}M params (d {d}, {layers} layers, SwiGLU {ffn})")
        print(f"| optimizer | state | state B/param | transient | step median | step max |")
        print(f"|---|---|---|---|---|---|")
        for label, state, bpp, tr, med, mx in rows:
            print(f"| {label} | {state / 2**20:.0f} MiB | {bpp:.2f} | {tr / 2**20:.0f} MiB | "
                  f"{1000 * med:.0f} ms | {1000 * mx:.0f} ms |")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
