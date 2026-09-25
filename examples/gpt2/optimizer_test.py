"""
optimizer_test.py -- continued pretraining of GPT-2 (124M) to compare LH-Muon, Muon, AdamW and Lion
from an already-trained starting point.

    python examples/gpt2/optimizer_test.py --opt muon --lr 3e-4          # -> runs/gpt2/muon_lr3e-4/
    examples/gpt2/run_queue.sh                                           # the full LR grid

Every run goes through LHMuon (bf16 weights with stochastic rounding, fp32 update maths), so only the
rule for the 48 block matrices differs; the tied embedding/head always gets factored Adam and
LayerNorms/biases fp32 AdamW. One optimizer step = 32 x 1024 tokens in a single forward/backward
(each block compiled, output layer + cross-entropy fused by Liger so the logits are never stored).

Writes <out>/log.csv (step, tokens, lr, train_loss, eval_loss, per-source eval, seconds) and
<out>/final.json. Eval = 64 held-out 1024-token windows (interleaved by source) at step 0 (the base
model) and every 50 steps.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers import AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lhmuon import LHMuon, build_param_groups  # noqa: E402

MODEL = "openai-community/gpt2"
DATA = os.path.join(ROOT, "data", "gpt2_mix")      # examples/qwen/prep_data.py --model openai-community/gpt2
STEPS, SEQS, WARMUP, DECAY_FRAC, EVAL_EVERY, CLIP = 600, 32, 30, 0.2, 50, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["lhmuon", "muon", "adamw", "lion"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=None, help="default 0.1, Lion 0.5")
    ap.add_argument("--alpha", type=float, default=0.25, help="LH-Muon slow-momentum weight")
    ap.add_argument("--slow-horizon", type=int, default=200, help="LH-Muon slow EMA horizon (steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default runs/gpt2/<opt>_lr<lr>")
    a = ap.parse_args()
    wd = a.wd if a.wd is not None else (0.5 if a.opt == "lion" else 0.1)
    out = a.out or os.path.join(ROOT, "runs", "gpt2", f"{a.opt}_lr{a.lr:g}")
    os.makedirs(out, exist_ok=True)

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model.config.use_cache = False
    for blk in model.transformer.h:
        blk.compile()
    head = model.lm_head.weight                    # tied to transformer.wte
    fused_ce = LigerFusedLinearCrossEntropyLoss()

    def loss_of(ids, reduction="mean"):
        h = model.transformer(input_ids=ids[:, :-1]).last_hidden_state.flatten(0, 1)
        if reduction == "mean":
            return fused_ce(head, h, ids[:, 1:].flatten())
        return torch.nn.functional.cross_entropy((h @ head.T).float(), ids[:, 1:].flatten(), reduction="none")

    groups = build_param_groups(model, weight_decay=wd)
    if a.opt in ("adamw", "lion"):
        for g in groups:
            if g["kind"] == "spectral":
                g["kind"] = a.opt
    lh = a.opt == "lhmuon"
    opt = LHMuon(groups, lr=a.lr, alpha=a.alpha if lh else 0.0, slow_horizon=a.slow_horizon if lh else None,
                 ns_dtype="bf16", compile_updates=True, seed=a.seed)
    params = [p for p in model.parameters() if p.requires_grad]

    man = json.load(open(os.path.join(DATA, "manifest.json")))
    src = [np.memmap(v["path"], dtype=np.uint32, mode="r") for v in man["sources"].values()]
    ev = torch.from_numpy(np.load(os.path.join(DATA, "eval.npy")).astype(np.int64))
    ev_sources = man["eval_sources"]
    rng = np.random.default_rng(a.seed)

    def batch():                                   # equal mix of the sources, random 1025-token windows
        out_ = np.empty((SEQS, 1025), dtype=np.int64)
        for i, s in enumerate(rng.integers(0, len(src), SEQS)):
            o = rng.integers(0, len(src[s]) - 1026)
            out_[i] = src[s][o:o + 1025]
        return torch.from_numpy(out_).to(dev, non_blocking=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        per_row = torch.cat([loss_of(ev[i:i + 8].to(dev), "none").view(-1, 1024).mean(1) for i in range(0, len(ev), 8)])
        model.train()
        per = [per_row[k::len(ev_sources)].mean().item() for k in range(len(ev_sources))]
        return float(np.mean(per)), per

    decay_start = int(STEPS * (1 - DECAY_FRAC))

    def lr_at(s):                                  # WSD: linear warmup, flat, linear decay to 0
        if s < WARMUP:
            return a.lr * (s + 1) / WARMUP
        return a.lr * min(1.0, (STEPS - s) / (STEPS - decay_start))

    log = open(os.path.join(out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "tokens", "lr", "train_loss", "eval_loss"] + [f"eval_{s}" for s in ev_sources] + ["seconds"])
    t0 = time.time()
    e0, per0 = evaluate()
    w.writerow([0, 0, 0, "", f"{e0:.5f}"] + [f"{v:.5f}" for v in per0] + [f"{time.time() - t0:.1f}"])
    log.flush()
    print(f"{a.opt} lr {a.lr:g} wd {wd:g}: base-model eval {e0:.4f} | " + " ".join(f"{s} {v:.3f}" for s, v in zip(ev_sources, per0)), flush=True)
    train_acc, e, per, done = [], e0, per0, 0
    for s in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        loss = loss_of(batch())
        loss.backward()
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in params])).item()
        if not math.isfinite(norm):
            print(f"non-finite gradient at step {s}", flush=True)
            break
        opt.step(grad_coef=min(1.0, CLIP / (norm + 1e-6)))
        opt.zero_grad(set_to_none=True)
        train_acc.append(loss.item())
        done = s + 1
        if done % EVAL_EVERY == 0 or done == STEPS:
            e, per = evaluate()
            tl = float(np.mean(train_acc))
            w.writerow([done, done * SEQS * 1024, f"{lr_at(s):.3e}", f"{tl:.5f}", f"{e:.5f}"] + [f"{v:.5f}" for v in per]
                       + [f"{time.time() - t0:.1f}"])
            log.flush()
            print(f"step {done}/{STEPS} | lr {lr_at(s):.2e} | train {tl:.4f} | eval {e:.4f} ({e - e0:+.4f} vs base) | "
                  f"gnorm {norm:.3f} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            train_acc = []
    json.dump({"opt": a.opt, "lr": a.lr, "wd": wd, "steps": done, "tokens": done * SEQS * 1024, "base_eval": e0,
               "eval": e, "eval_by_source": dict(zip(ev_sources, per)), "base_by_source": dict(zip(ev_sources, per0)),
               "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "args": {**vars(a), "model": MODEL, "steps": STEPS, "seqs": SEQS, "warmup": WARMUP, "decay_frac": DECAY_FRAC}},
              open(os.path.join(out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
