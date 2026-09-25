"""
optimizer_test.py -- continued pretraining of a pretrained model (Qwen3.5-0.8B-Base, GPT-2, ...), to
compare LH-Muon, Muon, AdamW and Lion from an already-trained starting point rather than from scratch.

Every run goes through LHMuon (bf16 weights with stochastic rounding, fp32 update maths, compiled
element-wise updates), so precision handling is identical. What differs is the rule for the hidden
matrices; the tied 248K-vocab embedding/head always gets factored Adam (int8 momentum + row/column
second moment) and norms / DeltaNet decay params fp32 AdamW, as Muon itself routes them, which also
keeps AdamW's state inside a 12 GB GPU.

    python examples/qwen/optimizer_test.py --opt lhmuon --lr 2e-5 --out runs/qwen/lhmuon_2e-5
    python examples/qwen/optimizer_test.py --model openai-community/gpt2 --data data/gpt2_mix --no-ckpt --compile \
        --opt muon --lr 3e-4 --steps 1000 --out runs/gpt2/muon_lr3e-4

Writes <out>/log.csv (step, tokens, lr, train_loss, eval_loss, per-source eval, seconds) and
<out>/final.json. Eval = 64 held-out 1024-token windows (13 per source, interleaved), evaluated at
step 0 (the untouched base model) and every --eval-every steps.
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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lhmuon import LHMuon, build_param_groups  # noqa: E402

CE_CHUNK = 512


def chunked_loss(model, ids):
    """Mean next-token CE without materializing [tokens, 248K] logits: 512 tokens at a time, each
    chunk recomputed in backward."""
    hidden = model.base_model(input_ids=ids[:, :-1]).last_hidden_state.flatten(0, 1)
    tgt = ids[:, 1:].reshape(-1)
    w = model.lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)

    def piece(h, t):
        return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")

    for s in range(0, hidden.shape[0], CE_CHUNK):
        total = total + checkpoint(piece, hidden[s:s + CE_CHUNK], tgt[s:s + CE_CHUNK], use_reentrant=False)
    return total / tgt.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["lhmuon", "muon", "adamw", "lion"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--no-ckpt", action="store_true", help="no activation checkpointing (small models)")
    ap.add_argument("--compile", action="store_true", help="torch.compile each transformer block")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=None, help="default 0.1, Lion 0.5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "qwen_mix"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seqs", type=int, default=32, help="sequences per optimizer step (1024 tokens each)")
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--slow-horizon", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    wd = a.wd if a.wd is not None else (0.5 if a.opt == "lion" else 0.1)

    from transformers import AutoModelForCausalLM
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    if not a.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.compile:
        blocks = getattr(model.base_model, "h", None) or getattr(model.base_model, "layers", None)
        for blk in blocks:
            blk.compile()
    model.config.use_cache = False
    model.train()

    groups = build_param_groups(model, weight_decay=wd)
    if a.opt in ("adamw", "lion"):
        for g in groups:
            if g["kind"] == "spectral":
                g["kind"] = a.opt
    opt = LHMuon(groups, lr=a.lr, alpha=a.alpha if a.opt == "lhmuon" else 0.0,
                 slow_horizon=a.slow_horizon if a.opt == "lhmuon" else None,
                 ns_dtype="bf16", compile_updates=True, seed=a.seed)
    params = [p for p in model.parameters() if p.requires_grad]

    man = json.load(open(os.path.join(a.data, "manifest.json")))
    src = {k: np.memmap(v["path"], dtype=np.uint32, mode="r") for k, v in man["sources"].items()}
    names = list(src)
    weights = np.full(len(names), 1.0 / len(names))              # equal mix: c4, cc, books, wiki, arxiv
    ev = torch.from_numpy(np.load(os.path.join(a.data, "eval.npy")).astype(np.int64))
    ev_sources = man["eval_sources"]
    rng = np.random.default_rng(a.seed)

    def batch(n):
        out = np.empty((n, 1025), dtype=np.int64)
        for i, s in enumerate(rng.choice(len(names), size=n, p=weights)):
            arr = src[names[s]]
            o = rng.integers(0, len(arr) - 1026)
            out[i] = arr[o:o + 1025]
        return torch.from_numpy(out)

    @torch.no_grad()
    def evaluate():
        model.eval()
        per = np.zeros(len(ev_sources))
        cnt = np.zeros(len(ev_sources))
        for i in range(0, len(ev), 4):
            x = ev[i:i + 4].to(dev)
            for j in range(len(x)):
                l = chunked_loss(model, x[j:j + 1]).item()
                per[(i + j) % len(ev_sources)] += l
                cnt[(i + j) % len(ev_sources)] += 1
        model.train()
        per /= np.maximum(cnt, 1)
        return float(per.mean()), per.tolist()

    decay_start = int(a.steps * (1 - a.decay_frac))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if s < decay_start:
            return a.lr
        return a.lr * max(0.0, (a.steps - s) / max(1, a.steps - decay_start))

    log = open(os.path.join(a.out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "tokens", "lr", "train_loss", "eval_loss"] + [f"eval_{s}" for s in ev_sources] + ["seconds"])
    t0 = time.time()
    e0, per0 = evaluate()
    w.writerow([0, 0, 0, "", f"{e0:.5f}"] + [f"{v:.5f}" for v in per0] + [f"{time.time() - t0:.1f}"])
    log.flush()
    print(f"{a.opt} lr {a.lr:g} wd {wd:g}: base-model eval {e0:.4f} | " + " ".join(f"{s} {v:.3f}" for s, v in zip(ev_sources, per0)), flush=True)
    train_acc, n_acc = 0.0, 0
    for s in range(a.steps):
        lr = lr_at(s)
        for g in opt.param_groups:
            g["lr"] = lr
        xb = batch(a.seqs)
        loss_sum = 0.0
        for i in range(0, a.seqs, a.micro):
            loss = chunked_loss(model, xb[i:i + a.micro].to(dev, non_blocking=True))
            (loss * a.micro / a.seqs).backward()
            loss_sum += loss.item() * a.micro / a.seqs
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in params if p.grad is not None])).item()
        if not math.isfinite(norm) or not math.isfinite(loss_sum):
            print(f"non-finite at step {s}", flush=True)
            break
        opt.step(grad_coef=min(1.0, a.clip / (norm + 1e-6)))
        opt.zero_grad(set_to_none=True)
        train_acc += loss_sum
        n_acc += 1
        done = s + 1
        if done % a.eval_every == 0 or done == a.steps:
            e, per = evaluate()
            w.writerow([done, done * a.seqs * 1024, f"{lr:.3e}", f"{train_acc / n_acc:.5f}", f"{e:.5f}"] + [f"{v:.5f}" for v in per]
                       + [f"{time.time() - t0:.1f}"])
            log.flush()
            print(f"step {done}/{a.steps} | lr {lr:.2e} | train {train_acc / n_acc:.4f} | eval {e:.4f} ({e - e0:+.4f} vs base) | "
                  f"gnorm {norm:.3f} | {time.time() - t0:.0f}s | peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
            train_acc, n_acc = 0.0, 0
    e, per = evaluate() if done != a.steps else (e, per)
    json.dump({"opt": a.opt, "lr": a.lr, "wd": wd, "steps": done, "tokens": done * a.seqs * 1024, "base_eval": e0,
               "eval": e, "eval_by_source": dict(zip(ev_sources, per)), "base_by_source": dict(zip(ev_sources, per0)),
               "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30, "args": vars(a)},
              open(os.path.join(a.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
