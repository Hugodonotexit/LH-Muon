"""
train_lm.py -- a small GPT on tokenized uint16 shards, to compare optimizer arms like-for-like.

All three arms run through the SAME LHMuon machinery (fp32 update math, stochastic rounding into
fp16/bf16 weights, loss scaling via grad_coef), so a precision artifact cannot masquerade as an
optimizer difference:

    --opt adamw    every parameter on fp32 AdamW
    --opt muon     spectral matrices on Muon (alpha = 0), the rest as LHMuon routes them
    --opt lhmuon   the full method (alpha > 0)
    --opt lion     every parameter on fp32 Lion (use ~1/7 of AdamW's LR and ~7x its weight decay)

Schedule: linear warmup, constant ("stable"), then linear decay to 0 over the last --decay-frac of
THIS run's budget (WSD). For the loss-vs-tokens frontier of the experiment plan, run one long job
with --save-at, then launch decay branches from those checkpoints with --init-from and a smaller
--tokens budget (the branch point must lie before that budget's decay start). Give the long run
and its branches the same explicit --slow-horizon: the default is derived from each run's budget.

    python examples/train_lm.py --opt lhmuon --tokens 400e6 --out runs/lh_400M
    python examples/train_lm.py --opt muon --tokens 1.6e9 --save-at 100e6,200e6,400e6,800e6 --out runs/mu_long
    python examples/train_lm.py --opt muon --tokens 200e6 --init-from runs/mu_long/stable_100000000.pt --out runs/mu_200M

Data (--data, default ./data/tokenized, or $LHMUON_DATA): a directory with manifest.json
    {"categories": {name: {"files": [{"path": ..., "num_tokens": ...}, ...]}}, ...}
pointing at flat uint16 token files, and optionally eval_set.npy (rows of >= seq+1 token ids).

Two held-out evals: eval_mix = <data>/eval_set.npy (a fixed eval set; falls back to eval_indist if absent) and
eval_indist = fixed windows from the TRAINING categories drawn with a separate RNG (the distribution
the model is trained on). Periodic evals use --eval-rows of each; the final eval uses
--final-eval-rows. Writes <out>/log.csv and <out>/final.json; a run whose loss goes non-finite or
above its starting value writes final.json with "diverged": true.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lhmuon import LHMuon, build_param_groups  # noqa: E402

DATA = os.environ.get("LHMUON_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data", "tokenized"))


# ---------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x, base=10000.0):
    T, hd = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.attn_norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d // heads), RMSNorm(d // heads)
        self.o = nn.Linear(d, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.up = nn.Linear(d, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(B, T, 3, self.heads, D // self.heads).unbind(2)
        q, k, v = rope(self.q_norm(q).transpose(1, 2)), rope(self.k_norm(k).transpose(1, 2)), v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        g, u = self.up(self.ffn_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ffn):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, ffn) for _ in range(layers))
        self.norm = RMSNorm(d)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers) if n.endswith(("o.weight", "down.weight")) else 0.02)

    def forward(self, idx, targets):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = F.linear(self.norm(x), self.embed.weight).float()      # tied head
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


# ---------------------------------------------------------------------------- data
def set_data(path):
    global DATA
    DATA = path


class Sampler:
    def __init__(self, categories, seq, seed):
        man = json.load(open(os.path.join(DATA, "manifest.json")))
        self.files, weights = [], []
        for c in categories:
            for f in man["categories"][c]["files"]:
                self.files.append(np.memmap(f["path"], dtype=np.uint16, mode="r"))
                weights.append(f["num_tokens"])
        self.p = np.array(weights, dtype=np.float64) / sum(weights)
        self.seq = seq
        self.rng = np.random.default_rng(seed)

    def batch(self, n):
        out = np.empty((n, self.seq + 1), dtype=np.int64)
        for i, f in enumerate(self.rng.choice(len(self.files), size=n, p=self.p)):
            s = self.rng.integers(0, len(self.files[f]) - self.seq - 1)
            out[i] = self.files[f][s:s + self.seq + 1]
        return torch.from_numpy(out)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["adamw", "muon", "lhmuon", "lion"], required=True)
    ap.add_argument("--tokens", type=float, required=True, help="token budget of this run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1408)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64, help="sequences per optimizer step")
    ap.add_argument("--micro", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp16",
                    help="weight + compute dtype; fp16 uses a dynamic loss scale")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--slow-every", type=int, default=None)
    ap.add_argument("--slow-horizon", type=int, default=None,
                    help="default min(steps/10, 10000); set it explicitly for a long run and its branches")
    ap.add_argument("--soft-kappa", type=float, default=0.0)
    ap.add_argument("--alpha-decay", action="store_true", help="scale alpha by lr/peak during the decay phase")
    ap.add_argument("--combine", default="sum")
    ap.add_argument("--norm-control", default="wd")
    ap.add_argument("--state-dtype", default="int8")
    ap.add_argument("--slow-dtype", default="int8")
    ap.add_argument("--slow-master", default="device")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--ns-dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"],
                    help="Newton-Schulz compute dtype (auto: bf16 on sm_80+, fp32 otherwise)")
    ap.add_argument("--categories", default="cc_en_head,c4,books,arxiv")
    ap.add_argument("--eval-rows", type=int, default=128)
    ap.add_argument("--final-eval-rows", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-at", default="", help="comma-separated token counts to save stable checkpoints at")
    ap.add_argument("--init-from", default="", help="stable checkpoint to branch from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default=DATA, help="tokenized data directory (see module docstring)")
    a = ap.parse_args()
    set_data(a.data)

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.precision]
    os.makedirs(a.out, exist_ok=True)
    tok_per_step = a.batch * a.seq
    steps = int(a.tokens // tok_per_step)
    decay_start = int(steps * (1 - a.decay_frac))
    save_steps = {int(float(x) // tok_per_step) for x in a.save_at.split(",") if x}

    def lr_at(s):  # s = optimizer steps completed before this one
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if s < decay_start:
            return a.lr
        return a.lr * max(0.0, (steps - s) / max(1, steps - decay_start))

    model = GPT(42000, a.d, a.layers, a.heads, a.ffn).to(dev).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    n_matrix = sum(p.numel() for n, p in model.named_parameters() if p.dim() == 2 and "embed" not in n)
    groups = build_param_groups(model, weight_decay=a.wd, verbose=True)
    if a.opt in ("adamw", "lion"):
        for g in groups:
            g["kind"] = a.opt
    opt = LHMuon(groups, lr=a.lr, total_steps=steps, slow_horizon=a.slow_horizon, alpha=a.alpha if a.opt == "lhmuon" else 0.0,
                 slow_every=a.slow_every, soft_kappa=a.soft_kappa if a.opt == "lhmuon" else 0.0,
                 combine=a.combine, norm_control=a.norm_control, state_dtype=a.state_dtype,
                 slow_dtype=a.slow_dtype, slow_master=a.slow_master, offload=a.offload, seed=a.seed,
                 ns_dtype=a.ns_dtype)
    scale = 2.0 ** 14 if dtype == torch.float16 else 1.0
    clean = 0
    start = 0
    sampler = Sampler(a.categories.split(","), a.seq, a.seed)
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, scale = ck["step"], ck["scale"]
        sampler.rng = np.random.default_rng([a.seed, start])
        if start > decay_start:
            sys.exit(f"branch point {start} is past this budget's decay start {decay_start}")
    n_ev = max(a.eval_rows, a.final_eval_rows)
    held = Sampler(a.categories.split(","), a.seq, seed=987654321)     # same for every run and every --seed
    ev_ind = held.batch(n_ev)
    mix_path = os.path.join(DATA, "eval_set.npy")
    ev_mix = (torch.from_numpy(np.load(mix_path, mmap_mode="r")[:n_ev, :a.seq + 1].astype(np.int64))
              if os.path.exists(mix_path) else ev_ind)

    @torch.no_grad()
    def evaluate(rows):
        model.eval()
        out = []
        for ev in (ev_mix, ev_ind):
            tot = 0.0
            for i in range(0, rows, a.micro):
                x = ev[i:min(rows, i + a.micro)].to(dev)
                tot += model(x[:, :-1], x[:, 1:]).item() * len(x)
            out.append(tot / rows)
        model.train()
        return out

    def finish(diverged, em, ei):
        el = time.time() - t0
        json.dump({"opt": a.opt, "params": n_params, "tokens": done * tok_per_step, "steps": done,
                   "tok_per_param": done * tok_per_step / n_params, "diverged": diverged,
                   "eval_mix": em, "eval_indist": ei, "final_eval_rows": a.final_eval_rows,
                   "seconds": el, "opt_fraction": opt_time / max(el, 1e-9), "args": vars(a)},
                  open(os.path.join(a.out, "final.json"), "w"), indent=1)

    print(f"{a.opt}: {n_params / 1e6:.1f}M params ({n_matrix / 1e6:.1f}M in hidden matrices), {steps} steps x "
          f"{tok_per_step} tokens = {steps * tok_per_step / 1e6:.0f}M tokens "
          f"({steps * tok_per_step / n_params:.1f} tok/param)", flush=True)
    log = open(os.path.join(a.out, "log.csv"), "a", newline="")
    w = csv.writer(log)
    if start == 0:
        w.writerow(["step", "tokens", "lr", "train_loss", "eval_mix", "eval_indist", "seconds", "opt_seconds"])
    t0 = time.time()
    opt_time = 0.0
    done = start
    first_loss = None
    train_sum, train_n = 0.0, 0
    for s in range(start, steps):
        lr = lr_at(s)
        for g in opt.param_groups:
            g["lr"] = lr
        if a.alpha_decay:
            opt.alpha_mult = min(1.0, lr / a.lr) if s >= decay_start else 1.0
        loss_acc = 0.0
        for _ in range(a.batch // a.micro):
            xb = sampler.batch(a.micro).to(dev, non_blocking=True)
            loss = model(xb[:, :-1], xb[:, 1:])
            (loss * (scale * a.micro / a.batch)).backward()
            loss_acc += loss.item() * a.micro / a.batch
        train_sum, train_n = train_sum + loss_acc, train_n + 1
        first_loss = first_loss if first_loss is not None else loss_acc
        if not math.isfinite(loss_acc) or loss_acc > first_loss + 0.5:
            done = s
            print(f"DIVERGED at step {s}: loss {loss_acc}", flush=True)
            finish(True, float("nan"), float("nan"))
            return
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])).item() / scale
        if math.isfinite(norm):
            torch.cuda.synchronize(dev)
            t1 = time.time()
            opt.step(grad_coef=min(1.0, a.clip / (norm + 1e-6)) / scale)
            torch.cuda.synchronize(dev)
            opt_time += time.time() - t1
            clean += 1
            if dtype == torch.float16 and clean >= 500:
                scale, clean = min(scale * 2, 2.0 ** 24), 0
        else:
            scale, clean = max(scale / 2, 1.0), 0
        opt.zero_grad(set_to_none=True)
        done = s + 1
        if done in save_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": done, "scale": scale},
                       os.path.join(a.out, f"stable_{done * tok_per_step}.pt"))
        last = done == steps
        if done % a.eval_every == 0 or last or done == start + 1:
            em, ei = evaluate(a.final_eval_rows if last else a.eval_rows)
            tl = train_sum / max(1, train_n)
            train_sum, train_n = 0.0, 0
            w.writerow([done, done * tok_per_step, f"{lr:.3e}", f"{tl:.4f}", f"{em:.4f}", f"{ei:.4f}",
                        f"{time.time() - t0:.1f}", f"{opt_time:.1f}"])
            log.flush()
            print(f"step {done}/{steps} | tok {done * tok_per_step / 1e6:.0f}M | lr {lr:.2e} | train {tl:.4f} | "
                  f"eval mix {em:.4f} indist {ei:.4f} | gnorm {norm:.3f} | scale {scale:g} | {time.time() - t0:.0f}s "
                  f"(optimizer {100 * opt_time / max(1e-9, time.time() - t0):.1f}%)", flush=True)
            if not (math.isfinite(em) and math.isfinite(ei)):
                finish(True, em, ei)
                return
    finish(False, em, ei)


if __name__ == "__main__":
    main()
