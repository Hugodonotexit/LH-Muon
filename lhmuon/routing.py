r"""
routing.py -- which parameter gets which update rule.

    spectral   2D+ weight matrices whose smaller side is >= min_dim: LH-Muon.
               conv weights (out, in, k, ...) are flattened to (out, in*k*...);
               MoE expert stacks (E, m, n) whose name contains "experts" are orthogonalized per expert.
    factored   embeddings, LM heads (tied or not) and MoE routers: Adam with an int8/int4 momentum
               and an Adafactor row/column second moment (a few fp32 vectors per matrix).
    adamw      everything else -- norms, biases, SSM/GDN decay/dt/A params, depthwise convs, gates,
               thin matrices: full fp32 AdamW. These are <1% of a model, so the memory is free.

Name patterns are regexes searched in the parameter name. `overrides` is checked first and maps a
pattern to a kind, e.g. {r"embed\.adaptive\.proj": "spectral"}.

The router default is a heuristic, not something validated: an MoE router is a small matrix whose
rows compete through a softmax, and Muon's equalised singular values seem a poor fit for that.
"""

import re

import torch.nn as nn

SPECTRAL, FACTORED, ADAMW, LION = "spectral", "factored", "adamw", "lion"
KINDS = (SPECTRAL, FACTORED, ADAMW, LION)                 # lion is never routed to; set a group's kind to use it

EMBED_PATTERNS = (r"embed", r"lm_head", r"(^|\.)wte\.", r"(^|\.)wpe\.", r"(^|\.)output\.weight$",
                  r"\.tables\.", r"cluster_vectors", r"(^|\.)head\.")
ROUTER_PATTERNS = (r"(^|\.)router\.", r"(^|\.)gate\.weight$")     # Mixtral/Qwen-MoE routers are "...mlp.gate.weight"
ADAMW_PATTERNS = (r"A_log", r"dt_bias", r"(^|\.)D$", r"norm")


def _search(patterns, name):
    return any(re.search(p, name) for p in patterns)


def matrix_view(name: str, shape) -> str:
    """How a spectral parameter is viewed: 'matrix' (2D), 'flatten' (conv), 'batched' (experts)."""
    if len(shape) == 2:
        return "matrix"
    if len(shape) == 3 and "experts" in name:
        return "batched"
    return "flatten"


def matrix_shape(view: str, shape):
    if view == "matrix":
        return tuple(shape)
    if view == "batched":
        return tuple(shape)
    rest = 1
    for d in shape[1:]:
        rest *= d
    return (shape[0], rest)


def classify(name: str, p, embedding_params=frozenset(), min_dim: int = 32, overrides=None) -> str:
    for pattern, kind in (overrides or {}).items():
        if re.search(pattern, name):
            if kind not in KINDS:
                raise ValueError(f"override {pattern!r} -> {kind!r}; kind must be one of {KINDS}")
            return kind
    if p.dim() < 2 or _search(ADAMW_PATTERNS, name):
        return ADAMW
    if id(p) in embedding_params or _search(EMBED_PATTERNS, name) or _search(ROUTER_PATTERNS, name):
        return FACTORED
    if min(matrix_shape(matrix_view(name, p.shape), p.shape)[-2:]) >= min_dim:
        return SPECTRAL
    return ADAMW


def build_param_groups(model: nn.Module, weight_decay: float = 0.1, lr_mult=None, no_decay_1d: bool = True,
                       min_dim: int = 32, overrides=None, wd_overrides=None, verbose: bool = False):
    """Param groups for LHMuon. Each group carries `kind`, `lr_mult`, `weight_decay` and `names`.

    lr_mult: {pattern: multiplier}; the first matching pattern wins. The effective LR of a group is
             group["lr"] * group["lr_mult"], so a trainer can keep setting every group's "lr" to the
             same scheduled value (LR is RMS-matched across kinds, so one value serves all).
    weight_decay: decoupled decay for factored/adamw groups, and for spectral ones when
             norm_control="wd" (with norm_control="sphere" spectral groups ignore it).
    no_decay_1d: 1D params (norm gains, biases) get weight_decay 0.
    wd_overrides: {pattern: weight_decay}; the first matching pattern wins over both rules above."""
    embedding_params = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    groups = {}
    for name, p in model.named_parameters():                       # tied weights appear once
        if not p.requires_grad:
            continue
        kind = classify(name, p, embedding_params, min_dim, overrides)
        mult = 1.0
        for pattern, m in (lr_mult or {}).items():
            if re.search(pattern, name):
                mult = float(m)
                break
        wd = 0.0 if (no_decay_1d and p.dim() < 2) else weight_decay
        for pattern, v in (wd_overrides or {}).items():
            if re.search(pattern, name):
                wd = float(v)
                break
        key = (kind, mult, wd)
        g = groups.setdefault(key, {"params": [], "names": [], "kind": kind, "lr_mult": mult, "weight_decay": wd})
        g["params"].append(p)
        g["names"].append(name)
    out = list(groups.values())
    if verbose:
        for g in out:
            n = sum(p.numel() for p in g["params"])
            print(f"[lhmuon] {g['kind']:9s} lr_mult {g['lr_mult']:g} wd {g['weight_decay']:g}: "
                  f"{len(g['params'])} tensors, {n / 1e6:.2f}M params")
    return out
