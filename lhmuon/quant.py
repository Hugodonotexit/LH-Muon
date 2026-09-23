"""
quant.py -- the block-scaled state container and stochastic rounding.

Every optimizer state tensor the optimizer keeps for a large parameter is stored as two tensors:
a payload `q` and one fp32 absmax per block `s`. All arithmetic happens in fp32 after decode(); the
payload is only a storage format. Formats:

    fmt    payload                       block   bits/param (payload + scale)
    fp32   the fp32 tensor itself        -       32
    bf16   x / absmax in bf16            256     16.125
    fp16   x / absmax in fp16            64      16.5
    int8   round(x / absmax * 127)       256     8.125
    int4   round(x / absmax * 7), packed 64      4.5

The fp16/bf16 payloads are normalised by their block absmax too, so an fp16 state tensor can
neither overflow nor flush its small entries to zero: the payload lives in [-1, 1] where fp16 has
its full 11-bit mantissa.

Integer payloads are written with stochastic rounding by default (E[decode(encode(x))] == x), so a
slowly moving EMA drifts correctly instead of freezing when its per-step change is below half a
quantization step. It still adds noise; see the README ("int4") for when that matters.
"""

import torch
import torch.nn.functional as F

FORMATS = ("fp32", "bf16", "fp16", "int8", "int4")
BLOCK = {"bf16": 256, "fp16": 64, "int8": 256, "int4": 64}
QMAX = {"int8": 127.0, "int4": 7.0}


def _rand(shape, like, generator):
    return torch.rand(shape, generator=generator, device=like.device, dtype=torch.float32)


def stochastic_round(x: torch.Tensor, dtype: torch.dtype, generator: torch.Generator = None) -> torch.Tensor:
    """Round fp32 `x` onto the grid of `dtype` (fp16 or bf16) with E[result] == x.

    Takes the round-to-nearest value and its neighbour on the far side of x (torch.nextafter works
    in the target dtype, so binade edges and subnormals are handled by construction), then moves
    to the neighbour with probability |x - nearest| / spacing. fp16 values beyond +-65504 are
    clamped rather than sent to inf."""
    x = x.float()
    if dtype == torch.float16:
        x = x.clamp(-65504.0, 65504.0)
    near = x.to(dtype)
    diff = x - near.float()
    toward = torch.where(diff >= 0, torch.full_like(near, float("inf")), torch.full_like(near, float("-inf")))
    far = torch.nextafter(near, toward)
    spacing = (far.float() - near.float()).abs()
    prob = torch.where(spacing > 0, diff.abs() / spacing, torch.zeros_like(diff))
    return torch.where(_rand(x.shape, x, generator) < prob, far, near)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    flat = x.reshape(-1).float()
    pad = (-flat.numel()) % block
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat.view(-1, block)


def encode(x: torch.Tensor, fmt: str, generator: torch.Generator = None, stochastic: bool = True):
    """fp32 tensor -> (payload, scale). `stochastic` only affects the integer formats."""
    if fmt == "fp32":
        return x.float().clone(), x.new_empty(0, dtype=torch.float32)
    if fmt not in BLOCK:
        raise ValueError(f"unknown state format {fmt!r}; expected one of {FORMATS}")
    xb = _blocks(x, BLOCK[fmt])
    absmax = xb.abs().amax(dim=1, keepdim=True)
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    y = xb / safe                                                    # in [-1, 1]
    if fmt in ("bf16", "fp16"):
        return y.to(torch.bfloat16 if fmt == "bf16" else torch.float16), absmax
    qmax = QMAX[fmt]
    y = y * qmax
    y = torch.floor(y + _rand(y.shape, y, generator)) if stochastic else torch.round(y)
    q = y.clamp_(-qmax, qmax).to(torch.int8)
    if fmt == "int4":
        u = (q + 8).to(torch.uint8).view(-1, 2)                      # 1..15, two nibbles per byte
        q = (u[:, 0] | (u[:, 1] << 4)).view(xb.shape[0], -1)
    return q, absmax


def decode(q: torch.Tensor, s: torch.Tensor, fmt: str, shape) -> torch.Tensor:
    """(payload, scale) -> fp32 tensor of `shape`."""
    if fmt == "fp32":
        return q.float().view(shape).clone()
    numel = 1
    for d in shape:
        numel *= d
    if fmt == "int4":
        lo = (q & 0x0F).to(torch.int8) - 8
        hi = (q >> 4).to(torch.int8) - 8
        y = torch.stack((lo, hi), dim=-1).view(q.shape[0], -1).float() / QMAX["int4"]
    elif fmt == "int8":
        y = q.float() / QMAX["int8"]
    else:
        y = q.float()
    return (y * s).reshape(-1)[:numel].view(shape)


def zeros(shape, fmt: str, device):
    """The encoded form of an all-zero tensor, allocated directly."""
    return encode(torch.zeros(shape, device=device), fmt, stochastic=False)


def bytes_per_param(fmt: str) -> float:
    if fmt == "fp32":
        return 4.0
    payload = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}[fmt]
    return payload + 4.0 / BLOCK[fmt]
