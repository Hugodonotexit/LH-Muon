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
    int4b16  as int4, bf16 scale          16      5
    nf4    16 levels denser near 0 (NF4)  64      4.5

The fp16/bf16 payloads are normalised by their block absmax too, so an fp16 state tensor can
neither overflow nor flush its small entries to zero: the payload lives in [-1, 1] where fp16 has
its full 11-bit mantissa.

Integer payloads are written with stochastic rounding by default (E[decode(encode(x))] == x), so a
slowly moving EMA drifts correctly instead of freezing when its per-step change is below half a
quantization step. It still adds noise; see the README ("int4") for when that matters.
"""

import torch
import torch.nn.functional as F

FORMATS = ("fp32", "bf16", "fp16", "int8", "int4", "int4b16", "nf4")
BLOCK = {"bf16": 256, "fp16": 64, "int8": 256, "int4": 64, "int4b16": 16, "nf4": 64}
QMAX = {"int8": 127.0, "int4": 7.0, "int4b16": 7.0}
# NF4 levels (QLoRA): quantiles of a normal distribution, scaled to [-1, 1]; 0 is exact.
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236328, 1.0)
_NF4_CACHE = {}


def _nf4_levels(device):
    if device not in _NF4_CACHE:
        _NF4_CACHE[device] = torch.tensor(NF4, device=device, dtype=torch.float32)
    return _NF4_CACHE[device]


def _pack4(u, rows):
    """uint8 codes 0..15 -> two per byte."""
    u = u.to(torch.uint8).view(-1, 2)
    return (u[:, 0] | (u[:, 1] << 4)).view(rows, -1)


def _unpack4(q):
    """two per byte -> uint8 codes 0..15 (uint8, not int64: this runs on every decode)."""
    return torch.stack((q & 0x0F, q >> 4), dim=-1).view(q.shape[0], -1)


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
    if fmt == "int4b16":
        # bf16 scale, rounded UP so |x / scale| <= 1 still holds (bf16, not fp16: momentum block
        # maxima can sit below fp16's normal range and lose precision there)
        sc = absmax.to(torch.bfloat16)
        sc = torch.where(sc.float() < absmax, torch.nextafter(sc, torch.full_like(sc, float("inf"))), sc)
        absmax, store = sc.float(), sc
    else:
        store = absmax
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    y = xb / safe                                                    # in [-1, 1]
    if fmt in ("bf16", "fp16"):
        return y.to(torch.bfloat16 if fmt == "bf16" else torch.float16), store
    if fmt == "nf4":
        lv = _nf4_levels(y.device)
        hi = torch.searchsorted(lv, y.clamp(-1, 1).contiguous()).clamp_(1, 15)   # levels[hi-1] <= y <= levels[hi]
        lo_v, hi_v = lv[hi - 1], lv[hi]
        if stochastic:
            up = _rand(y.shape, y, generator) < (y - lo_v) / (hi_v - lo_v)
        else:
            up = (y - lo_v) > (hi_v - y)
        return _pack4(torch.where(up, hi, hi - 1), xb.shape[0]), store
    qmax = QMAX[fmt]
    y = y * qmax
    y = torch.floor(y + _rand(y.shape, y, generator)) if stochastic else torch.round(y)
    q = y.clamp_(-qmax, qmax).to(torch.int8)
    if fmt in ("int4", "int4b16"):
        q = _pack4(q + 8, xb.shape[0])                               # 1..15, two nibbles per byte
    return q, store


def decode(q: torch.Tensor, s: torch.Tensor, fmt: str, shape) -> torch.Tensor:
    """(payload, scale) -> fp32 tensor of `shape`."""
    if fmt == "fp32":
        return q.float().view(shape).clone()
    numel = 1
    for d in shape:
        numel *= d
    if fmt in ("int4", "int4b16"):
        y = (_unpack4(q).to(torch.int8) - 8).float() / QMAX[fmt]
    elif fmt == "nf4":
        y = _nf4_levels(q.device)[_unpack4(q).long()]
    elif fmt == "int8":
        y = q.float() / QMAX["int8"]
    else:
        y = q.float()
    return (y * s.float()).reshape(-1)[:numel].view(shape)


def zeros(shape, fmt: str, device):
    """The encoded form of an all-zero tensor, allocated directly."""
    return encode(torch.zeros(shape, device=device), fmt, stochastic=False)


def bytes_per_param(fmt: str) -> float:
    if fmt == "fp32":
        return 4.0
    payload = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5, "int4b16": 0.5, "nf4": 0.5}[fmt]
    return payload + (2.0 if fmt == "int4b16" else 4.0) / BLOCK[fmt]


# ---------------------------------------------------------------------------
# ranges of a flattened tensor, for chunked updates
# ---------------------------------------------------------------------------
# A chunk is the flat element range [start, start + n) of a tensor whose container was made by
# encode(). For the block formats, `start` must be a multiple of the block size (every chunk but
# the last is a whole number of blocks), so chunk k owns container rows [start/B, (start+n)/B) and
# encoding chunk by chunk gives the same layout as encoding the whole tensor at once.

def chunk_unit(fmt: str) -> int:
    return BLOCK.get(fmt, 1)


def decode_range(q: torch.Tensor, s: torch.Tensor, fmt: str, start: int, n: int) -> torch.Tensor:
    """Elements [start, start + n) of the tensor that (q, s) encodes, as a flat fp32 tensor."""
    if fmt == "fp32":
        return q.reshape(-1)[start:start + n].float().clone()
    B = BLOCK[fmt]
    if start % B:
        raise ValueError(f"chunk start {start} is not a multiple of the {fmt} block size {B}")
    b0, b1 = start // B, -(-(start + n) // B)
    return decode(q[b0:b1], s[b0:b1], fmt, (min(n, (b1 - b0) * B),))[:n]


def encode_into(q: torch.Tensor, s: torch.Tensor, fmt: str, x: torch.Tensor, start: int,
                generator: torch.Generator = None, stochastic: bool = True):
    """Write the flat fp32 chunk `x` (elements [start, start + len(x)) of the tensor) into (q, s)."""
    n = x.numel()
    if fmt == "fp32":
        q.reshape(-1)[start:start + n].copy_(x.reshape(-1))
        return
    B = BLOCK[fmt]
    if start % B:
        raise ValueError(f"chunk start {start} is not a multiple of the {fmt} block size {B}")
    qc, sc = encode(x, fmt, generator, stochastic)
    b0 = start // B
    q[b0:b0 + qc.shape[0]].copy_(qc)
    s[b0:b0 + sc.shape[0]].copy_(sc)
