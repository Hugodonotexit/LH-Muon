"""Unit tests. `python -m pytest tests -q` (GPU tests use the emptiest CUDA device and skip without one)."""

import copy
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lhmuon import LHMuon, build_param_groups, newton_schulz, soft_polar, quant as Q  # noqa: E402
from lhmuon.routing import ADAMW, FACTORED, SPECTRAL  # noqa: E402


def _emptiest_cuda():
    if not torch.cuda.is_available():
        return None
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{max(range(len(free)), key=free.__getitem__)}")


CUDA = _emptiest_cuda()
needs_cuda = pytest.mark.skipif(CUDA is None, reason="needs CUDA")
DEV = CUDA or torch.device("cpu")


# ---------------------------------------------------------------------------- quant
@pytest.mark.parametrize("fmt", Q.FORMATS)
def test_roundtrip_error_bound(fmt):
    torch.manual_seed(0)
    x = torch.randn(1000, 77, device=DEV) * torch.logspace(-6, 2, 77, device=DEV)   # wide dynamic range
    q, s = Q.encode(x, fmt, stochastic=False)
    y = Q.decode(q, s, fmt, x.shape)
    assert y.shape == x.shape and y.dtype == torch.float32
    if fmt == "fp32":
        assert torch.equal(x, y)
        return
    xb = Q._blocks(x, Q.BLOCK[fmt])
    step = xb.abs().amax(1, keepdim=True) / Q.QMAX.get(fmt, 1.0)
    err = Q._blocks(y - x, Q.BLOCK[fmt]).abs()
    if fmt == "nf4":                                              # half the widest gap between NF4 levels
        assert (err <= 0.153 * xb.abs().amax(1, keepdim=True) + 1e-30).all()
    elif fmt in Q.QMAX:
        slack = 1.01 if fmt == "int4b16" else 1.0 + 1e-6            # int4b16's bf16 scale is rounded up
        assert (err <= step / 2 * slack).all()
    else:
        rel = {"fp16": 2 ** -11, "bf16": 2 ** -8}[fmt]
        assert (err <= step * rel + 1e-30).all()


@pytest.mark.parametrize("fmt", ["int8", "int4", "int4b16", "nf4"])
def test_stochastic_encode_is_unbiased(fmt):
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.linspace(-1, 1, 256, device=DEV)
    acc = torch.zeros_like(x)
    n = 4000
    for _ in range(n):
        q, s = Q.encode(x, fmt, g)
        acc += Q.decode(q, s, fmt, x.shape)
    step = 1.0 / Q.QMAX[fmt] if fmt in Q.QMAX else 0.33                 # nf4's widest gap between levels
    assert (acc / n - x).abs().max() < 4 * step * 0.5 / math.sqrt(n) + 1e-6


def test_int4_packing_exact_on_grid():
    x = (torch.randint(-7, 8, (5, 64), device=DEV).float() / 7)
    x[:, 0] = 1.0                                                   # pin absmax to 1
    q, s = Q.encode(x, "int4", stochastic=False)
    assert q.dtype == torch.uint8 and q.numel() == x.numel() // 2
    assert torch.allclose(Q.decode(q, s, "int4", x.shape), x, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stochastic_round_weights_unbiased(dtype):
    g = torch.Generator(device=DEV).manual_seed(2)
    # far below half an ulp at 1.0 (fp16 ulp 9.8e-4, bf16 7.8e-3), a subnormal, a negative, a binade edge
    x = torch.tensor([1.0 + 1e-5, 3e-7, -2.5 - 1e-4, 2.0 - 1e-6], device=DEV)
    n = 200000
    xs = x.expand(n, -1).contiguous()
    r = Q.stochastic_round(xs, dtype, g)
    assert r.dtype == dtype
    mean = r.float().mean(0)
    ulp = (torch.nextafter(x.to(dtype), torch.full_like(x, float("inf")).to(dtype)).float() - x.to(dtype).float()).abs()
    assert ((mean - x).abs() <= 5 * ulp / math.sqrt(n)).all(), (mean - x, ulp)
    assert torch.isfinite(Q.stochastic_round(torch.tensor([1e6], device=DEV), torch.float16, g)).all()


# ---------------------------------------------------------------------------- polar
def test_soft_polar_identity_exact():
    """polar([C; eps I]) top block == C (C^T C + eps^2 I)^(-1/2), via SVD (the maths, not the iteration)."""
    torch.manual_seed(0)
    C = torch.randn(40, 24, dtype=torch.float64) @ torch.diag(torch.logspace(-3, 0, 24, dtype=torch.float64))
    eps = 0.05
    X = torch.cat([C, eps * torch.eye(24, dtype=torch.float64)])
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    top = (U @ Vh)[:40]
    u, s, vh = torch.linalg.svd(C, full_matrices=False)
    want = u @ torch.diag(s / torch.sqrt(s ** 2 + eps ** 2)) @ vh
    assert torch.allclose(top, want, atol=1e-10)


@pytest.mark.parametrize("shape", [(96, 64), (64, 96), (3, 48, 80)])
def test_soft_polar_shrinks_by_spectrum(shape):
    torch.manual_seed(0)
    *b, m, n = shape
    k = min(m, n)
    u, _ = torch.linalg.qr(torch.randn(*b, m, k, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(*b, n, k, dtype=torch.float64))
    sig = torch.logspace(-2, 0, k, dtype=torch.float64)
    C = (u * sig) @ v.mT
    eps = torch.tensor(0.1)
    out = soft_polar(C.float(), eps).double()
    d = torch.diagonal(u.mT @ out @ v, dim1=-2, dim2=-1)          # values along C's singular directions
    want = sig / torch.sqrt(sig ** 2 + 0.01)
    ratio = d / want
    # Jordan's NS coefficients map singular values into ~[0.7, 1.2] rather than to exactly 1
    assert ((ratio > 0.55) & (ratio < 1.3)).all(), ratio
    offdiag = u.mT @ out @ v - torch.diag_embed(d)
    assert offdiag.abs().max() < 0.05
    # eps -> 0 recovers Muon
    assert torch.allclose(soft_polar(C.float(), torch.tensor(1e-8)), newton_schulz(C.float()), atol=2e-3)


# ---------------------------------------------------------------------------- routing
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.proj = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)
        self.dwconv = nn.Conv1d(128, 128, 4, groups=128)
        self.conv = nn.Conv2d(16, 64, 3)
        self.experts = nn.Parameter(torch.randn(4, 64, 128) * 0.02)
        self.gate = nn.Linear(64, 4, bias=False)
        self.thin = nn.Linear(64, 8)
        self.lm_head = nn.Linear(128, 100, bias=False)


def test_routing():
    m = Toy()
    kinds = {n: k for g in build_param_groups(m) for n, k in zip(g["names"], [g["kind"]] * len(g["names"]))}
    assert kinds["embed.weight"] == FACTORED and kinds["lm_head.weight"] == FACTORED
    assert kinds["gate.weight"] == FACTORED                         # router
    assert kinds["proj.weight"] == SPECTRAL and kinds["conv.weight"] == SPECTRAL
    assert kinds["experts"] == SPECTRAL
    for n in ("proj.bias", "norm.weight", "norm.bias", "dwconv.weight", "thin.weight"):
        assert kinds[n] == ADAMW, n
    groups = build_param_groups(m, overrides={r"^proj\.weight$": "factored"}, lr_mult={r"norm": 3.0},
                                wd_overrides={r"^conv\.": 0.0})
    kinds = {n: (g["kind"], g["lr_mult"], g["weight_decay"]) for g in groups for n in g["names"]}
    assert kinds["proj.weight"][0] == FACTORED
    assert kinds["norm.weight"] == (ADAMW, 3.0, 0.0)
    assert kinds["conv.weight"] == (SPECTRAL, 1.0, 0.0) and kinds["proj.weight"][2] == 0.1


# ---------------------------------------------------------------------------- optimizer
def _ref_muon(W, grads, lr, beta, wd):
    m = torch.zeros_like(W)
    for g in grads:
        m = beta * m + (1 - beta) * g
        c = beta * m + (1 - beta) * g
        U = newton_schulz(c)
        W = W * (1 - lr * wd) - lr * 0.2 * math.sqrt(max(W.shape)) * U
    return W


def test_alpha0_is_muon():
    torch.manual_seed(0)
    W0 = torch.randn(64, 96, device=DEV) * 0.05
    grads = [torch.randn(64, 96, device=DEV) for _ in range(6)]
    p = nn.Parameter(W0.clone())
    opt = LHMuon([{"params": [p], "kind": "spectral"}], lr=0.02, alpha=0.0, norm_control="wd", weight_decay=0.1,
                 state_dtype="fp32", ns_dtype="fp32")
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_muon(W0, grads, 0.02, 0.95, 0.1)
    assert torch.allclose(p.data, ref, atol=1e-5), (p.data - ref).abs().max()


def test_grad_coef_and_skip():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(48, 48, device=DEV) * 0.05)
    q = nn.Parameter(p.data.clone())
    o1 = LHMuon([p], lr=0.01, alpha=0.0, state_dtype="fp32")
    o2 = LHMuon([q], lr=0.01, alpha=0.0, state_dtype="fp32")
    g = torch.randn(48, 48, device=DEV)
    p.grad, q.grad = g * 1024, g.clone()
    o1.step(grad_coef=1 / 1024)
    o2.step()
    assert torch.allclose(p.data, q.data, atol=1e-6)
    before = copy.deepcopy(o1.state_dict())
    w = p.data.clone()
    p.grad = torch.full_like(p, float("inf"))
    o1.step(check_finite=True)
    assert o1.last_step_skipped and o1._t == before["t"] and torch.equal(p.data, w)


def test_sphere_keeps_norm_and_zero_init_falls_back():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(64, 64, device=DEV) * 0.05)
    z = nn.Parameter(torch.zeros(64, 64, device=DEV))
    r0 = p.data.norm().item()
    opt = LHMuon([p, z], lr=0.05, alpha=1.0, total_steps=100, slow_every=2, norm_control="sphere")
    for _ in range(10):
        p.grad, z.grad = torch.randn_like(p), torch.randn_like(z)
        opt.step()
    assert abs(p.data.norm().item() - r0) < 1e-3 * r0
    assert opt.state[z]["sphere"] is False and z.data.norm() > 0


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16", "int8", "int4"])
@pytest.mark.parametrize("combine", ["sum", "separate"])
def test_trains_toy_regression(dtype, combine):
    """A linear map to recover; loss must drop well below the start with every format."""
    torch.manual_seed(0)
    A = torch.randn(64, 64, device=DEV) / 8
    net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(DEV)
    opt = LHMuon(build_param_groups(net, weight_decay=0.0), lr=0.01, alpha=0.5, total_steps=300, slow_every=2,
                 combine=combine, state_dtype=dtype, slow_dtype=dtype, soft_kappa=1.0, norm_control="wd")
    losses = []
    for _ in range(300):
        x = torch.randn(256, 64, device=DEV)
        loss = (net(x) - x @ A).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses)
    assert sum(losses[-20:]) / 20 < 0.2 * sum(losses[:5]) / 5, (losses[:5], losses[-5:])


def _mixed_model(dtype):
    torch.manual_seed(0)
    m = Toy().to(DEV)
    return m.to(dtype)


def _run(model, opt, steps, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        for p in model.parameters():
            p.grad = (torch.randn(p.shape, generator=g, device=DEV) * 1024).to(p.dtype)
        opt.step(grad_coef=1 / 1024)


OPT_KW = dict(lr=0.01, alpha=2.0, total_steps=100, slow_every=2, soft_kappa=1.0, state_dtype="int8", slow_dtype="int4",
              norm_control="sphere")


@needs_cuda
@pytest.mark.parametrize("combine", ["sum", "separate"])
@pytest.mark.parametrize("slow_master", ["device", "host"])
def test_offload_is_bitwise_identical(combine, slow_master):
    kw = dict(OPT_KW, combine=combine, slow_master=slow_master)
    a, b = _mixed_model(torch.float16), _mixed_model(torch.float16)
    oa = LHMuon(build_param_groups(a), **kw)
    ob = LHMuon(build_param_groups(b), offload=True, **kw)
    _run(a, oa, 7)
    _run(b, ob, 7)
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(pa, pb), n
    sb = ob.state_bytes()
    assert sb.get((SPECTRAL, "host"), 0) > 0


@pytest.mark.parametrize("offload", [False, True])
def test_resume_is_exact(offload):
    if offload and CUDA is None:
        pytest.skip("needs CUDA")
    kw = dict(OPT_KW, slow_master="host" if offload else "device", offload=offload)
    a = _mixed_model(torch.float16)
    oa = LHMuon(build_param_groups(a), **kw)
    _run(a, oa, 3, seed=0)
    ckpt = {"model": copy.deepcopy(a.state_dict()), "opt": oa.state_dict()}
    _run(a, oa, 4, seed=1)

    b = _mixed_model(torch.float16)
    b.load_state_dict(ckpt["model"])
    ob = LHMuon(build_param_groups(b), **kw)
    ob.load_state_dict(ckpt["opt"])
    _run(b, ob, 4, seed=1)
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(pa, pb), n

    other = LHMuon(build_param_groups(_mixed_model(torch.float16)), **dict(kw, state_dtype="int4"))
    with pytest.raises(ValueError):
        other.load_state_dict(ckpt["opt"])


def test_fp16_weights_small_updates_land():
    """An fp16 weight at 1.0 with a per-step update 50x below half its spacing must still move."""
    p = nn.Parameter(torch.ones(4096, device=DEV, dtype=torch.float16))
    opt = LHMuon([{"params": [p], "kind": "adamw"}], lr=1e-5, alpha=0.0, weight_decay=0.0)
    for _ in range(200):
        p.grad = torch.ones_like(p)
        opt.step()
    moved = 1.0 - p.data.float().mean().item()
    assert abs(moved - 200 * 1e-5) < 0.25 * 200 * 1e-5, moved


def test_lion_matches_reference():
    torch.manual_seed(0)
    W0 = torch.randn(32, 16, device=DEV)
    grads = [torch.randn(32, 16, device=DEV) for _ in range(5)]
    p = nn.Parameter(W0.clone())
    opt = LHMuon([{"params": [p], "kind": "lion"}], lr=1e-3, alpha=0.0, weight_decay=0.5)
    W, m = W0.clone(), torch.zeros_like(W0)
    for g in grads:
        p.grad = g.clone()
        opt.step()
        W = W * (1 - 1e-3 * 0.5) - 1e-3 * torch.sign(0.9 * m + 0.1 * g)
        m = 0.99 * m + 0.01 * g
    assert torch.allclose(p.data, W, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_newton_schulz_low_precision_tiny_inputs(dtype):
    """Momenta can have entries ~1e-6; they must be normalized before the cast to fp16/bf16."""
    torch.manual_seed(0)
    C = torch.randn(512, 1536, device=DEV) * 1e-6
    ref = newton_schulz(C.double(), dtype=torch.float64)
    out = newton_schulz(C, dtype=dtype)
    err = ((out.double() - ref).norm() / ref.norm()).item()
    assert err < {torch.float16: 0.01, torch.bfloat16: 0.05}[dtype], err


CHUNK_CASES = {
    "muon": dict(alpha=0.0),
    "lhmuon": dict(alpha=0.5, slow_every=2),
    "lhmuon host": dict(alpha=0.5, slow_every=2, slow_master="host"),
    "separate": dict(alpha=0.5, slow_every=2, combine="separate"),
    "soft": dict(alpha=0.5, slow_every=2, soft_kappa=1.0),
    "sphere": dict(alpha=0.5, slow_every=2, norm_control="sphere"),
}


@pytest.mark.parametrize("case", list(CHUNK_CASES))
def test_chunked_equals_unchunked(case):
    """Many small chunks must give the same result as one chunk per tensor. fp32 weights and state
    make the update deterministic (no stochastic rounding), so only summation order differs."""
    kw = dict(lr=0.01, total_steps=100, slow_horizon=16, state_dtype="fp32", slow_dtype="fp32", ns_dtype="fp32",
              **CHUNK_CASES[case])
    a, b = Toy().to(DEV), Toy().to(DEV)
    b.load_state_dict(a.state_dict())
    oa = LHMuon(build_param_groups(a), chunk_elements=1 << 30, **kw)
    ob = LHMuon(build_param_groups(b), chunk_elements=512, **kw)      # every tensor split into many chunks
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(7):
        for pa, pb in zip(a.parameters(), b.parameters()):
            grad = torch.randn(pa.shape, generator=g, device=DEV)
            pa.grad, pb.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=2e-5, rtol=1e-5), (n, (pa - pb).abs().max().item())


@pytest.mark.parametrize("kind", ["adamw", "lion"])
def test_chunked_elementwise_kinds(kind):
    a, b = Toy().to(DEV), Toy().to(DEV)
    b.load_state_dict(a.state_dict())
    ga, gb = build_param_groups(a), build_param_groups(b)
    for g in ga + gb:
        g["kind"] = kind
    oa = LHMuon(ga, lr=1e-3, alpha=0.0, chunk_elements=1 << 30)
    ob = LHMuon(gb, lr=1e-3, alpha=0.0, chunk_elements=512)
    gen = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(5):
        for pa, pb in zip(a.parameters(), b.parameters()):
            grad = torch.randn(pa.shape, generator=gen, device=DEV)
            pa.grad, pb.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), n
