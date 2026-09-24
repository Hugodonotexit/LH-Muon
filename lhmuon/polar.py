"""
polar.py -- orthogonalization: Muon's Newton-Schulz polar factor and the noise-floored "soft" polar.

Both work on (..., m, n) tensors, so a stack of MoE experts (E, m, n) is orthogonalized per expert
in one batched call.

newton_schulz(C)            ~ U V^T                         (Muon; singular values -> ~1)
soft_polar(C, eps)          ~ U diag(s / sqrt(s^2 + eps^2)) V^T

soft_polar uses the identity polar([C; eps*I]) = [C (C^T C + eps^2 I)^(-1/2); ...]: the top block
of the polar factor of C stacked on eps*I is exactly the soft map, so the same Newton-Schulz
iteration computes it with Muon's numerics. Cost is Muon's times (m + n) / m with n <= m, i.e. 2x
for square matrices, 1.25x for a 4:1 MLP matrix.
"""

import torch

# Jordan et al.'s quintic coefficients (modded-nanogpt). They do not converge to the exact polar
# factor -- singular values land in roughly [0.7, 1.2] -- which is what Muon is tuned with.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def resolve_dtype(ns_dtype: str, device: torch.device) -> torch.dtype:
    """'auto' -> bf16 on sm_80+ (Ampere/Hopper), fp32 elsewhere (V100 has no bf16, and an
    earlier fp16 implementation was observed to underflow on real LM gradients; fp16 is available as
    an explicit choice but unverified on real momenta)."""
    if ns_dtype == "auto":
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
            return torch.bfloat16
        return torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[ns_dtype]


def _iterate(X: torch.Tensor, steps: int) -> torch.Tensor:
    """X is WIDE (rows <= cols) and already normalised to Frobenius norm <= 1."""
    a, b, c = NS_COEFFS
    for _ in range(steps):
        A = X @ X.mT
        if A.dim() == 2:
            B = torch.addmm(A, A, A, beta=b, alpha=c)
        elif A.dim() == 3:
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        else:
            B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def newton_schulz(C: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Approximate polar factor of C (..., m, n), returned in fp32.

    The Frobenius normalization is done in fp32 (fp64 for fp64 input) BEFORE casting to `dtype`:
    raw momenta can have entries ~1e-6, which flush to zero if cast to fp16 first. On real training
    momenta, casting first gave up to 24% error vs fp64; normalizing first gives <= 0.5%."""
    X = C.to(torch.promote_types(C.dtype, torch.float32))
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)).to(dtype)
    X = _iterate(X, steps)
    if tall:
        X = X.mT
    return X.float()


def soft_polar(C: torch.Tensor, eps: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Approximate C (C^T C + eps^2 I)^(-1/2), returned in fp32. `eps` is a 0-d tensor (or one
    shared by every matrix in the batch) in the same units as C's singular values."""
    wide = C.size(-2) < C.size(-1)
    X = C.float().mT if wide else C.float()                          # tall: (M, N), N <= M
    M, N = X.shape[-2:]
    eye = torch.eye(N, device=X.device, dtype=torch.float32).expand(*X.shape[:-2], N, N)
    X = torch.cat([X, eps.float().reshape(*([1] * (X.dim() - 2)), 1, 1) * eye], dim=-2)   # (M+N, N)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = _iterate(X.to(dtype).mT, steps).mT[..., :M, :]               # iterate in the wide orientation
    return (X.mT if wide else X).float()
