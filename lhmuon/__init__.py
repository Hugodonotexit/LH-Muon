"""LH-Muon: Muon with a long-horizon slow momentum, norm control and an optional noise-floored polar map,
with block-scaled int8/int4 state, fp16-safe updates and optional CPU offload of the state."""

from .optimizer import LHMuon
from .routing import build_param_groups, classify, SPECTRAL, FACTORED, ADAMW
from .polar import newton_schulz, soft_polar
from . import quant

__all__ = ["LHMuon", "build_param_groups", "classify", "newton_schulz", "soft_polar", "quant",
           "SPECTRAL", "FACTORED", "ADAMW"]
