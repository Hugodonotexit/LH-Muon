"""
optimizer.py -- LHMuon: long-horizon-momentum spectral descent.

Per spectral matrix W (m x n), with unscaled gradient g:

    m_f <- beta m_f + (1 - beta) g                           fast EMA (Muon's momentum), beta = 0.95
    c_f  = beta m_f + (1 - beta) g                           Nesterov lookahead
    every K steps:
        m_s <- lerp(m_s, m_f, 1 / min(H / K, n_refresh))     slow EMA of horizon H steps, sampled from m_f;
                                                             starts as a running mean so it fills unbiased
    c    = c_f + alpha_t m_s                                 combine="sum" (default)
    U    = polar(c)            or  soft_polar(c, eps_t)      soft_kappa > 0: noise-floored polar map
         (combine="separate": U = polar(c_f) + alpha_t S, S = polar(m_s) cached at each refresh)
    W   <- W - lr (rms_scale sqrt(max(m, n)) U)              RMS-matched: one LR for every kind
    W   <- W (1 - lr wd) before the step                    norm_control="wd" (default)
         (norm_control="sphere": W <- W * r / ||W||_F after it, r = ||W||_F at init, or
          sphere_rms * sqrt(m n). This FREEZES each matrix's scale: only safe where norm layers absorb
          it. On a 2-layer MLP without norms it stalls at loss 0.25 where wd reaches 0.001.)

alpha_t ramps linearly from 0 to alpha over the first `alpha_warmup` steps (default: H).
H defaults to min(total_steps / 10, 10000), K to min(20, H / 32).

eps_t (soft_kappa > 0) is kappa times the Marchenko-Pastur edge of the noise in c:
    eps_t = kappa * s_c * (sqrt(m) + sqrt(n)),
where s_c^2 is the per-entry noise variance of c, derived from a per-tensor running estimate of the
per-step gradient noise, E[(g - m_f)^2] = s^2 * 2 / (1 + beta). Directions whose singular value sits
at the noise edge are shrunk by 1/sqrt(2); well above it the map is Muon's.

Embeddings / heads / routers get factored Adam (int8 momentum + Adafactor second moment);
1D and small params get fp32 AdamW. See routing.py. A group with kind="lion" runs fp32 Lion (a
baseline: same precision handling, so optimizer comparisons don't measure rounding).

Precision: every update is computed in fp32. fp16/bf16 weights are written back with stochastic
rounding (so updates far below the weight's spacing still land in expectation). The caller passes
`grad_coef` (1 / loss_scale, times a clip factor), which is applied to an fp32 copy of the gradient,
so a loss-scaled fp16 gradient is never unscaled in fp16.
"""

import copy
import math

import torch

from . import polar as P
from . import quant as Q
from .routing import ADAMW, FACTORED, KINDS, LION, SPECTRAL, matrix_shape, matrix_view


class LHMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 2e-4, *,
                 total_steps: int = None,
                 momentum: float = 0.95, nesterov: bool = True,
                 alpha: float = 2.0, slow_horizon: int = None, slow_every: int = None,
                 alpha_warmup: int = None, combine: str = "sum",
                 norm_control: str = "wd", sphere_rms: float = None, weight_decay: float = 0.1, rms_scale: float = 0.2,
                 ns_steps: int = 5, ns_dtype: str = "auto",
                 soft_kappa: float = 0.0, noise_beta: float = 0.99,
                 adam_betas=(0.9, 0.95), adam_eps: float = 1e-8, factored_clip: float = 1.0,
                 state_dtype: str = "int8", slow_dtype: str = "int8", slow_master: str = "device",
                 offload: bool = False, stochastic_weights: bool = True, min_dim: int = 32, seed: int = 0,
                 chunk_elements: int = 1 << 22):
        if combine not in ("sum", "separate"):
            raise ValueError(f"combine must be 'sum' or 'separate', not {combine!r}")
        if norm_control not in ("sphere", "wd", "none"):
            raise ValueError(f"norm_control must be 'sphere', 'wd' or 'none', not {norm_control!r}")
        if slow_master not in ("device", "host"):
            raise ValueError(f"slow_master must be 'device' or 'host', not {slow_master!r}")
        for f in (state_dtype, slow_dtype):
            if f not in Q.FORMATS:
                raise ValueError(f"state format {f!r} is not one of {Q.FORMATS}")
        if alpha > 0 and slow_horizon is None:
            if total_steps is None:
                raise ValueError("alpha > 0 needs slow_horizon, or total_steps to derive it from "
                                 "(slow_horizon = min(total_steps / 10, 10000))")
            slow_horizon = max(32, min(total_steps // 10, 10000))
        if slow_every is None:                     # >= 32 samples per horizon: the slow buffer must be an
            slow_every = max(1, min(20, (slow_horizon or 640) // 32))   # average, not a stale copy of m_f
        if alpha > 0 and slow_horizon < 8 * slow_every:
            raise ValueError(f"slow_horizon {slow_horizon} holds fewer than 8 samples at slow_every {slow_every}")
        self.alpha = float(alpha)
        self.slow_horizon = slow_horizon
        self.slow_every = int(slow_every)
        self.alpha_warmup = alpha_warmup if alpha_warmup is not None else slow_horizon
        self.combine = combine
        self.norm_control = norm_control
        self.sphere_rms = sphere_rms
        self.rms_scale = rms_scale
        self.ns_steps = ns_steps
        self.ns_dtype = ns_dtype
        self.soft_kappa = soft_kappa
        self.noise_beta = noise_beta
        self.factored_clip = factored_clip
        self.state_dtype = state_dtype
        self.slow_dtype = slow_dtype
        self.slow_master = slow_master
        self.offload = offload
        self.stochastic_weights = stochastic_weights
        self.min_dim = min_dim
        self.chunk_elements = int(chunk_elements)   # element-wise work runs on chunks of this many elements
        self.seed = seed
        self.alpha_mult = 1.0      # a trainer may scale alpha per step, e.g. with the LR during the decay phase
        self._t = 0
        self._gens = {}
        self._streams = {}
        self._stage = {}
        self.last_step_skipped = False
        defaults = dict(lr=lr, lr_mult=1.0, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        adam_betas=tuple(adam_betas), adam_eps=adam_eps, lion_betas=(0.9, 0.99), kind=None, names=None)
        super().__init__(params, defaults)
        for g in self.param_groups:
            if g["kind"] is not None and g["kind"] not in KINDS:
                raise ValueError(f"group kind {g['kind']!r} is not one of {KINDS}")
            if g["names"] is not None and len(g["names"]) != len(g["params"]):
                raise ValueError("a group's 'names' must line up with its 'params'")

    # ------------------------------------------------------------------ schedule
    def alpha_at(self, t: int) -> float:
        if self.alpha <= 0:
            return 0.0
        return self.alpha * self.alpha_mult * min(1.0, t / max(1, self.alpha_warmup))

    # ------------------------------------------------------------------ state
    def _kind_view(self, p, group, i):
        name = group["names"][i] if group["names"] is not None else ""
        kind = group["kind"]
        if kind is None:
            ok = p.dim() >= 2 and min(matrix_shape(matrix_view(name, p.shape), p.shape)[-2:]) >= self.min_dim
            kind = SPECTRAL if ok else ADAMW
        if kind == SPECTRAL and p.dim() < 2:
            kind = ADAMW
        view = matrix_view(name, p.shape) if p.dim() >= 2 else None
        return kind, view

    def _put(self, st, name, fmt, shape, device):
        q, s = Q.zeros(shape, fmt, device)
        if self.offload:
            q, s = q.cpu().pin_memory(), s.cpu().pin_memory()
        st[name + ".q"], st[name + ".s"] = q, s

    @torch.no_grad()
    def _init_state(self, p, group, i):
        st = self.state[p]
        kind, view = self._kind_view(p, group, i)
        st["kind"] = kind
        dev = p.device
        if self.offload and dev.type != "cuda":
            raise ValueError("offload=True needs CUDA parameters")
        if kind == SPECTRAL:
            mshape = matrix_shape(view, p.shape)
            st["view"], st["mshape"] = view, tuple(mshape)
            self._put(st, "mf", self.state_dtype, mshape, dev)
            if self.alpha > 0:
                st["n_slow"] = 0                                    # refreshes this tensor has seen
                if self.slow_master == "host":
                    st["ms_host"] = torch.zeros(mshape, dtype=torch.float32).pin_memory() if dev.type == "cuda" \
                        else torch.zeros(mshape, dtype=torch.float32)
                if self.slow_master == "device" or self.combine == "sum":
                    self._put(st, "ms", self.slow_dtype, mshape, dev)  # master (device) or snapshot (host)
                if self.combine == "separate":
                    self._put(st, "S", self.slow_dtype, mshape, dev)
            if self.soft_kappa > 0:
                st["s2"] = torch.zeros((), device=dev)
                st["s2_init"] = False
            if self.norm_control == "sphere":
                r = p.detach().float().reshape(mshape).norm(dim=(-2, -1), keepdim=True)
                if self.sphere_rms is not None:
                    r = torch.full_like(r, self.sphere_rms * math.sqrt(mshape[-1] * mshape[-2]))
                tiny = 1e-6 * math.sqrt(mshape[-1] * mshape[-2])
                st["sphere"] = bool((r > tiny).all())               # zero-init matrices fall back to wd
                st["radius"] = r
        elif kind == FACTORED:
            mshape = (p.shape[0], p.numel() // p.shape[0])
            st["mshape"] = mshape
            self._put(st, "mf", self.state_dtype, mshape, dev)
            st["vr"] = torch.zeros(mshape[0], device=dev)
            st["vc"] = torch.zeros(mshape[1], device=dev)
            st["t"] = 0
        elif kind == LION:
            st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        else:
            st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            st["t"] = 0

    def _fmt(self, name):
        return self.state_dtype if name == "mf" else self.slow_dtype

    # ------------------------------------------------------------------ offload streaming
    def _stream(self, dev, which):
        key = (str(dev), which)
        if key not in self._streams:
            self._streams[key] = torch.cuda.Stream(dev)
        return self._streams[key]

    @staticmethod
    def _container_keys(st):
        return [k for k in st if k.endswith(".q") or k.endswith(".s")]

    def _fetch(self, p):
        st = self.state[p]
        keys = self._container_keys(st)
        if not self.offload or not keys:
            return {k: st[k] for k in keys}, None
        h2d = self._stream(p.device, "h2d")
        h2d.wait_stream(self._stream(p.device, "d2h"))             # last step's write-back of these buffers
        with torch.cuda.stream(h2d):
            out = {k: st[k].to(p.device, non_blocking=True) for k in keys}
            ev = torch.cuda.Event()
            ev.record(h2d)
        return out, ev

    def _ready(self, fetched, dev):
        out, ev = fetched
        if ev is not None:
            cur = torch.cuda.current_stream(dev)
            cur.wait_event(ev)
            for t in out.values():
                t.record_stream(cur)
        return out

    def _store(self, p, new):
        st = self.state[p]
        if not self.offload:
            st.update(new)
            return
        cur = torch.cuda.current_stream(p.device)
        d2h = self._stream(p.device, "d2h")
        d2h.wait_stream(cur)
        with torch.cuda.stream(d2h):
            for k, v in new.items():
                st[k].copy_(v, non_blocking=True)
                v.record_stream(d2h)

    def _sync(self):
        for (dev, which), s in self._streams.items():
            s.synchronize()

    # ------------------------------------------------------------------ helpers
    def _generator(self, device, generator=None):
        if generator is not None:
            return generator
        key = str(device)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gens[key] = g
        return self._gens[key]

    # ------------------------------------------------------------------ chunking
    # Every element-wise stage runs over flat chunks of about `chunk_elements` elements, so the
    # fp32 temporaries of one step (gradient copy, decoded momentum, update, weight copy, the
    # intermediates of stochastic rounding) exist for one chunk at a time instead of the whole
    # tensor. Chunks are whole rows and whole state blocks, so row statistics and the block-scaled
    # containers come out as if the tensor had been processed in one piece.

    def _chunks(self, numel, row_len):
        unit = math.lcm(max(1, row_len), 256)                   # 256 is a multiple of every block size
        size = max(unit, (self.chunk_elements // unit) * unit)
        for a in range(0, numel, size):
            yield a, min(numel, a + size)

    @staticmethod
    def _flat(p):
        """(flat fp16/fp32 view of the weights, flat view of the gradient); both must be contiguous."""
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
        g = p.grad if p.grad.is_contiguous() else p.grad.contiguous()
        return p.data.view(-1), g.view(-1)

    def _write(self, dst, W, gen):
        """dst (a flat slice of the parameter) <- W (fp32), with stochastic rounding for fp16/bf16."""
        if dst.dtype in (torch.float16, torch.bfloat16) and self.stochastic_weights:
            dst.copy_(Q.stochastic_round(W, dst.dtype, gen))
        else:
            dst.copy_(W)

    def _staging(self, device, n):
        """A pinned fp32 host buffer of >= n elements, reused across tensors (one per device)."""
        key = str(device)
        buf = self._stage.get(key)
        if buf is None or buf.numel() < n:
            buf = torch.empty(n, dtype=torch.float32, pin_memory=device.type == "cuda")
            self._stage[key] = buf
        return buf[:n]

    def _new_like(self, bufs, name):
        return torch.empty_like(bufs[name + ".q"]), torch.empty_like(bufs[name + ".s"])

    # ------------------------------------------------------------------ updates
    def _spectral(self, p, group, bufs, coef, gen, alpha_t, refresh):
        st = self.state[p]
        mshape = st["mshape"]
        m_, n_ = mshape[-2:]
        numel = p.numel()
        lr = group["lr"] * group["lr_mult"]
        beta, nesterov = group["momentum"], group["nesterov"]
        ns_dtype = P.resolve_dtype(self.ns_dtype, p.device)
        fmt_f, fmt_s = self.state_dtype, self.slow_dtype
        flat_w, flat_g = self._flat(p)
        use_ms = alpha_t > 0 and self.combine == "sum"
        mf_q, mf_s = self._new_like(bufs, "mf")
        new = {"mf.q": mf_q, "mf.s": mf_s}
        slow_w = None
        if refresh:
            st["n_slow"] += 1
            slow_w = 1.0 / min(self.slow_horizon / self.slow_every, st["n_slow"])
            if self.slow_master == "device":
                ms_q, ms_s = self._new_like(bufs, "ms")
                new["ms.q"], new["ms.s"] = ms_q, ms_s

        # pass 1, per chunk: momentum update, the direction c, the slow-buffer refresh
        c = torch.empty(numel, device=p.device, dtype=torch.float32)
        host_refresh = refresh and self.slow_master == "host"
        mf_new = torch.empty(numel, device=p.device, dtype=torch.float32) if host_refresh else None
        noise = torch.zeros((), device=p.device) if self.soft_kappa > 0 else None
        for a, b in self._chunks(numel, n_):
            g = flat_g[a:b].float()
            if coef != 1.0:
                g.mul_(coef)
            mf = Q.decode_range(bufs["mf.q"], bufs["mf.s"], fmt_f, a, b - a)
            if noise is not None:
                noise += (g - mf).square().sum()
            mf.lerp_(g, 1 - beta)
            ca = c[a:b]
            if nesterov:
                torch.lerp(mf, g, 1 - beta, out=ca)
            else:
                ca.copy_(mf)
            del g
            if use_ms:
                ca.add_(Q.decode_range(bufs["ms.q"], bufs["ms.s"], fmt_s, a, b - a), alpha=alpha_t)
            Q.encode_into(mf_q, mf_s, fmt_f, mf, a, gen)
            if refresh:
                if self.slow_master == "device":
                    ms = Q.decode_range(bufs["ms.q"], bufs["ms.s"], fmt_s, a, b - a).lerp_(mf, slow_w)
                    Q.encode_into(ms_q, ms_s, fmt_s, ms, a, gen)          # a master: stochastic rounding
                else:
                    mf_new[a:b].copy_(mf)                                 # to CPU in one pinned transfer below
            del mf

        if noise is not None:
            r2 = noise / numel
            if st["s2_init"]:
                st["s2"].lerp_(r2, 1 - self.noise_beta)
            else:
                st["s2"], st["s2_init"] = r2, True

        # orthogonalize the whole matrix (Newton-Schulz needs it in one piece)
        c = c.view(mshape)
        if self.soft_kappa > 0:
            if nesterov:
                var_c = beta ** 4 * (1 - beta) / (1 + beta) + ((1 - beta) * (1 + beta)) ** 2
            else:
                var_c = (1 - beta) / (1 + beta)
            if use_ms:
                var_c += alpha_t ** 2 / (2 * self.slow_horizon - 1)   # (1-b_s)/(1+b_s), b_s = 1 - 1/H
            s2 = st["s2"] * ((1 + beta) / 2)                           # E[(g - m_f)^2] -> per-step noise
            eps = self.soft_kappa * torch.sqrt(s2 * var_c) * (math.sqrt(m_) + math.sqrt(n_))
            U = P.soft_polar(c, eps, self.ns_steps, ns_dtype)
        else:
            U = P.newton_schulz(c, self.ns_steps, ns_dtype)
        del c
        U = U.reshape(-1)

        # the slow buffer's snapshot (host master) or polar factor (combine="separate") at a refresh
        if refresh and (self.slow_master == "host" or self.combine == "separate"):
            if self.slow_master == "device":
                ms_full = Q.decode(new["ms.q"], new["ms.s"], fmt_s, mshape)
            else:
                # The fp32 master is updated on the CPU. Everything crosses PCIe as fp32 through pinned
                # memory: measured on a V100 host, that is ~20x faster than unpinned copies, and cheaper
                # than bf16 transfers because this CPU converts bf16 <-> fp32 slowly.
                stage = self._staging(p.device, numel)
                stage.copy_(mf_new, non_blocking=True)
                del mf_new
                if p.device.type == "cuda":
                    torch.cuda.current_stream(p.device).synchronize()
                st["ms_host"].view(-1).lerp_(stage, slow_w)
                ms_full = st["ms_host"].to(p.device, non_blocking=True)   # ms_host is pinned
                if self.combine == "sum":
                    new["ms.q"], new["ms.s"] = Q.encode(ms_full, fmt_s, gen, stochastic=False)   # snapshot: nearest
            if self.combine == "separate":
                new["S.q"], new["S.s"] = Q.encode(P.newton_schulz(ms_full, self.ns_steps, ns_dtype), fmt_s,
                                                  gen, stochastic=False)
            del ms_full

        # pass 2 (and 3 for the sphere), per chunk: the weight update
        wd = group["weight_decay"]
        sphere = self.norm_control == "sphere" and st["sphere"]
        decay = 1 - lr * wd if (self.norm_control == "wd" or (self.norm_control == "sphere" and not sphere)) and wd else 1.0
        step = -lr * self.rms_scale * math.sqrt(max(m_, n_))
        use_S = alpha_t > 0 and self.combine == "separate"

        def updated(a, b, copy=False):
            # copy=True for the sphere's measuring pass: for fp32 weights .float() is the weight itself
            W = flat_w[a:b].to(torch.float32, copy=copy)
            if decay != 1.0:
                W.mul_(decay)
            u = U[a:b]
            if use_S:
                u = u + alpha_t * Q.decode_range(bufs["S.q"], bufs["S.s"], fmt_s, a, b - a)
            return W.add_(u, alpha=step)

        factor = None
        if sphere:                                                  # norm of each updated matrix first
            rows = torch.zeros(numel // n_, device=p.device)
            for a, b in self._chunks(numel, n_):
                rows[a // n_:b // n_] = updated(a, b, copy=True).view(-1, n_).square().sum(1)
            norms = rows.view(-1, m_).sum(1).sqrt()                  # one per matrix (per expert if batched)
            factor = (st["radius"].reshape(-1) / norms.clamp_min(1e-12)).repeat_interleave(m_)   # one per row
        for a, b in self._chunks(numel, n_):
            W = updated(a, b)
            if factor is not None:
                W.view(-1, n_).mul_(factor[a // n_:b // n_, None])
            self._write(flat_w[a:b], W, gen)
        return new

    def _factored(self, p, group, bufs, coef, gen):
        st = self.state[p]
        R, C = st["mshape"]
        numel = p.numel()
        lr = group["lr"] * group["lr_mult"]
        b1, b2 = group["adam_betas"]
        st["t"] += 1
        t = st["t"]
        fmt = self.state_dtype
        flat_w, flat_g = self._flat(p)

        # pass 1: the factored second moment (row means are per chunk, column means are summed)
        vr_new = torch.empty(R, device=p.device)
        col = torch.zeros(C, device=p.device)
        for a, b in self._chunks(numel, C):
            g2 = flat_g[a:b].float()
            if coef != 1.0:
                g2.mul_(coef)
            g2 = g2.square_().add_(1e-30).view(-1, C)
            vr_new[a // C:b // C] = g2.mean(dim=1)
            col += g2.sum(dim=0)
            del g2
        st["vr"].lerp_(vr_new, 1 - b2)
        st["vc"].lerp_(col / R, 1 - b2)
        norm = st["vr"].mean().clamp_min(1e-30) * (1 - b2 ** t)

        def momentum_and_update(a, b):
            g = flat_g[a:b].float()
            if coef != 1.0:
                g.mul_(coef)
            m = Q.decode_range(bufs["mf.q"], bufs["mf.s"], fmt, a, b - a).lerp_(g, 1 - b1)
            del g
            v = torch.outer(st["vr"][a // C:b // C], st["vc"]).div_(norm).reshape(-1)
            upd = (m / (1 - b1 ** t)).div_(v.sqrt_().add_(group["adam_eps"]))
            return m, upd

        # pass 2 (only with clipping): the RMS of the whole update
        clip = None
        if self.factored_clip:
            sq = torch.zeros((), device=p.device)
            for a, b in self._chunks(numel, C):
                _, upd = momentum_and_update(a, b)
                sq += upd.square().sum()
            clip = (torch.sqrt(sq / numel) / self.factored_clip).clamp_min(1.0)

        # pass 3: apply
        mf_q, mf_s = self._new_like(bufs, "mf")
        for a, b in self._chunks(numel, C):
            m, upd = momentum_and_update(a, b)
            if clip is not None:
                upd.div_(clip)
            W = flat_w[a:b].float()
            if group["weight_decay"]:
                W.mul_(1 - lr * group["weight_decay"])
            W.add_(upd, alpha=-lr)
            del upd
            self._write(flat_w[a:b], W, gen)
            Q.encode_into(mf_q, mf_s, fmt, m, a, gen)
        return {"mf.q": mf_q, "mf.s": mf_s}

    def _adamw(self, p, group, coef, gen):
        st = self.state[p]
        lr = group["lr"] * group["lr_mult"]
        b1, b2 = group["adam_betas"]
        st["t"] += 1
        t = st["t"]
        flat_w, flat_g = self._flat(p)
        ea, es = st["exp_avg"].view(-1), st["exp_avg_sq"].view(-1)
        for a, b in self._chunks(p.numel(), 1):
            g = flat_g[a:b].float()
            if coef != 1.0:
                g.mul_(coef)
            ea[a:b].lerp_(g, 1 - b1)
            es[a:b].mul_(b2).addcmul_(g, g, value=1 - b2)
            del g
            denom = (es[a:b] / (1 - b2 ** t)).sqrt_().add_(group["adam_eps"])
            W = flat_w[a:b].float()
            if group["weight_decay"]:
                W.mul_(1 - lr * group["weight_decay"])
            W.addcdiv_(ea[a:b], denom, value=-lr / (1 - b1 ** t))
            self._write(flat_w[a:b], W, gen)

    def _lion(self, p, group, coef, gen):
        """Chen et al. 2023: W <- W(1 - lr wd) - lr sign(b1 m + (1-b1) g);  m <- b2 m + (1-b2) g."""
        st = self.state[p]
        lr = group["lr"] * group["lr_mult"]
        b1, b2 = group["lion_betas"]
        flat_w, flat_g = self._flat(p)
        ea = st["exp_avg"].view(-1)
        for a, b in self._chunks(p.numel(), 1):
            g = flat_g[a:b].float()
            if coef != 1.0:
                g.mul_(coef)
            W = flat_w[a:b].float()
            if group["weight_decay"]:
                W.mul_(1 - lr * group["weight_decay"])
            W.add_(torch.lerp(ea[a:b], g, 1 - b1).sign_(), alpha=-lr)
            ea[a:b].lerp_(g, 1 - b2)
            self._write(flat_w[a:b], W, gen)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, closure=None, grad_coef: float = 1.0, generator: torch.Generator = None,
             check_finite: bool = False):
        """One optimizer step on p.grad * grad_coef.

        grad_coef:    1 / loss_scale (times a clip factor); applied in fp32.
        generator:    for stochastic rounding; pass one seeded identically on every rank if ranks must
                      stay bit-identical. Default: one per device owned by the optimizer (saved in
                      state_dict), seeded with `seed`.
        check_finite: if True and any gradient is inf/NaN, nothing is touched -- not the momenta, not
                      the step count -- and self.last_step_skipped is set."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        items = [(p, group, i) for group in self.param_groups for i, p in enumerate(group["params"])
                 if p.grad is not None]
        self.last_step_skipped = False
        if check_finite and items:
            per_dev = {}
            for p, _, _ in items:
                per_dev.setdefault(p.grad.device, []).append(torch.isfinite(p.grad).all())
            if not all(bool(torch.stack(v).all()) for v in per_dev.values()):
                self.last_step_skipped = True
                return loss

        for p, group, i in items:
            if not self.state[p]:
                self._init_state(p, group, i)
        self._t += 1
        alpha_t = self.alpha_at(self._t)
        refresh = self.alpha > 0 and self._t % self.slow_every == 0

        streamed = [(p, group) for p, group, _ in items if self.state[p]["kind"] in (SPECTRAL, FACTORED)]
        pending = self._fetch(streamed[0][0]) if streamed else None
        for k, (p, group) in enumerate(streamed):
            bufs = self._ready(pending, p.device)
            if k + 1 < len(streamed):
                pending = self._fetch(streamed[k + 1][0])            # overlaps with this tensor's update
            gen = self._generator(p.device, generator)
            if self.state[p]["kind"] == SPECTRAL:
                new = self._spectral(p, group, bufs, grad_coef, gen, alpha_t, refresh)
            else:
                new = self._factored(p, group, bufs, grad_coef, gen)
            del bufs
            self._store(p, new)
        for p, group, _ in items:
            kind = self.state[p]["kind"]
            if kind == ADAMW:
                self._adamw(p, group, grad_coef, self._generator(p.device, generator))
            elif kind == LION:
                self._lion(p, group, grad_coef, self._generator(p.device, generator))
        return loss

    # ------------------------------------------------------------------ checkpointing
    def _all_params(self):
        return [(p, g, i) for g in self.param_groups for i, p in enumerate(g["params"])]

    def state_dict(self):
        """State keyed by parameter position (group order, then order within the group), with every
        tensor copied to CPU. Buffers keep their storage format (int8 stays int8)."""
        self._sync()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        index = {id(p): n for n, (p, _, _) in enumerate(self._all_params())}
        state = {}
        for p, st in self.state.items():
            state[index[id(p)]] = {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else copy.deepcopy(v))
                                   for k, v in st.items()}
        groups = [{**{k: v for k, v in g.items() if k != "params"}, "n_params": len(g["params"])}
                  for g in self.param_groups]
        return {"state": state, "param_groups": groups, "t": self._t,
                "generators": {k: g.get_state() for k, g in self._gens.items()},
                "config": {"state_dtype": self.state_dtype, "slow_dtype": self.slow_dtype,
                           "combine": self.combine, "slow_master": self.slow_master}}

    @torch.no_grad()
    def load_state_dict(self, sd):
        """Copies INTO freshly allocated buffers, so placement (device / pinned host) and storage
        format follow the current settings; a checkpoint written with a different state_dtype or
        combine mode is refused rather than silently reinterpreted."""
        if len(sd["param_groups"]) != len(self.param_groups):
            raise ValueError("checkpoint has a different number of param groups")
        for g, sg in zip(self.param_groups, sd["param_groups"]):
            if sg["n_params"] != len(g["params"]):
                raise ValueError("checkpoint param group sizes differ from this optimizer's")
            for k, v in sg.items():
                if k not in ("params", "names", "n_params"):
                    g[k] = v
        self._t = sd["t"]
        for n, (p, group, i) in enumerate(self._all_params()):
            saved = sd["state"].get(n)
            if saved is None:
                continue
            self.state[p].clear()
            self._init_state(p, group, i)
            st = self.state[p]
            for k, v in saved.items():
                if torch.is_tensor(v) and torch.is_tensor(st.get(k)):
                    if st[k].shape != v.shape or st[k].dtype != v.dtype:
                        raise ValueError(f"state {k!r} is {tuple(v.shape)}/{v.dtype} in the checkpoint but "
                                         f"{tuple(st[k].shape)}/{st[k].dtype} here (saved with {sd.get('config')})")
                    st[k].copy_(v)
                elif torch.is_tensor(v):
                    raise ValueError(f"checkpoint has state {k!r} that the current settings do not use "
                                     f"(saved with {sd.get('config')})")
                else:
                    st[k] = v
        for key, s in sd.get("generators", {}).items():
            self._generator(torch.device(key)).set_state(s)

    # ------------------------------------------------------------------ introspection
    def state_bytes(self):
        """{(kind, 'device'|'host'): bytes} of optimizer state, plus parameter counts per kind."""
        out = {}
        for p, st in self.state.items():
            kind = st["kind"]
            out.setdefault((kind, "params"), 0)
            out[(kind, "params")] += p.numel()
            for k, v in st.items():
                if torch.is_tensor(v):
                    where = "host" if v.device.type == "cpu" and p.device.type != "cpu" else "device"
                    out.setdefault((kind, where), 0)
                    out[(kind, where)] += v.numel() * v.element_size()
        return out
