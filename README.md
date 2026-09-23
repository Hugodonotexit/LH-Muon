# LH-Muon

**Muon with a long-horizon slow momentum.** It adds an AdEMAMix-style slow EMA to the direction
Muon orthogonalizes, with an optional noise-floored polar map and optional norm control. It is
packaged with a memory-lean state container (block-scaled int8/int4 optimizer state, CPU
offload) and fp16-safe updates. It is a drop-in `torch.optim.Optimizer`, and it also includes
AdamW, Lion and factored-Adam update rules, so every baseline runs through the same code path.

> **Result: in the regime tested, LH-Muon does not beat Muon.** On a 47M-parameter GPT trained on
> 131M tokens, every slow-momentum setting was worse than plain Muon. The loss got steadily worse
> as the slow term's weight α grew. The best variant (α = 0.25) is **+0.018 ± 0.002 nats** worse
> than Muon, paired over 2 seeds, where seed noise is ≈ 0.005. The default (α = 2) is +0.104
> worse. Details and all numbers are [below](#results). The repository is published as a
> documented negative result plus reusable infrastructure: the quantized/offloaded state, fp16
> stochastic-rounding updates, soft polar map, and a paired-seed optimizer benchmark harness.

## Contents

```
lhmuon/
  optimizer.py   LHMuon: spectral (LH-Muon / Muon), factored Adam, AdamW and Lion update rules
  polar.py       Newton–Schulz polar factor; soft_polar = C (CᵀC + ε²I)^(-1/2) via augmented NS
  quant.py       block-scaled state container (fp32/bf16/fp16/int8/int4) + stochastic rounding
  routing.py     which parameter gets which rule; build_param_groups()
examples/train_lm.py        small GPT on tokenized uint16 shards; adamw / lion / muon / lhmuon arms
scripts/run_suite.py        the benchmark: LR sweep → ablations → seeds → loss-vs-tokens frontier
scripts/analyze_suite.py    tables + figures from a suite directory
scripts/fit_multiplier.py   token multiplier from fully decayed endpoints (L = E + B·D^−β fit)
tests/test_lhmuon.py        36 unit tests
results/suite/              the published run: report, figures, per-run logs and final evals
```

## Results

**Setup.**
- **Model:** a 47.2M-parameter GPT: d_model 512, 8 layers, 8 heads, SwiGLU FFN with width 1408,
  RMSNorm, QK-norm, RoPE, tied embeddings, 42k vocabulary.
- **Training:** fp16 weights with stochastically rounded updates, sequence length 1024, 65,536
  tokens per step, WSD schedule (warmup 200 steps, constant, linear decay over the last 20%),
  gradient clipping at 1.0.
- **Data:** Dolma subsets `cc_en_head, c4, books, arxiv`.
- **Metric:** final loss on 512 held-out 1024-token windows from the training distribution.
- **Fairness controls:**
  - All four optimizers run through the same code path: fp32 update math, loss scaling,
    stochastic rounding and clipping are identical.
  - A given seed fixes both the initialization and the data order, so differences are compared
    **paired by seed**.
  - Every number is a fully decayed endpoint.
- **Hardware:** V100s.

**Headline**: best LR per optimizer, 131M tokens (2.8 tokens/param):

| optimizer | peak LR | seeds | loss (mean ± sd) | Δ vs Muon (paired) | Δ vs AdamW (paired) |
|---|---|---|---|---|---|
| **Muon** | 8e-3 | 2 | **3.592 ± 0.002** | – | −0.115 ± 0.000 |
| LH-Muon, α = 0.25 | 2e-3 | 2 | 3.610 ± 0.002 | +0.018 ± 0.002 | −0.097 ± 0.003 |
| LH-Muon, α = 2 (default) | 2e-3 | 2 | 3.696 ± 0.008 | +0.104 ± 0.005 | −0.011 ± 0.005 |
| AdamW | 2e-3 | 3 | 3.707 ± 0.001 | +0.115 ± 0.000 | – |
| Lion | 1.5e-4 | 2 | 3.931 ± 0.006 | +0.340 ± 0.005 | +0.225 ± 0.005 |

![loss curves and difference from Muon](results/suite/report/fig1_curves.png)

**LR sweep** (seed 0). Every optimum is bracketed; the harness extends a grid automatically when
its best LR sits on an edge. Muon is flat from 2e-3 to 8e-3. LH-Muon at α = 2 is almost
insensitive to LR (3.705 / 3.701 / 3.702 over 1e-3 to 4e-3).

![LR sweep](results/suite/report/fig2_lr_sweep.png)

**Ablations** (seed 0, LH-Muon at LR 2e-3; Muon at the same LR scores 3.596):

| variant | loss | Δ vs Muon (same LR) |
|---|---|---|
| α = 0.25 | 3.608 | +0.012 |
| α = 0.5 | 3.622 | +0.026 |
| α = 1 | 3.651 | +0.055 |
| α = 2 (default) | 3.701 | +0.105 |
| α = 5 | 3.761 | +0.165 |
| slow horizon 100 steps (instead of 300) | 3.636 | +0.040 |
| α scaled down with the LR during decay | 3.697 | +0.101 |
| soft ε floor (κ = 1) | 3.692 | +0.096 |
| `combine="separate"`, α = 0.5 | 3.671 | +0.075 |
| sphere norm control (fixed Frobenius radius) | 3.762 | +0.166 |
| int4 state (fast and slow buffers) | 3.885 | +0.289 |

![ablations](results/suite/report/fig3_ablations.png)

**Reading it:**
- **The slow momentum hurts in proportion to its weight**, and a fresher (shorter-horizon) slow
  buffer hurts less. The orthogonalization gives a stale direction a full-magnitude step. In
  AdEMAMix, Adam's normalization tames that term; here nothing does.
- **The damage is done mostly during the LR decay phase.** At α = 2, LH-Muon was within 0.02 of
  Muon at ~105M tokens and lost ~0.09 over the last 20% of training. Scaling α down together with
  the LR did not recover that loss.
- **The soft ε floor** changed the loss by −0.010 against LH-Muon's own default. That is within
  about 2σ of seed noise.
- **int4 optimizer state is not usable** in this setting (+0.18 vs int8). int8 is fine; the
  default runs use it.
- **The sphere constraint caps capacity** when matrices need to grow. It also stalls a 2-layer MLP
  without norm layers at loss 0.25, where weight decay reaches 0.001.
- **Muon beats AdamW by 0.115 nats**, which confirms the baseline harness behaves as expected.
- **Lion is probably under-tuned.** Only its LR was swept; weight decay 0.7 and betas
  (0.9, 0.99) were fixed, and the 65k-token batch is small for sign updates.

**Limits of this evidence:**
- One model size (47M).
- Short budgets (2.8 tokens/param rather than ~20). Slow-momentum methods reportedly gain more
  with longer horizons, and the horizon here was 300 steps.
- Ablations are single-seed.
- Seed stage partially complete at publication, and the loss-vs-tokens frontier stage
  (token multipliers) was not run. Full per-run logs are in [`results/suite/runs`](results/suite/runs),
  and the generated report is [`results/suite/report/report.md`](results/suite/report/report.md).

## Install

```bash
git clone https://github.com/Hugodonotexit/LH-Muon && cd LH-Muon
pip install -e .            # torch, numpy; matplotlib + pytest for the scripts/tests: pip install -e ".[dev]"
python -m pytest tests -q   # GPU tests run on the emptiest CUDA device and skip without one
```

## Usage

```python
from lhmuon import LHMuon, build_param_groups

groups = build_param_groups(model, weight_decay=0.1, verbose=True)
opt = LHMuon(groups, lr=2e-3, total_steps=total_steps,
             alpha=0.0)                                        # alpha=0: plain Muon (+ the rest of the routing)

loss_scaled.backward()                                         # fp16: loss * scale
opt.step(grad_coef=clip_factor / scale, check_finite=True)     # unscale + clip, applied in fp32
if opt.last_step_skipped: ...                                  # inf/NaN: nothing was touched
opt.zero_grad()
```

- **`grad_coef`** multiplies an fp32 copy of each gradient, so a loss-scaled fp16 gradient is never
  unscaled in fp16.
- **fp16/bf16 weights** are written back with stochastic rounding (`stochastic_weights=True`), so
  updates far below half an fp16 spacing still land in expectation. Round-to-nearest can silently
  delete a large fraction of Muon's small per-element updates.
- **`generator=`** takes a `torch.Generator` for all stochastic rounding. Seed it identically on every
  rank to keep ranks bit-identical. By default the optimizer owns one per device, and it is saved
  in `state_dict()`.
- **Under DDP** every rank runs the full optimizer on identical all-reduced gradients. There is no
  ZeRO sharding.
- **`opt.alpha_mult`** can be set per step by the trainer to scale α, e.g. with the LR.
- **To use AdamW or Lion for everything**, set every group's `"kind"` to `"adamw"` / `"lion"`.

### Parameter routing

| kind | what | rule | state |
|---|---|---|---|
| `spectral` | 2D+ matrices with min side ≥ 32; conv `(o,i,k..)` flattened to `(o, i·k..)`; `(E,m,n)` params named `*experts*` orthogonalized per expert | LH-Muon / Muon | m_f (+ m_s) in `state_dtype` / `slow_dtype` |
| `factored` | embeddings, LM heads (tied or not), MoE routers (`router.`, `*.gate.weight`) | Adam with Adafactor row/col second moment, update-RMS clip 1 | m in `state_dtype`, v as two fp32 vectors |
| `adamw` | 1D params, names containing `norm` / `A_log` / `dt_bias` / `D`, depthwise convs, thin matrices | fp32 AdamW | 8 B/param (tiny) |
| `lion` | only when a group's `kind` is set to it | fp32 Lion | 4 B/param |

`overrides={pattern: kind}`, `lr_mult={pattern: mult}` and `wd_overrides={pattern: wd}` are regexes
matched against parameter names; the first match wins. A group's effective LR is
`group["lr"] * group["lr_mult"]`, so a trainer can write the same scheduled LR into every group.
Updates are RMS-matched (0.2·√max(m, n)), so one LR serves every kind. Routing MoE routers to
Adam is a heuristic that hasn't been validated.

## The algorithm

For each hidden weight matrix W (m×n), with g the unscaled gradient:

```
m_f ← β m_f + (1−β) g                        fast momentum, β = 0.95 (Muon's)
c_f = β m_f + (1−β) g                        Nesterov
every K steps:  m_s ← lerp(m_s, m_f, 1/min(H/K, n))   slow EMA, horizon H steps, sampled from m_f
c   = c_f + α_t · m_s                        α_t ramps 0 → α over the first H steps
U   = polar(c)                               5 Newton–Schulz iterations (Jordan et al. coefficients)
    | soft_polar(c, ε_t)                     if soft_kappa > 0: σ → σ/√(σ²+ε²)
W  ← W(1 − lr·wd) − lr · 0.2·√max(m,n) · U   decoupled wd, RMS-matched step
```

- **Slow buffer.** It is sampled from the fast momentum every K steps, not updated from g every
  step. With K ≤ 20 ≈ the fast horizon, each gradient still enters with a weight within a factor
  0.36–1. Sampling means no per-step host transfer and no per-step requantization of a slow EMA.
  For its first H/K updates the buffer is a running mean, so it starts unbiased. Defaults are
  `H = min(total_steps/10, 10000)` and `K = min(20, H/32)`. The constructor refuses fewer than 8
  samples per horizon.
- **Soft polar (ε floor).** `ε_t = κ · s_c · (√m + √n)` is the Marchenko–Pastur edge of the noise in
  c, estimated from a per-tensor running estimate of E[(g − m_f)²]. A direction at the noise edge
  is shrunk by 1/√2; well above it the map is Muon's. It is computed as the top block of
  `polar([c; εI])`, which costs (m+n)/m times Muon's NS (2× for square matrices).
- **`combine="separate"`**: `U = polar(c_f) + α_t · polar(m_s)`, with the second term cached at each
  refresh. Per-step cost is Muon's plus one add.
- **Norm control.** `norm_control="wd"` (default) is decoupled weight decay. `"sphere"` rescales W
  to a fixed Frobenius radius (its init norm or `sphere_rms·√(mn)`) after every step, which
  freezes each matrix's scale; see the results. Zero-initialized matrices fall back to wd.

### Options

| option | default | notes |
|---|---|---|
| `lr` | 2e-4 | all kinds, RMS-matched; sweep it — optima don't transfer between optimizers |
| `momentum`, `nesterov` | 0.95, True | fast buffer |
| `alpha` | 2.0 | slow-buffer weight; `0` = Muon exactly (tested). In our runs smaller was always better |
| `slow_horizon` | min(T/10, 10k) | in steps; set explicitly when branching decays off a long run |
| `slow_every` | min(20, H/32) | K |
| `alpha_warmup` | H | linear ramp of α |
| `combine` | `"sum"` | or `"separate"` |
| `soft_kappa` | 0 (off) | ε floor |
| `norm_control` | `"wd"` | `"sphere"`, `"none"` |
| `weight_decay` | 0.1 | per group via `build_param_groups` |
| `ns_steps`, `ns_dtype` | 5, `"auto"` | auto = bf16 on sm_80+, fp32 otherwise |
| `state_dtype` | `"int8"` | fast momentum (and factored m): fp32 / bf16 / fp16 / int8 / int4 |
| `slow_dtype` | `"int8"` | slow buffer on device, or its snapshot when `slow_master="host"` |
| `slow_master` | `"device"` | `"host"`: fp32 slow EMA in pinned host RAM, device holds a round-to-nearest snapshot |
| `offload` | False | all quantized state in pinned host RAM, streamed per tensor |
| `adam_betas`, `adam_eps`, `factored_clip` | (0.9, 0.95), 1e-8, 1.0 | factored + adamw kinds |

## State, precision and memory

- **Block-scaled state.** Every large state tensor is a payload plus one fp32 absmax per block
  (256 elements for int8/bf16, 64 for int4/fp16). Integer payloads are written with stochastic
  rounding, so a slow EMA drifts correctly instead of freezing. fp16 payloads are normalized to
  [−1, 1], so they can neither overflow nor flush to zero.
- **Offload.** With `offload=True`, all quantized state lives in pinned host memory. The next
  tensor's state is prefetched on a side stream while the current one updates. Results are
  bit-identical to on-device (tested). Transfers are not overlapped with the backward pass.
- **Checkpointing.** `state_dict()` keeps the storage format. `load_state_dict()` allocates fresh
  buffers under the current settings, and it refuses a checkpoint saved with a different format
  rather than reinterpreting it. Resume is bit-exact (tested).

**Optimizer step cost** on a 963.8M-parameter hybrid-attention transformer (880.8M params in 392
spectral matrices): fp16 weights, one V100-SXM2-16GB, synthetic gradients, median step. The
transient memory is ~1.35 GiB in every configuration, from the per-tensor fp32 working copies;
there are no fused kernels.

| configuration | step | refresh step | device state | host state |
|---|---|---|---|---|
| Muon (α = 0), int8 | 3.59 s | – | 0.92 GiB | – |
| LH-Muon, int8 + int8 slow | 3.83 s | 4.00 s | 1.75 GiB | – |
| LH-Muon, int8 + int4 snapshot, host master | 3.95 s | 5.60 s | 1.38 GiB | 3.28 GiB |
| LH-Muon + soft ε (κ = 1) | 6.13 s | 6.25 s | 1.75 GiB | – |
| LH-Muon `separate`, host master | 3.92 s | 8.64 s | 1.75 GiB | 3.28 GiB |
| LH-Muon, `offload=True` | 3.96 s | 5.35 s | 0.01 GiB | 5.03 GiB |
| LH-Muon, fp16 state + fp16 slow | 3.96 s | 4.02 s | 3.55 GiB | – |

Most of that time is fp32 Newton–Schulz, since the V100 has no bf16. On a V100, fp16 NS on
Frobenius-normalized input is ~5× faster (2048² in 3.8 ms vs 19.2 ms). It showed 0.6–0.9%
relative error vs fp64 on synthetic spectra, but it hasn't been verified on real momenta, so
`ns_dtype="auto"` keeps fp32 on pre-Ampere GPUs.

## Reproducing the benchmark

Data is a directory with a `manifest.json` that points at flat uint16 token files:

```json
{"categories": {"c4": {"files": [{"path": "/data/c4_0000.bin", "num_tokens": 123456789}]}}}
```

An optional `eval_set.npy` (rows of ≥ 1025 token ids) adds a second held-out eval. Point
`--data` or `$LHMUON_DATA` at the directory (default `./data/tokenized`). The model's vocabulary is
42,000; change `GPT(42000, ...)` in `examples/train_lm.py` for other tokenizers.

```bash
# the whole benchmark, resumable, one job per GPU (~11 h on 2 V100s for all four stages)
nohup python scripts/run_suite.py --gpus cuda:0,cuda:1 > runs/suite/suite.log 2>&1 &
python scripts/analyze_suite.py            # regenerate runs/suite/report/ from finished runs

# a single run
python examples/train_lm.py --opt lhmuon --alpha 0.25 --lr 2e-3 --tokens 131072000 --out runs/lh
```

| stage | what |
|---|---|
| 1 LR sweep | each optimizer on a ×2 grid; auto-extends while the best LR is on the grid's edge |
| 2 ablations | LH-Muon variants at its best LR (the table above) |
| 3 seeds | seeds 1 and 2 for every optimizer's best config and for the best LH-Muon variant |
| 4 frontier | per optimizer, a long WSD run with decay branches at 4 budgets → L(D) fit → token multiplier |

`scripts/fit_multiplier.py endpoints.csv --baseline muon --candidate lhmuon` fits
L = E + B·D^−β to a baseline's decayed endpoints and reports the token multiplier
M = D_baseline(L_c) / D_c, and it flags extrapolation. On synthetic data with a planted 1.30×
it recovers 1.32×.

## Tests

`python -m pytest tests -q` runs 36 tests. They cover:
- quantization error bounds, unbiased stochastic encoding (int8, int4) and weight rounding
  (fp16, bf16, including subnormals);
- the soft-polar identity (SVD, fp64) and its per-direction shrinkage;
- `alpha=0` matching a reference Muon to 1e-5, and Lion matching a reference implementation;
- loss-scale unscaling and skip-on-inf;
- sphere radius and the zero-init fallback;
- offload bit-identical to on-device;
- bit-exact resume and refusal of mismatched checkpoints;
- a toy regression that trains with every state format.

## Related work

- **Muon:** Jordan et al. 2024; Liu et al. 2025, "Muon is Scalable for LLM Training" (RMS
  matching, weight decay).
- **AdEMAMix:** Pagliardini et al. 2024.
- **Lion:** Chen et al. 2023.
- **Adafactor:** Shazeer & Stern 2018.
- **Schedules:** WSD schedules (Hägele et al. 2024).
- **Low-precision optimizer state:** 8-bit optimizers (Dettmers et al. 2021); 4-bit optimizer
  states (Li et al. 2023).
- **Mixed precision and stochastic rounding:** Micikevicius et al. 2017; Gupta et al. 2015.
- **Scaling-law fitting:** Hoffmann et al. 2022.
- **Closest relative:** MuonM (2026) combines a slow momentum with Muon and a sphere constraint,
  but restricts the slow buffer to a subspace. The results here suggest that restriction matters.

## License

[MIT](LICENSE)
