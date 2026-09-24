# Optimizer evaluation: LH-Muon vs Muon vs AdamW vs Lion

Model: 47.2M-param GPT (d 512, 8 layers, 8 heads, SwiGLU 1408, tied embeddings, QK-norm, RoPE), fp16 weights with stochastic-rounded updates, seq 1024, 64 seqs/step (65,536 tokens). Data: Dolma categories cc_en_head,c4,books,arxiv (42k-token vocabulary). Schedule: warmup 200, constant, linear decay over the last 20% (WSD). Grad clip 1.0. All optimizers run through the same LHMuon code path (fp32 update math, same loss scaling, same rounding), same init and data order per seed.

Primary metric: final **eval_indist**: 512 held-out 1024-token windows from the training distribution. Secondary: **eval_mix**: 512 rows of a fixed eval set drawn from the full, broader Dolma mixture. Lower is better; Δ values are paired by seed.

Finished runs: stage 1 (LR sweep) 15, stage 2 (ablations) 10, stage 3 (seeds) 10, stage 4 (frontier) 18.

## 1. Headline: best config per optimizer at 131M tokens (2.8 tok/param)

| optimizer | peak LR | seeds | eval_indist mean ± sd | Δ vs Muon (paired) | Δ vs AdamW (paired) | eval_mix | token multiplier vs Muon | optimizer time | tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| LH-Muon | 0.002 | 3 | 3.6982 ± 0.0070 | +0.1060 ± 0.0034 | -0.0084 ± 0.0036 | 4.0402 | 0.67× | 7.8% | 61,195 |
| LH-Muon, α = 0.25 | 0.002 | 3 | 3.6094 ± 0.0015 | +0.0172 ± 0.0015 | -0.0972 ± 0.0015 | 3.9015 | 0.92× | 8.7% | 72,449 |
| Muon | 0.008 | 3 | 3.5922 ± 0.0012 | – | -0.1144 ± 0.0003 | 3.8420 | 1 (reference) | 8.4% | 72,906 |
| AdamW | 0.002 | 3 | 3.7066 ± 0.0011 | +0.1144 ± 0.0003 | – | 4.0501 | 0.66× | 3.8% | 65,074 |
| Lion | 0.00015 | 3 | 3.9279 ± 0.0074 | +0.3356 ± 0.0049 | +0.2213 ± 0.0047 | 4.4423 | 0.36× | 3.5% | 64,777 |

Seed noise: the SD of a paired difference between two optimizers is 0.0050 nats; a single run's SD across seeds is 0.0042. With 3 seeds, a paired Δ smaller than about 0.0057 nats (2 SE) is not distinguishable from noise.

![curves](fig1_curves.png)

## 2. LR sweep (final eval_indist, 131M tokens, seed 0)

| optimizer | 7.5e-05 | 0.00015 | 0.0003 | 0.0006 | 0.001 | 0.002 | 0.004 | 0.008 | 0.016 |
|---|---|---|---|---|---|---|---|---|---|
| LH-Muon |  |  |  |  | 3.7045 | **3.7013** | 3.7015 |  |  |
| Muon |  |  |  |  | 3.6244 | 3.5961 | 3.5960 | **3.5929** | 3.6361 |
| AdamW |  |  |  |  | 3.7237 | **3.7077** | 3.7145 |  |  |
| Lion | 4.0313 | **3.9273** | 3.9448 | 4.0668 |  |  |  |  |  |

A best LR on the edge of its grid means the optimum wasn't bracketed; the suite extends the grid automatically in that case (up to twice).

![lr sweep](fig2_lr_sweep.png)

## 3. LH-Muon ablations (at LH-Muon's best LR, seed 0)

| variant | eval_indist | Δ vs Muon | Δ vs LH-Muon default | optimizer time |
|---|---|---|---|---|
| default (α 2, sum, wd, int8) | 3.7013 | +0.1084 | – | 7.8% |
| α = 0.25 | 3.6084 | +0.0155 | -0.0929 | 8.7% |
| α = 0.5 | 3.6219 | +0.0289 | -0.0794 | 7.8% |
| α = 1 | 3.6505 | +0.0576 | -0.0508 | 8.7% |
| α = 5 | 3.7611 | +0.1682 | +0.0598 | 7.8% |
| soft ε (κ = 1) | 3.6915 | +0.0986 | -0.0098 | 9.0% |
| separate, α = 0.5 | 3.6714 | +0.0784 | -0.0299 | 8.3% |
| sphere norm control | 3.7616 | +0.1686 | +0.0603 | 8.7% |
| int4 state + slow | 3.8847 | +0.2918 | +0.1834 | 8.8% |
| slow horizon 100 | 3.6362 | +0.0433 | -0.0651 | 8.4% |
| α decays with LR | 3.6969 | +0.1040 | -0.0044 | 7.9% |

These are single seeds. Read them against the seed-noise line in section 1.

![ablations](fig3_ablations.png)

## 4. Loss vs tokens and the token multiplier (fully decayed endpoints)

| optimizer | 32M | 65M | 131M | 262M | 524M | fit L = E + B·D^−β |
|---|---|---|---|---|---|---|
| LH-Muon | 4.2266 | 3.8318 | 3.6067 | 3.4417 | 3.3250 | E 3.124, B 3.96e+04, β 0.606 |
| Muon | 4.1163 | 3.7928 | 3.5916 | 3.4520 | 3.3534 | E 3.170, B 2.45e+04, β 0.587 |
| AdamW | 4.5027 | 3.9913 | 3.7135 | 3.5130 |  | E 3.235, B 3.45e+05, β 0.723 |
| Lion | 5.2415 | 4.5664 | 3.9474 | 3.5788 |  | E 2.190, B 2.36e+03, β 0.384 |

Token multiplier M = D_Muon(L) / D, at each budget of the other optimizers:

| optimizer | 32M | 65M | 131M | 262M | 524M |
|---|---|---|---|---|---|
| LH-Muon | 0.83×* | 0.92× | 0.93× | 1.04× | 1.35×* |
| AdamW | 0.56×* | 0.64× | 0.64× | 0.70× |  |
| Lion | 0.26×* | 0.26×* | 0.35× | 0.52× |  |

\* extrapolated beyond the baseline's measured token range.

Token multiplier M = D_AdamW(L) / D, at each budget of the other optimizers:

| optimizer | 32M | 65M | 131M | 262M | 524M |
|---|---|---|---|---|---|
| LH-Muon | 1.40× | 1.42× | 1.36× | 1.54×* | 2.43×* |
| Muon | 1.65× | 1.56× | 1.44× | 1.44×* | 1.66×* |
| Lion | 0.53×* | 0.47×* | 0.55× | 0.76× |  |

\* extrapolated beyond the baseline's measured token range.

![frontier](fig4_frontier.png)

## 5. Verdict

LH-Muon (α = 0.25) − Muon at 131M tokens, paired over 3 seed(s): **+0.0172 ± 0.0015 nats** (more than 2 SE from zero). That corresponds to a token multiplier of **0.92×** over Muon at 131M tokens.

Pre-registered bar (from the design discussion): the gain counts if M ≥ 1.10× over tuned Muon, the 2σ band on Δ excludes zero, and the gain doesn't shrink with scale. That last check needs a second model size; this suite has one.

LH-Muon's stage-3 seeds and stage-4 frontier use the best stage-2 variant (α = 0.25), chosen on a single seed. That selection biases it slightly in LH-Muon's favour.

Lion's weight decay (0.7) and betas (0.9, 0.99) were not swept, only its LR, so Lion may be under-tuned relative to the others.

Caveats: a single 47M scale; budgets of 0.7–5.6 tok/param, not the 20 tok/param target; the slow-momentum horizon is fixed at 300 steps, and its gains reportedly grow with horizon. Periodic evals in fig. 1 use 128 windows, so they are noisier than the final 512-window evals in the tables.
