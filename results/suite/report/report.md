# Optimizer evaluation: LH-Muon vs Muon vs AdamW vs Lion

Model: 47.2M-param GPT (d 512, 8 layers, 8 heads, SwiGLU 1408, tied embeddings, QK-norm, RoPE), fp16 weights with stochastic-rounded updates, seq 1024, 64 seqs/step (65,536 tokens). Data: Dolma categories cc_en_head,c4,books,arxiv (42k-token vocabulary). Schedule: warmup 200, constant, linear decay over the last 20% (WSD). Grad clip 1.0. All optimizers run through the same LHMuon code path (fp32 update math, same loss scaling, same rounding), same init and data order per seed.

Primary metric: final **eval_indist**: 512 held-out 1024-token windows from the training distribution. Secondary: **eval_mix**: 512 rows of a fixed eval set drawn from the full, broader Dolma mixture. Lower is better; Δ values are paired by seed.

Finished runs: stage 1 (LR sweep) 15, stage 2 (ablations) 10, stage 3 (seeds) 6, stage 4 (frontier) 0.

## 1. Headline: best config per optimizer at 131M tokens (2.8 tok/param)

| optimizer | peak LR | seeds | eval_indist mean ± sd | Δ vs Muon (paired) | Δ vs AdamW (paired) | eval_mix | token multiplier vs Muon | optimizer time | tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| LH-Muon | 0.002 | 2 | 3.6957 ± 0.0079 | +0.1038 ± 0.0045 | -0.0109 ± 0.0045 | 4.0402 | needs stage 4 | 7.8% | 61,195 |
| LH-Muon, α = 0.25 | 0.002 | 2 | 3.6098 ± 0.0020 | +0.0179 ± 0.0024 | -0.0968 ± 0.0025 | 3.9015 | needs stage 4 | 8.7% | 72,449 |
| Muon | 0.008 | 2 | 3.5919 ± 0.0015 | – | -0.1147 ± 0.0001 | 3.8420 | 1 (reference) | 8.4% | 72,906 |
| AdamW | 0.002 | 3 | 3.7066 ± 0.0011 | +0.1147 ± 0.0001 | – | 4.0501 | needs stage 4 | 3.8% | 65,074 |
| Lion | 0.00015 | 2 | 3.9314 ± 0.0058 | +0.3395 ± 0.0052 | +0.2248 ± 0.0052 | 4.4423 | needs stage 4 | 3.5% | 64,777 |

Seed noise: the SD of a paired difference between two optimizers is 0.0046 nats; a single run's SD across seeds is 0.0041. With 3 seeds, a paired Δ smaller than about 0.0053 nats (2 SE) is not distinguishable from noise.

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

Not run yet (stage 4).

## 5. Verdict

LH-Muon (α = 0.25) − Muon at 131M tokens, paired over 2 seed(s): **+0.0179 ± 0.0024 nats** (more than 2 SE from zero).

Pre-registered bar (from the design discussion): the gain counts if M ≥ 1.10× over tuned Muon, the 2σ band on Δ excludes zero, and the gain doesn't shrink with scale. That last check needs a second model size; this suite has one.

LH-Muon's stage-3 seeds and stage-4 frontier use the best stage-2 variant (α = 0.25), chosen on a single seed. That selection biases it slightly in LH-Muon's favour.

Lion's weight decay (0.7) and betas (0.9, 0.99) were not swept, only its LR, so Lion may be under-tuned relative to the others.

Caveats: a single 47M scale; budgets of 0.7–5.6 tok/param, not the 20 tok/param target; the slow-momentum horizon is fixed at 300 steps, and its gains reportedly grow with horizon. Periodic evals in fig. 1 use 128 windows, so they are noisier than the final 512-window evals in the tables.
