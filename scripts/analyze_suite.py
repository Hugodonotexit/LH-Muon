"""
analyze_suite.py -- figures + tables for runs/suite (see run_suite.py). Safe to run at any point: it
reports whatever has finished and says what is still missing.

Writes runs/suite/report/{report.md, summary.csv, fig1_curves.png, fig2_lr_sweep.png,
fig3_ablations.png, fig4_frontier.png}.

Primary metric: final eval_indist (512 held-out 1024-token windows from the training distribution),
lower is better. Comparisons across optimizers are PAIRED by seed (same init + data order).
"""

import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.get("SUITE_DIR", os.path.join(ROOT, "runs", "suite"))
REPORT = os.path.join(RUNS, "report")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fit_multiplier import fit  # noqa: E402

ORDER = ["lhmuon", "muon", "adamw", "lion"]
LABEL = {"lhmuon": "LH-Muon", "muon": "Muon", "adamw": "AdamW", "lion": "Lion"}
COLOR = {"lhmuon": "#2a78d6", "muon": "#eb6834", "adamw": "#1baf7a", "lion": "#eda100"}   # fixed per entity
MARK = {"lhmuon": "o", "muon": "s", "adamw": "^", "lion": "D"}                            # secondary encoding
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ABL_LABEL = {"alpha0.25": "α = 0.25", "alpha0.5": "α = 0.5", "alpha1": "α = 1", "alpha5": "α = 5", "soft1": "soft ε (κ = 1)", "separate": "separate, α = 0.5",
             "sphere": "sphere norm control", "int4": "int4 state + slow", "h100": "slow horizon 100",
             "alphadecay": "α decays with LR"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 7, "figure.dpi": 150,
    })


# ------------------------------------------------------------------------------------------ data
def load_runs():
    runs = {}
    if not os.path.isdir(RUNS):
        return runs
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        fj = os.path.join(d, "final.json")
        if not os.path.exists(fj):
            continue
        f = json.load(open(fj))
        rows = list(csv.DictReader(open(os.path.join(d, "log.csv")))) if os.path.exists(os.path.join(d, "log.csv")) else []
        runs[name] = {"final": f, "log": rows}
    return runs


def ok(r):
    return r is not None and not r["final"]["diverged"] and math.isfinite(r["final"]["eval_indist"])


def fmt_lr(lr):
    return f"{lr:.2e}".replace("e-0", "e-")


def seeds_of(runs, opt, best_name, variant=None):
    """{seed: final eval_indist} for an optimizer's best config (or an LH-Muon ablation variant)."""
    out = {}
    first = f"s2_lhmuon_{variant}" if variant else best_name
    if ok(runs.get(first)):
        out[0] = runs[first]["final"]["eval_indist"]
    for s in (1, 2, 3, 4):
        n = f"s3_lhmuon_{variant}_seed{s}" if variant else f"s3_{opt}_seed{s}"
        if ok(runs.get(n)):
            out[s] = runs[n]["final"]["eval_indist"]
    return out


def paired(a, b):
    """mean, SE, n of a - b over shared seeds (SE None when n < 2)."""
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common])
    if len(d) == 0:
        return None, None, 0
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
    return float(d.mean()), se, len(d)


def curve(r, key="eval_indist"):
    x = np.array([float(row["tokens"]) for row in r["log"]])
    y = np.array([float(row[key]) for row in r["log"]])
    return x, y


# ------------------------------------------------------------------------------------------ figures
def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)


def label_ends(ax, items, gap_frac=0.055):
    """Direct labels at line ends, pushed apart vertically so they never overlap. items: [(x, y, text)]."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    items = sorted(items, key=lambda t: t[1])
    ys = [items[0][1]]
    for _, y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    for (x, y, text), ly in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(x, ly), textcoords="data", va="center", fontsize=9, color=INK,
                    ha="left", transform=ax.transData,
                    arrowprops=None)


def plain_log_axis(ax, ticks):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def legend_below(ax, n):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n, handlelength=1.6)


def fig_curves(runs, best_name, variant=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    have = [o for o in ORDER if ok(runs.get(best_name.get(o)))]
    ends = []
    for o in have:
        x, y = curve(runs[best_name[o]])
        m = x >= 10e6
        a1.plot(x[m] / 1e6, y[m], color=COLOR[o], label=LABEL[o])
        a1.plot(x[-1] / 1e6, y[-1], MARK[o], color=COLOR[o], markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append((x[-1] / 1e6, y[-1], f"{LABEL[o]} {y[-1]:.3f}"))
    vname = f"s2_lhmuon_{variant}" if variant else None
    if vname and ok(runs.get(vname)):
        x, y = curve(runs[vname])
        m = x >= 10e6
        vlab = f"LH-Muon, {ABL_LABEL[variant]}"
        a1.plot(x[m] / 1e6, y[m], color=COLOR["lhmuon"], linestyle="--", label=vlab)
        ends.append((x[-1] / 1e6, y[-1], f"{vlab} {y[-1]:.3f}"))
    if ends:
        xmax = a1.get_xlim()[1]
        a1.set_xlim(right=xmax * 1.16)
        label_ends(a1, [(x + xmax * 0.012, y, t) for x, y, t in ends])
    a1.set_title("Held-out loss during training (best LR, seed 0)")
    a1.set_xlabel("tokens (M)")
    a1.set_ylabel("eval loss, training distribution (nats)")
    legend_below(a1, min(3, len(have) + (1 if vname else 0)))
    if "muon" in have:
        a2.axhline(0, color=INK2, linewidth=1)
        a2.annotate("Muon", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, -11), textcoords="offset points",
                    ha="left", fontsize=9, color=INK2)
        ends = []
        series = [(o, seed_pairs(best_name, o), LABEL[o], "-") for o in have if o != "muon"]
        if vname and ok(runs.get(vname)):
            pairs = [(vname, best_name["muon"])] + [(f"s3_lhmuon_{variant}_seed{s}", f"s3_muon_seed{s}") for s in (1, 2, 3, 4)]
            series.append(("lhmuon", pairs, f"LH-Muon, {ABL_LABEL[variant]}", "--"))
        for o, pairs, lab, ls in series:
            deltas = []
            for na, nb in pairs:
                if ok(runs.get(na)) and ok(runs.get(nb)):
                    xa, ya = curve(runs[na])
                    xb, yb = curve(runs[nb])
                    n = min(len(xa), len(xb))
                    deltas.append((xa[:n], ya[:n] - yb[:n]))
            if not deltas:
                continue
            n = min(len(d[0]) for d in deltas)
            x = deltas[0][0][:n]
            d = np.mean([d[1][:n] for d in deltas], axis=0)
            m = x >= 10e6
            a2.plot(x[m] / 1e6, d[m], color=COLOR[o], linestyle=ls, label=f"{lab} ({len(deltas)} seed{'s' if len(deltas) > 1 else ''})")
            a2.plot(x[m][-1] / 1e6, d[m][-1], MARK[o], color=COLOR[o], markeredgecolor=SURFACE, markeredgewidth=2)
            ends.append((x[m][-1] / 1e6, d[m][-1], f"{d[m][-1]:+.3f}"))
        a2.set_title("Difference from Muon (below 0 = better than Muon)")
        a2.set_xlabel("tokens (M)")
        a2.set_ylabel("Δ eval loss vs Muon (nats)")
        xmax = a2.get_xlim()[1]
        a2.set_xlim(right=xmax * 1.06)
        label_ends(a2, [(x + xmax * 0.012, y, t) for x, y, t in ends])
        legend_below(a2, 2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig1_curves.png"))
    plt.close(fig)


def seed_pairs(best_name, o):
    yield best_name[o], best_name["muon"]
    for s in (1, 2, 3, 4):
        yield f"s3_{o}_seed{s}", f"s3_muon_seed{s}"


def fig_lr(runs, sweep):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    worst = []
    for o in ORDER:
        if o not in sweep:
            continue
        pts = [(lr, runs.get(f"s1_{o}_lr{fmt_lr(lr)}")) for lr in sweep[o]["grid"]]
        good = [(lr, r["final"]["eval_indist"]) for lr, r in pts if ok(r)]
        worst += [v for _, v in good]
        if good:
            xs, ys = zip(*good)
            ax.plot(xs, ys, "-" + MARK[o], color=COLOR[o], label=LABEL[o], markeredgecolor=SURFACE, markeredgewidth=1.5)
            b = min(good, key=lambda t: t[1])
            ax.plot(*b, MARK[o], color=COLOR[o], markersize=12, markerfacecolor="none", markeredgewidth=2)
    top = max(worst) + 0.05 if worst else 1
    for o in ORDER:
        for lr in (sweep.get(o, {}).get("grid") or []):
            r = runs.get(f"s1_{o}_lr{fmt_lr(lr)}")
            if r is not None and not ok(r):
                ax.plot(lr, top, "x", color=COLOR[o], markersize=9, markeredgewidth=2)
                ax.annotate("diverged", (lr, top), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=8, color=INK2)
    ax.set_xscale("log")
    plain_log_axis(ax, sorted({lr for o in ORDER for lr in (sweep.get(o, {}).get("grid") or [])}))
    ax.set_title("LR sweep: final held-out loss (131M tokens, seed 0; ring = best)")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final eval loss, training distribution (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig2_lr_sweep.png"))
    plt.close(fig)


def fig_ablations(runs, best_name, noise):
    ref = runs.get(best_name.get("muon"))
    rows = []
    lh = runs.get(best_name.get("lhmuon"))
    if not ok(ref):
        return
    r0 = ref["final"]["eval_indist"]
    if ok(lh):
        rows.append(("LH-Muon (default: α = 2, sum, wd, int8)", lh["final"]["eval_indist"] - r0, True))
    for k, lab in ABL_LABEL.items():
        r = runs.get(f"s2_lhmuon_{k}")
        if ok(r):
            rows.append((lab, r["final"]["eval_indist"] - r0, False))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.8))
    ys = np.arange(len(rows))[::-1]
    if noise:
        ax.axvspan(-2 * noise, 2 * noise, color=GRID, alpha=0.7, linewidth=0)
        ax.annotate("±2σ seed noise (paired Δ)", (2 * noise, ys[0] + 0.55), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.axvline(0, color=COLOR["muon"], linewidth=1.5)
    ax.annotate("Muon", (0, ys[-1] - 0.75), xytext=(3, 0), textcoords="offset points", fontsize=9, color=INK2)
    for o in ("adamw", "lion"):
        r = runs.get(best_name.get(o))
        if ok(r):
            v = r["final"]["eval_indist"] - r0
            ax.axvline(v, color=COLOR[o], linewidth=1.5, linestyle="--")
            ax.annotate(LABEL[o], (v, ys[-1] - 0.75), xytext=(3, 0), textcoords="offset points", fontsize=9, color=INK2)
    for y, (lab, v, main) in zip(ys, rows):
        ax.plot([0, v], [y, y], color=COLOR["lhmuon"], linewidth=1.5, alpha=0.5)
        ax.plot(v, y, "o", color=COLOR["lhmuon"], markersize=9 if main else 7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{v:+.3f}", (v, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(ys[-1] - 1, ys[0] + 1)
    ax.grid(axis="y", visible=False)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.08 * (x1 - x0), x1)
    ax.set_title("LH-Muon ablations vs Muon (seed 0; left = better)")
    ax.set_xlabel("Δ eval loss vs Muon (nats)")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig3_ablations.png"))
    plt.close(fig)


def frontier_points(runs, o):
    pts = []
    for name, r in runs.items():
        if name.startswith(f"s4_{o}_") and ok(r):
            pts.append((r["final"]["tokens"], r["final"]["eval_indist"]))
    return sorted(pts)


def fig_frontier(runs, fits, variant=None, noise=None):
    have = [o for o in ORDER if len(frontier_points(runs, o)) >= 2]
    if not have:
        return
    fig, (ax, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for o in have:
        D, L = zip(*frontier_points(runs, o))
        lab = f"LH-Muon, {ABL_LABEL[variant]}" if (o == "lhmuon" and variant) else LABEL[o]
        ax.plot(np.array(D) / 1e6, L, MARK[o], color=COLOR[o], label=lab, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if o in fits:
            E, B, beta = fits[o]
            xs = np.geomspace(min(D) * 0.9, max(D) * 1.1, 100)
            ax.plot(xs / 1e6, E + B * xs ** -beta, color=COLOR[o], linewidth=1.4)
    ticks = sorted({p[0] / 1e6 for o in have for p in frontier_points(runs, o)})
    for a in (ax, a2):
        a.set_xscale("log")
        plain_log_axis(a, ticks)
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
        a.set_xlabel("training tokens (M, log scale)")
    ax.set_title("Loss vs tokens: decayed endpoints + fit L = E + B·D^−β")
    ax.set_ylabel("final eval loss, training distribution (nats)")
    ax.legend(loc="upper right")

    # right: each optimizer minus Muon at every budget (the gap the left panel hides)
    mu = dict(frontier_points(runs, "muon"))
    if noise:
        a2.axhspan(-2 * noise, 2 * noise, color=GRID, alpha=0.7, linewidth=0)
    a2.axhline(0, color=COLOR["muon"], linewidth=1.5)
    a2.annotate("Muon", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, -11), textcoords="offset points",
                fontsize=9, color=INK2)
    for o in ("lhmuon", "adamw"):
        pts = [(d, l - mu[d]) for d, l in frontier_points(runs, o) if d in mu]
        if not pts:
            continue
        x, y = zip(*pts)
        lab = f"LH-Muon, {ABL_LABEL[variant]}" if (o == "lhmuon" and variant) else LABEL[o]
        a2.plot(np.array(x) / 1e6, y, "-" + MARK[o], color=COLOR[o], label=lab, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        for xi, yi in pts:
            a2.annotate(f"{yi:+.3f}", (xi / 1e6, yi), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)
    a2.set_title("Difference from Muon at each budget (below 0 = better)")
    a2.set_ylabel("Δ final eval loss vs Muon (nats)")
    lo, hi = a2.get_ylim()
    a2.set_ylim(lo - 0.02, hi + 0.03)
    x0, x1 = a2.get_xlim()
    a2.set_xlim(x0 / 1.12, x1 * 1.12)
    a2.legend(loc="upper right")
    if noise:
        a2.annotate("±2σ seed noise", (0.01, 2 * noise), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="left", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig4_frontier.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------ report
def pm(mean, se, digits=4):
    if mean is None:
        return "–"
    return f"{mean:+.{digits}f}" + (f" ± {se:.{digits}f}" if se is not None else "")


def multiplier(fitp, D_c, L_c):
    E, B, beta = fitp
    if L_c <= E:
        return None
    return (B / (L_c - E)) ** (1 / beta) / D_c


def main():
    style()
    os.makedirs(REPORT, exist_ok=True)
    runs = load_runs()
    sp = os.path.join(RUNS, "lr_sweep.json")
    sweep = json.load(open(sp)) if os.path.exists(sp) else {}
    # before stage 1 finishes: best so far among finished sweep runs
    best_name, best_lr = {}, {}
    for o in ORDER:
        if o in sweep:
            best_lr[o] = sweep[o]["best"]
        else:
            cands = [(r["final"]["eval_indist"], r["final"]["args"]["lr"]) for n, r in runs.items()
                     if n.startswith(f"s1_{o}_") and ok(r)]
            if cands:
                best_lr[o] = min(cands)[1]
        if o in best_lr:
            best_name[o] = f"s1_{o}_lr{fmt_lr(best_lr[o])}"
    if not sweep:
        sweep = {o: {"grid": sorted({r["final"]["args"]["lr"] for n, r in runs.items() if n.startswith(f"s1_{o}_")})}
                 for o in ORDER}

    seeds = {o: seeds_of(runs, o, best_name.get(o)) for o in ORDER}
    vp = os.path.join(RUNS, "lh_variant.json")
    variant = json.load(open(vp))["variant"] if os.path.exists(vp) else None
    vseeds = seeds_of(runs, "lhmuon", None, variant=variant) if variant else {}
    # seed noise: pooled SD of paired differences vs Muon, else per-run SD
    diffs = []
    for o in ORDER:
        if o == "muon":
            continue
        common = sorted(set(seeds[o]) & set(seeds["muon"]))
        if len(common) >= 2:
            d = np.array([seeds[o][s] - seeds["muon"][s] for s in common])
            diffs.append(d.std(ddof=1))
    noise = float(np.mean(diffs)) if diffs else None
    run_sd = [np.std(list(v.values()), ddof=1) for v in seeds.values() if len(v) >= 2]
    run_sd = float(np.mean(run_sd)) if run_sd else None

    fits = {}
    for o in ORDER:
        pts = frontier_points(runs, o)
        if len(pts) >= 3:
            try:
                fits[o] = fit(*zip(*pts))
            except ValueError:
                pass

    fig_curves(runs, best_name, variant)
    fig_lr(runs, sweep)
    fig_ablations(runs, best_name, noise)
    fig_frontier(runs, fits, variant, noise)

    any_run = next(iter(runs.values()), None)
    L = []
    L.append("# Optimizer evaluation: LH-Muon vs Muon vs AdamW vs Lion\n")
    if any_run:
        a = any_run["final"]["args"]
        L.append(f"Model: {any_run['final']['params'] / 1e6:.1f}M-param GPT (d {a['d']}, {a['layers']} layers, "
                 f"{a['heads']} heads, SwiGLU {a['ffn']}, tied embeddings, QK-norm, RoPE), {a['precision']} weights with "
                 f"stochastic-rounded updates, seq {a['seq']}, {a['batch']} seqs/step ({a['batch'] * a['seq']:,} tokens). "
                 f"Data: Dolma categories {a['categories']} (42k-token vocabulary). Schedule: warmup {a['warmup']}, constant, linear decay over the "
                 f"last {a['decay_frac']:.0%} (WSD). Grad clip {a['clip']}. All optimizers run through the same LHMuon code "
                 f"path (fp32 update math, same loss scaling, same rounding), same init and data order per seed.\n")
    L.append("Primary metric: final **eval_indist**: 512 held-out 1024-token windows from the training distribution. "
             "Secondary: **eval_mix**: 512 rows of a fixed eval set drawn from the full, broader Dolma mixture. "
             "Lower is better; Δ values are paired by seed.\n")
    done = {k: sum(1 for n in runs if n.startswith(k)) for k in ("s1_", "s2_", "s3_", "s4_")}
    L.append(f"Finished runs: stage 1 (LR sweep) {done['s1_']}, stage 2 (ablations) {done['s2_']}, "
             f"stage 3 (seeds) {done['s3_']}, stage 4 (frontier) {done['s4_']}.\n")

    # headline
    L.append("## 1. Headline: best config per optimizer at 131M tokens (2.8 tok/param)\n")
    L.append("| optimizer | peak LR | seeds | eval_indist mean ± sd | Δ vs Muon (paired) | Δ vs AdamW (paired) | "
             "eval_mix | token multiplier vs Muon | optimizer time | tokens/s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    summary = []
    for o in ORDER:
        s = seeds[o]
        if not s:
            L.append(f"| {LABEL[o]} | – | 0 | not run yet | | | | | | |")
            continue
        vals = np.array(list(s.values()))
        sd = f" ± {vals.std(ddof=1):.4f}" if len(vals) > 1 else ""
        dm, sem, nm = paired(s, seeds["muon"]) if o != "muon" else (None, None, 0)
        da, sea, na = paired(s, seeds["adamw"]) if o != "adamw" else (None, None, 0)
        r = runs[best_name[o]]["final"]
        M = "needs stage 4"
        if "muon" in fits and o != "muon":
            m = multiplier(fits["muon"], 131072000, float(vals.mean()))
            M = f"{m:.2f}×" if m else "below fitted floor"
        elif o == "muon":
            M = "1 (reference)"
        L.append(f"| {LABEL[o]} | {best_lr[o]:g} | {len(vals)} | {vals.mean():.4f}{sd} | {pm(dm, sem)} | {pm(da, sea)} | "
                 f"{r['eval_mix']:.4f} | {M} | {100 * r['opt_fraction']:.1f}% | {r['tokens'] / r['seconds']:,.0f} |")
        if o == "lhmuon" and vseeds:
            vv = np.array(list(vseeds.values()))
            vsd = f" ± {vv.std(ddof=1):.4f}" if len(vv) > 1 else ""
            vdm, vsem, _ = paired(vseeds, seeds["muon"])
            vda, vsea, _ = paired(vseeds, seeds["adamw"])
            vr = runs[f"s2_lhmuon_{variant}"]["final"]
            vM = "(see section 4)" if fits.get("lhmuon") else "needs stage 4"
            if "muon" in fits:
                m = multiplier(fits["muon"], 131072000, float(vv.mean()))
                vM = f"{m:.2f}×" if m else "below fitted floor"
            L.append(f"| LH-Muon, {ABL_LABEL[variant]} | {best_lr[o]:g} | {len(vv)} | {vv.mean():.4f}{vsd} | {pm(vdm, vsem)} | "
                     f"{pm(vda, vsea)} | {vr['eval_mix']:.4f} | {vM} | {100 * vr['opt_fraction']:.1f}% | "
                     f"{vr['tokens'] / vr['seconds']:,.0f} |")
        summary.append({"optimizer": o, "lr": best_lr[o], "seeds": len(vals), "eval_indist_mean": vals.mean(),
                        "eval_indist_sd": vals.std(ddof=1) if len(vals) > 1 else "", "delta_vs_muon": dm if dm is not None else "",
                        "delta_vs_muon_se": sem if sem is not None else "", "eval_mix_seed0": r["eval_mix"],
                        "opt_fraction": r["opt_fraction"]})
    L.append("")
    if noise:
        L.append(f"Seed noise: the SD of a paired difference between two optimizers is {noise:.4f} nats; "
                 f"a single run's SD across seeds is {run_sd:.4f}. With 3 seeds, a paired Δ smaller than about "
                 f"{2 * noise / math.sqrt(3):.4f} nats (2 SE) is not distinguishable from noise.\n")
    else:
        L.append("Seed noise: not measured yet (stage 3). Until it is, differences below ~0.01 nats should be "
                 "treated as noise.\n")
    L.append("![curves](fig1_curves.png)\n")

    # LR sweep table
    L.append("## 2. LR sweep (final eval_indist, 131M tokens, seed 0)\n")
    grid_all = sorted({lr for o in ORDER for lr in sweep.get(o, {}).get("grid", [])})
    if grid_all:
        L.append("| optimizer | " + " | ".join(f"{lr:g}" for lr in grid_all) + " |")
        L.append("|---|" + "---|" * len(grid_all))
        for o in ORDER:
            cells = []
            for lr in grid_all:
                r = runs.get(f"s1_{o}_lr{fmt_lr(lr)}")
                if r is None:
                    cells.append("")
                elif not ok(r):
                    cells.append("diverged")
                else:
                    v = f"{r['final']['eval_indist']:.4f}"
                    cells.append(f"**{v}**" if best_lr.get(o) == lr else v)
            L.append(f"| {LABEL[o]} | " + " | ".join(cells) + " |")
        L.append("\nA best LR on the edge of its grid means the optimum wasn't bracketed; the suite extends the grid "
                 "automatically in that case (up to twice).\n")
    L.append("![lr sweep](fig2_lr_sweep.png)\n")

    # ablations
    L.append("## 3. LH-Muon ablations (at LH-Muon's best LR, seed 0)\n")
    if ok(runs.get(best_name.get("muon"))):
        r0 = runs[best_name["muon"]]["final"]["eval_indist"]
        L.append("| variant | eval_indist | Δ vs Muon | Δ vs LH-Muon default | optimizer time |")
        L.append("|---|---|---|---|---|")
        lh = runs.get(best_name.get("lhmuon"))
        lh0 = lh["final"]["eval_indist"] if ok(lh) else None
        if ok(lh):
            L.append(f"| default (α 2, sum, wd, int8) | {lh0:.4f} | {lh0 - r0:+.4f} | – | {100 * lh['final']['opt_fraction']:.1f}% |")
        for k, lab in ABL_LABEL.items():
            r = runs.get(f"s2_lhmuon_{k}")
            if r is None:
                continue
            if not ok(r):
                L.append(f"| {lab} | diverged | | | |")
                continue
            v = r["final"]["eval_indist"]
            L.append(f"| {lab} | {v:.4f} | {v - r0:+.4f} | {(v - lh0) if lh0 else float('nan'):+.4f} | "
                     f"{100 * r['final']['opt_fraction']:.1f}% |")
        L.append("\nThese are single seeds. Read them against the seed-noise line in section 1.\n")
        L.append("![ablations](fig3_ablations.png)\n")

    # frontier
    L.append("## 4. Loss vs tokens and the token multiplier (fully decayed endpoints)\n")
    if fits:
        budgets = sorted({p[0] for o in ORDER for p in frontier_points(runs, o)})
        L.append("| optimizer | " + " | ".join(f"{b // 1_000_000}M" for b in budgets) + " | fit L = E + B·D^−β |")
        L.append("|---|" + "---|" * (len(budgets) + 1))
        for o in ORDER:
            pts = dict(frontier_points(runs, o))
            cells = [f"{pts[b]:.4f}" if b in pts else "" for b in budgets]
            f = fits.get(o)
            L.append(f"| {LABEL[o]} | " + " | ".join(cells) + " | " +
                     (f"E {f[0]:.3f}, B {f[1]:.3g}, β {f[2]:.3f}" if f else "–") + " |")
        L.append("")
        for base in ("muon", "adamw"):
            if base not in fits:
                continue
            L.append(f"Token multiplier M = D_{LABEL[base]}(L) / D, at each budget of the other optimizers:\n")
            L.append("| optimizer | " + " | ".join(f"{b // 1_000_000}M" for b in budgets) + " |")
            L.append("|---|" + "---|" * len(budgets))
            Dmin, Dmax = min(p[0] for p in frontier_points(runs, base)), max(p[0] for p in frontier_points(runs, base))
            for o in ORDER:
                if o == base:
                    continue
                pts = dict(frontier_points(runs, o))
                cells = []
                for b in budgets:
                    if b not in pts:
                        cells.append("")
                        continue
                    m = multiplier(fits[base], b, pts[b])
                    if m is None:
                        cells.append("below floor")
                    else:
                        ext = "" if Dmin <= m * b <= Dmax else "*"
                        cells.append(f"{m:.2f}×{ext}")
                L.append(f"| {LABEL[o]} | " + " | ".join(cells) + " |")
            L.append("\n\\* extrapolated beyond the baseline's measured token range.\n")
        L.append("![frontier](fig4_frontier.png)\n")
    else:
        L.append("Not run yet (stage 4).\n")

    # verdict
    L.append("## 5. Verdict\n")
    lh_seeds = vseeds if vseeds else seeds["lhmuon"]
    lh_name = f"LH-Muon ({ABL_LABEL[variant]})" if vseeds else "LH-Muon"
    dm, sem, nm = paired(lh_seeds, seeds["muon"])
    if dm is None:
        L.append("Not enough runs yet.\n")
    else:
        sig = sem is not None and abs(dm) > 2 * sem
        mtxt = ""
        if "muon" in fits and lh_seeds:
            m = multiplier(fits["muon"], 131072000, float(np.mean(list(lh_seeds.values()))))
            if m:
                mtxt = f" That corresponds to a token multiplier of **{m:.2f}×** over Muon at 131M tokens."
        L.append(f"{lh_name} − Muon at 131M tokens, paired over {nm} seed(s): **{pm(dm, sem)} nats**"
                 + (" (more than 2 SE from zero)." if sig else " (within 2 SE of zero, or SE unknown).") + mtxt + "\n")
        L.append("Pre-registered bar (from the design discussion): the gain counts if M ≥ 1.10× over tuned Muon, "
                 "the 2σ band on Δ excludes zero, and the gain doesn't shrink with scale. That last check needs a "
                 "second model size; this suite has one.\n")
    if variant:
        L.append(f"LH-Muon's stage-3 seeds and stage-4 frontier use the best stage-2 variant ({ABL_LABEL[variant]}), "
                 "chosen on a single seed. That selection biases it slightly in LH-Muon's favour.\n")
    L.append("Lion's weight decay (0.7) and betas (0.9, 0.99) were not swept, only its LR, so Lion may be under-tuned "
             "relative to the others.\n")
    L.append("Caveats: a single 47M scale; budgets of 0.7–5.6 tok/param, not the 20 tok/param target; the slow-momentum "
             "horizon is fixed at 300 steps, and its gains reportedly grow with horizon. Periodic evals in fig. 1 use 128 "
             "windows, so they are noisier than the final 512-window evals in the tables.\n")

    open(os.path.join(REPORT, "report.md"), "w").write("\n".join(L))
    if summary:
        with open(os.path.join(REPORT, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    print(f"report: {os.path.join(REPORT, 'report.md')} ({len(runs)} finished runs)")


if __name__ == "__main__":
    main()
