"""
analyze.py -- table + figure for a continued-pretraining optimizer test (examples/qwen/optimizer_test.py).

    python examples/qwen/analyze.py              # reads runs/qwen/*/log.csv, writes runs/qwen/report.md + optimizers.png
    python examples/qwen/analyze.py runs/gpt2

Reports whatever has finished. For each optimizer the best LR (lowest final eval) is plotted; the
table lists every run. Lower eval loss = better; "vs base" is against the untouched base model.
"""

import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "qwen")
ORDER = ["lhmuon", "muon", "adamw", "lion"]
LABEL = {"lhmuon": "LH-Muon (α 0.25)", "muon": "Muon", "adamw": "AdamW", "lion": "Lion"}
COLOR = {"lhmuon": "#2a78d6", "muon": "#eb6834", "adamw": "#1baf7a", "lion": "#eda100"}
MARK = {"lhmuon": "o", "muon": "s", "adamw": "^", "lion": "D"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load():
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_lr*")) + glob.glob(os.path.join(RUNS, "runs", "*_lr*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "log.csv")):
            continue
        rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
        if not rows:
            continue
        fin = json.load(open(os.path.join(d, "final.json"))) if os.path.exists(os.path.join(d, "final.json")) else None
        opt, lr = os.path.basename(d).split("_lr")
        runs.append({"opt": opt, "lr": float(lr), "rows": rows, "final": fin, "name": os.path.basename(d)})
    return runs


def main():
    runs = load()
    if not runs:
        print("no runs yet")
        return
    done = [r for r in runs if r["final"]]
    best = {}
    for r in done:
        if r["opt"] not in best or r["final"]["eval"] < best[r["opt"]]["final"]["eval"]:
            best[r["opt"]] = r
    base = float(runs[0]["rows"][0]["eval_loss"])
    args = (done or runs)[0]["final"]["args"] if done else {}
    model = args.get("model", "Qwen/Qwen3.5-0.8B-Base").split("/")[-1]
    steps, seqs, warm, dfrac = args.get("steps", 200), args.get("seqs", 32), args.get("warmup", 20), args.get("decay_frac", 0.2)
    srcs = [k[5:] for k in runs[0]["rows"][0] if k.startswith("eval_") and k != "eval_loss"]

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0, color=INK2, linewidth=1)
    ax.annotate("base model", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                fontsize=9, color=INK2)
    shown = best if best else {r["opt"]: r for r in runs}
    ends = []
    for o in ORDER:
        if o not in shown:
            continue
        r = shown[o]
        x = [int(row["tokens"]) / 1e6 for row in r["rows"]]
        y = [float(row["eval_loss"]) - base for row in r["rows"]]
        ax.plot(x, y, "-" + MARK[o], color=COLOR[o], label=f"{LABEL[o]}, lr {r['lr']:g}", markeredgecolor=SURF,
                markeredgewidth=1.5)
        ends.append([y[-1], x[-1], f"{y[-1]:+.3f}", COLOR[o]])
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)                        # spread the end labels so equal finishes stay readable
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for ly, lx, txt, col in ends:
        ax.annotate(txt, (lx, ly), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=col,
                    fontweight="bold")
    ax.set_title(f"{model}: held-out loss vs the base model (best LR per optimizer)")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("Δ eval loss vs base model (nats; below 0 = better)")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 * 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "optimizers.png"), dpi=150)

    L = [f"# {model}: optimizers on continued pretraining\n",
         f"Continued pretraining on an equal mix of C4, Common Crawl, books, MegaWika and arXiv (Dolma v1.7), "
         f"{steps} steps × {seqs} × 1024 tokens = {steps * seqs * 1024 / 1e6:.1f}M tokens, WSD (warmup {warm}, linear decay over "
         f"the last {dfrac:.0%}), bf16 weights "
         f"with stochastic rounding, all optimizers through LHMuon. Embeddings/head: factored Adam in every run. "
         f"Base-model eval loss: **{base:.4f}** (64 held-out windows).\n",
         "| optimizer | LR | final eval | vs base | " + " | ".join(srcs) + " | time | peak GiB |",
         "|---|---|---|---|" + "---|" * len(srcs) + "---|---|"]
    for r in sorted(done, key=lambda r: (ORDER.index(r["opt"]), r["lr"])):
        f = r["final"]
        mark = "**" if best.get(r["opt"]) is r else ""
        L.append(f"| {LABEL[r['opt']]} | {r['lr']:g} | {mark}{f['eval']:.4f}{mark} | {f['eval'] - base:+.4f} | "
                 + " | ".join(f"{f['eval_by_source'][s] - f['base_by_source'][s]:+.3f}" for s in srcs)
                 + f" | {f['seconds'] / 60:.0f} min | {f['peak_gib']:.1f} |")
    L.append(f"\nPer-source columns are the change vs the base model on that source. {len(done)} of {len(runs)} runs finished.\n")
    L.append("![curves](optimizers.png)\n")
    open(os.path.join(RUNS, "report.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
