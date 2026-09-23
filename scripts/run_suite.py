"""
run_suite.py -- the systematic optimizer comparison: AdamW, Lion, Muon, LH-Muon on the 47M GPT.

Stages (each one uses the previous stage's results; the report is regenerated after every stage):

  1 LR sweep     every optimizer on a x2 grid at 131M tokens (2.8 tok/param), seed 0. If the best LR is
                 on the grid's edge the grid is extended in that direction (up to twice): a sweep
                 whose optimum sits on its boundary has not found the optimum.
  2 ablations    LH-Muon components at its best LR: alpha 0.25 / 0.5 / 1 / 5, alpha decaying with the LR
                 in the decay phase, slow horizon 100, soft eps
                 (kappa 1), combine=separate, norm_control=sphere, int4 state. If a variant beats the
                 default, stages 3 and 4 use that variant for LH-Muon (runs/suite/lh_variant.json).
  3 seeds        seeds 1 and 2 for every optimizer's best config (and the best LH-Muon ablation if it
                 beat the default). Same seed = same init and same data order across optimizers, so
                 differences are compared PAIRED by seed.
  4 frontier     per optimizer (best config): one 262M-token WSD run that saves stable checkpoints at
                 the decay start of the 33M / 66M / 131M budgets, then a decay branch from each. The
                 fully decayed endpoints give each optimizer's loss-vs-tokens curve, which is what the
                 token multiplier is measured on.

Runs go to runs/suite/<name>/; a run with final.json is skipped, so the suite is resumable (kill it
and start it again). Two jobs at a time, one per GPU in --gpus.

    nohup python scripts/run_suite.py > runs/suite/suite.log 2>&1 &
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "suite")
TRAIN = os.path.join(ROOT, "examples", "train_lm.py")
TOK_STEP = 64 * 1024
BUDGET = 131072000                         # 2000 steps
FRONTIER = [32768000, 65536000, 131072000, 262144000]   # 500 / 1000 / 2000 / 4000 steps
SLOW_HORIZON = 300                          # fixed for every LH-Muon run, so branches share it

OPTS = {
    #          LR grid (x2 steps)        extra args
    "adamw":  ([1e-3, 2e-3, 4e-3], ["--wd", "0.1"]),
    "muon":   ([1e-3, 2e-3, 4e-3], ["--wd", "0.1"]),
    "lhmuon": ([1e-3, 2e-3, 4e-3], ["--wd", "0.1", "--alpha", "2", "--slow-horizon", str(SLOW_HORIZON)]),
    "lion":   ([1.5e-4, 3e-4, 6e-4], ["--wd", "0.7"]),     # lr x wd ~ AdamW's at the grid centre
}
ABLATIONS = {
    "alpha0.25": ["--alpha", "0.25"],
    "alpha0.5": ["--alpha", "0.5"],
    "alpha1": ["--alpha", "1"],
    "alpha5": ["--alpha", "5"],
    "soft1": ["--soft-kappa", "1"],
    "separate": ["--combine", "separate", "--alpha", "0.5"],
    "sphere": ["--norm-control", "sphere"],
    "int4": ["--state-dtype", "int4", "--slow-dtype", "int4"],
    "h100": ["--slow-horizon", "100"],
    "alphadecay": ["--alpha-decay"],
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fmt_lr(lr):
    return f"{lr:.2e}".replace("e-0", "e-")


def final(name):
    p = os.path.join(OUT, name, "final.json")
    return json.load(open(p)) if os.path.exists(p) else None


def score(name):
    f = final(name)
    if f is None or f["diverged"]:
        return float("inf")
    return f["eval_indist"]


def run_jobs(jobs, gpus):
    """jobs: [(name, [args...])]. Runs each on a free GPU; skips finished ones."""
    todo = [(n, a) for n, a in jobs if final(n) is None]
    for n, _ in jobs:
        if final(n) is not None:
            log(f"skip {n} (done)")
    running = {}
    while todo or running:
        for g in gpus:
            if g not in running and todo:
                name, args = todo.pop(0)
                d = os.path.join(OUT, name)
                if os.path.isdir(d):
                    shutil.rmtree(d)                       # a crashed run: start clean (log.csv appends)
                os.makedirs(d, exist_ok=True)
                cmd = [sys.executable, TRAIN, "--out", d, "--device", g] + args
                log(f"start {name} on {g}: {' '.join(args)}")
                running[g] = (name, subprocess.Popen(cmd, stdout=open(os.path.join(d, "stdout.txt"), "w"),
                                                      stderr=subprocess.STDOUT))
        time.sleep(10)
        for g, (name, proc) in list(running.items()):
            if proc.poll() is not None:
                f = final(name)
                state = "FAILED (no final.json)" if f is None else ("diverged" if f["diverged"] else
                                                                     f"eval_indist {f['eval_indist']:.4f}")
                log(f"done  {name} rc={proc.returncode}: {state}")
                del running[g]


def analyze():
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze_suite.py")], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="cuda:0,cuda:1")
    ap.add_argument("--stages", default="1,2,3,4")
    a = ap.parse_args()
    gpus = a.gpus.split(",")
    stages = set(a.stages.split(","))
    os.makedirs(OUT, exist_ok=True)

    def base(opt, lr, seed=0, tokens=BUDGET):
        return ["--opt", opt, "--lr", str(lr), "--tokens", str(tokens), "--seed", str(seed)] + OPTS[opt][1]

    # ---------------- stage 1: LR sweep
    grids = {o: list(OPTS[o][0]) for o in OPTS}

    def sweep_name(o, lr):
        return f"s1_{o}_lr{fmt_lr(lr)}"

    def best_lr(o):
        return min(grids[o], key=lambda lr: score(sweep_name(o, lr)))

    if "1" in stages:
        log("=== stage 1: LR sweep")
        run_jobs([(sweep_name(o, lr), base(o, lr)) for o in OPTS for lr in grids[o]], gpus)
        for _ in range(2):
            ext = []
            for o in OPTS:
                b = best_lr(o)
                if b == min(grids[o]):
                    grids[o].insert(0, b / 2)
                    ext.append((sweep_name(o, b / 2), base(o, b / 2)))
                elif b == max(grids[o]):
                    grids[o].append(b * 2)
                    ext.append((sweep_name(o, b * 2), base(o, b * 2)))
            if not ext:
                break
            log(f"best LR on the grid edge -> extending: {[n for n, _ in ext]}")
            run_jobs(ext, gpus)
        json.dump({o: {"grid": sorted(grids[o]), "best": best_lr(o)} for o in OPTS},
                  open(os.path.join(OUT, "lr_sweep.json"), "w"), indent=1)
        analyze()
    sweep = json.load(open(os.path.join(OUT, "lr_sweep.json")))
    best = {o: sweep[o]["best"] for o in OPTS}
    best_name = {o: sweep_name(o, best[o]) for o in OPTS}
    best_args = {o: base(o, best[o]) for o in OPTS}
    log(f"best LRs: {best}")

    # ---------------- stage 2: LH-Muon ablations
    if "2" in stages:
        log("=== stage 2: LH-Muon ablations")
        run_jobs([(f"s2_lhmuon_{k}", best_args["lhmuon"] + v) for k, v in ABLATIONS.items()], gpus)
        analyze()
    abl = {k: score(f"s2_lhmuon_{k}") for k in ABLATIONS}
    best_abl = min(abl, key=abl.get) if abl else None
    promote = best_abl is not None and abl[best_abl] < score(best_name["lhmuon"])
    lh_extra = ABLATIONS[best_abl] if promote else []       # stages 3-4 use the best LH-Muon variant
    json.dump({"variant": best_abl if promote else None, "args": lh_extra},
              open(os.path.join(OUT, "lh_variant.json"), "w"), indent=1)
    if promote:
        log(f"LH-Muon variant {best_abl} beat the default ({abl[best_abl]:.4f} vs {score(best_name['lhmuon']):.4f}); "
            f"stages 3-4 use it")

    # ---------------- stage 3: seeds
    if "3" in stages:
        log("=== stage 3: seeds")
        jobs = []
        for seed in (1, 2):
            for o in OPTS:
                jobs.append((f"s3_{o}_seed{seed}", base(o, best[o], seed)))
            if promote:
                jobs.append((f"s3_lhmuon_{best_abl}_seed{seed}", base("lhmuon", best["lhmuon"], seed) + ABLATIONS[best_abl]))
        run_jobs(jobs, gpus)
        analyze()

    # ---------------- stage 4: frontier
    if "4" in stages:
        log("=== stage 4: frontier (WSD branches)")
        top = FRONTIER[-1]
        points = {b: int(int(b // TOK_STEP) * 0.8) * TOK_STEP for b in FRONTIER[:-1]}   # decay start of each budget
        save = ",".join(str(t) for t in points.values())
        extra = {o: (lh_extra if o == "lhmuon" else []) for o in OPTS}
        run_jobs([(f"s4_{o}_long", base(o, best[o], tokens=top) + extra[o] + ["--save-at", save]) for o in OPTS], gpus)
        run_jobs([(f"s4_{o}_b{b // 1_000_000}M",
                   base(o, best[o], tokens=b) + extra[o] + ["--init-from", os.path.join(OUT, f"s4_{o}_long", f"stable_{t}.pt")])
                  for o in OPTS for b, t in points.items()], gpus)
        analyze()
    log("suite finished")


if __name__ == "__main__":
    main()
