"""
fit_multiplier.py -- token multiplier of a candidate over a baseline, from fully decayed runs.

Input: a CSV with columns  arm,tokens,loss  -- one row per FULLY DECAYED endpoint (a WSD branch or a
separate run), never an intermediate point of a scheduled run. For each baseline arm, fits

    L(D) = E + B * D^(-beta)

to its endpoints, then for every candidate endpoint (D_c, L_c) reports

    M = D_baseline(L_c) / D_c        with  D_baseline(L) = (B / (L - E))^(1 / beta)

plus the local-slope conversion M ~= exp(dL / (beta (L - E))) as a sanity check. M is only
meaningful when L_c lies inside (or just beyond) the baseline's fitted range; the script flags
extrapolation.

    python scripts/fit_multiplier.py endpoints.csv --baseline muon --baseline adamw --candidate lhmuon
"""

import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def fit(D, L):
    """Least squares in L-space over a grid of E; for each E, log(L - E) is linear in log D."""
    D, L = np.asarray(D, float), np.asarray(L, float)
    best = None
    for E in np.linspace(0.0, L.min() - 1e-4, 4000):
        y = np.log(L - E)
        A = np.stack([np.ones_like(D), -np.log(D)], 1)
        (logB, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
        if beta <= 0:
            continue
        pred = E + np.exp(logB) * D ** (-beta)
        sse = float(((pred - L) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, E, math.exp(logB), beta)
    if best is None:
        raise ValueError("no decreasing power law fits these points")
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--baseline", action="append", required=True)
    ap.add_argument("--candidate", action="append", required=True)
    a = ap.parse_args()
    runs = defaultdict(list)
    for row in csv.DictReader(open(a.csv)):
        runs[row["arm"]].append((float(row["tokens"]), float(row["loss"])))
    for base in a.baseline:
        pts = sorted(runs[base])
        if len(pts) < 3:
            raise SystemExit(f"baseline {base!r} needs >= 3 decayed endpoints, has {len(pts)} (4-5 recommended)")
        D, L = zip(*pts)
        E, B, beta = fit(D, L)
        resid = max(abs(E + B * d ** -beta - l) for d, l in pts)
        print(f"{base}: L = {E:.4f} + {B:.4g} * D^-{beta:.4f}   (max residual {resid:.4f}, "
              f"range {min(D) / 1e6:.0f}M..{max(D) / 1e6:.0f}M tokens)")
        for cand in a.candidate:
            for Dc, Lc in sorted(runs[cand]):
                if Lc <= E:
                    print(f"  {cand} @ {Dc / 1e6:.0f}M: loss {Lc:.4f} is below the fitted floor E -- no multiplier")
                    continue
                Db = (B / (Lc - E)) ** (1 / beta)
                Lb = E + B * Dc ** -beta
                slope_M = math.exp((Lb - Lc) / (beta * (Lb - E)))
                flag = "" if min(D) <= Db <= max(D) else "  (EXTRAPOLATED beyond the baseline's range)"
                print(f"  {cand} @ {Dc / 1e6:.0f}M tokens: loss {Lc:.4f} vs baseline {Lb:.4f} (dL {Lb - Lc:+.4f}) "
                      f"-> M = {Db / Dc:.3f}x  (slope approx {slope_M:.3f}x){flag}")


if __name__ == "__main__":
    main()
