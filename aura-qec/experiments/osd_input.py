"""
Which BP posterior should OSD-0 receive?  (BB codes: BP oscillates, so the
posterior at the moment BP stops is not necessarily the best one.)

For each stopping rule (run to the budget, or the stall heuristic with K),
OSD-0 is run on the following posteriors of every non-converged shot:
  final   posterior at the iteration BP stopped           (current decoder)
  best    posterior at the iteration with the fewest unsatisfied checks
  avg2    mean of the last two iterations' posteriors      (cancels period-2 oscillation)
  mean    running mean over all iterations (likelihood accumulation, Jiang et al. 2007 /
          Li & Feng arXiv:0710.5230)
  minllr  running minimum of each LLR over all iterations (NVIDIA cudaq-qec "min_llr" OSD init)
The hardware cost of best/avg2/mean is one extra V-entry register bank.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/osd_input.py --code bb:72 --scales 0.5,1 --shots 10000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, StallPolicy, default_rounds, load_dem, scaled, wilson
from aura.decoding import MinSumBP, TannerGraph, osd0

INPUTS = ("final", "best", "avg2", "mean", "minllr")


class Tracker:
    """BP observer keeping per-shot best / previous / running-sum posteriors."""

    def __init__(self, shots: int, num_vars: int):
        self.best_unsat = np.full(shots, np.iinfo(np.int32).max, np.int64)
        self.best = np.zeros((shots, num_vars), np.float32)
        self.prev = np.zeros((shots, num_vars), np.float32)
        self.cur = np.zeros((shots, num_vars), np.float32)
        self.sum = np.zeros((shots, num_vars), np.float32)
        self.min = np.full((shots, num_vars), np.inf, np.float32)
        self.count = np.zeros(shots, np.int64)

    def __call__(self, f, done, halted, tot):
        idx = f.shot_index
        post = tot.T                                   # (A, V)
        better = f.unsat < self.best_unsat[idx]
        if better.any():
            self.best[idx[better]] = post[better]
            self.best_unsat[idx[better]] = f.unsat[better]
        self.prev[idx] = np.where((self.count[idx] > 0)[:, None], self.cur[idx], post)
        self.cur[idx] = post
        self.sum[idx] += post
        self.min[idx] = np.minimum(self.min[idx], post)
        self.count[idx] += 1

    def posterior(self, kind: str, i: int, final: np.ndarray) -> np.ndarray:
        if kind == "final" or self.count[i] == 0:
            return final
        if kind == "best":
            return self.best[i]
        if kind == "avg2":
            return 0.5 * (self.prev[i] + self.cur[i])
        if kind == "minllr":
            return self.min[i]
        return self.sum[i] / self.count[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--scales", default="0.5,1")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--stall-k", default="0,4,8", help="0 = no halting (run to the budget)")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "osd_input.csv"))
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        bp = MinSumBP(g, max_iter=a.max_iter)
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        for k in [int(x) for x in a.stall_k.split(",")]:
            tr = Tracker(a.shots, g.num_vars)
            res = bp.decode(s, policy=StallPolicy(k) if k > 0 else None, observer=tr)
            fb = np.nonzero(~res.converged)[0]
            base = g.logical(res.errors)
            for kind in INPUTS:
                pred = base.copy()
                for i in fb:
                    pred[i], _, _ = osd0(g, s[i], tr.posterior(kind, i, res.posterior[i]))
                wrong = int((pred != o).any(axis=1).sum())
                lo, hi = wilson(wrong, a.shots)
                rule = f"stall{k}" if k else "budget"
                rows.append({"code": a.code, "rounds": rounds, "scale": lam, "stop": rule, "osd_input": kind,
                             "shots": a.shots, "errors": wrong, "ler": wrong / a.shots, "ci_lo": lo, "ci_hi": hi,
                             "mean_iter": float(res.iterations.mean()), "fallback": float(len(fb) / a.shots)})
                print(f"lambda={lam:<4g} {rule:<8} osd({kind:<5}) LER {wrong / a.shots:.2e} [{lo:.2e},{hi:.2e}]"
                      f"  iters {res.iterations.mean():.2f}  fallback {len(fb) / a.shots:.3f}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
