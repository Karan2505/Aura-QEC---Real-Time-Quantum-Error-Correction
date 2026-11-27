"""
Two-table OAM ensemble: run the 16-state OAM-BP table and the check-aware 48-state
table (both already learned, no training here) on the same shots and keep the
lowest-weight solution (weight = sum of prior LLRs of the flipped variables).

Variants
  pair_both     always run both members; iterations = sum of both runs
  pair_seq      run member 2 only when member 1 did not converge
                (the cheap version; it cannot rescue converged-but-wrong shots)

Test shots are the same seeds as oam_bp.py / oam2.py, so all McNemar tests are paired
against the single 16-state OAM-BP.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/oam_pair.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from oam_bp import ler_of
from oam_ensemble import mcnemar
from oam2 import OAM2
from aura.decoding import TannerGraph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--scales", default="1,0.7,0.5")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--cap", type=int, default=200, help="iteration cap per member")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    res = HERE / "results"
    t16 = np.array(json.loads((res / "oam_gamma_bb72.json").read_text())["gamma"], np.float32)
    t48 = np.array(json.loads((res / "oam2_tables_unsat_L1_S1.json").read_text())["tables"][0], np.float32)
    dem = load_dem(a.code, default_rounds(a.code), "sota", a.seed, HERE / "cache")
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        t0 = time.perf_counter()
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        p1, ok1, it1, w1 = OAM2(g, "base", (a.cap,)).decode(s, t16[None])
        p2, ok2, it2, w2 = OAM2(g, "unsat", (a.cap,)).decode(s, t48[None])
        wr1 = ler_of(p1, ok1, s, o)
        wr2 = ler_of(p2, ok2, s, o)
        take2 = w2 < w1                                   # inf when unsolved, so solved beats unsolved
        pb = np.where(take2[:, None], p2, p1)
        wr_both = ler_of(pb, ok1 | ok2, s, o)
        seq2 = ~ok1                                       # member 2 only on non-converged shots
        ps = np.where((seq2 & take2)[:, None], p2, p1)
        wr_seq = ler_of(ps, ok1 | (seq2 & ok2), s, o)
        variants = (("oam16", wr1, it1), ("oam48", wr2, it2), ("pair_both", wr_both, it1 + it2),
                    ("pair_seq", wr_seq, it1 + np.where(seq2, it2, 0)))
        for name, w, its in variants:
            k = int(w.sum())
            lo, hi = wilson(k, len(s))
            p, b, c = mcnemar(w, wr1)
            rows.append({"code": a.code, "scale": lam, "decoder": name, "shots": len(s), "errors": k,
                         "ler": k / len(s), "ci_lo": lo, "ci_hi": hi, "mean_iter": float(its.mean()),
                         "p99_iter": float(np.percentile(its, 99)), "max_iter": int(its.max()),
                         "mcnemar_vs_oam16_p": p, "only_this_wrong": b, "only_oam16_wrong": c})
            print(f"lambda={lam:<4g} {name:<10} LER {k / len(s):.2e} [{lo:.2e},{hi:.2e}]  iters mean "
                  f"{its.mean():5.1f} p99 {np.percentile(its, 99):4.0f} max {its.max():4d}  "
                  f"vs oam16: {b}/{c} p={p:.3g}", flush=True)
        print(f"  members disagree on {int((wr1 != wr2).sum())} shots ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    out = res / f"oam_pair_cap{a.cap}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
