"""
OAM pair on surface codes with the hardware settings (alpha 0.75, 30 iterations,
averaged OSD-0 input): does the two-table pair learned on BB-72 help, and what does
it cost in OSD fallbacks? Decides whether the two-core HLS kernel is worth building.

Pair rule (as in hardware): among the members that converged, take the solution with
the lowest prior-LLR weight; if neither converged, OSD-0 on member A's averaged posterior.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/surface_pair.py
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from eval_rlbp import HERE, build_matcher, default_rounds, load_dem, scaled, wilson
from oam_ensemble import mcnemar
from aura.decoding import MinSumBP, TannerGraph, finish_with_fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default="surface:3,surface:5")
    ap.add_argument("--scales", default="0.5,1,2")
    ap.add_argument("--shots", default="100000,50000", help="per code")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    res = HERE / "results"
    t16 = np.array(json.loads((res / "oam_gamma_bb72.json").read_text())["gamma"], np.float32)
    t48 = np.array(json.loads((res / "oam2_tables_unsat_L1_S1.json").read_text())["tables"][0], np.float32)
    rows = []
    for code, shots in zip(a.codes.split(","), [int(x) for x in a.shots.split(",")]):
        dem = load_dem(code, default_rounds(code), "sota", a.seed, HERE / "cache")
        for lam in [float(x) for x in a.scales.split(",")]:
            g = TannerGraph(scaled(dem, lam))
            _, s, o = g.sample(shots, np.random.default_rng([a.seed, 7, int(lam * 1000)]))   # = surface_oam.py
            mk = lambda t: MinSumBP(g, max_iter=30, alpha=0.75, posterior_avg=2, memory_table=t)
            ra, rb = mk(t16).decode(s), mk(t48).decode(s)
            pa, fa = finish_with_fallback(g, s, ra)
            pb_, fb_ = finish_with_fallback(g, s, rb)
            ca = ra.converged | ~s.any(axis=1)
            cb = rb.converged | ~s.any(axis=1)
            wa = np.where(ca, ra.errors.astype(np.float32) @ g.prior_llr, np.inf)
            wb = np.where(cb, rb.errors.astype(np.float32) @ g.prior_llr, np.inf)
            pp = np.where((wb < wa)[:, None], pb_, pa)          # neither converged -> pa = OSD on member A
            fp = ~(ca | cb)
            it_a, it_b = ra.iterations, rb.iterations
            dec = {"oam16_osd": ((pa != o).any(1), fa, it_a), "oam48_osd": ((pb_ != o).any(1), fb_, it_b),
                   "pair_osd": ((pp != o).any(1), fp, np.maximum(it_a, it_b))}
            mt = build_matcher(g)
            if mt is not None:
                dec["mwpm"] = ((mt.decode_batch(s) != o).any(1), np.zeros(len(s), bool), np.zeros(len(s)))
            ref = dec["oam16_osd"][0]
            for name, (w, fb, it) in dec.items():
                k = int(w.sum())
                lo, hi = wilson(k, len(s))
                p, b, c = mcnemar(w, ref)
                rows.append({"code": code, "scale": lam, "decoder": name, "shots": len(s), "errors": k,
                             "ler": k / len(s), "ci_lo": lo, "ci_hi": hi, "fallback": float(fb.mean()),
                             "mean_iter_parallel": float(it.mean()), "mcnemar_vs_oam16_p": p,
                             "only_this_wrong": b, "only_oam16_wrong": c})
                print(f"{code} lambda={lam:<4g} {name:<10} LER {k / len(s):.2e} [{lo:.2e},{hi:.2e}]  fallback "
                      f"{fb.mean():.4f}  iters(par) {it.mean():5.2f}  vs oam16 {b}/{c} p={p:.3g}", flush=True)
    out = res / "surface_pair.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
