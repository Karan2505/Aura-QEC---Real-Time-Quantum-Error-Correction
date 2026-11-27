"""
OAM-BP as ensemble members: run a learned OAM-BP member first, then fixed-strength
memory-BP members, keep the lowest-weight solution, stop early on agreement.
Also reports paired (McNemar) significance against single-run baselines, since
every decoder sees the same test shots.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/oam_ensemble.py --code bb:72 --scales 1,0.5,0.7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from oam_bp import NSTATE, OAMBP, ler_of
from rl_ensemble import Episodes, evaluate
from aura.decoding import TannerGraph


def mcnemar(wrong_a, wrong_b):
    """Exact two-sided McNemar p-value for paired error indicators."""
    b = int((wrong_a & ~wrong_b).sum())
    c = int((~wrong_a & wrong_b).sum())
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), b, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--scales", default="1,0.5,0.7")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--gamma-file", default=str(HERE / "results" / "oam_gamma_bb72.json"))
    ap.add_argument("--members", default="oam,0.125,0.4,0.06", help="'oam' = learned table, numbers = fixed gamma")
    ap.add_argument("--member-budget", type=int, default=50)
    ap.add_argument("--single-budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "oam_ensemble.csv"))
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    learned = np.array(json.loads(Path(a.gamma_file).read_text())["gamma"])
    specs = [m.strip() for m in a.members.split(",")]
    tables = [learned if m == "oam" else np.full(NSTATE, float(m)) for m in specs]
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        # single-run references at the full worst-case budget (paired: same shots)
        single = {}
        for name, G in (("oam_single", learned), ("fixed_0.25_single", np.full(NSTATE, 0.25)),
                        ("fixed_0.125_single", np.full(NSTATE, 0.125))):
            pred, ok, it, _ = OAMBP(g, a.single_budget).decode(s, G)
            single[name] = (ler_of(pred, ok, s, o), it)
        # ensemble members
        K, B = len(tables), len(s)
        P = np.zeros((K, B, g.num_observables), np.uint8)
        W = np.zeros((K, B))
        IT = np.zeros((K, B), np.int64)
        OK = np.zeros((K, B), bool)
        for k, G in enumerate(tables):
            P[k], OK[k], IT[k], W[k] = OAMBP(g, a.member_budget).decode(s, G)
        ep = Episodes((P, W, IT, OK), o)

        def add(name, wrong_or_res, it=None, members=1.0):
            if it is not None:
                wrong = wrong_or_res
                err, mean_it, p99, mx = int(wrong.sum()), it.mean(), np.percentile(it, 99), int(it.max())
            else:
                ler_, err, mean_it, p99, mx, members = wrong_or_res
            lo, hi = wilson(err, B)
            rows.append({"code": a.code, "scale": lam, "decoder": name, "shots": B, "errors": err, "ler": err / B,
                         "ci_lo": lo, "ci_hi": hi, "mean_iter": float(mean_it), "p99_iter": float(p99),
                         "max_iter": mx, "mean_members": float(members), "members": a.members})
            print(f"lambda={lam:<4g} {name:<26} LER {err / B:.2e} [{lo:.2e},{hi:.2e}]  iters mean {mean_it:5.1f} "
                  f"p99 {p99:4.0f} max {mx:4d}  members {members:.2f}", flush=True)

        for name, (wrong, it) in single.items():
            add(name, wrong, it)
        for k in range(K):
            add(f"ens_fixed{k + 1}", evaluate(ep, lambda kk, e, k=k: np.full(e.B, kk >= k)))
        add("ens_agree_stop", evaluate(ep, lambda kk, e: e.agree[kk] == 1))
        # paired significance of the headline comparisons
        w_oam = single["oam_single"][0]
        for ref in ("fixed_0.25_single", "fixed_0.125_single"):
            p, b, c = mcnemar(w_oam, single[ref][0])
            print(f"lambda={lam:<4g} McNemar oam_single vs {ref}: only-OAM-wrong {b}, only-ref-wrong {c}, p = {p:.3g}")
        final = np.full(B, K - 1)
        w_ens = ep.wrong[K - 1]
        p, b, c = mcnemar(w_ens, w_oam)
        print(f"lambda={lam:<4g} McNemar ens_all vs oam_single: only-ens-wrong {b}, only-oam-wrong {c}, p = {p:.3g}")
    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
