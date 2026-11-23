"""
OAM-BP extensions (all deterministic, all per-variable, all learned by policy search):

  --feat unsat   check-aware state: the 16 OAM states x (number of unsatisfied checks
                 next to the variable: 0, 1, 2+) -> 48 states. The unsatisfied count is
                 the bit-flipping decoders' signal; in hardware it is a popcount over
                 the syndrome-check bits the decoder already computes.
  --legs K       OAM-Relay: K legs with their own learned tables. A leg ends when it
                 converges or after --leg-iters iterations; the next leg starts warm
                 (messages and memory kept), its flip history reset. Up to --solutions
                 solutions are collected and the lowest-weight one is returned.

Tables are learned leg by leg with the cross-entropy method (earlier legs fixed).
Test shots are the same seeds as oam_bp.py / oam_ensemble.py, so results are paired.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/oam2.py --feat unsat --legs 1
  PYTHONPATH=src python experiments/oam2.py --feat unsat --legs 3 --solutions 2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from oam_bp import OAMBP, POP4, ler_of
from oam_ensemble import mcnemar
from aura.decoding import TannerGraph


class OAM2(OAMBP):
    def __init__(self, g, feat="base", leg_iters=(200,), solutions=1, t_split=8, alpha=1.0):
        super().__init__(g, sum(leg_iters), t_split, alpha)
        self.feat, self.leg_iters, self.S = feat, np.array(leg_iters), solutions
        self.nstate = 48 if feat == "unsat" else 16

    def decode(self, syndromes, tables):
        """tables: (K, nstate). Returns pred, solved, total iterations, weight."""
        g = self.g
        T = np.asarray(tables, np.float32).reshape(len(self.leg_iters), self.nstate)
        K = len(T)
        B, V = len(syndromes), g.num_vars
        lam0 = g.prior_llr.astype(np.float32)
        lam0c = lam0[:, None]
        pred = np.zeros((B, g.num_observables), np.uint8)
        solved = ~syndromes.any(axis=1)
        iters = np.zeros(B, np.int64)
        weight = np.where(solved, 0.0, np.inf)
        todo = np.nonzero(~solved)[0]
        for c0 in range(0, len(todo), 256):
            idx = todo[c0:c0 + 256]
            A = len(idx)
            s = np.ascontiguousarray(syndromes[idx][:, g.active_checks].T)
            lam = np.repeat(lam0c, A, axis=1)
            m_vc = lam[g.e_var]
            hard_prev = np.zeros((V, A), bool)
            hist = np.zeros((V, A), np.int64)
            leg = np.zeros(A, np.int64)
            t_leg = np.zeros(A, np.int64)
            nsol = np.zeros(A, np.int64)
            alive = np.arange(A)
            t = 0
            while len(alive):
                t += 1
                t_leg[alive] += 1
                m_cv = self._check(m_vc[:, alive], s[:, alive])
                M = (g.var_sum @ m_cv) + lam[:, alive]
                m_vc[:, alive] = M[g.e_var] - m_cv
                hard = M < 0
                flips = hard != hard_prev[:, alive]
                hard_prev[:, alive] = hard
                h = ((hist[:, alive] << 1) | flips) & 0xF
                hist[:, alive] = h
                unsat = (g.h_active @ hard.view(np.uint8).astype(np.float32)).astype(np.uint8) & 1 ^ s[:, alive]
                ok = ~unsat.any(axis=0)
                st = (np.minimum(POP4[h], 3) * 2 + hard) * 2 + (t_leg[alive] > self.t_split)[None, :]
                if self.feat == "unsat":
                    cnt = np.asarray(g.h_active.T @ unsat.astype(np.float32)).astype(np.int64)
                    st = st * 3 + np.minimum(cnt, 2)
                gam = T[leg[alive][None, :], st]
                lam[:, alive] = (1 - gam) * lam0c + gam * M
                # record solutions
                rows_ok = np.nonzero(ok)[0]
                if len(rows_ok):
                    a_ok = alive[rows_ok]
                    hd = hard[:, rows_ok]
                    w = lam0 @ hd.astype(np.float32)
                    better = w < weight[idx[a_ok]]
                    if better.any():
                        r = idx[a_ok[better]]
                        weight[r] = w[better]
                        pred[r] = g.logical(hd[:, better].T.astype(np.uint8))
                        solved[r] = True
                    nsol[a_ok] += 1
                # leg transitions: converged or leg budget spent -> next leg (warm), history reset
                end_leg = ok | (t_leg[alive] >= self.leg_iters[leg[alive]])
                fin = end_leg & ((leg[alive] + 1 >= K) | (nsol[alive] >= self.S))
                nxt = end_leg & ~fin
                if nxt.any():
                    a_n = alive[nxt]
                    leg[a_n] += 1
                    t_leg[a_n] = 0
                    hist[:, a_n] = 0
                if fin.any():
                    a_f = alive[fin]
                    iters[idx[a_f]] = t
                    nos = ~solved[idx[a_f]]
                    if nos.any():                     # never solved: report the last hard decision
                        pred[idx[a_f[nos]]] = g.logical(hard[:, fin][:, nos].T.astype(np.uint8))
                    alive = alive[~fin]
        return pred, solved, iters, weight


def cem(evalf, dim, init, gens, pop, elite, c_iter, seed, lo=-0.25, hi=0.7, sd0=0.15, log=print, ckpt=None):
    rng = np.random.default_rng(seed)
    mu, sd = init.astype(float).copy(), np.full(dim, sd0)
    start = 0
    if ckpt is not None and ckpt.exists():                 # resume after an interruption
        c = json.loads(ckpt.read_text())
        mu, sd, start = np.array(c["mu"]), np.array(c["sd"]), c["gen"]
        rng = np.random.default_rng([seed, start])
        log(f"  resuming at generation {start + 1}")
    for gen in range(start, gens):
        cand = np.clip(mu + sd * rng.standard_normal((pop, dim)), lo, hi)
        cand[0] = mu
        sc = [evalf(c) for c in cand]
        obj = np.array([l + c_iter * it for l, it in sc])
        order = np.argsort(obj)
        el = cand[order[:elite]]
        mu = 0.7 * el.mean(axis=0) + 0.3 * mu
        sd = np.maximum(0.7 * el.std(axis=0) + 0.3 * sd, 0.02)
        if ckpt is not None:
            ckpt.write_text(json.dumps({"gen": gen + 1, "mu": mu.tolist(), "sd": sd.tolist()}))
            rng = np.random.default_rng([seed, gen + 1])
        log(f"  gen {gen + 1}/{gens}: best LER {sc[order[0]][0]:.4f} it {sc[order[0]][1]:.1f}; "
            f"mean-candidate LER {sc[0][0]:.4f} it {sc[0][1]:.1f}")
    return mu


def expand16(t16, feat):
    t16 = np.asarray(t16, float)
    return np.repeat(t16, 3) if feat == "unsat" else t16


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--feat", default="unsat", choices=["base", "unsat"])
    ap.add_argument("--legs", type=int, default=1)
    ap.add_argument("--leg-iters", default="", help="comma list; default: 200 for 1 leg, else 80,60,60,...")
    ap.add_argument("--solutions", type=int, default=1)
    ap.add_argument("--init", default=str(HERE / "results" / "oam_gamma_bb72.json"))
    ap.add_argument("--fixed", default="", help="JSON file with already-learned leading leg tables")
    ap.add_argument("--train-scale", type=float, default=1.0)
    ap.add_argument("--train-shots", type=int, default=4000)
    ap.add_argument("--train-cap-scale", type=float, default=0.5, help="training leg budgets = this x test budgets")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=28)
    ap.add_argument("--elite", type=int, default=7)
    ap.add_argument("--c-iter", type=float, default=1e-4)
    ap.add_argument("--test-scales", default="1,0.7,0.5")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dem = load_dem(a.code, default_rounds(a.code), "sota", a.seed, HERE / "cache")
    legs = [int(x) for x in a.leg_iters.split(",")] if a.leg_iters else (
        [200] if a.legs == 1 else [80] + [int(120 / (a.legs - 1))] * (a.legs - 1))
    assert len(legs) == a.legs
    tag = a.tag or f"{a.feat}_L{a.legs}_S{a.solutions}"
    res = HERE / "results"
    init16 = np.array(json.loads(Path(a.init).read_text())["gamma"])
    ns = 48 if a.feat == "unsat" else 16
    tables = np.tile(expand16(init16, a.feat), (a.legs, 1))
    first = 0
    if a.fixed:
        fx = np.array(json.loads(Path(a.fixed).read_text())["tables"])
        tables[:len(fx)] = fx
        first = len(fx)

    gt = TannerGraph(scaled(dem, a.train_scale))
    _, s_tr, o_tr = gt.sample(a.train_shots, np.random.default_rng([a.seed, 555, int(a.train_scale * 1000)]))
    train_legs = [max(10, int(round(x * a.train_cap_scale))) for x in legs]
    dec_tr = OAM2(gt, a.feat, train_legs, a.solutions)
    t0 = time.perf_counter()
    for k in range(first, a.legs):
        print(f"training leg {k + 1}/{a.legs} table ({ns} states), legs {train_legs}, S={a.solutions}", flush=True)

        def evalf(c, k=k):
            tb = tables.copy()
            tb[k] = c
            pred, ok, it, _ = dec_tr.decode(s_tr, tb)
            return ler_of(pred, ok, s_tr, o_tr).mean(), it.mean()

        init = tables[k] if k == 0 else np.zeros(ns)            # later legs start from plain BP (explore)
        tables[k] = cem(evalf, ns, init, a.gens, a.pop, a.elite, a.c_iter, a.seed + k,
                        log=lambda m: print(m, flush=True), ckpt=HERE / "cache" / f"ckpt_{tag}_leg{k + 1}.json")
        (res / f"oam2_tables_{tag}.json").write_text(json.dumps(
            {"tables": tables[:k + 1].tolist(), "feat": a.feat, "legs": legs, "solutions": a.solutions,
             "train_scale": a.train_scale, "minutes": (time.perf_counter() - t0) / 60}, indent=1))
    print(f"trained in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    base16 = OAMBP  # reference: the published single-run OAM-BP (16 states, cap 200)
    rows = []
    for lam in [float(x) for x in a.test_scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        pr, okr, itr, _ = base16(g, 200).decode(s, init16)
        w_ref = ler_of(pr, okr, s, o)
        t1 = time.perf_counter()
        pred, ok, it, _ = OAM2(g, a.feat, legs, a.solutions).decode(s, tables)
        wrong = ler_of(pred, ok, s, o)
        for name, w, its in (("oam16_single", w_ref, itr), (f"oam2_{tag}", wrong, it)):
            k_ = int(w.sum())
            lo, hi = wilson(k_, len(s))
            rows.append({"code": a.code, "scale": lam, "decoder": name, "shots": len(s), "errors": k_,
                         "ler": k_ / len(s), "ci_lo": lo, "ci_hi": hi, "mean_iter": float(its.mean()),
                         "p99_iter": float(np.percentile(its, 99)), "max_iter": int(its.max()),
                         "legs": ",".join(map(str, legs)), "solutions": a.solutions, "feat": a.feat})
            print(f"lambda={lam:<4g} {name:<22} LER {k_ / len(s):.2e} [{lo:.2e},{hi:.2e}]  iters mean "
                  f"{its.mean():5.1f} p99 {np.percentile(its, 99):4.0f} max {its.max()}", flush=True)
        p, b, c = mcnemar(wrong, w_ref)
        print(f"lambda={lam:<4g} McNemar new vs oam16: only-new-wrong {b}, only-oam16-wrong {c}, p = {p:.3g}  "
              f"({time.perf_counter() - t1:.0f} s)", flush=True)
    out = res / f"oam2_{tag}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
