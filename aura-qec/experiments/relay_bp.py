"""
Relay-BP baseline (re-implemented from the description in Müller et al.,
arXiv:2506.01779; not IBM's code), compared with BP+OSD-0 on the same shots.

Relay-BP: min-sum BP whose per-variable prior carries a memory of the previous
marginal,  Lambda_j(t) = (1 - g_j) Lambda_j(0) + g_j M_j(t - 1).
  * leg 0: uniform memory strength g0, up to T0 iterations;
  * legs 1..R: g_j drawn i.i.d. uniform in [g_lo, g_hi] (negative values allowed),
    messages restarted, memory warm-started from the previous leg's marginals,
    up to Tr iterations each;
  * every converged leg is a candidate solution; a shot stops after S solutions
    (or R legs) and returns the lowest-weight one (weight = sum of prior LLRs of
    the flipped mechanisms). No solution -> the last hard decision (usually wrong).

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/relay_bp.py --code bb:72 --scales 0.5,1 --shots 10000
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from aura.decoding import MinSumBP, TannerGraph, osd0


class RelayBP:
    def __init__(self, g: TannerGraph, alpha=1.0, g0=0.125, g_lo=-0.24, g_hi=0.66, t0=80, tr=60, legs=50,
                 solutions=5, seed=0):
        self.g, self.alpha = g, np.float32(alpha)
        self.g0, self.g_lo, self.g_hi = g0, g_lo, g_hi
        self.t0, self.tr, self.legs, self.S = t0, tr, legs, solutions
        self.rng = np.random.default_rng(seed)

    def _check_update(self, m_vc, s):
        g = self.g
        ptr, loc = g.chk_ptr, g.e_chk_local
        mag = np.abs(m_vc)
        neg = (m_vc < 0).view(np.uint8)
        min1 = np.minimum.reduceat(mag, ptr, axis=0)
        min1_e = min1[loc]
        is_min = mag == min1_e
        cnt = np.add.reduceat(is_min.astype(np.int16), ptr, axis=0)
        mag2 = np.where(is_min, np.float32(np.inf), mag)
        min2 = np.minimum.reduceat(mag2, ptr, axis=0)
        uniq = (cnt[loc] == 1) & is_min
        ext = np.where(uniq, min2[loc], min1_e)
        par = np.bitwise_xor.reduceat(neg, ptr, axis=0) ^ s
        flip = (par[loc] ^ neg).astype(bool)
        out = ext * self.alpha
        np.negative(out, out=out, where=flip)
        return out

    def decode(self, syndromes: np.ndarray):
        """Returns (pred (B, K), solved (B,), total iterations (B,), legs used (B,))."""
        g = self.g
        B = len(syndromes)
        V = g.num_vars
        lam0 = g.prior_llr.astype(np.float32)
        pred = np.zeros((B, g.num_observables), np.uint8)
        solved = np.zeros(B, bool)
        iters = np.zeros(B, np.int64)
        legs_used = np.zeros(B, np.int64)
        w0 = syndromes.any(axis=1)
        solved[~w0] = True
        idx = np.nonzero(w0)[0]
        chunk = 256
        for c0 in range(0, len(idx), chunk):
            self._run(syndromes, idx[c0:c0 + chunk], lam0, pred, solved, iters, legs_used)
        return pred, solved, iters, legs_used

    def _run(self, syn_all, idx, lam0, pred, solved, iters, legs_used):
        g = self.g
        V, A = g.num_vars, len(idx)
        s = np.ascontiguousarray(syn_all[idx][:, g.active_checks].T)            # (C, A)
        gam = np.full((V, A), self.g0, np.float32)
        lam = np.repeat(lam0[:, None], A, axis=1)                                # memory-mixed prior
        m_vc = lam[g.e_var]
        leg = np.zeros(A, np.int64)
        t_leg = np.zeros(A, np.int64)
        nsol = np.zeros(A, np.int64)
        best_w = np.full(A, np.inf)
        best_pred = np.zeros((A, g.num_observables), np.uint8)
        last_hard = np.zeros((V, A), bool)
        alive = np.ones(A, bool)
        lam0c = lam0[:, None]
        while alive.any():
            a = np.nonzero(alive)[0]
            m_cv = self._check_update(m_vc[:, a], s[:, a])
            M = (g.var_sum @ m_cv) + lam[:, a]                                   # marginals (V, n)
            m_vc[:, a] = M[g.e_var] - m_cv
            hard = M < 0
            last_hard[:, a] = hard
            synd = (g.h_active @ hard.view(np.uint8).astype(np.float32)).astype(np.uint8) & 1
            ok = (synd == s[:, a]).all(axis=0)
            t_leg[a] += 1
            iters[idx[a]] += 1
            # memory for the next iteration
            lam[:, a] = (1 - gam[:, a]) * lam0c + gam[:, a] * M
            # converged legs -> candidate solutions
            for j in np.nonzero(ok)[0]:
                p = a[j]
                w = float(lam0[hard[:, j]].sum())
                if w < best_w[p]:
                    best_w[p] = w
                    best_pred[p] = g.logical(hard[:, j][None, :].astype(np.uint8))[0]
                nsol[p] += 1
            limit = np.where(leg[a] == 0, self.t0, self.tr)
            leg_end = ok | (t_leg[a] >= limit)
            if not leg_end.any():
                continue
            ends = a[leg_end]
            finished = ends[(nsol[ends] >= self.S) | (leg[ends] + 1 >= self.legs)]
            alive[finished] = False
            restart = np.setdiff1d(ends, finished)
            if len(restart):
                # new leg: disordered memory strengths, memory warm-started from the last marginals
                k = len(restart)
                gam[:, restart] = self.rng.uniform(self.g_lo, self.g_hi, (V, k)).astype(np.float32)
                Mr = M[:, np.searchsorted(a, restart)]
                lam[:, restart] = (1 - gam[:, restart]) * lam0c + gam[:, restart] * Mr
                m_vc[:, restart] = lam[:, restart][g.e_var]
                leg[restart] += 1
                t_leg[restart] = 0
        found = nsol > 0
        rows = idx
        pred[rows[found]] = best_pred[found]
        if (~found).any():
            pred[rows[~found]] = g.logical(last_hard[:, ~found].T.astype(np.uint8))
        solved[rows] = found
        legs_used[rows] = leg + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--scales", default="0.5,1")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--legs", type=int, default=50)
    ap.add_argument("--solutions", type=int, default=5)
    ap.add_argument("--t0", type=int, default=80)
    ap.add_argument("--tr", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "relay_bp.csv"))
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))   # same shots as osd_input.py
        res = {}
        for avg in (1, 2):
            t0 = time.perf_counter()
            r = MinSumBP(g, max_iter=30, posterior_avg=avg).decode(s)
            p = g.logical(r.errors)
            for i in np.nonzero(~r.converged)[0]:
                p[i], _, _ = osd0(g, s[i], r.posterior[i])
            res[f"bposd_avg{avg}"] = (p, r.iterations.astype(np.int64), time.perf_counter() - t0, None)
        t0 = time.perf_counter()
        rb = RelayBP(g, alpha=a.alpha, t0=a.t0, tr=a.tr, legs=a.legs, solutions=a.solutions, seed=a.seed)
        p, ok, it, legs = rb.decode(s)
        res[f"relay{a.solutions}"] = (p, it, time.perf_counter() - t0, (ok, legs))
        for name, (p, it, wall, extra) in res.items():
            wrong = int((p != o).any(axis=1).sum())
            lo, hi = wilson(wrong, a.shots)
            row = {"code": a.code, "rounds": rounds, "scale": lam, "decoder": name, "shots": a.shots,
                   "errors": wrong, "ler": wrong / a.shots, "ci_lo": lo, "ci_hi": hi,
                   "mean_iter": float(it.mean()), "p99_iter": float(np.percentile(it, 99)),
                   "max_iter": int(it.max()), "unsolved": float((~extra[0]).mean()) if extra else 0.0,
                   "mean_legs": float(extra[1].mean()) if extra else 1.0, "wall_s": wall}
            rows.append(row)
            print(f"lambda={lam:<4g} {name:<11} LER {row['ler']:.2e} [{lo:.2e},{hi:.2e}]  iters mean "
                  f"{row['mean_iter']:.1f} p99 {row['p99_iter']:.0f} max {row['max_iter']}"
                  f"  unsolved {row['unsolved']:.3f}  ({wall:.0f} s)", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
