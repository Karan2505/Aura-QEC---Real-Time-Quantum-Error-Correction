"""
Oscillation-Aware Memory BP (OAM-BP) with a learned per-variable memory policy.

Memory BP mixes each variable's channel prior with its previous marginal,
    Lambda_j(t) = (1 - g_j(t)) Lambda_j(0) + g_j(t) M_j(t-1),
but uses one global strength g. On BB codes BP oscillates (period 2) and the
best g differs from shot to shot. OAM-BP lets every variable pick its own
strength every iteration from what it observes locally:

  state s_j(t) = (flips of its hard decision in the last 4 iterations: 0,1,2,3+)
               x (current hard decision: 0 / 1)
               x (phase: t <= T_split / later)                     -> 16 states
  g_j(t) = Gamma[s_j(t)]                                           (a 16-entry ROM)

Gamma is learned with the cross-entropy method (policy search) on training
shots: objective = LER + c * mean iterations. A constant Gamma reproduces
ordinary memory BP exactly.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/oam_bp.py --code bb:72 --train-scale 1 --test-scales 0.5,0.7,1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from aura.decoding import TannerGraph

NSTATE = 16
POP4 = np.array([bin(i).count("1") for i in range(16)], np.int64)


class OAMBP:
    def __init__(self, g: TannerGraph, budget: int = 100, t_split: int = 8, alpha: float = 1.0):
        self.g, self.budget, self.t_split, self.alpha = g, budget, t_split, np.float32(alpha)

    def _check(self, m_vc, s):
        g = self.g
        ptr, loc = g.chk_ptr, g.e_chk_local
        mag = np.abs(m_vc)
        neg = (m_vc < 0).view(np.uint8)
        min1 = np.minimum.reduceat(mag, ptr, axis=0)
        min1_e = min1[loc]
        is_min = mag == min1_e
        cnt = np.add.reduceat(is_min.astype(np.int16), ptr, axis=0)
        min2 = np.minimum.reduceat(np.where(is_min, np.float32(np.inf), mag), ptr, axis=0)
        ext = np.where((cnt[loc] == 1) & is_min, min2[loc], min1_e)
        flip = ((np.bitwise_xor.reduceat(neg, ptr, axis=0) ^ s)[loc] ^ neg).astype(bool)
        out = ext * self.alpha
        np.negative(out, out=out, where=flip)
        return out

    def decode(self, syndromes, gamma_table):
        """Returns (pred (B, K), solved (B,), iterations (B,), weight (B,))."""
        g = self.g
        G = np.asarray(gamma_table, np.float32)
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
            alive = np.arange(A)
            for t in range(1, self.budget + 1):
                m_cv = self._check(m_vc[:, alive], s[:, alive])
                M = (g.var_sum @ m_cv) + lam[:, alive]
                m_vc[:, alive] = M[g.e_var] - m_cv
                hard = M < 0
                flips = hard != hard_prev[:, alive]
                hard_prev[:, alive] = hard
                h = ((hist[:, alive] << 1) | flips) & 0xF
                hist[:, alive] = h
                synd = (g.h_active @ hard.view(np.uint8).astype(np.float32)).astype(np.uint8) & 1
                ok = (synd == s[:, alive]).all(axis=0)
                # per-variable memory strength from the learned table
                st = (np.minimum(POP4[h], 3) * 2 + hard) * 2 + (1 if t > self.t_split else 0)
                gam = G[st]
                lam[:, alive] = (1 - gam) * lam0c + gam * M
                done = ok | (t == self.budget)
                if done.any():
                    rows = idx[alive[done]]
                    hd = hard[:, done]
                    pred[rows] = g.logical(hd.T.astype(np.uint8))
                    solved[rows] = ok[done]
                    iters[rows] = t
                    weight[rows] = np.where(ok[done], lam0 @ hd.astype(np.float32), np.inf)
                    alive = alive[~done]
                    if not len(alive):
                        break
        return pred, solved, iters, weight


def ler_of(pred, solved, s, o):
    wrong = (~solved & s.any(axis=1)) | (pred != o).any(axis=1)
    return wrong


def cem(dec, s, o, gens, pop, elite, c_iter, seed, lo=-0.25, hi=0.7, init=0.25, log=print):
    rng = np.random.default_rng(seed)
    mu = np.full(NSTATE, init)
    sd = np.full(NSTATE, 0.2)
    best = (np.inf, mu.copy(), None)
    for gen in range(gens):
        cand = np.clip(mu + sd * rng.standard_normal((pop, NSTATE)), lo, hi)
        cand[0] = mu                                         # always re-evaluate the mean
        scores = []
        for k in range(pop):
            pred, ok, it, _ = dec.decode(s, cand[k])
            ler = ler_of(pred, ok, s, o).mean()
            scores.append((ler + c_iter * it.mean(), ler, it.mean()))
        order = np.argsort([x[0] for x in scores])
        el = cand[order[:elite]]
        mu = 0.7 * el.mean(axis=0) + 0.3 * mu
        sd = np.maximum(0.7 * el.std(axis=0) + 0.3 * sd, 0.02)
        top = scores[order[0]]
        if top[0] < best[0]:
            best = (top[0], cand[order[0]].copy(), top)
        log(f"  gen {gen + 1}/{gens}: best obj {top[0]:.4f} (LER {top[1]:.4f}, iters {top[2]:.1f}); "
            f"mean-candidate LER {scores[0][1]:.4f}")
    return mu, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--train-scale", type=float, default=1.0)
    ap.add_argument("--test-scales", default="0.5,0.7,1")
    ap.add_argument("--train-shots", type=int, default=3000)
    ap.add_argument("--shots", type=int, default=10000, help="test shots (same seeds as rl_ensemble / relay_bp)")
    ap.add_argument("--budget", type=int, default=100, help="iteration cap used for training")
    ap.add_argument("--test-budgets", default="100,200")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--c-iter", type=float, default=1e-4, help="objective weight per mean iteration")
    ap.add_argument("--gamma", default="", help="skip training: use this JSON Gamma file")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    tag = a.tag or f"{a.code.replace(':', '')}"
    gfile = res_dir / f"oam_gamma_{tag}.json"

    if a.gamma:
        learned = np.array(json.loads(Path(a.gamma).read_text())["gamma"])
    else:
        gt = TannerGraph(scaled(dem, a.train_scale))
        _, s_tr, o_tr = gt.sample(a.train_shots, np.random.default_rng([a.seed, 555, int(a.train_scale * 1000)]))
        t0 = time.perf_counter()
        print(f"training Gamma on {a.code} lambda={a.train_scale}, {a.train_shots} shots, cap {a.budget}", flush=True)
        mu, best = cem(OAMBP(gt, a.budget), s_tr, o_tr, a.gens, a.pop, a.elite, a.c_iter, a.seed,
                       log=lambda m: print(m, flush=True))
        learned = mu
        gfile.write_text(json.dumps({"gamma": [float(x) for x in learned], "best_candidate": [float(x) for x in best[1]],
                                     "train_scale": a.train_scale, "train_shots": a.train_shots, "budget": a.budget,
                                     "state": "index = (min(flips4,3)*2 + hard)*2 + (t > t_split)",
                                     "minutes": (time.perf_counter() - t0) / 60}, indent=1))
        print(f"learned Gamma -> {gfile}", flush=True)
    print("Gamma table (flips, hard, late):")
    for i in range(NSTATE):
        f, hd, late = i // 4, (i // 2) % 2, i % 2
        print(f"  flips={f}{'+' if f == 3 else ''} hard={hd} late={late}: {learned[i]:+.3f}")

    hand = np.array([0.1 if (i // 4) == 0 else 0.25 if (i // 4) == 1 else 0.5 for i in range(NSTATE)])
    policies = {"fixed_0.125": np.full(NSTATE, 0.125), "fixed_0.25": np.full(NSTATE, 0.25),
                "oam_hand": hand, "oam_learned": learned}
    rows = []
    for lam in [float(x) for x in a.test_scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        for B in [int(x) for x in a.test_budgets.split(",")]:
            dec = OAMBP(g, B)
            for name, G in policies.items():
                t0 = time.perf_counter()
                pred, ok, it, _ = dec.decode(s, G)
                wrong = ler_of(pred, ok, s, o)
                k = int(wrong.sum())
                lo, hi = wilson(k, a.shots)
                row = {"code": a.code, "scale": lam, "budget": B, "decoder": name, "shots": a.shots, "errors": k,
                       "ler": k / a.shots, "ci_lo": lo, "ci_hi": hi, "mean_iter": float(it.mean()),
                       "p99_iter": float(np.percentile(it, 99)), "max_iter": int(it.max()),
                       "unsolved": float((~ok).mean()), "train_scale": a.train_scale}
                rows.append(row)
                print(f"lambda={lam:<4g} cap={B:<4d} {name:<12} LER {row['ler']:.2e} [{lo:.2e},{hi:.2e}]  "
                      f"iters mean {row['mean_iter']:.1f} p99 {row['p99_iter']:.0f}  unsolved {row['unsolved']:.3f}  "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
    out = res_dir / f"oam_{tag}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
