"""
RL-Relay: a learned controller for memory-BP restarts under a fixed iteration budget.

BP with a memory term (Relay-BP, arXiv:2506.01779) escapes oscillations by
restarting "legs" with disordered memory strengths, but on a fixed schedule
(leg lengths T0 / Tr). Here a reinforcement-learned policy decides, after every
iteration, whether to CONTINUE the current leg or RESTART a new leg now. The
policy is a Q-table over a small discrete state (hardware: a ROM lookup, like
the RL-BP HALT table):

  leg iteration  1, 2, 3-4, 5-8, 9-16, 17+                 (6)
  unsatisfied    1, 2, 3-4, 5-8, 9-16, 17-32, 33+          (7)
  stall in leg   0, 1, 2-3, 4-7, 8+                        (5)
  flips          0, 1-2, 3-8, 9+   (hard-decision bits changed this iteration; oscillation)
  budget used    <25%, <50%, <75%, >=75%                   (4)

Reward: -1 per iteration; at the end -F if the shot is not solved within the
budget or converges to a wrong logical class (training uses the true flips).
Training: fitted Q-iteration on trajectories from an exploratory behaviour
policy (random restarts with probability eps per iteration).

Compared at the same budget, on the same test shots:
  fixed    Relay-BP leg schedule T0 / Tr
  stallK   heuristic: restart when the leg has not improved for K iterations
  rl       the learned controller

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/rl_relay.py --code bb:72 --scales 0.5,1 --budget 100
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from aura.decoding import TannerGraph

CONT, RESTART = 0, 1
NS = 6 * 7 * 5 * 4 * 4


def _bucket(x, edges):
    return np.searchsorted(np.asarray(edges), x, side="right")


def encode(t_leg, unsat, stall, flips, used_frac):
    tl = np.minimum(_bucket(t_leg, [1, 2, 4, 8, 16]) - 1, 5)          # 1->0, 2->1, 3-4->2, 5-8->3, 9-16->4, 17+->5
    tl = np.clip(tl, 0, 5)
    u = np.clip(_bucket(unsat, [1, 2, 4, 8, 16, 32]) - 1, 0, 6)
    st = np.clip(_bucket(stall, [0, 1, 3, 7]) - 1, 0, 4)
    fl = np.clip(_bucket(flips, [0, 2, 8]) - 1, 0, 3)
    bu = np.clip((used_frac * 4).astype(np.int64), 0, 3)
    return (((tl * 7 + u) * 5 + st) * 4 + fl) * 4 + bu


class ControlledRelay:
    """Memory BP with restarts chosen by ``controller(features) -> restart mask``."""

    def __init__(self, g: TannerGraph, budget: int, alpha=1.0, g0=0.125, g_lo=-0.24, g_hi=0.66, seed=0):
        self.g, self.budget, self.alpha = g, budget, np.float32(alpha)
        self.g0, self.g_lo, self.g_hi = g0, g_lo, g_hi
        self.seed = seed

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

    def decode(self, syndromes, controller, log=None):
        """Returns (pred (B, K), solved (B,), iterations (B,)). ``log`` (list) receives
        (shot, state, action) rows for every decision when given."""
        g = self.g
        B, V = len(syndromes), g.num_vars
        rng = np.random.default_rng(self.seed)
        lam0 = g.prior_llr.astype(np.float32)
        lam0c = lam0[:, None]
        pred = np.zeros((B, g.num_observables), np.uint8)
        solved = np.zeros(B, bool)
        iters = np.zeros(B, np.int64)
        self.weight = np.full(B, np.inf)       # sum of prior LLRs of the returned error pattern (solved shots)
        self.weight[~syndromes.any(axis=1)] = 0.0
        solved[~syndromes.any(axis=1)] = True
        todo = np.nonzero(syndromes.any(axis=1))[0]
        for c0 in range(0, len(todo), 256):
            idx = todo[c0:c0 + 256]
            A = len(idx)
            s = np.ascontiguousarray(syndromes[idx][:, g.active_checks].T)
            gam = np.full((V, A), self.g0, np.float32)
            lam = np.repeat(lam0c, A, axis=1)
            m_vc = lam[g.e_var]
            hard_prev = np.zeros((V, A), bool)
            t_leg = np.zeros(A, np.int64)
            best = np.full(A, np.iinfo(np.int64).max)
            stall = np.zeros(A, np.int64)
            it = np.zeros(A, np.int64)
            alive = np.ones(A, bool)
            last_hard = np.zeros((V, A), bool)
            while alive.any():
                a = np.nonzero(alive)[0]
                m_cv = self._check(m_vc[:, a], s[:, a])
                M = (g.var_sum @ m_cv) + lam[:, a]
                m_vc[:, a] = M[g.e_var] - m_cv
                hard = M < 0
                flips = (hard != hard_prev[:, a]).sum(axis=0)
                hard_prev[:, a] = hard
                last_hard[:, a] = hard
                synd = (g.h_active @ hard.view(np.uint8).astype(np.float32)).astype(np.uint8) & 1
                unsat = (synd != s[:, a]).sum(axis=0)
                ok = unsat == 0
                t_leg[a] += 1
                it[a] += 1
                imp = unsat < best[a]
                best[a] = np.where(imp, unsat, best[a])
                stall[a] = np.where(imp, 0, stall[a] + 1)
                lam[:, a] = (1 - gam[:, a]) * lam0c + gam[:, a] * M
                done = ok | (it[a] >= self.budget)
                if done.any():
                    fin = a[done]
                    alive[fin] = False
                    rows = idx[fin]
                    solved[rows] = ok[done]
                    pred[rows] = g.logical(hard[:, done].T.astype(np.uint8))
                    hd = hard[:, done]
                    self.weight[rows] = np.where(ok[done], lam0 @ hd.astype(np.float32), np.inf)
                    iters[rows] = it[fin]
                live = ~done
                if not live.any():
                    continue
                al = a[live]
                state = encode(t_leg[al], unsat[live], stall[al], flips[live], it[al] / self.budget)
                act = np.asarray(controller(state, t_leg[al], stall[al], rng), bool)
                if log is not None:
                    log.extend(zip(idx[al].tolist(), state.tolist(), act.astype(int).tolist(),
                                   it[al].tolist()))
                rs = al[act]
                if len(rs):
                    k = len(rs)
                    gam[:, rs] = rng.uniform(self.g_lo, self.g_hi, (V, k)).astype(np.float32)
                    Mr = M[:, np.searchsorted(a, rs)]
                    lam[:, rs] = (1 - gam[:, rs]) * lam0c + gam[:, rs] * Mr
                    m_vc[:, rs] = lam[:, rs][g.e_var]
                    t_leg[rs] = 0
                    best[rs] = np.iinfo(np.int64).max
                    stall[rs] = 0
        return pred, solved, iters


# ------------------------------------------------------------------ controllers
def stall_rule(k):
    return lambda state, t_leg, stall, rng: stall >= k


def leg_rule(tr):
    """Relay-BP style fixed schedule: restart every tr iterations (uniform leg length)."""
    return lambda state, t_leg, stall, rng: t_leg >= tr


def eps_random(eps):
    return lambda state, t_leg, stall, rng: rng.random(len(state)) < eps


class QPolicy:
    def __init__(self, q):
        self.q = q

    def __call__(self, state, t_leg, stall, rng):
        return self.q[state, RESTART] > self.q[state, CONT]


def train(g, budget, episodes, fail_cost, eps, seed, scale_rng, min_visits=100):
    """Fitted Q-iteration on exploratory trajectories."""
    _, s, o = g.sample(episodes, scale_rng)
    dec = ControlledRelay(g, budget, seed=seed + 1)
    log = []
    pred, solved, iters = dec.decode(s, eps_random(eps), log=log)
    wrong = (~solved) | (pred != o).any(axis=1)
    if not log:
        return QPolicy(np.zeros((NS, 2)))
    shot = np.array([r[0] for r in log])
    st = np.array([r[1] for r in log])
    act = np.array([r[2] for r in log])
    itn = np.array([r[3] for r in log])
    order = np.lexsort((itn, shot))
    shot, st, act, itn = shot[order], st[order], act[order], itn[order]
    nxt_same = np.zeros(len(shot), bool)
    nxt_same[:-1] = shot[1:] == shot[:-1]
    nxt_state = np.where(nxt_same, np.roll(st, -1), 0)
    # cost to go from this decision: 1 iteration, then either the next decision or the end of the shot
    term = np.where(nxt_same, 0.0, -fail_cost * wrong[shot].astype(float) - (iters[shot] - itn - 1))
    q = np.zeros((NS, 2))
    cnt = np.zeros((NS, 2))
    np.add.at(cnt, (st, act), 1)
    # actions with too few samples are masked *inside* the iteration: a max over noisy
    # low-count estimates is optimistically biased and that bias propagates through V
    rare = cnt < min_visits
    for _ in range(200):
        v = q.max(axis=1)
        tgt = -1.0 + np.where(nxt_same, v[nxt_state], term)
        acc = np.zeros((NS, 2))
        np.add.at(acc, (st, act), tgt)
        new = np.where(cnt > 0, acc / np.maximum(cnt, 1), -1e9)
        # unseen actions: never chosen; unseen states: CONTINUE
        new[rare[:, RESTART], RESTART] = -1e9
        new[(cnt[:, CONT] == 0), CONT] = 0.0
        if np.allclose(new, q):
            q = new
            break
        q = new
    return QPolicy(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--scales", default="0.5,1")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--fail-cost", type=float, default=2000.0)
    ap.add_argument("--legs", default="20,30", help="fixed leg lengths Tr to compare")
    ap.add_argument("--stall-k", default="2,4,6,8")
    ap.add_argument("--no-rl", action="store_true", help="only the fixed/heuristic controllers")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "rl_relay.csv"))
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s, o = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        t0 = time.perf_counter()
        pol = None if a.no_rl else train(g, a.budget, a.train, a.fail_cost, a.eps, a.seed, np.random.default_rng([a.seed, 99, int(lam * 1000)]))
        n_restart = 0 if pol is None else int((pol.q[:, RESTART] > pol.q[:, CONT]).sum())
        print(f"lambda={lam:<4g} trained RL-Relay in {time.perf_counter() - t0:.0f} s, {n_restart} RESTART states",
              flush=True)
        ctls = {f"legs{t}": leg_rule(t) for t in (int(x) for x in a.legs.split(","))}
        ctls.update({f"stall{k}": stall_rule(k) for k in (int(x) for x in a.stall_k.split(","))})
        ctls["never"] = lambda state, t_leg, stall, rng: np.zeros(len(state), bool)
        if pol is not None:
            ctls["rl"] = pol
        for name, ctl in ctls.items():
            t0 = time.perf_counter()
            pred, ok, it = ControlledRelay(g, a.budget, seed=a.seed).decode(s, ctl)
            wrong = int(((~ok & s.any(axis=1)) | (pred != o).any(axis=1)).sum())
            lo, hi = wilson(wrong, a.shots)
            row = {"code": a.code, "scale": lam, "budget": a.budget, "controller": name, "shots": a.shots,
                   "errors": wrong, "ler": wrong / a.shots, "ci_lo": lo, "ci_hi": hi,
                   "mean_iter": float(it.mean()), "p99_iter": float(np.percentile(it, 99)),
                   "unsolved": float((~ok).mean()), "wall_s": time.perf_counter() - t0}
            rows.append(row)
            print(f"lambda={lam:<4g} b={a.budget} {name:<8} LER {row['ler']:.2e} [{lo:.2e},{hi:.2e}]"
                  f"  iters mean {row['mean_iter']:.1f} p99 {row['p99_iter']:.0f}  unsolved {row['unsolved']:.3f}",
                  flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
