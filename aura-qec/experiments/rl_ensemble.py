"""
RL early exit for a sequential memory-BP ensemble ("RL-Ensemble").

Memory BP (prior mixed with the previous marginal, strength gamma) converges to
different solutions for different gamma; the lowest-weight one is usually right
(oracle headroom ~45% on BB-72). Running every gamma is expensive, so the
members run one after another (fixed order) and a learned policy decides, after
each member, to STOP (answer with the lowest-weight solution so far) or to
CONTINUE with the next member. This is the RL-BP early-exit idea lifted from
"stop iterating" to "stop trying".

State after member k (all discrete, a ROM lookup in hardware):
  k                       1 .. K-1
  solved so far           0, 1, 2+
  agreement               <2 solved / all solutions in one logical class / disagree
  last member iterations  <=5, <=10, <=20, >20 or unsolved
  best weight             bucket by training quantiles (25/50/75/95%) of the best solution weight, or none
Reward: -c per BP iteration spent, -F if the final answer is wrong. The policy
is solved exactly by backward induction on training shots (finite horizon K);
the member results are precomputed, so evaluation on fresh test shots is exact.

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/rl_ensemble.py --code bb:72 --scales 1,0.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, default_rounds, load_dem, scaled, wilson
from rl_relay import ControlledRelay
from aura.decoding import TannerGraph

STOP, CONT = 0, 1


def run_members(g, s, gammas, budget, seed):
    never = lambda state, t_leg, stall, rng: np.zeros(len(state), bool)
    K, B = len(gammas), len(s)
    pred = np.zeros((K, B, g.num_observables), np.uint8)
    w = np.zeros((K, B))
    it = np.zeros((K, B), np.int64)
    ok = np.zeros((K, B), bool)
    for k, gm in enumerate(gammas):
        dec = ControlledRelay(g, budget, g0=gm, seed=seed)
        pred[k], ok[k], it[k] = dec.decode(s, never)
        w[k] = dec.weight
    return pred, w, it, ok


class Episodes:
    """Per-shot member results plus the running statistics a policy needs."""

    def __init__(self, members, truth):
        self.pred, self.w, self.it, self.ok = members
        self.truth = truth
        K, B = self.w.shape
        self.K, self.B = K, B
        # after member k (0-based, inclusive): index of the best member, solved count, agreement
        self.best = np.zeros((K, B), np.int64)
        self.nsol = np.zeros((K, B), np.int64)
        self.agree = np.zeros((K, B), np.int64)
        cls = self.pred.reshape(K, B, -1)
        for k in range(K):
            wk = self.w[:k + 1]
            self.best[k] = wk.argmin(axis=0)
            self.nsol[k] = np.isfinite(wk).sum(axis=0)
            sol = np.isfinite(wk)
            # agreement among solved members: 0 = fewer than 2 solved, 1 = all agree, 2 = disagree
            same = np.ones(B, bool)
            ref = np.full(B, -1)
            for j in range(k + 1):
                key = cls[j].dot(1 << np.arange(cls.shape[2]))
                first = sol[j] & (ref < 0)
                ref = np.where(first, key, ref)
                same &= ~sol[j] | (key == ref)
            self.agree[k] = np.where(self.nsol[k] < 2, 0, np.where(same, 1, 2))
        self.wrong = np.stack([(self.pred[self.best[k], np.arange(B)] != truth).any(axis=1) for k in range(K)])
        self.cost = np.cumsum(self.it, axis=0)            # iterations spent after member k

    def state(self, k, wq):
        itb = np.where(~self.ok[k], 3, np.searchsorted([5, 10, 20], self.it[k], side="left"))
        itb = np.minimum(itb, 3)
        bw = self.w[self.best[k], np.arange(self.B)]
        wb = np.where(np.isfinite(bw), np.searchsorted(wq, bw), 5)      # 0..4 solved, 5 = nothing solved
        return ((((k * 3 + np.minimum(self.nsol[k], 2)) * 3 + self.agree[k]) * 4 + itb) * 6 + wb)


def solve_policy(ep: Episodes, wq, c, F, min_visits=30):
    """Backward induction: returns stop[k][state] (bool tables) for k = 0..K-2."""
    K, B = ep.K, ep.B
    ns = K * 3 * 3 * 4 * 6
    V = -F * ep.wrong[K - 1].astype(float)            # value after the last member: must stop
    tables = [None] * (K - 1)
    for k in range(K - 2, -1, -1):
        st = ep.state(k, wq)
        q_stop = -F * ep.wrong[k].astype(float)
        q_cont = -c * ep.it[k + 1] + V
        cnt = np.bincount(st, minlength=ns)
        m_stop = np.bincount(st, q_stop, ns) / np.maximum(cnt, 1)
        m_cont = np.bincount(st, q_cont, ns) / np.maximum(cnt, 1)
        stop = m_stop >= m_cont
        stop[cnt < min_visits] = False                   # rare state: keep going (safe for accuracy)
        tables[k] = stop
        V = np.where(stop[st], q_stop, q_cont)
    return tables


def evaluate(ep: Episodes, decide):
    """decide(k, ep) -> bool stop mask for the shots still running. Returns (ler, mean iters, max iters)."""
    B = ep.B
    final = np.full(B, ep.K - 1)
    running = np.ones(B, bool)
    for k in range(ep.K - 1):
        stop = decide(k, ep) & running
        final[stop] = k
        running &= ~stop
    wrong = ep.wrong[final, np.arange(B)]
    cost = ep.cost[final, np.arange(B)]
    return wrong.mean(), int(wrong.sum()), cost.mean(), np.percentile(cost, 99), cost.max(), (final + 1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="bb:72")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--scales", default="1,0.5")
    ap.add_argument("--gammas", default="0.25,0.125,0.4,0.06", help="member order")
    ap.add_argument("--member-budget", type=int, default=50)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--fail-cost", type=float, default=2000.0)
    ap.add_argument("--iter-costs", default="0.25,0.5,1,2,4", help="RL trade-off sweep (cost per iteration)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "rl_ensemble.csv"))
    a = ap.parse_args()
    rounds = a.rounds or default_rounds(a.code)
    dem = load_dem(a.code, rounds, "sota", a.seed, HERE / "cache")
    gammas = [float(x) for x in a.gammas.split(",")]
    rows = []
    for lam in [float(x) for x in a.scales.split(",")]:
        g = TannerGraph(scaled(dem, lam))
        _, s_tr, o_tr = g.sample(a.train, np.random.default_rng([a.seed, 77, int(lam * 1000)]))
        _, s_te, o_te = g.sample(a.shots, np.random.default_rng([a.seed, int(lam * 1000)]))
        tr = Episodes(run_members(g, s_tr, gammas, a.member_budget, a.seed), o_tr)
        te = Episodes(run_members(g, s_te, gammas, a.member_budget, a.seed), o_te)
        bw = tr.w[tr.best[0], np.arange(tr.B)]
        wq = np.quantile(bw[np.isfinite(bw) & (bw > 0)], [0.25, 0.5, 0.75, 0.95])

        def report(name, res):
            ler, err, mean_it, p99, mx, members = res
            lo, hi = wilson(err, te.B)
            rows.append({"code": a.code, "scale": lam, "policy": name, "shots": te.B, "errors": err, "ler": ler,
                         "ci_lo": lo, "ci_hi": hi, "mean_iter": mean_it, "p99_iter": p99, "max_iter": mx,
                         "mean_members": members, "member_budget": a.member_budget, "gammas": a.gammas})
            print(f"lambda={lam:<4g} {name:<16} LER {ler:.2e} [{lo:.2e},{hi:.2e}]  iters mean {mean_it:5.1f} "
                  f"p99 {p99:4.0f} max {mx:4d}  members {members:.2f}", flush=True)

        for k in range(te.K):
            report(f"fixed{k + 1}", evaluate(te, lambda kk, ep, k=k: np.full(ep.B, kk >= k)))
        report("agree-stop", evaluate(te, lambda kk, ep: ep.agree[kk] == 1))
        for T in (5, 10, 20):
            report(f"fast-stop{T}", evaluate(te, lambda kk, ep, T=T: ep.ok[kk] & (ep.it[kk] <= T)))
        for c in [float(x) for x in a.iter_costs.split(",")]:
            tabs = solve_policy(tr, wq, c, a.fail_cost)
            report(f"rl_c{c:g}", evaluate(te, lambda kk, ep, tabs=tabs: tabs[kk][ep.state(kk, wq)]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
