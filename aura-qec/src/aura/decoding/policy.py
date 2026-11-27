"""
Reinforcement-learned halting policy for BP ("RL-BP").

Decoding a shot is an episode.  After each BP iteration the agent sees the
convergence state of that shot and chooses

    CONTINUE  -- run another BP iteration (cost ``iter_cost``), or
    HALT      -- stop BP now and hand the shot to the OSD fallback
                 (cost ``osd_cost``).

The episode also ends when BP converges (no extra cost) or hits ``max_iter``
(forced fallback).  A penalty ``fail_cost`` is charged if the final logical
prediction is wrong.  The return is therefore

    -(iterations * iter_cost + [fallback] * osd_cost + [logical error] * fail_cost)

i.e. the agent minimises expected decoder latency subject to a (heavily
weighted) accuracy term.  Converging shots already exit at the first iteration
that explains the syndrome; what the agent learns is to recognise *hopeless*
trajectories (stalled, oscillating, residual syndrome not shrinking) and cut
them short instead of burning the full iteration budget, while letting
trajectories that are still making progress run.

State (discretised; Q-table of ``NUM_STATES`` x 2):
    iteration bucket, residual unsatisfied-check bucket, residual / initial
    syndrome-weight ratio bucket, stall bucket (iterations since the best
    residual last improved).

Training methods:

* ``"fqi"`` (default) -- batch RL by fitted Q-iteration.  BP's trajectory does
  not depend on the agent's choices until it halts, so every episode is
  recorded once to ``max_iter`` together with the OSD outcome at *every*
  possible halting point; value iteration on the resulting empirical MDP then
  gives the optimal policy for the state abstraction.  Data-efficient and
  stable, which matters because logical failures are rare, high-cost events.
* ``"online"`` -- classic tabular Q-learning with epsilon-greedy exploration
  against the live BP engine.

Episodes come from the detector error model (``dem_episodes``) or from
recorded experiment shots (``shot_episodes``: detection events + the logical
flips actually observed), so the policy can adapt to the noise the hardware
really produces rather than to the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .bp import EXIT_CONVERGED, EXIT_HALTED, IterFeatures, MinSumBP, TannerGraph
from .osd import osd0

CONTINUE, HALT = 0, 1

_T_EDGES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 21])   # iterations 1..8, 9-12, 13-20, 21+
_W_EDGES = np.array([1, 2, 3, 5, 9, 17])                     # residual unsat: 1, 2, 3-4, 5-8, 9-16, 17+
_R_EDGES = np.array([0.0, 0.34, 0.67, 1.0])                  # residual / initial: <.34, <.67, <1, >=1
_S_EDGES = np.array([0, 1, 2, 4, 8])                         # stall: 0, 1, 2-3, 4-7, 8+
NUM_STATES = len(_T_EDGES) * len(_W_EDGES) * len(_R_EDGES) * len(_S_EDGES)
_FORMAT = "aura-rlbp-q/2"


def encode_state(f: IterFeatures) -> np.ndarray:
    tb = np.searchsorted(_T_EDGES, f.t, side="right") - 1
    wb = np.clip(np.searchsorted(_W_EDGES, np.maximum(f.unsat, 1), side="right") - 1, 0, len(_W_EDGES) - 1)
    ratio = f.unsat / np.maximum(f.unsat0, 1)
    rb = np.clip(np.searchsorted(_R_EDGES, ratio, side="right") - 1, 0, len(_R_EDGES) - 1)
    sb = np.clip(np.searchsorted(_S_EDGES, f.stall, side="right") - 1, 0, len(_S_EDGES) - 1)
    return (((tb * len(_W_EDGES) + wb) * len(_R_EDGES) + rb) * len(_S_EDGES) + sb).astype(np.int64)


@dataclass
class RewardModel:
    iter_cost: float = 1.0           # one BP iteration
    osd_cost: float = 20.0           # one OSD-0 fallback, in iteration-equivalents
    fail_cost: float = 2000.0        # one logical error, in iteration-equivalents


@dataclass
class TrainReport:
    method: str = "fqi"
    episodes: int = 0
    decision_points: int = 0
    states_visited: int = 0
    halt_states: int = 0
    history: list = field(default_factory=list)     # per-epoch / final greedy evaluation

    def summary(self) -> str:
        lines = [f"RL-BP training ({self.method}): {self.episodes} episodes, {self.decision_points} decision points, "
                 f"{self.states_visited} states visited, HALT chosen in {self.halt_states}"]
        for h in self.history:
            lines.append(f"  {h.get('label', 'eval'):>10s}  mean iters {h['mean_iterations']:.2f}  "
                         f"p99 {h['p99_iterations']:.0f}  fallback {h['fallback_rate']:.3f}  "
                         f"LER {h['ler']:.4f}  cost {h['mean_cost']:.2f}")
        return "\n".join(lines)


class QHaltingPolicy:
    """Greedy policy over a learned Q-table; implements ``bp.HaltingPolicy``.

    States never visited in training default to CONTINUE (safe: behaves like
    plain BP with the iteration budget)."""

    def __init__(self, q: Optional[np.ndarray] = None, min_iter: int = 1, meta: Optional[dict] = None):
        self.q = np.zeros((NUM_STATES, 2)) if q is None else np.asarray(q, float)
        if self.q.shape != (NUM_STATES, 2):
            raise ValueError(f"Q-table must have shape ({NUM_STATES}, 2)")
        self.min_iter = min_iter
        self.meta = dict(meta or {})
        self.visits = np.zeros((NUM_STATES, 2), np.int64)

    @property
    def halt_table(self) -> np.ndarray:
        return self.q[:, HALT] > self.q[:, CONTINUE]

    def halt(self, f: IterFeatures) -> np.ndarray:
        if f.t < self.min_iter:
            return np.zeros(len(f.unsat), bool)
        return self.halt_table[encode_state(f)]

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        nz = np.nonzero(self.q.any(axis=1))[0]
        return {"format": _FORMAT, "num_states": NUM_STATES, "min_iter": self.min_iter, "meta": self.meta,
                "q": {int(i): [float(a), float(b)] for i, (a, b) in zip(nz, self.q[nz])}}

    @classmethod
    def from_dict(cls, d: dict) -> "QHaltingPolicy":
        if d.get("format") != _FORMAT or d.get("num_states") != NUM_STATES:
            raise ValueError("not an aura RL-BP policy file (or an incompatible state encoding)")
        q = np.zeros((NUM_STATES, 2))
        for k, v in d["q"].items():
            q[int(k)] = v
        return cls(q, d.get("min_iter", 1), d.get("meta"))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "QHaltingPolicy":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- shared helpers
def finish_with_fallback(graph: TannerGraph, syndromes: np.ndarray, res) -> tuple[np.ndarray, np.ndarray]:
    """Logical predictions for a BP result: BP's answer where it converged,
    OSD-0 on the posterior everywhere else.  Returns (pred (B, K), fallback mask (B,))."""
    pred = graph.logical(res.errors)
    fb = ~res.converged
    for i in np.nonzero(fb)[0]:
        pred[i], _, _ = osd0(graph, syndromes[i], res.posterior[i])
    return pred, fb


def episode_cost(res, fallback: np.ndarray, wrong: np.ndarray, rm: RewardModel) -> np.ndarray:
    return res.iterations * rm.iter_cost + fallback * rm.osd_cost + wrong * rm.fail_cost


def evaluate(bp: MinSumBP, syndromes: np.ndarray, obs_true: np.ndarray, policy=None,
             reward: Optional[RewardModel] = None) -> dict:
    """Decode with BP (+policy) and OSD fallback; report iterations, fallback rate, LER, cost."""
    rm = reward or RewardModel()
    res = bp.decode(syndromes, policy=policy)
    pred, fb = finish_with_fallback(bp.g, syndromes, res)
    wrong = (pred != obs_true).any(axis=1)
    it = res.iterations
    nontriv = np.asarray(syndromes).any(axis=1)
    return {"shots": int(len(syndromes)), "mean_iterations": float(it.mean()),
            "mean_iterations_nontrivial": float(it[nontriv].mean()) if nontriv.any() else 0.0,
            "p99_iterations": float(np.percentile(it, 99)), "max_iterations": int(it.max(initial=0)),
            "total_iterations": int(it.sum()),
            "fallback_rate": float(fb.mean()), "halted_rate": float((res.exit_reason == EXIT_HALTED).mean()),
            "ler": float(wrong.mean()), "logical_errors": int(wrong.sum()),
            "mean_cost": float(episode_cost(res, fb, wrong, rm).mean())}


# ---------------------------------------------------------------- episode sources
def dem_episodes(graph: TannerGraph, noise_scale: float = 1.0):
    """Episodes sampled from the detector error model (``noise_scale`` multiplies all priors)."""
    def fn(n, rng):
        _, s, o = graph.sample(n, rng, noise_scale)
        return s, o
    return fn


def shot_episodes(detection_events: np.ndarray, observable_flips: np.ndarray):
    """Episodes resampled from recorded shots (detection events restricted to the
    decoding graph's detectors, and the logical flips actually observed)."""
    det = np.asarray(detection_events, np.uint8)
    obs = np.asarray(observable_flips, np.uint8)
    if len(det) != len(obs) or not len(det):
        raise ValueError("need matching, non-empty detection-event and observable arrays")

    def fn(n, rng):
        i = rng.integers(0, len(det), n)
        return det[i], obs[i]
    return fn


def _nontrivial(episodes_fn, n, rng, max_draws: int = 50):
    """Draw ``n`` episodes with a non-empty syndrome (trivial ones carry no decision)."""
    ss, oo, got = [], [], 0
    for _ in range(max_draws):
        s, o = episodes_fn(max(n, 256), rng)
        m = s.any(axis=1)
        ss.append(s[m]); oo.append(o[m]); got += int(m.sum())
        if got >= n:
            break
    if not got:
        raise ValueError("episode source produced only empty syndromes; nothing to learn")
    return np.concatenate(ss)[:n], np.concatenate(oo)[:n]


# ---------------------------------------------------------------- fitted Q-iteration
@dataclass
class Trajectories:
    """Decision points of recorded BP episodes (one row per (shot, iteration))."""
    state: np.ndarray           # (R,) encoded state after the iteration
    done: np.ndarray            # (R,) BP converged at this iteration (no decision)
    r_halt: np.ndarray          # (R,) reward of HALT here: -(osd + fail * OSD wrong)
    r_done: np.ndarray          # (R,) terminal reward when done here: -fail * BP wrong
    nxt: np.ndarray             # (R,) row of the next iteration of the same shot, -1 if last
    last: np.ndarray            # (R,) final allowed iteration (max_iter): HALT is forced
    episodes: int = 0


def collect_trajectories(bp: MinSumBP, syndromes: np.ndarray, obs_true: np.ndarray,
                         reward: Optional[RewardModel] = None) -> Trajectories:
    rm = reward or RewardModel()
    g = bp.g
    rows: list[tuple] = []          # (shot, t, state, done, osd_wrong)
    syn = np.asarray(syndromes, np.uint8)

    def observer(f: IterFeatures, done, halted, post):
        st = encode_state(f)
        for j, shot in enumerate(f.shot_index.tolist()):
            wrong = 0
            if not done[j]:
                pred, _, _ = osd0(g, syn[shot], post[:, j])
                wrong = int((pred != obs_true[shot]).any())
            rows.append((shot, f.t, int(st[j]), bool(done[j]), wrong))

    res = bp.decode(syn, observer=observer)
    bp_wrong = (g.logical(res.errors) != obs_true).any(axis=1)
    rows.sort()
    shot = np.array([r[0] for r in rows], np.int64)
    t = np.array([r[1] for r in rows], np.int64)
    nxt = np.arange(1, len(rows) + 1)
    same = np.zeros(len(rows), bool)
    same[:-1] = shot[1:] == shot[:-1]
    nxt = np.where(same, nxt, -1)
    done = np.array([r[3] for r in rows], bool)
    return Trajectories(
        state=np.array([r[2] for r in rows], np.int64), done=done,
        r_halt=-(rm.osd_cost + rm.fail_cost * np.array([r[4] for r in rows], float)),
        r_done=np.where(done, -rm.fail_cost * bp_wrong[shot], 0.0),
        nxt=nxt, last=t == bp.max_iter, episodes=len(syn))


def fitted_q_iteration(tr: Trajectories, reward: Optional[RewardModel] = None, gamma: float = 1.0,
                       min_visits: int = 20, sweeps: int = 200, tol: float = 1e-6) -> QHaltingPolicy:
    """Value iteration on the empirical MDP defined by recorded trajectories.

    Q(s, HALT)     = mean over visits of r_halt
    Q(s, CONTINUE) = mean over visits of  -iter_cost + gamma * [r_done(next) if next converged else V(s_next)]
    (at the last allowed iteration CONTINUE is unavailable and equals HALT).
    States with fewer than ``min_visits`` decision points keep CONTINUE.
    """
    rm = reward or RewardModel()
    dec = ~tr.done                                     # rows where a decision is taken
    s = tr.state[dec]
    r_halt = tr.r_halt[dec]
    nxt = tr.nxt[dec]
    last = tr.last[dec] | (nxt < 0)
    nxt_safe = np.where(nxt >= 0, nxt, 0)
    nxt_done = tr.done[nxt_safe] & ~last
    nxt_r = tr.r_done[nxt_safe]
    nxt_state = tr.state[nxt_safe]
    cnt = np.bincount(s, minlength=NUM_STATES).astype(float)
    seen = cnt > 0
    qh = np.zeros(NUM_STATES)
    qh[seen] = np.bincount(s, weights=r_halt, minlength=NUM_STATES)[seen] / cnt[seen]
    v = qh.copy()
    qc = np.full(NUM_STATES, -np.inf)
    for _ in range(sweeps):
        tgt = np.where(last, r_halt, -rm.iter_cost + gamma * np.where(nxt_done, nxt_r, v[nxt_state]))
        new_qc = np.full(NUM_STATES, -np.inf)
        new_qc[seen] = np.bincount(s, weights=tgt, minlength=NUM_STATES)[seen] / cnt[seen]
        new_v = np.where(seen, np.maximum(qh, new_qc), 0.0)
        delta = np.max(np.abs(new_v - v)) if seen.any() else 0.0
        qc, v = new_qc, new_v
        if delta < tol:
            break
    pol = QHaltingPolicy()
    pol.q[:, CONTINUE] = np.where(seen, qc, 0.0)
    pol.q[:, HALT] = np.where(seen & (cnt >= min_visits), qh, -1e9)
    pol.visits[:, HALT] = cnt.astype(np.int64)
    pol.visits[:, CONTINUE] = cnt.astype(np.int64)
    return pol


# ---------------------------------------------------------------- online Q-learning
class _EpsGreedy:
    """Exploring policy used during online training; records (state, action) per shot."""

    def __init__(self, agent: QHaltingPolicy, epsilon: float, rng: np.random.Generator):
        self.agent, self.eps, self.rng = agent, epsilon, rng
        self.trace: dict[int, list[tuple[int, int]]] = {}

    def halt(self, f: IterFeatures) -> np.ndarray:
        st = encode_state(f)
        q = self.agent.q[st]
        greedy = q[:, HALT] > q[:, CONTINUE]
        explore = self.rng.random(len(st)) < self.eps
        act = np.where(explore, self.rng.random(len(st)) < 0.5, greedy)
        for shot, s, a in zip(f.shot_index.tolist(), st.tolist(), act.tolist()):
            self.trace.setdefault(shot, []).append((s, int(a)))
        return act


def _online_q_learning(bp, episodes_fn, rng, rm, epochs, episodes_per_epoch, lr, gamma, eps_start, eps_end,
                       eval_fn, rep, verbose) -> QHaltingPolicy:
    agent = QHaltingPolicy()
    agent.q[:, CONTINUE] = -rm.iter_cost
    agent.q[:, HALT] = -rm.osd_cost
    g = bp.g
    for ep in range(epochs):
        eps = eps_start + (eps_end - eps_start) * (ep / max(1, epochs - 1))
        syn, obs_true = _nontrivial(episodes_fn, episodes_per_epoch, rng)
        pol = _EpsGreedy(agent, eps, rng)
        res = bp.decode(syn, policy=pol)
        pred, fb = finish_with_fallback(g, syn, res)
        wrong = (pred != obs_true).any(axis=1)
        for shot, steps in pol.trace.items():
            reason = res.exit_reason[shot]
            term = -(rm.osd_cost if fb[shot] else 0.0) - (rm.fail_cost if wrong[shot] else 0.0)
            n = len(steps)
            for i in range(n - 1, -1, -1):
                s, a = steps[i]
                if i == n - 1:
                    # the episode ended here: halted (OSD outcome), converged or forced at max_iter
                    target = term
                    a = HALT if reason == EXIT_HALTED else (CONTINUE if reason == EXIT_CONVERGED else a)
                else:
                    target = -rm.iter_cost + gamma * agent.q[steps[i + 1][0]].max()
                    a = CONTINUE
                agent.q[s, a] += lr * (target - agent.q[s, a])
                agent.visits[s, a] += 1
            rep.decision_points += n
        rep.episodes += len(syn)
        if eval_fn is not None:
            h = dict(eval_fn(agent))
            h.update(label=f"epoch {ep + 1}", epsilon=eps)
            rep.history.append(h)
            if verbose:
                print(rep.summary().splitlines()[-1])
    agent.q[agent.visits[:, HALT] == 0, HALT] = -1e9          # never-tried HALT never wins
    return agent


# ---------------------------------------------------------------- entry point
def train_q_policy(bp: MinSumBP, episodes_fn, *, method: str = "fqi", episodes: int = 20000,
                   reward: Optional[RewardModel] = None, gamma: float = 1.0, min_visits: int = 20,
                   seed: int = 0, eval_fn=None, verbose: bool = False,
                   # online Q-learning only
                   epochs: int = 20, lr: float = 0.1, eps_start: float = 0.5,
                   eps_end: float = 0.02) -> tuple[QHaltingPolicy, TrainReport]:
    """Learn a halting policy for ``bp``.

    ``episodes_fn(n, rng) -> (syndromes (n, C), true observable flips (n, K))``
    supplies training episodes (``dem_episodes(graph)`` or ``shot_episodes(...)``).
    ``eval_fn(policy) -> dict`` (optional) is called on the result (and after
    every epoch for the online method).
    """
    rm = reward or RewardModel()
    rng = np.random.default_rng(seed)
    rep = TrainReport(method=method)
    if method == "fqi":
        syn, obs = _nontrivial(episodes_fn, episodes, rng)
        tr = collect_trajectories(bp, syn, obs, rm)
        agent = fitted_q_iteration(tr, rm, gamma=gamma, min_visits=min_visits)
        rep.episodes = tr.episodes
        rep.decision_points = int((~tr.done).sum())
        if eval_fn is not None:
            h = dict(eval_fn(agent))
            h.update(label="final")
            rep.history.append(h)
    elif method == "online":
        agent = _online_q_learning(bp, episodes_fn, rng, rm, epochs, max(1, episodes // max(1, epochs)), lr,
                                   gamma, eps_start, eps_end, eval_fn, rep, verbose)
    else:
        raise ValueError("method must be 'fqi' or 'online'")
    rep.states_visited = int((agent.visits.sum(axis=1) > 0).sum())
    rep.halt_states = int(agent.halt_table.sum())
    agent.meta.update(method=method, reward=asdict(rm), max_iter=bp.max_iter, alpha=float(bp.alpha),
                      episodes=rep.episodes)
    if verbose:
        print(rep.summary())
    return agent, rep
