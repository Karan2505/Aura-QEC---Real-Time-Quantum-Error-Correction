"""
Batched belief-propagation (normalised min-sum) decoding on a detector error model.

The Tanner graph has one *check* per detector and one *variable* per DEM
mechanism.  Decoding a shot means finding the most likely set of mechanisms
that explains its detection events; the logical prediction is the XOR of the
observables those mechanisms flip.

Iteration control (the part that saves compute):

* **zero-syndrome fast path** -- a shot with no detection events needs 0 iterations;
* **convergence early exit** -- after every iteration the hard decision is
  checked against the syndrome; a shot whose syndrome is explained leaves the
  active set immediately (if it converges at iteration 3, it costs 3 iterations,
  not ``max_iter``);
* **halting policy hook** -- an optional ``policy`` sees per-shot convergence
  features after each iteration and may *halt* shots it judges hopeless,
  handing them to the fallback decoder early instead of running to ``max_iter``
  (see ``aura.decoding.policy`` for the reinforcement-learned policy).

Implementation: arrays are laid out (edges, shots) so gathers are row gathers
and segment reductions run over contiguous rows; all per-iteration temporaries
live in a preallocated workspace (fresh multi-MB temporaries page-fault on
every iteration otherwise); per-variable sums and the syndrome check are
sparse matrix products.  The active set shrinks as shots converge or halt, so
the work done tracks the number of *unresolved* shots, not the batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import scipy.sparse as sp

from ..host.compiler.dem import DetectorErrorModel

# per-shot termination reasons
EXIT_TRIVIAL = 0        # empty syndrome, nothing to do
EXIT_CONVERGED = 1      # BP hard decision explains the syndrome
EXIT_HALTED = 2         # halting policy stopped BP early
EXIT_MAX_ITER = 3       # iteration budget exhausted (worst-case execution time)


@dataclass
class IterFeatures:
    """Convergence state of the active shots after iteration ``t`` (1-based)."""
    t: int
    unsat: np.ndarray           # (A,) unsatisfied checks now
    unsat0: np.ndarray          # (A,) syndrome weight before BP
    unsat_prev: np.ndarray      # (A,) unsatisfied checks after the previous iteration
    stall: np.ndarray           # (A,) consecutive iterations without improving the best unsat
    flips: np.ndarray           # (A,) hard-decision bits that changed in this iteration
    shot_index: np.ndarray      # (A,) index of each active shot in the batch


class HaltingPolicy(Protocol):
    def halt(self, f: IterFeatures) -> np.ndarray:  # (A,) bool
        ...


@dataclass
class BPResult:
    errors: np.ndarray          # (B, M) uint8 hard decision (meaningful when converged)
    posterior: np.ndarray       # (B, M) float32 posterior LLRs at exit (input to OSD)
    converged: np.ndarray       # (B,) bool
    iterations: np.ndarray      # (B,) int32 iterations spent
    exit_reason: np.ndarray     # (B,) int8 EXIT_*


class TannerGraph:
    """Check matrix of a DEM in edge-list form, with the index plumbing min-sum needs.

    Mechanisms that flip no detector (undetectable) or have p outside (0, 0.5)
    are dropped: BP cannot infer them.
    """

    def __init__(self, dem: DetectorErrorModel, min_prob: float = 1e-12):
        keep = [j for j, (p, ds) in enumerate(zip(dem.probs, dem.dets)) if ds and min_prob < p < 0.5]
        self.num_checks = dem.num_detectors
        self.num_vars = len(keep)
        self.num_observables = dem.num_observables
        self.priors = np.array([dem.probs[j] for j in keep], dtype=np.float64)
        self.var_dets = [tuple(dem.dets[j]) for j in keep]
        self.var_obs = [tuple(dem.obs[j]) for j in keep]
        chk = np.array([c for ds in self.var_dets for c in ds], dtype=np.int64)
        var = np.array([v for v, ds in enumerate(self.var_dets) for _ in ds], dtype=np.int64)
        order = np.lexsort((var, chk))             # edges sorted by check
        self.e_chk = chk[order]
        self.e_var = var[order]
        self.num_edges = E = len(order)
        deg_c = np.bincount(self.e_chk, minlength=self.num_checks)
        self.empty_checks = np.nonzero(deg_c == 0)[0]
        self.active_checks = np.nonzero(deg_c > 0)[0]
        nca = len(self.active_checks)
        self.chk_ptr = np.concatenate([[0], np.cumsum(deg_c[self.active_checks])[:-1]]).astype(np.int64)
        self.e_chk_local = np.repeat(np.arange(nca), deg_c[self.active_checks])
        self.prior_llr = np.log((1 - self.priors) / self.priors).astype(np.float32)
        # sparse operators: per-variable sum over its edges, and syndrome of a hard decision
        ones = np.ones(E, np.float32)
        self.var_sum = sp.csr_matrix((ones, (self.e_var, np.arange(E))), shape=(self.num_vars, E))
        self.h_active = sp.csr_matrix((ones, (self.e_chk_local, self.e_var)), shape=(nca, self.num_vars))
        # observable matrix and bitset column views (for OSD)
        self.obs_matrix = np.zeros((self.num_observables, self.num_vars), np.uint8)
        for v, os_ in enumerate(self.var_obs):
            if os_:
                self.obs_matrix[list(os_), v] = 1
        self.col_bits = [sum(1 << c for c in ds) for ds in self.var_dets]
        self.col_obs = [sum(1 << o for o in os_) for os_ in self.var_obs]

    def syndrome(self, errors: np.ndarray) -> np.ndarray:
        """(B, M) -> (B, C) syndrome of a batch of error patterns."""
        e = np.atleast_2d(np.asarray(errors, np.float32))
        out = np.zeros((e.shape[0], self.num_checks), np.uint8)
        out[:, self.active_checks] = (self.h_active @ e.T).T.astype(np.int64) & 1
        return out

    def logical(self, errors: np.ndarray) -> np.ndarray:
        """(B, M) -> (B, K) observable flips."""
        return (np.asarray(errors, np.int64) @ self.obs_matrix.T.astype(np.int64) & 1).astype(np.uint8)

    def sample(self, shots: int, rng: np.random.Generator, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``shots`` error patterns from the (independent-mechanism) model.
        Returns (errors (B, M), syndromes (B, C), observable flips (B, K))."""
        p = np.clip(self.priors * scale, 0.0, 0.5)
        e = (rng.random((shots, self.num_vars)) < p).astype(np.uint8)
        return e, self.syndrome(e), self.logical(e)


class _Workspace:
    """Named flat buffers; ``get`` returns a C-contiguous (rows, n) view of the prefix."""

    def __init__(self):
        self._buf: dict[str, np.ndarray] = {}

    def get(self, name: str, rows: int, n: int, dtype) -> np.ndarray:
        size = rows * n
        b = self._buf.get(name)
        if b is None or b.size < size or b.dtype != dtype:
            b = self._buf[name] = np.empty(max(size, 1), dtype)
        return b[:size].reshape(rows, n)


class MinSumBP:
    """Normalised min-sum BP with convergence early exit and an optional halting policy."""

    def __init__(self, graph: TannerGraph, max_iter: int = 30, alpha: float = 0.75, damping: float = 0.0,
                 chunk: int = 64, posterior_avg: int = 1):
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= damping < 1.0:
            raise ValueError("damping must be in [0, 1)")
        if posterior_avg not in (1, 2):
            raise ValueError("posterior_avg must be 1 (final iteration) or 2 (mean of the last two)")
        self.g = graph
        self.max_iter = max_iter
        self.alpha = np.float32(alpha)
        self.damping = np.float32(damping)
        self.chunk = chunk
        # Posterior reported at exit (the OSD input): the last iteration's (1), or the mean of the
        # last two (2), which cancels the period-2 oscillation BP shows on BB codes. At t = 1 the
        # mean falls back to the current posterior.
        self.posterior_avg = posterior_avg
        self._ws = _Workspace()

    def decode(self, syndromes: np.ndarray, policy: Optional[HaltingPolicy] = None,
               prior_llr: Optional[np.ndarray] = None,
               observer: Optional[Callable[..., None]] = None) -> BPResult:
        """Decode a (B, C) batch of syndromes.

        ``observer(features, converged_mask, halted_mask, posterior)`` is called
        after every iteration with the (V, A) posterior LLRs of the active shots
        (used to record trajectories when training a policy).
        """
        s = np.atleast_2d(np.asarray(syndromes, np.uint8))
        B = s.shape[0]
        M = self.g.num_vars
        res = BPResult(errors=np.zeros((B, M), np.uint8), posterior=np.empty((B, M), np.float32),
                       converged=np.zeros(B, bool), iterations=np.zeros(B, np.int32),
                       exit_reason=np.full(B, EXIT_TRIVIAL, np.int8))
        prior = self.g.prior_llr if prior_llr is None else np.asarray(prior_llr, np.float32)
        res.posterior[:] = prior
        w0 = s.sum(axis=1)
        # syndrome bits on checks no mechanism touches cannot be explained by BP
        unexplainable = s[:, self.g.empty_checks].any(axis=1) if len(self.g.empty_checks) else np.zeros(B, bool)
        res.converged[w0 == 0] = True
        res.exit_reason[unexplainable] = EXIT_MAX_ITER
        todo = np.nonzero((w0 > 0) & ~unexplainable)[0]
        for i in range(0, len(todo), self.chunk):
            self._run(s, todo[i:i + self.chunk], prior, res, policy, observer)
        return res

    # ------------------------------------------------------------------
    def _run(self, s_all, idx, prior, res: BPResult, policy, observer) -> None:
        g, ws = self.g, self._ws
        E, V = g.num_edges, g.num_vars
        C = len(g.active_checks)
        e_var, chk_ptr, e_loc = g.e_var, g.chk_ptr, g.e_chk_local
        alpha, damp = self.alpha, self.damping
        inf = np.float32(np.inf)
        prior_col = prior[:, None]
        # per-shot state (re-allocated only when the active set shrinks)
        s = np.ascontiguousarray(s_all[idx][:, g.active_checks].T)               # (C, A) uint8
        A = len(idx)
        m_vc = np.repeat(prior[e_var][:, None], A, axis=1)                         # (E, A) float32
        m_cv = np.zeros_like(m_vc)
        hard = np.zeros((V, A), bool)
        unsat0 = s.sum(axis=0, dtype=np.int32)
        unsat_prev = unsat0.copy()
        best = unsat0.copy()
        stall = np.zeros(A, np.int32)
        alive = np.arange(A)                     # positions (into idx) still iterating
        prev = None                              # previous iteration's posterior (posterior_avg == 2)
        avg = self.posterior_avg == 2
        for t in range(1, self.max_iter + 1):
            n = len(alive)
            # ---- check-node update (normalised min-sum, min1/min2 trick) ----
            mag = np.abs(m_vc, out=ws.get("mag", E, n, np.float32))
            neg = np.less(m_vc, 0, out=ws.get("neg", E, n, bool))
            min1 = np.minimum.reduceat(mag, chk_ptr, axis=0, out=ws.get("min1", C, n, np.float32))
            min1_e = np.take(min1, e_loc, axis=0, mode="clip", out=ws.get("min1e", E, n, np.float32))
            is_min = np.equal(mag, min1_e, out=ws.get("ismin", E, n, bool))
            cnt = np.add.reduceat(is_min, chk_ptr, axis=0, dtype=np.int16, out=ws.get("cnt", C, n, np.int16))
            np.copyto(mag, inf, where=is_min)                      # mag now = magnitudes without the minimum
            min2 = np.minimum.reduceat(mag, chk_ptr, axis=0, out=ws.get("min2", C, n, np.float32))
            uniq = np.equal(np.take(cnt, e_loc, axis=0, mode="clip", out=ws.get("cnte", E, n, np.int16)), 1,
                            out=ws.get("uniq", E, n, bool))
            uniq &= is_min                                         # the unique minimum sees the second minimum
            ext = np.take(min2, e_loc, axis=0, mode="clip", out=ws.get("ext", E, n, np.float32))
            np.copyto(ext, min1_e, where=~uniq)
            negu8 = neg.view(np.uint8)
            par = np.bitwise_xor.reduceat(negu8, chk_ptr, axis=0, out=ws.get("par", C, n, np.uint8))
            par ^= s
            flip = np.take(par, e_loc, axis=0, mode="clip", out=ws.get("flip", E, n, np.uint8))
            flip ^= negu8
            new_cv = np.multiply(ext, alpha, out=ws.get(f"mcv{t & 1}", E, n, np.float32))
            np.negative(new_cv, out=new_cv, where=flip.view(bool))
            if damp:
                new_cv *= (1 - damp)
                new_cv += damp * m_cv
            m_cv = new_cv
            # ---- variable-node update ----
            tot = g.var_sum @ m_cv                                 # (V, n)
            tot += prior_col
            m_vc = np.take(tot, e_var, axis=0, mode="clip", out=ws.get("mvc", E, n, np.float32))
            m_vc -= m_cv
            new_hard = np.less(tot, 0, out=ws.get(f"hard{t & 1}", V, n, bool))
            flips = np.not_equal(new_hard, hard, out=ws.get("flips", V, n, bool)).sum(axis=0, dtype=np.int32)
            hard = new_hard
            # ---- syndrome check ----
            syn = (g.h_active @ hard.view(np.uint8).astype(np.float32)).astype(np.uint8)
            syn &= 1
            unsat = np.not_equal(syn, s).sum(axis=0, dtype=np.int32)
            improved = unsat < best
            best = np.minimum(best, unsat)
            stall = np.where(improved, 0, stall + 1)
            done = unsat == 0
            halted = np.zeros(n, bool)
            if policy is not None or observer is not None:
                f = IterFeatures(t=t, unsat=unsat, unsat0=unsat0[alive], unsat_prev=unsat_prev, stall=stall,
                                 flips=flips, shot_index=idx[alive])
                if policy is not None:
                    halted = np.asarray(policy.halt(f), bool) & ~done
                if observer is not None:
                    observer(f, done, halted, tot)
            leaving = done | halted if t < self.max_iter else np.ones(n, bool)
            if leaving.any():
                rows = idx[alive[leaving]]
                res.errors[rows] = hard[:, leaving].T
                if avg and prev is not None:
                    res.posterior[rows] = ((prev[:, leaving] + tot[:, leaving]) * np.float32(0.5)).T
                else:
                    res.posterior[rows] = tot[:, leaving].T
                res.iterations[rows] = t
                res.converged[rows] = done[leaving]
                res.exit_reason[rows] = np.where(done[leaving], EXIT_CONVERGED,
                                                 np.where(halted[leaving], EXIT_HALTED, EXIT_MAX_ITER))
                keep = ~leaving
                if not keep.any():
                    return
                alive = alive[keep]
                # compact the state into fresh arrays (they must not alias workspace buffers)
                s, m_vc, m_cv, hard = s[:, keep], m_vc[:, keep], m_cv[:, keep], hard[:, keep]
                unsat, best, stall = unsat[keep], best[keep], stall[keep]
                if avg:
                    tot = tot[:, keep]
            if avg:
                prev = tot
            unsat_prev = unsat
