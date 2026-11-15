"""
Far-end stand-ins for Layers 2-3 (Router & Preprocessing Unit + decoder).

They implement the ``IngressSink`` interface of the AURA-Link gateway and
send corrections back through the gateway's egress path, so the whole
Layer 0 + Layer 1 loop can be exercised before the real decoder exists.

* ``NullDecoderStub``      -- answers "no correction" (measures the raw loop)
* ``OracleDecoderStub``    -- perfect answer using the noiseless reference
                              parities (validates every piece of plumbing:
                              logical error rate after correction must be 0)
* ``MatchingDecoderStub``  -- a real software decoder (PyMatching MWPM on the
                              compiler's detector error model), optional
                              dependency; gives meaningful logical error rates
* ``BPOSDDecoderStub``     -- min-sum BP + OSD-0 fallback (any CSS code incl. BB),
                              with convergence early exit
* ``RLBPDecoderStub``      -- BP+OSD with a reinforcement-learned halting policy
                              that cuts hopeless BP runs short

Decoder latency is modelled in virtual time: ``base_ps + per_event_ps *
(#detection events)`` for the single-pass decoders and ``base_ps + iter_ps *
iterations + osd_ps * [fallback]`` for the iterative ones (Python compute time
is irrelevant to the twin).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.telemetry import MetricsRegistry
from ..core.units import NS
from ..link.records import (STATUS_CONVERGED, STATUS_ERASURE_SEEN, STATUS_FALLBACK, STATUS_ORACLE,
                            STATUS_WCET_FORCED, CorrectionRecord, ErasureRecord, SyndromeRecord)


@dataclass
class _ShotBuf:
    rounds: dict = field(default_factory=dict)      # round -> bits
    erased: set = field(default_factory=set)
    last_seq: int = -1
    done: bool = False


class _BaseStub:
    name = "stub"

    def __init__(self, sched, gateway=None, base_latency_ps: int = 400 * NS, per_event_ps: int = 0,
                 metrics: Optional[MetricsRegistry] = None):
        self.s = sched
        self.gw = gateway
        self.base = base_latency_ps
        self.per_event = per_event_ps
        self.m = metrics or MetricsRegistry(self.name)
        self.manifest = None
        self._bufs: dict[tuple[int, int], _ShotBuf] = {}
        self._patch = {}
        self.decoded = 0
        self.records: list[SyndromeRecord] = []
        self.erasures: list[ErasureRecord] = []
        self.keep_records = False

    def attach(self, gateway) -> None:
        self.gw = gateway

    # ---------------------------------------------------------- IngressSink
    def on_session(self, event: str, info: dict) -> None:
        if event == "config":
            self.manifest = info["manifest"]
            self._patch = {pm.patch_id: pm for pm in self.manifest.patches}
            self._bufs.clear()
            self._configured()

    def _configured(self) -> None:
        pass

    def on_syndrome(self, rec: SyndromeRecord) -> None:
        if self.keep_records:
            self.records.append(rec)
        pm = self._patch.get(rec.patch)
        if pm is None:
            self.m.counter("unknown_patch").inc()
            return
        b = self._bufs.setdefault((rec.patch, rec.shot), _ShotBuf())
        if b.done:
            self.m.counter("after_decode").inc()
            return
        b.rounds[rec.round] = rec.bits
        b.last_seq = max(b.last_seq, rec.seq)
        # decode when the final frame arrives or every round is accounted for
        if rec.round == pm.rounds or len(b.rounds) + len(b.erased) >= pm.rounds + 1:
            self._decode(rec.patch, rec.shot, b)

    def on_erasure(self, e: ErasureRecord) -> None:
        self.erasures.append(e)
        self.m.counter(f"erasure.{e.reason}").inc()
        pm = self._patch.get(e.patch)
        if pm is None or e.shot < 0:
            return
        b = self._bufs.setdefault((e.patch, e.shot), _ShotBuf())
        if b.done:
            return
        b.erased.add(e.round)
        if len(b.rounds) + len(b.erased) >= pm.rounds + 1:
            self._decode(e.patch, e.shot, b)

    # ---------------------------------------------------------- decoding
    def _assemble(self, pm, b: _ShotBuf) -> np.ndarray:
        """Bits indexed by acquisition id (missing rounds -> 0)."""
        bits = np.zeros(self.manifest.num_acq, dtype=np.uint8)
        for r, accs in enumerate(pm.round_acqs):
            if r in b.rounds:
                bits[accs] = b.rounds[r]
        return bits

    def _decode(self, patch: int, shot: int, b: _ShotBuf) -> None:
        b.done = True
        pm = self._patch[patch]
        bits = self._assemble(pm, b)
        out = self.decode(pm, bits)
        obs, status, nevents = out[:3]
        if b.erased:
            status |= STATUS_ERASURE_SEEN
        # decoders may report their own modelled latency (4th element), e.g. per BP iteration
        lat = int(out[3]) if len(out) > 3 else self.base + self.per_event * nevents
        cor = CorrectionRecord(patch=patch, shot=shot, round=pm.rounds, obs_flips=np.asarray(obs, np.uint8),
                               status=status, decoder_latency_ns=lat // 1000, ref_seq=b.last_seq)
        self.decoded += 1
        self.m.counter("decoded").inc()
        self.s.call_later(lat, self._emit, cor)
        # bounded memory
        if len(self._bufs) > 4096:
            for k in list(self._bufs)[:1024]:
                if self._bufs[k].done:
                    del self._bufs[k]

    def _emit(self, cor: CorrectionRecord) -> None:
        if self.gw is not None:
            self.gw.send_correction(cor)

    def decode(self, pm, bits: np.ndarray) -> tuple:
        """One shot: bits by acquisition id -> (obs flips, status, #events[, latency_ps])."""
        raise NotImplementedError

    def decode_batch(self, pm, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Many shots (S, num_acq) -> (obs flips (S, K), status (S,), #events (S,)).
        Subclasses override this with a vectorised path; the default loops."""
        bits = np.atleast_2d(bits)
        outs = [self.decode(pm, b) for b in bits]
        k = len(pm.observables)
        obs = np.array([o[0] for o in outs], np.uint8).reshape(len(bits), k)
        return obs, np.array([o[1] for o in outs], np.int64), np.array([o[2] for o in outs], np.int64)

    @staticmethod
    def _basis_detectors(pm) -> np.ndarray:
        """Detectors that see the memory basis' logical errors (the decoding graph)."""
        return np.array([i for i, b in enumerate(pm.det_basis) if b == pm.basis], dtype=np.int64)


class NullDecoderStub(_BaseStub):
    name = "null-decoder"

    def decode(self, pm, bits):
        return np.zeros(len(pm.observables), np.uint8), 0, 0

    def decode_batch(self, pm, bits):
        n = len(np.atleast_2d(bits))
        return np.zeros((n, len(pm.observables)), np.uint8), np.zeros(n, np.int64), np.zeros(n, np.int64)


class OracleDecoderStub(_BaseStub):
    """Returns the exact observable flip (measured parity XOR noiseless reference)."""
    name = "oracle-decoder"

    def decode(self, pm, bits):
        return pm.observable_flips(bits), STATUS_ORACLE | STATUS_CONVERGED, 0

    def decode_batch(self, pm, bits):
        obs = pm.observable_flips(np.atleast_2d(bits))
        n = len(obs)
        return obs, np.full(n, STATUS_ORACLE | STATUS_CONVERGED, np.int64), np.zeros(n, np.int64)


class MatchingDecoderStub(_BaseStub):
    """Minimum-weight perfect matching on the DEM restricted to the detectors
    that see the memory basis' logical errors (graph-like for surface /
    repetition codes).  Requires ``pymatching``."""
    name = "matching-decoder"

    def __init__(self, *a, per_event_ps: int = 20 * NS, **kw):
        super().__init__(*a, per_event_ps=per_event_ps, **kw)
        self._matchers: dict[int, tuple] = {}

    def _configured(self) -> None:
        try:
            import pymatching
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("MatchingDecoderStub needs the optional 'pymatching' package "
                               "(pip install 'aura-qec[decoders]')") from e
        self._matchers = {}
        for pm in self.manifest.patches:
            if pm.dem is None:
                raise RuntimeError("manifest has no DEM; compile with build_dem=True")
            keep = self._basis_detectors(pm)
            dem = pm.dem.restrict(keep)
            h_cols, weights, l_cols = [], [], []
            split = 0
            for p, ds, os_ in zip(dem.probs, dem.dets, dem.obs):
                if p <= 0 or p >= 0.5:
                    continue
                w = float(np.log((1 - p) / p))
                if len(ds) <= 2:
                    h_cols.append(ds); l_cols.append(os_); weights.append(w)
                else:                                   # rare: split a hyperedge into pairs
                    split += 1
                    ds = list(ds)
                    for i in range(0, len(ds), 2):
                        part = tuple(ds[i:i + 2])
                        h_cols.append(part); l_cols.append(os_ if i == 0 else ()); weights.append(w)
            nd, nm = len(keep), len(h_cols)
            H = np.zeros((nd, nm), np.uint8)
            L = np.zeros((pm.dem.num_observables, nm), np.uint8)
            for j, (ds, os_) in enumerate(zip(h_cols, l_cols)):
                H[list(ds), j] = 1
                if os_:
                    L[list(os_), j] = 1
            mt = pymatching.Matching.from_check_matrix(H, weights=np.array(weights), faults_matrix=L,
                                                       use_virtual_boundary_node=True)
            self._matchers[pm.patch_id] = (mt, keep)
            self.m.gauge(f"hyperedges_split.p{pm.patch_id}").set(split)

    def decode(self, pm, bits):
        mt, keep = self._matchers[pm.patch_id]
        ev = pm.detection_events(bits)[keep]
        pred = mt.decode(ev)
        return np.asarray(pred, np.uint8), STATUS_CONVERGED, int(ev.sum())

    def decode_batch(self, pm, bits):
        mt, keep = self._matchers[pm.patch_id]
        ev = pm.detection_events(np.atleast_2d(bits))[:, keep]
        pred = np.asarray(mt.decode_batch(ev), np.uint8).reshape(len(ev), -1)
        return pred, np.full(len(ev), STATUS_CONVERGED, np.int64), ev.sum(axis=1).astype(np.int64)


# ---------------------------------------------------------------------- iterative decoders
_POLICY_CACHE: dict[tuple, object] = {}


def _dem_key(dem) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(np.asarray(dem.probs, np.float64).tobytes())
    h.update(repr((dem.dets, dem.obs)).encode())
    return h.hexdigest()


class BPOSDDecoderStub(_BaseStub):
    """Min-sum belief propagation on the basis-restricted DEM with an OSD-0
    fallback for shots BP does not resolve (works for any CSS code, including
    BB codes whose DEMs are hypergraphs that matching cannot represent).

    Iteration control: zero-syndrome shots cost 0 iterations; every shot exits
    at the first iteration whose hard decision explains its syndrome.

    Modelled latency: ``base + iter_ps * iterations + osd_ps * [fallback]``.
    Status bits: CONVERGED (BP converged), FALLBACK (OSD used), WCET_FORCED
    (iteration budget exhausted).
    """
    name = "bposd-decoder"

    def __init__(self, *a, max_iter: int = 30, alpha: float = 0.75, iter_ps: int = 20 * NS,
                 osd_ps: int = 400 * NS, **kw):
        super().__init__(*a, **kw)
        self.max_iter = max_iter
        self.alpha = alpha
        self.iter_ps = iter_ps
        self.osd_ps = osd_ps
        self._bp: dict[int, tuple] = {}

    def _configured(self) -> None:
        from ..decoding import MinSumBP, TannerGraph
        self._bp = {}
        for pm in self.manifest.patches:
            if pm.dem is None:
                raise RuntimeError("manifest has no DEM; compile with build_dem=True")
            keep = self._basis_detectors(pm)
            g = TannerGraph(pm.dem.restrict(keep))
            self._bp[pm.patch_id] = (MinSumBP(g, max_iter=self.max_iter, alpha=self.alpha), keep)
        self._configured_policy()

    def _configured_policy(self) -> None:
        pass

    def _policy(self, patch_id: int):
        return None

    def _run(self, pm, ev: np.ndarray):
        from ..decoding import EXIT_HALTED, EXIT_MAX_ITER, finish_with_fallback
        bp, _ = self._bp[pm.patch_id]
        res = bp.decode(ev, policy=self._policy(pm.patch_id))
        pred, fb = finish_with_fallback(bp.g, ev, res)
        status = np.where(res.converged, STATUS_CONVERGED, 0) | np.where(fb, STATUS_FALLBACK, 0) \
            | np.where(res.exit_reason == EXIT_MAX_ITER, STATUS_WCET_FORCED, 0)
        lat = self.base + self.iter_ps * res.iterations.astype(np.int64) + self.osd_ps * fb
        # telemetry
        m = self.m
        h = m.histogram("bp.iterations")
        for it, c in zip(*np.unique(res.iterations, return_counts=True)):
            h.record(int(it), int(c))
        m.counter("bp.shots").inc(len(ev))
        m.counter("bp.iterations_total").inc(int(res.iterations.sum()))
        m.counter("bp.converged").inc(int((res.converged & ev.any(axis=1)).sum()))
        m.counter("bp.trivial").inc(int((~ev.any(axis=1)).sum()))
        m.counter("bp.fallback_osd").inc(int(fb.sum()))
        m.counter("bp.halted_early").inc(int((res.exit_reason == EXIT_HALTED).sum()))
        m.counter("bp.wcet_forced").inc(int((res.exit_reason == EXIT_MAX_ITER).sum()))
        return pred, status.astype(np.int64), lat

    def decode(self, pm, bits):
        _, keep = self._bp[pm.patch_id]
        ev = pm.detection_events(bits)[keep][None, :]
        pred, status, lat = self._run(pm, ev)
        return pred[0], int(status[0]), int(ev.sum()), int(lat[0])

    def decode_batch(self, pm, bits):
        _, keep = self._bp[pm.patch_id]
        ev = pm.detection_events(np.atleast_2d(bits))[:, keep]
        pred, status, _ = self._run(pm, ev)
        return pred, status, ev.sum(axis=1).astype(np.int64)


class RLBPDecoderStub(BPOSDDecoderStub):
    """BP+OSD whose iteration budget is controlled per shot by a reinforcement-
    learned halting policy (``aura.decoding.policy``): trajectories the policy
    judges hopeless are handed to OSD early instead of running to ``max_iter``.

    ``policy``: a ``QHaltingPolicy``, a path to a saved one, or ``None`` to train
    one from the manifest's DEM at configuration time (cached per DEM, so
    repeated runs of the same program reuse it).  The reward is the modelled
    latency itself: one iteration costs ``iter_ps``, a fallback ``osd_ps``.
    """
    name = "rlbp-decoder"

    def __init__(self, *a, policy=None, train_episodes: int = 4000, fail_cost: float = 2000.0,
                 seed: int = 0, **kw):
        super().__init__(*a, **kw)
        self.policy_src = policy
        self.train_episodes = train_episodes
        self.fail_cost = fail_cost
        self.seed = seed
        self._policies: dict[int, object] = {}

    def reward_model(self):
        from ..decoding import RewardModel
        return RewardModel(iter_cost=1.0, osd_cost=self.osd_ps / max(1, self.iter_ps), fail_cost=self.fail_cost)

    def _configured_policy(self) -> None:
        from ..decoding import QHaltingPolicy, dem_episodes, train_q_policy
        self._policies = {}
        if isinstance(self.policy_src, (str, Path)) and str(self.policy_src):
            pol = QHaltingPolicy.load(self.policy_src)
            self._policies = {pm.patch_id: pol for pm in self.manifest.patches}
            return
        if isinstance(self.policy_src, QHaltingPolicy):
            self._policies = {pm.patch_id: self.policy_src for pm in self.manifest.patches}
            return
        rm = self.reward_model()
        for pm in self.manifest.patches:
            bp, keep = self._bp[pm.patch_id]
            key = (_dem_key(pm.dem), tuple(keep.tolist()), self.max_iter, float(self.alpha),
                   tuple(rm.__dict__.values()), self.train_episodes, self.seed)
            pol = _POLICY_CACHE.get(key)
            if pol is None:
                pol, rep = train_q_policy(bp, dem_episodes(bp.g), episodes=self.train_episodes, reward=rm,
                                          seed=self.seed)
                _POLICY_CACHE[key] = pol
                self.m.gauge(f"rl.halt_states.p{pm.patch_id}").set(rep.halt_states)
            self._policies[pm.patch_id] = pol

    def _policy(self, patch_id: int):
        return self._policies.get(patch_id)
