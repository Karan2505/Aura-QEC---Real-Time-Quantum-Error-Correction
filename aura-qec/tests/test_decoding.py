"""Iterative decoding: BP early exit, OSD fallback, RL-learned halting policy, lab integration."""

import numpy as np
import pytest

from conftest import make_compiled

from aura.decoding import (EXIT_CONVERGED, EXIT_HALTED, EXIT_MAX_ITER, EXIT_TRIVIAL, MinSumBP, QHaltingPolicy,
                           RewardModel, TannerGraph, dem_episodes, evaluate, osd0, train_q_policy)


def _graph(spec="surface:3", rounds=3, p=3e-3, basis="Z"):
    _, _, ce, _ = make_compiled(spec, rounds, basis, p=p)
    pm = ce.manifest.patches[0]
    keep = [i for i, b in enumerate(pm.det_basis) if b == pm.basis]
    return TannerGraph(pm.dem.restrict(keep)), pm.dem.restrict(keep), ce


@pytest.fixture(scope="module")
def s3():
    return _graph("surface:3", 3)


@pytest.fixture(scope="module")
def s5():
    return _graph("surface:5", 5)


def test_tanner_graph_syndrome_matches_check_matrix(s3):
    g, dem, _ = s3
    h = dem.check_matrix()
    keep = [j for j, (p, ds) in enumerate(zip(dem.probs, dem.dets)) if ds and 0 < p < 0.5]
    e = np.random.default_rng(0).integers(0, 2, (50, g.num_vars), dtype=np.uint8)
    ref = (e.astype(int) @ h[:, keep].T.astype(int)) % 2
    assert np.array_equal(g.syndrome(e), ref)
    _, s, o = g.sample(200, np.random.default_rng(1))
    assert s.shape == (200, g.num_checks) and o.shape == (200, g.num_observables)


def test_bp_trivial_shots_cost_zero_iterations(s3):
    g, _, _ = s3
    res = MinSumBP(g).decode(np.zeros((5, g.num_checks), np.uint8))
    assert (res.iterations == 0).all() and res.converged.all() and (res.exit_reason == EXIT_TRIVIAL).all()


def test_bp_exits_at_first_converging_iteration(s3):
    """Single-fault syndromes converge quickly and stop there instead of running max_iter."""
    g, _, _ = s3
    e = np.eye(g.num_vars, dtype=np.uint8)
    s = g.syndrome(e)
    bp = MinSumBP(g, max_iter=50)
    res = bp.decode(s)
    # most single faults converge, and those exit within a few iterations, far below the budget
    assert res.converged.mean() > 0.75
    assert (res.exit_reason[res.converged] == EXIT_CONVERGED).all()
    assert res.iterations[res.converged].max() <= 5 < bp.max_iter
    # converged hard decisions explain their syndromes
    assert np.array_equal(g.syndrome(res.errors[res.converged]), s[res.converged])
    # the rest (short-cycle oscillation / exact ties: classic BP failure modes) is what OSD is for
    for i in np.nonzero(~res.converged)[0]:
        assert osd0(g, s[i], res.posterior[i])[2]


def test_bp_iteration_budget_and_halting_hook(s5):
    g, _, _ = s5
    _, s, _ = g.sample(600, np.random.default_rng(3), scale=3.0)

    class HaltAll:
        def halt(self, f):
            return np.ones(len(f.unsat), bool)

    res = MinSumBP(g, max_iter=7).decode(s, policy=HaltAll())
    nz = s.any(axis=1)
    assert (res.iterations[nz] == 1).all()
    assert set(np.unique(res.exit_reason[nz])) <= {EXIT_CONVERGED, EXIT_HALTED}
    res2 = MinSumBP(g, max_iter=7).decode(s)
    assert res2.iterations.max() == 7 and (res2.exit_reason == EXIT_MAX_ITER).any()


def test_posterior_avg_reports_mean_of_last_two_iterations(s5):
    g, _, _ = s5
    _, s, _ = g.sample(400, np.random.default_rng(5), scale=3.0)
    hist = {}

    def observer(f, done, halted, tot):
        for j, shot in enumerate(f.shot_index.tolist()):
            hist.setdefault(shot, []).append(tot[:, j].copy())

    res1 = MinSumBP(g, max_iter=9).decode(s)
    res2 = MinSumBP(g, max_iter=9, posterior_avg=2).decode(s, observer=observer)
    # BP itself is unchanged: same decisions, iterations and exits
    assert (res1.errors == res2.errors).all() and (res1.iterations == res2.iterations).all()
    assert (res1.exit_reason == res2.exit_reason).all()
    checked = 0
    for shot, posts in hist.items():
        want = posts[-1] if len(posts) == 1 else (posts[-2] + posts[-1]) * np.float32(0.5)
        assert np.array_equal(res2.posterior[shot], want)
        assert np.array_equal(res1.posterior[shot], posts[-1])
        checked += len(posts) > 1
    assert checked > 20                           # many shots ran several iterations
    with pytest.raises(ValueError):
        MinSumBP(g, posterior_avg=3)


def test_memory_table_zero_is_plain_bp_and_nonzero_changes_decoding(s5):
    g, _, _ = s5
    _, s, _ = g.sample(400, np.random.default_rng(7), scale=3.0)
    plain = MinSumBP(g, max_iter=20, alpha=1.0).decode(s)
    zero = MinSumBP(g, max_iter=20, alpha=1.0, memory_table=np.zeros(16)).decode(s)
    assert (plain.errors == zero.errors).all() and (plain.iterations == zero.iterations).all()
    assert np.array_equal(plain.posterior, zero.posterior)
    table = np.linspace(0.05, 0.4, 16)
    mem = MinSumBP(g, max_iter=20, alpha=1.0, memory_table=table).decode(s)
    assert (mem.iterations != plain.iterations).any()           # the memory term actually acts
    with pytest.raises(ValueError):
        MinSumBP(g, memory_table=np.zeros(8))


def test_osd0_solves_every_explainable_syndrome(s5):
    g, _, _ = s5
    rng = np.random.default_rng(4)
    e, s, o = g.sample(300, rng, scale=4.0)
    for i in range(len(s)):
        pred, examined, ok = osd0(g, s[i], g.prior_llr)
        assert ok and examined <= g.num_vars
    # with a posterior that points at the true (single) fault, OSD-0 returns its logical effect
    for j in rng.choice(g.num_vars, 40, replace=False):
        post = g.prior_llr.copy()
        post[j] = -10.0
        ej = np.zeros(g.num_vars, np.uint8)
        ej[j] = 1
        pred, _, ok = osd0(g, g.syndrome(ej[None])[0], post)
        assert ok and np.array_equal(pred, g.logical(ej[None])[0])


def test_bp_osd_matches_mwpm_on_surface_code(s3):
    pymatching = pytest.importorskip("pymatching")
    g, dem, _ = s3
    _, s, o = g.sample(6000, np.random.default_rng(5))
    h = np.zeros((g.num_checks, g.num_vars), np.uint8)
    for v, ds in enumerate(g.var_dets):
        h[list(ds), v] = 1
    mt = pymatching.Matching.from_check_matrix(h, weights=g.prior_llr.astype(float), faults_matrix=g.obs_matrix)
    ler_mwpm = (mt.decode_batch(s) != o).any(axis=1).mean()
    ler_bp = evaluate(MinSumBP(g), s, o)["ler"]
    assert ler_bp <= 1.25 * ler_mwpm + 2e-3


def test_rl_policy_cuts_iterations_without_hurting_accuracy(s5):
    g, _, _ = s5
    bp = MinSumBP(g, max_iter=30)
    pol, rep = train_q_policy(bp, dem_episodes(g), episodes=6000, seed=0)
    assert rep.halt_states > 0 and rep.decision_points > 0
    _, s, o = g.sample(8000, np.random.default_rng(77))
    base = evaluate(bp, s, o)
    rl = evaluate(bp, s, o, pol)
    assert rl["total_iterations"] < 0.7 * base["total_iterations"]
    assert rl["p99_iterations"] < base["p99_iterations"]
    # accuracy: no worse than the baseline beyond statistical noise (paired, same shots)
    assert rl["logical_errors"] <= base["logical_errors"] + 3 * np.sqrt(base["logical_errors"] + 1)


def test_online_q_learning_runs(s3):
    g, _, _ = s3
    bp = MinSumBP(g, max_iter=20)
    pol, rep = train_q_policy(bp, dem_episodes(g), method="online", episodes=1500, epochs=3, seed=1)
    assert rep.episodes == 1500 and pol.q.shape[1] == 2


def test_policy_roundtrip(tmp_path, s3):
    g, _, _ = s3
    pol, _ = train_q_policy(MinSumBP(g), dem_episodes(g), episodes=1500, seed=2,
                            reward=RewardModel(osd_cost=5.0))
    path = tmp_path / "p.json"
    pol.save(path)
    back = QHaltingPolicy.load(path)
    assert np.array_equal(back.halt_table, pol.halt_table)
    assert back.meta["reward"]["osd_cost"] == 5.0
    with pytest.raises(ValueError):
        QHaltingPolicy.from_dict({"format": "nope"})


def test_vectorised_parities_match_reference_loop():
    _, _, ce, _ = make_compiled("bb:72", 2, "X")
    pm = ce.manifest.patches[0]
    b = np.random.default_rng(6).integers(0, 2, (3, 4, ce.num_acq), dtype=np.uint8)

    def ref(rows, refv):
        out = np.zeros(b.shape[:-1] + (len(rows),), np.uint8)
        for i, accs in enumerate(rows):
            out[..., i] = np.bitwise_xor.reduce(b[..., accs], axis=-1) ^ refv[i]
        return out
    assert np.array_equal(pm.detection_events(b), ref(pm.detectors, pm.det_ref))
    assert np.array_equal(pm.observable_flips(b), ref(pm.observables, pm.obs_ref))


# ---------------------------------------------------------------- lab integration
@pytest.fixture(scope="module")
def lab():
    from aura.lab import LabConfig, PatchConfig, VirtualLab
    lab = VirtualLab(LabConfig(seed=11, patches=[PatchConfig(0, "surface:3")], cosmic_rays=False,
                               rlbp_train_episodes=2000))
    lab.calibrate("oracle")
    return lab


@pytest.mark.parametrize("decoder", ["bposd", "rlbp"])
def test_iterative_decoders_in_batch_mode(lab, decoder):
    r = lab.run(lab.experiment(rounds=3), shots=3000, decoder=decoder, mode="batch")
    p = r.patches[0]
    assert p.missing_corrections == 0
    assert p.logical_errors < 0.5 * p.raw_logical_errors
    c = r.decoder_stats["counters"]
    assert c["bp.shots"] == 3000 and c["bp.trivial"] > 0
    assert "decoder iterations" in r.summary()


def test_rlbp_streams_with_iteration_dependent_latency(lab):
    from aura.core import US
    r = lab.run(lab.experiment(rounds=3, branching=True), shots=40, decoder="rlbp", mode="stream")
    p = r.patches[0]
    assert p.missing_corrections == 0
    assert r.branching["timeouts"] == 0 and r.branching["flag_matches_decision"] == 1.0
    assert r.latency["measurement_to_correction_ps"]["max"] < 5 * US


def test_decode_batch_agrees_with_single_shot_decode(lab):
    ce = lab.compile(lab.experiment(rounds=3))
    bits = np.random.default_rng(8).integers(0, 2, (40, ce.num_acq), dtype=np.uint8)
    pm = ce.manifest.patches[0]
    for name in ("null", "oracle", "matching", "bposd"):
        stub = lab._make_stub(name)
        stub.on_session("config", {"manifest": ce.manifest, "epoch": ce.config_epoch})
        obs, status, nev = stub.decode_batch(pm, bits)
        for i in range(len(bits)):
            o1 = stub.decode(pm, bits[i])
            assert np.array_equal(obs[i], o1[0]) and status[i] == o1[1] and nev[i] == o1[2], name
