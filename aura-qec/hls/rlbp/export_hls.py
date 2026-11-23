"""
Export an RL-BP decoder instance from the Python golden model to Vitis HLS.

Produces, in --out (default: this directory):

* ``rlbp_graph.h`` -- compile-time Tanner graph (CSR by check and by variable),
  prior LLRs, observable masks and the learned 1320-entry HALT table;
* ``vectors.txt``  -- golden test vectors: per shot the syndrome, the true
  logical flip, and what the Python model (``aura.decoding``) produced:
  exit reason, BP iterations, BP-only logical prediction, and the final
  prediction after the OSD-0 fallback.

The C++ kernel is a line-by-line port of ``MinSumBP`` + ``QHaltingPolicy``;
in float mode it must match the Python model bit for bit on every shot.

Usage (from aura-qec/):
    PYTHONPATH=src python hls/rlbp/export_hls.py --code surface:3 --rounds 3
    PYTHONPATH=src python hls/rlbp/export_hls.py --code surface:5 --rounds 5 --shots 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aura.decoding import (EXIT_CONVERGED, EXIT_HALTED, EXIT_MAX_ITER, EXIT_TRIVIAL, MinSumBP, QHaltingPolicy,
                           RewardModel, TannerGraph, dem_episodes, finish_with_fallback, shot_episodes,
                           train_q_policy)
from aura.decoding.policy import NUM_STATES
from aura.host.compiler.dem import DetectorErrorModel
from aura.lab import LabConfig, PatchConfig, VirtualLab


def device_shots(lab, ce, pm, keep, n):
    """Detection events + observed logical flips from the virtual device (as in examples/07)."""
    lab.qpu.upload_settings(ce.settings)
    lab.qpu.load_program(ce.qpu_program)
    iq = lab.qpu.run_batch(n, t_start_ps=lab.sim.now)["iq"]
    bits = np.zeros((len(iq), ce.num_acq), np.uint8)
    for a, q in enumerate(ce.lowered.acq_qubit):
        bits[:, a] = ce.discriminators[q].classify(iq[:, a])[0]
    return pm.detection_events(bits)[:, keep].astype(np.uint8), pm.observable_flips(bits).astype(np.uint8)


def c_array(ctype, name, values, per_line=16):
    vals = list(values)
    body = ",\n    ".join(", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))
    return f"static const {ctype} {name}[{max(len(vals), 1)}] = {{\n    {body if vals else '0'}\n}};\n"


def f32(x) -> str:
    """Exact float32 literal (shortest round-trip repr)."""
    s = repr(float(np.float32(x)))
    if s in ("inf", "-inf", "nan"):
        raise ValueError("non-finite prior LLR")
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default="surface:3")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--profile", default="sota")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--posterior-avg", type=int, choices=[1, 2], default=2,
                    help="OSD input: 1 = final-iteration posterior, 2 = mean of the last two iterations")
    ap.add_argument("--source", choices=["device", "dem"], default="device",
                    help="where training episodes and test vectors come from")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="multiply every DEM mechanism probability (--source dem only)")
    ap.add_argument("--train", type=int, default=8000, help="policy training episodes")
    ap.add_argument("--shots", type=int, default=1000, help="test vectors to write")
    ap.add_argument("--policy", default="", help="load a saved policy (aura train-decoder --out ...) instead")
    ap.add_argument("--fail-cost", type=float, default=2000.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- build the decoding problem exactly as the RLBP decoder stub does
    lab = VirtualLab(LabConfig(seed=a.seed, profile=a.profile, patches=[PatchConfig(0, a.code)], cosmic_rays=False))
    lab.calibrate("oracle")
    ce = lab.compile(lab.experiment(rounds=a.rounds))
    pm = ce.manifest.patches[0]
    keep = [i for i, b in enumerate(pm.det_basis) if b == pm.basis]
    dem = pm.dem.restrict(keep)
    if a.noise_scale != 1.0:
        if a.source != "dem":
            raise SystemExit("--noise-scale needs --source dem")
        dem = DetectorErrorModel(dem.num_detectors, dem.num_observables,
                                 np.clip(np.asarray(dem.probs) * a.noise_scale, 0.0, 0.499),
                                 list(dem.dets), list(dem.obs))
    g = TannerGraph(dem)
    bp = MinSumBP(g, max_iter=a.max_iter, alpha=a.alpha, posterior_avg=a.posterior_avg)
    if g.num_observables > 32:
        raise SystemExit("kernel packs observables into 32 bits; this code has more")
    print(f"{a.code} r={a.rounds}: checks {g.num_checks}, vars {g.num_vars}, edges {g.num_edges}, "
          f"observables {g.num_observables}")

    # ---- episodes / test vectors
    rng = np.random.default_rng(a.seed)
    if a.source == "device":
        det, obs = device_shots(lab, ce, pm, keep, a.train + a.shots)
        train_eps = shot_episodes(det[:a.train], obs[:a.train])
        ts, to = det[a.train:], obs[a.train:]
    else:
        train_eps = dem_episodes(g)
        _, ts, to = g.sample(a.shots, rng)

    # ---- halting policy
    rm = RewardModel(iter_cost=1.0, osd_cost=20.0, fail_cost=a.fail_cost)
    if a.policy:
        policy = QHaltingPolicy.load(a.policy)
    else:
        policy, rep = train_q_policy(bp, train_eps, episodes=a.train, reward=rm, seed=a.seed)
        print(rep.summary())
    halt = policy.halt_table.astype(np.uint8)
    assert len(halt) == NUM_STATES

    # ---- golden run of the Python model on the test vectors
    res = bp.decode(ts, policy=policy)
    pred, fb = finish_with_fallback(g, ts, res)
    bp_obs = g.logical(res.errors)
    wrong = (pred != to).any(axis=1)
    it = res.iterations
    print(f"golden: LER {wrong.mean():.4f}  mean iters {it.mean():.2f}  p99 {np.percentile(it, 99):.0f}  "
          f"fallback {fb.mean():.3f}  halted {(res.exit_reason == EXIT_HALTED).mean():.3f}")

    # ---- header
    deg_c = np.bincount(g.e_chk, minlength=g.num_checks)
    chk_ptr = np.concatenate([[0], np.cumsum(deg_c)])
    var_edges = [[] for _ in range(g.num_vars)]
    for e, v in enumerate(g.e_var):
        var_edges[v].append(e)                          # ascending edge order = scipy's summation order
    var_ptr = np.concatenate([[0], np.cumsum([len(x) for x in var_edges])])
    var_edge = [e for x in var_edges for e in x]
    var_obs = [sum(1 << o for o in os_) for os_ in g.var_obs]
    nw = (g.num_checks + 31) // 32                      # check-matrix column of each variable, 32 checks/word
    var_col = []
    for x in var_edges:
        m = sum(1 << int(g.e_chk[e]) for e in x)
        var_col += [(m >> (32 * w)) & 0xFFFFFFFF for w in range(nw)]

    h = [f"// Generated by hls/rlbp/export_hls.py -- do not edit.\n"
         f"// code {a.code}, rounds {a.rounds}, profile {a.profile}, seed {a.seed}, source {a.source}, "
         f"noise scale {a.noise_scale:g}, posterior avg {a.posterior_avg}\n"
         f"// halting policy: {int(halt.sum())} HALT states of {NUM_STATES}\n"
         "#ifndef RLBP_GRAPH_H\n#define RLBP_GRAPH_H\n\n",
         f"#define RLBP_NUM_CHECKS   {g.num_checks}\n",
         f"#define RLBP_NUM_VARS     {g.num_vars}\n",
         f"#define RLBP_NUM_EDGES    {g.num_edges}\n",
         f"#define RLBP_NUM_OBS      {g.num_observables}\n",
         f"#define RLBP_MAX_ITER     {a.max_iter}\n",
         f"#define RLBP_MIN_ITER     {policy.min_iter}\n",
         f"#define RLBP_ALPHA        {f32(a.alpha)}\n",
         f"#define RLBP_POST_AVG     {a.posterior_avg}   // OSD input: 1 final posterior, 2 mean of last two\n",
         f"#define RLBP_MAX_CHK_DEG  {int(deg_c.max())}\n",
         f"#define RLBP_MAX_VAR_DEG  {max(len(x) for x in var_edges)}\n",
         f"#define RLBP_NUM_STATES   {NUM_STATES}\n\n",
         "// edges sorted by check: edges of check c are CHK_PTR[c] .. CHK_PTR[c+1]-1\n",
         c_array("unsigned short", "CHK_PTR", map(str, chk_ptr)),
         c_array("unsigned short", "E_VAR", map(str, g.e_var)),
         c_array("unsigned short", "E_CHK", map(str, g.e_chk)),
         "// edges of variable v (ascending): VAR_EDGE[VAR_PTR[v] .. VAR_PTR[v+1]-1]\n",
         c_array("unsigned short", "VAR_PTR", map(str, var_ptr)),
         c_array("unsigned short", "VAR_EDGE", map(str, var_edge)),
         "// prior LLR log((1-p)/p) per variable (float32, exact). An initialiser list, so the kernel\n"
         "// can store it directly as llr_t (float or ap_fixed) without a runtime conversion.\n",
         "#define RLBP_PRIOR_INIT { \\\n    "
         + ", \\\n    ".join(", ".join(map(f32, g.prior_llr[i:i + 6])) for i in range(0, g.num_vars, 6))
         + " }\n",
         "// observables flipped by each variable (bit k = observable k)\n",
         c_array("unsigned int", "VAR_OBS", map(str, var_obs)),
         f"// check-matrix column of variable v (bit c = check c): words VAR_COL[v*{nw} .. v*{nw}+{nw - 1}]\n"
         f"#define RLBP_COL_WORDS    {nw}\n",
         c_array("unsigned int", "VAR_COL", (f"{x}u" for x in var_col), 8),
         "// learned policy: 1 = HALT (hand the shot to OSD), 0 = CONTINUE\n",
         c_array("unsigned char", "HALT_TABLE", map(str, halt), 40),
         "\n#endif\n"]
    (out / "rlbp_graph.h").write_text("".join(h))

    # ---- vectors
    bits = lambda row: "".join(str(int(x)) for x in row)
    lines = [f"# shots checks observables -- then per shot: syndrome true_obs exit iters bp_obs final_obs",
             f"{len(ts)} {g.num_checks} {g.num_observables}"]
    for i in range(len(ts)):
        lines.append(f"{bits(ts[i])} {bits(to[i])} {int(res.exit_reason[i])} {int(it[i])} "
                     f"{bits(bp_obs[i])} {bits(pred[i])}")
    (out / "vectors.txt").write_text("\n".join(lines) + "\n")
    policy.save(out / "policy.json")
    print(f"wrote {out / 'rlbp_graph.h'}, {out / 'vectors.txt'}, {out / 'policy.json'}")


if __name__ == "__main__":
    main()
