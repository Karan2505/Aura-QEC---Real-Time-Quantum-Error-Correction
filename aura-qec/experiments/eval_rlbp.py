"""
Evaluation harness for the RL-BP decoder: logical error rate and latency vs
noise strength, against baselines, with confidence intervals.

Noise model: the device detector error model (DEM) of each code, compiled once
from the virtual lab (profile ``sota``, oracle calibration) and restricted to
the decoded basis, is cached to ``experiments/cache``. A noise scale ``lambda``
multiplies every mechanism probability; the scaled DEM is used both to sample
shots and as the decoders' prior, so every decoder sees the same model.

Decoders per (code, lambda):
  mwpm    PyMatching on the same DEM (graph-like codes only; hyperedges split)
  bp      min-sum BP alone (convergence early exit, max_iter budget)
  bposd   BP + OSD-0 fallback when BP does not converge
  stall   BP + OSD-0, halting heuristic: stop after K iterations without progress
  rlbp    BP + OSD-0, learned halting policy (trained on DEM episodes)

Latency model (cycles of the surface-3 HLS kernel at 300 MHz, co-sim calibrated):
  5 + 13 * iterations + [OSD: 2 * vars + 18 * columns examined]

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/eval_rlbp.py --codes surface:3,surface:5 \
      --scales 0.5,1,2,3 --shots 100000 --plot
  # policy generalisation: train once at lambda=1, test at every lambda
  PYTHONPATH=src python experiments/eval_rlbp.py --codes surface:5 --train-scale 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from aura.codes.library import get_code
from aura.decoding import EXIT_HALTED, MinSumBP, RewardModel, TannerGraph, dem_episodes, osd0, train_q_policy
from aura.host.compiler.dem import DetectorErrorModel
from aura.lab import LabConfig, PatchConfig, VirtualLab

HERE = Path(__file__).resolve().parent
DECODERS = ("mwpm", "bp", "bposd", "stall", "rlbp")

# HLS kernel latency model (see module docstring)
CYC_BASE, CYC_ITER, CYC_OSD_VAR, CYC_OSD_COL = 5, 13, 2, 18


# ------------------------------------------------------------------ noise model
def load_dem(code: str, rounds: int, profile: str, seed: int, cache: Path) -> DetectorErrorModel:
    f = cache / f"{code.replace(':', '-')}_r{rounds}_{profile}_s{seed}.json"
    if f.exists():
        return DetectorErrorModel.from_json_dict(json.loads(f.read_text()))
    t0 = time.perf_counter()
    lab = VirtualLab(LabConfig(seed=seed, profile=profile, patches=[PatchConfig(0, code)], cosmic_rays=False))
    lab.calibrate("oracle")
    ce = lab.compile(lab.experiment(rounds=rounds))
    pm = ce.manifest.patches[0]
    keep = [i for i, b in enumerate(pm.det_basis) if b == pm.basis]
    dem = pm.dem.restrict(keep)
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dem.to_json_dict()))
    print(f"  compiled {code} r={rounds} DEM in {time.perf_counter() - t0:.0f} s -> {f.name}")
    return dem


def scaled(dem: DetectorErrorModel, lam: float) -> DetectorErrorModel:
    p = np.clip(np.asarray(dem.probs) * lam, 0.0, 0.499)
    return DetectorErrorModel(dem.num_detectors, dem.num_observables, p, list(dem.dets), list(dem.obs))


def default_rounds(code: str) -> int:
    d = getattr(get_code(code), "distance", None)
    return int(d) if d else 3


# ------------------------------------------------------------------ baselines
def build_matcher(g: TannerGraph):
    """PyMatching on the decoding graph; None if the code is not graph-like
    (more than 5% of mechanisms flip > 2 detectors, e.g. BB codes)."""
    try:
        import pymatching
    except ImportError:
        return None
    hyper = sum(len(ds) > 2 for ds in g.var_dets)
    if hyper > 0.05 * g.num_vars:
        return None
    cols, obs, w = [], [], []
    for ds, os_, llr in zip(g.var_dets, g.var_obs, g.prior_llr.astype(float)):
        if len(ds) <= 2:
            cols.append(ds); obs.append(os_); w.append(llr)
        else:                                       # split a rare hyperedge into pairs
            ds = list(ds)
            for i in range(0, len(ds), 2):
                cols.append(tuple(ds[i:i + 2])); obs.append(os_ if i == 0 else ()); w.append(llr)
    H = np.zeros((g.num_checks, len(cols)), np.uint8)
    L = np.zeros((g.num_observables, len(cols)), np.uint8)
    for j, (ds, os_) in enumerate(zip(cols, obs)):
        H[list(ds), j] = 1
        if os_:
            L[list(os_), j] = 1
    return pymatching.Matching.from_check_matrix(H, weights=np.array(w), faults_matrix=L,
                                                 use_virtual_boundary_node=True)


class StallPolicy:
    """Heuristic halting: stop once the best unsatisfied count has not improved for ``k`` iterations."""

    def __init__(self, k: int, min_iter: int = 1):
        self.k, self.min_iter = k, min_iter

    def halt(self, f) -> np.ndarray:
        return (f.t >= self.min_iter) & (f.stall >= self.k)


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def summarise(name, wrong, iters, fb, halted, cycles, wall) -> dict:
    n, k = len(wrong), int(wrong.sum())
    lo, hi = wilson(k, n)
    return {"decoder": name, "shots": n, "errors": k, "ler": k / n, "ci_lo": lo, "ci_hi": hi,
            "mean_iter": float(iters.mean()), "p99_iter": float(np.percentile(iters, 99)),
            "max_iter": int(iters.max(initial=0)), "fallback": float(fb.mean()), "halted": float(halted.mean()),
            "mean_cycles": float(cycles.mean()), "p99_cycles": float(np.percentile(cycles, 99)),
            "max_cycles": int(cycles.max(initial=0)), "wall_s": wall}


def run_bp(g, bp, s, o, policy, with_osd: bool, names: tuple[str, ...]) -> list[dict]:
    """Decode once; returns rows for BP alone (names[0], if with_osd also BP+OSD names[-1])."""
    t0 = time.perf_counter()
    res = bp.decode(s, policy=policy)
    t_bp = time.perf_counter() - t0
    iters = res.iterations.astype(np.int64)
    halted = res.exit_reason == EXIT_HALTED
    pred_bp = g.logical(res.errors)
    fb = ~res.converged
    rows = []
    if not with_osd or len(names) > 1:
        cyc = CYC_BASE + CYC_ITER * iters
        rows.append(summarise(names[0], (pred_bp != o).any(axis=1), iters, np.zeros_like(fb), halted, cyc, t_bp))
    if with_osd:
        pred = pred_bp.copy()
        examined = np.zeros(len(s), np.int64)
        t1 = time.perf_counter()
        for i in np.nonzero(fb)[0]:
            pred[i], examined[i], _ = osd0(g, s[i], res.posterior[i])
        t_osd = time.perf_counter() - t1
        cyc = CYC_BASE + CYC_ITER * iters + np.where(fb, CYC_OSD_VAR * g.num_vars + CYC_OSD_COL * examined, 0)
        rows.append(summarise(names[-1], (pred != o).any(axis=1), iters, fb, halted, cyc, t_bp + t_osd))
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default="surface:3,surface:5")
    ap.add_argument("--rounds", type=int, default=0, help="syndrome rounds (default: code distance)")
    ap.add_argument("--scales", default="0.5,1,1.5,2,3", help="noise multipliers lambda")
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--train", type=int, default=20000, help="RL training episodes per point")
    ap.add_argument("--train-scale", type=float, default=0.0,
                    help="train one policy at this lambda and reuse it at every lambda (0: train per point)")
    ap.add_argument("--decoders", default=",".join(DECODERS))
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--stall-k", type=int, default=4)
    ap.add_argument("--profile", default="sota")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "eval_rlbp.csv"))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    decs = [d.strip() for d in a.decoders.split(",") if d.strip()]
    bad = set(decs) - set(DECODERS)
    if bad:
        raise SystemExit(f"unknown decoders: {sorted(bad)}")
    scales = [float(x) for x in a.scales.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rm = RewardModel(iter_cost=1.0, osd_cost=20.0, fail_cost=2000.0)

    rows = []
    for code in [c.strip() for c in a.codes.split(",") if c.strip()]:
        rounds = a.rounds or default_rounds(code)
        dem = load_dem(code, rounds, a.profile, a.seed, HERE / "cache")
        g1 = TannerGraph(dem)
        print(f"{code} r={rounds}: checks {g1.num_checks}, vars {g1.num_vars}, edges {g1.num_edges}, "
              f"observables {g1.num_observables}")
        shared_policy = None
        if "rlbp" in decs and a.train_scale > 0:
            gt = TannerGraph(scaled(dem, a.train_scale))
            shared_policy, _ = train_q_policy(MinSumBP(gt, max_iter=a.max_iter, alpha=a.alpha), dem_episodes(gt),
                                              episodes=a.train, reward=rm, seed=a.seed)
        for lam in scales:
            g = TannerGraph(scaled(dem, lam))
            bp = MinSumBP(g, max_iter=a.max_iter, alpha=a.alpha)
            rng = np.random.default_rng([a.seed, int(lam * 1000)])
            _, s, o = g.sample(a.shots, rng)
            meta = {"code": code, "rounds": rounds, "scale": lam, "checks": g.num_checks, "vars": g.num_vars,
                    "edges": g.num_edges, "p_mean": float(np.mean(g.priors)),
                    "train_scale": a.train_scale or lam}
            point = []
            if "mwpm" in decs:
                mt = build_matcher(g)
                if mt is not None:
                    t0 = time.perf_counter()
                    pred = np.asarray(mt.decode_batch(s), np.uint8)
                    z = np.zeros(len(s), np.int64)
                    point.append(summarise("mwpm", (pred != o).any(axis=1), z, z.astype(bool), z.astype(bool),
                                           z, time.perf_counter() - t0))
            if "bp" in decs or "bposd" in decs:
                names = tuple(n for n in ("bp", "bposd") if n in decs)
                point += run_bp(g, bp, s, o, None, "bposd" in decs, names)
            if "stall" in decs:
                point += run_bp(g, bp, s, o, StallPolicy(a.stall_k), True, ("stall",))
            if "rlbp" in decs:
                pol = shared_policy
                if pol is None:
                    pol, _ = train_q_policy(bp, dem_episodes(g), episodes=a.train, reward=rm, seed=a.seed)
                point += run_bp(g, bp, s, o, pol, True, ("rlbp",))
            for r in point:
                r.update(meta)
                print(f"  lambda={lam:<5g} {r['decoder']:<6} LER {r['ler']:.2e} [{r['ci_lo']:.2e},{r['ci_hi']:.2e}]"
                      f"  iters mean {r['mean_iter']:.2f} p99 {r['p99_iter']:.0f} max {r['max_iter']:>2}"
                      f"  fallback {r['fallback']:.3f}  cycles mean {r['mean_cycles']:.0f} max {r['max_cycles']}"
                      f"  ({r['wall_s']:.1f} s)")
            rows += point

    cols = ["code", "rounds", "scale", "train_scale", "p_mean", "checks", "vars", "edges", "decoder", "shots",
            "errors", "ler", "ci_lo", "ci_hi", "mean_iter", "p99_iter", "max_iter", "fallback", "halted",
            "mean_cycles", "p99_cycles", "max_cycles", "wall_s"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    if a.plot:
        plot(rows, out.with_suffix(".png"))


def plot(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = list(dict.fromkeys(r["code"] for r in rows))
    decs = list(dict.fromkeys(r["decoder"] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for code in codes:
        for dec in decs:
            rs = [r for r in rows if r["code"] == code and r["decoder"] == dec]
            if not rs:
                continue
            x = [r["p_mean"] for r in rs]
            lbl = f"{code} {dec}"
            y = np.array([r["ler"] for r in rs])
            err = np.array([[r["ler"] - r["ci_lo"] for r in rs], [r["ci_hi"] - r["ler"] for r in rs]])
            ax[0].errorbar(x, y, yerr=err, marker="o", ms=3, capsize=2, label=lbl)
            if dec != "mwpm":
                ax[1].plot(x, [r["mean_iter"] for r in rs], marker="o", ms=3, label=lbl)
                ax[2].plot(x, [r["max_cycles"] for r in rs], marker="o", ms=3, label=lbl)
    ax[0].set(xscale="log", yscale="log", xlabel="mean mechanism probability", ylabel="logical error rate")
    ax[1].set(xscale="log", xlabel="mean mechanism probability", ylabel="mean BP iterations")
    ax[2].set(xscale="log", yscale="log", xlabel="mean mechanism probability", ylabel="max cycles (model)")
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
