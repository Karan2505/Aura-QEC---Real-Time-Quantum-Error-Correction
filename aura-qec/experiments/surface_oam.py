"""
OAM-BP on surface codes: does the memory table learned on BB-72 transfer?
Compares MWPM, BP+OSD-0 (averaged OSD input) and OAM-BP+OSD-0 (same, plus the
learned per-variable memory table) on the same shots, with the hardware
settings of the HLS kernel (alpha 0.75, 30 iterations, OSD-0 fallback).

Usage (from aura-qec/):
  PYTHONPATH=src python experiments/surface_oam.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval_rlbp import HERE, build_matcher, default_rounds, load_dem, run_bp, scaled, summarise
from aura.decoding import MinSumBP, TannerGraph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default="surface:3,surface:5")
    ap.add_argument("--scales", default="0.5,1,2")
    ap.add_argument("--shots", default="100000,50000", help="per code")
    ap.add_argument("--gamma-file", default=str(HERE / "results" / "oam_gamma_bb72.json"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results" / "surface_oam.csv"))
    a = ap.parse_args()
    table = np.array(json.loads(Path(a.gamma_file).read_text())["gamma"], np.float32)
    rows = []
    for code, shots in zip(a.codes.split(","), [int(x) for x in a.shots.split(",")]):
        rounds = default_rounds(code)
        dem = load_dem(code, rounds, "sota", a.seed, HERE / "cache")
        for lam in [float(x) for x in a.scales.split(",")]:
            g = TannerGraph(scaled(dem, lam))
            _, s, o = g.sample(shots, np.random.default_rng([a.seed, 7, int(lam * 1000)]))
            out = []
            mt = build_matcher(g)
            if mt is not None:
                wrong = (mt.decode_batch(s) != o).any(axis=1)
                z = np.zeros(len(s), np.int64)
                out.append(summarise("mwpm", wrong, z, z.astype(bool), z.astype(bool), z, 0.0))
            for name, mem in (("bposd_avg2", None), ("oam_osd_avg2", table)):
                bp = MinSumBP(g, max_iter=30, alpha=0.75, posterior_avg=2, memory_table=mem)
                out += run_bp(g, bp, s, o, None, True, (name,))
            for r in out:
                r.update(code=code, scale=lam)
                print(f"{code} lambda={lam:<4g} {r['decoder']:<14} LER {r['ler']:.2e} [{r['ci_lo']:.2e},"
                      f"{r['ci_hi']:.2e}]  iters {r.get('mean_iter', 0):5.2f}  fallback {r.get('fallback', 0):.4f}",
                      flush=True)
            rows += out
    keys = ["code", "scale"] + [k for k in rows[0] if k not in ("code", "scale")]
    with Path(a.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
