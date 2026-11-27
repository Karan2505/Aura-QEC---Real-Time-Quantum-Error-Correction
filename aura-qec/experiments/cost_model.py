"""
Latency / area cost model for BB-72 decoders on FPGA, combined with the
measured logical error rates (experiments/results/relay_*.csv).

All hardware constants are explicit assumptions, calibrated where possible on
the surface-3 RL-BP kernel measured after Vivado place & route on the KV260
(10-bit LLRs, 300 MHz): 11.7k LUT in total, of which ~77% is the BP datapath
(HLS breakdown) for 94 edges -> ~96 LUT per edge lane; 13 cycles per BP
iteration fully parallel; OSD-0 at 18 cycles per examined column for 16 checks.

Model, P = edge lanes (P = E is fully parallel):
  BP cycles / iteration     2 * ceil(E / P) + 13      (check then variable phase stream E edges)
  BP LUT                    96 * P (+15% for the Relay-BP memory term and gamma ROM)
  OSD cycles                2 * V + cols * t_col
      t_col naive      = 18 * ceil(C / 16)   (our bit-serial-chain OSD scaled to C checks)
      t_col optimistic = 18                  (C-bit wide single-pass reduction per column)
  latency = cycles / 300 MHz

Decoders (worst case = budget iterations + worst observed OSD):
  BP+OSD (avg2)       30 iterations + OSD on every non-converged shot
  Relay-BP budget B   B iterations, no OSD
  Hybrid  budget B    B iterations + OSD on shots Relay-BP did not solve

Usage (from aura-qec/):  PYTHONPATH=src python experiments/cost_model.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
E, V, C = 8244, 2232, 252                  # BB-72, 6 rounds, one basis
F_MHZ = 300.0
LUT_PER_LANE, RELAY_OVERHEAD, ITER_DEPTH = 96, 1.15, 13
DEVICES = [("KV260 (XCK26)", 117_120), ("ZCU102 (XCZU9EG)", 274_080), ("VCU118 (XCVU9P)", 1_182_240),
           ("XCVU19P", 4_085_760)]


def iter_cycles(p: int) -> int:
    return 13 if p >= E else 2 * math.ceil(E / p) + ITER_DEPTH


def osd_cycles(cols: float, t_col: float) -> float:
    return 2 * V + cols * t_col


def device_for(lut: float) -> str:
    for name, cap in DEVICES:
        if lut <= 0.8 * cap:                   # leave 20% for routing / the rest of the system
            return name
    return "none listed"


def load(name):
    f = HERE / "results" / name
    return list(csv.DictReader(f.open())) if f.exists() else []


def main():
    rows = load("relay_hybrid.csv") + [r for r in load("relay_budget.csv") if r["budget"] == "200"
                                       and r["decoder"].startswith("relay1_t30-20")]
    if not rows:
        raise SystemExit("run experiments/relay_bp.py first")
    t_naive, t_opt = 18 * math.ceil(C / 16), 18
    print(f"BB-72: E={E} edges, V={V} vars, C={C} checks; OSD t_col naive {t_naive}, optimistic {t_opt}\n")
    print(f"{'lanes P':>8} {'BP LUT':>9} {'device':>18} {'cyc/iter':>8}")
    for p in (E, 4096, 2048, 1024, 512, 256):
        lut = LUT_PER_LANE * min(p, E) * RELAY_OVERHEAD
        print(f"{p:>8} {lut:>9.0f} {device_for(lut):>18} {iter_cycles(p):>8}")

    out = []
    for lam in sorted({r["scale"] for r in rows}, key=float):
        print(f"\nlambda = {lam}: worst-case latency (us) per decoder, by lanes P "
              f"[naive OSD / optimistic OSD]")
        hdr = f"{'decoder':<26} {'LER':>9} {'max it':>6} {'OSD max cols':>12}"
        print(hdr + "".join(f"{'P=' + str(p):>17}" for p in (E, 2048, 512)))
        seen = set()
        for r in rows:
            if r["scale"] != lam or r["decoder"] in seen or r["decoder"] == "bposd_avg1":
                continue
            seen.add(r["decoder"])
            it = float(r["max_iter"])
            cols = float(r.get("osd_cols_max") or 0)
            has_osd = r["decoder"].startswith("bposd") or r["decoder"].endswith("+osd")
            cells = []
            for p in (E, 2048, 512):
                bp = it * iter_cycles(p)
                lo = bp + (osd_cycles(cols, t_opt) if has_osd and cols else 0)
                hi = bp + (osd_cycles(cols, t_naive) if has_osd and cols else 0)
                cells.append(f"{hi / F_MHZ:>8.1f}/{lo / F_MHZ:<8.1f}")
                out.append({"scale": lam, "decoder": r["decoder"], "ler": r["ler"], "lanes": p,
                            "worst_us_naive_osd": hi / F_MHZ, "worst_us_opt_osd": lo / F_MHZ,
                            "bp_lut": LUT_PER_LANE * min(p, E) * RELAY_OVERHEAD})
            print(f"{r['decoder']:<26} {float(r['ler']):>9.2e} {it:>6.0f} {cols:>12.0f}" + "".join(cells))
    with (HERE / "results" / "cost_model.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {HERE / 'results' / 'cost_model.csv'}")


if __name__ == "__main__":
    main()
