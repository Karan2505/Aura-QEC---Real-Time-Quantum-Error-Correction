# RL-BP decoder on FPGA (Vitis HLS)

A synthesisable C++ port of the Layer-3 reference decoder in `src/aura/decoding/`:
min-sum BP with a zero-syndrome fast path, convergence early exit, and the
learned RL halting policy. The Python model is the golden reference, and the
float build reproduces it **bit for bit**.

```
 Python (golden model)                         Vitis HLS
 ─────────────────────                         ─────────
 VirtualLab → compiler → DEM ──┐
 TannerGraph + MinSumBP        ├─ export_hls.py ─► rlbp_graph.h  (graph, priors, HALT table)
 train_q_policy (FQI) ─────────┤                   vectors.txt   (syndromes + golden outputs)
 device / DEM shots ───────────┘                        │
                                                        ▼
                                  rlbp_decoder.cpp  (kernel: BP + early exit + RL halt)
                                  tb_rlbp.cpp       (replays vectors, OSD-0 fallback in C++,
                                                     checks exit/iterations/logical = golden)
```

| File | Role |
|---|---|
| `export_hls.py` | builds the decoding problem the way `RLBPDecoderStub` does; trains or loads the policy; writes the header and the vectors |
| `rlbp_decoder.h/.cpp` | **top function `rlbp_decode`**, one shot per call |
| `tb_rlbp.cpp` | C and RTL co-simulation testbench (returns 0 on PASS) |
| `run_hls.tcl` | csim → csynth → cosim (→ IP export) |
| `rlbp_graph.h`, `vectors.txt`, `policy.json` | generated: surface-3, 3 rounds, device shots, 1000 vectors |

## 1. (Optional) regenerate for another code

```bash
cd aura-qec
PYTHONPATH=src python hls/rlbp/export_hls.py --code surface:3 --rounds 3            # default, already generated
PYTHONPATH=src python hls/rlbp/export_hls.py --code surface:5 --rounds 5 --shots 2000
PYTHONPATH=src python hls/rlbp/export_hls.py --code surface:3 --policy rlbp.json    # policy from `aura train-decoder`
```

Array sizes are compile-time constants, so the header and the vectors must
always be regenerated together.

## 2. Quick C check without Vitis (any g++)

```bash
cd aura-qec/hls/rlbp
g++ -std=c++14 -O2 -ffp-contract=off -Wno-unknown-pragmas -Wno-unused-label rlbp_decoder.cpp tb_rlbp.cpp -o tb
./tb vectors.txt        # -> PASS (float: bit-exact against the Python model)
```

## 3. Vitis HLS

```bash
cd aura-qec/hls/rlbp
vitis_hls -f run_hls.tcl                         # 2020.2 – 2024.x
vitis-run --mode hls --tcl run_hls.tcl           # 2024.2+ unified flow
```

To use the GUI instead, create a project with top function `rlbp_decode`.
Add `rlbp_decoder.cpp` as a source, add `tb_rlbp.cpp` and `vectors.txt` as testbench files, and set
the C-sim / co-sim argument to `vectors.txt`. Then run C Simulation, C Synthesis and
Co-Simulation.

Options (environment variables):

* `RLBP_MODE=fixed` switches LLRs to `ap_fixed<10,6>`; override the width with `RLBP_LLR="16 8"`.
  Use fixed for synthesis. Float is the bit-exact reference for csim; in fixed mode the
  pass criterion is "LER within 0.5 % of golden".
* `RLBP_PART` defaults to the Kria K26 (`xck26-sfvc784-2LV-c`).
* `RLBP_CLK` defaults to 3.333 ns, the 300 MHz AURA core clock.
* `RLBP_STEPS` defaults to `"csim csynth cosim"`. Add `export` to package the IP, or `impl`
  for Vivado synthesis and place & route of the kernel alone (real LUT/FF and timing;
  about 15 min, fits in 7 GB RAM with the default `RLBP_IMPL_REPORTS=0`).
* `RLBP_AXI=1` puts every port and the block control on one AXI4-Lite slave, for a Zynq
  block design.

Each mode gets its own solution (`sol_float`, `sol_fixed`). The project is not reset between runs.

Look at (replace `sol_fixed` with `sol_float` for float runs):

* `rlbp_prj/sol_fixed/csim/report/rlbp_decode_csim.log`: the PASS line and LER;
* `rlbp_prj/sol_fixed/syn/report/rlbp_decode_csynth.rpt`: latency per iteration, II, and resources;
* `rlbp_prj/sol_fixed/sim/report/rlbp_decode_cosim.rpt`: min/avg/max latency over the real shot mix;
* `rlbp_prj/sol_fixed/impl/report/verilog/rlbp_decode_export.rpt`: Vivado resources and timing (`impl` step).

## Kernel interface

```cpp
void rlbp_decode(syn_t syndrome,              // ap_uint<NUM_CHECKS>, bit c = detection event on check c
                 vbits_t *hard,               // ap_uint<NUM_VARS>, BP hard decision at exit
                 unsigned int *obs_bp,        // logical prediction of the BP hard decision
                 unsigned int *obs,           // final logical prediction (BP, or OSD-0 fallback)
                 unsigned char *iterations,
                 unsigned char *exit_reason); // 0 trivial, 1 converged, 2 RL-halted, 3 max-iter
```

The kernel is the complete decoder. When `exit_reason` is 2 or 3 it runs OSD-0
on chip, as a port of `osd.py`:
1. Rank the posteriors with 55 parallel comparators (V cycles).
2. Insert the columns into a GF(2) basis in that order, stopping as soon as the syndrome is solved.

The column bitmasks come from `VAR_COL` in `rlbp_graph.h`.

## Status

* **Bit-exact:** the float kernel, including the on-chip OSD-0, matches Python on surface-3 (device
  shots, 1000 vectors). The serial version also matched on surface-5 (DEM shots, 3000 vectors).
  The comparison covers exit reason, iterations, BP logical and final logical.
* **Fully parallel node array:** the message arrays are completely partitioned, and every check,
  variable and syndrome loop is unrolled over the compile-time graph. A BP iteration is a 13-cycle
  pipeline. I/O uses packed words, so the syndrome loads in one cycle.

Results on surface-3 r=3 (16 checks, 55 vars, 94 edges), KV260, 300 MHz, Vitis 2025.1:

| | serial float (first version) | parallel fixed 10-bit + on-chip OSD |
|---|---|---|
| cycles / BP iteration | 3938 | **13** |
| co-sim min / avg / max, 1000 shots | 119 / 2160 / 43,870 cycles (BP only) | **5 / 20 / 379 cycles** (incl. OSD) |
| time at 300 MHz (avg / max) | 7.2 us / 146 us | **0.07 us / 1.26 us** |
| worst case (30 iters + full OSD) | 118,362 cycles | 1526 cycles (~5.1 us) |
| clock (3.333 ns target) | 3.64 ns HLS estimate (missed) | **3.21 ns post-route (met, Fmax ~311 MHz)** |
| LUT / FF / DSP / BRAM | 4.3k / 2.9k / 5 / 6 (HLS estimate) | **11.2k (9.6%) / 8.6k (3.7%) / 2 / 1** (post-route, 2227 CLBs) |
| LER (golden 2.20%) | 2.20% (bit-exact) | 2.30% |

Notes:

* The HLS LUT estimate for the parallel design is 41k; Vivado synthesis gives 11.3k, so trust
  the `impl` numbers.
* LLR widths from 16 down to 8 bits gave the same LER (2.3–2.4%) on these vectors.
* OSD-0 accounts for about 1100 of the worst-case cycles, but runs on only about 4% of shots.
* Timing after place & route: 3.21 ns, against 2.80 ns post-synthesis, leaving 0.12 ns of slack.
  The routing margin is thin.
* `impl` runs with `vivado_report_level 0` by default. Level 2's parallel report generation
  ran out of memory (`bad allocation`) on a 7 GB-RAM PC. Set `RLBP_IMPL_REPORTS=2` on a bigger
  machine to get the full report set.

## Next steps

1. Surface-5 is about 5x larger. If it does not fit fully parallel, use partial unrolling
   (process checks in groups).
2. Speed up the OSD walk (about 18 cycles per column now). Options: pipeline the GF(2)
   reduction, or stop the walk after NUM_CHECKS independent pivots.
3. AXI-Stream syndrome input straight from AURA-Link. `RLBP_AXI` covers AXI4-Lite control but
   has not been synthesised yet.
