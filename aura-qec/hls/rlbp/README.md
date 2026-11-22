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

Options (environment variables): `RLBP_MODE=fixed` switches LLRs to
`ap_fixed<10,6>` (override with `RLBP_LLR="16 8"`). Use fixed for synthesis;
float is the bit-exact reference for csim. The pass criterion then becomes "LER within 0.5 % of golden",
since bit-exactness no longer applies. `RLBP_PART` defaults to the Kria K26
(`xck26-sfvc784-2LV-c`). `RLBP_CLK` defaults to 3.333 ns, the 300 MHz AURA core
clock. `RLBP_STEPS` defaults to `"csim csynth cosim"`; add `export` to package the IP.

Look at:

* `rlbp_prj/sol_float/csim/report/rlbp_decode_csim.log`: the PASS line and LER;
* `rlbp_prj/sol_float/syn/report/rlbp_decode_csynth.rpt`: latency per iteration, II, and resources;
* `rlbp_prj/sol_float/sim/report/rlbp_decode_cosim.rpt`: min/avg/max latency over the real shot mix.

The early exit only shows up here, because most shots are trivial or converge in 1–3 iterations.

## Kernel interface

```cpp
void rlbp_decode(const unsigned char syndrome[NUM_CHECKS],   // detection events (basis detectors)
                 llr_t posterior[NUM_VARS],                  // LLRs at exit -> OSD input
                 unsigned char hard[NUM_VARS],               // BP hard decision
                 unsigned int *obs_bp,                       // logical prediction (bit k = observable k)
                 unsigned char *iterations,
                 unsigned char *exit_reason);                // 0 trivial, 1 converged, 2 RL-halted, 3 max-iter
```

OSD-0 is required when `exit_reason` is 2 or 3. The kernel does not run it yet;
the testbench does, in C++, as a port of `osd.py`.

## Status and next steps

* **Done:** float kernel bit-exact against Python on surface-3 (device shots,
  1000 vectors) and surface-5 (DEM shots, 3000 vectors). The comparison
  covers exit reason, iterations, BP logical and post-OSD logical.
* **Fully parallel node array.** The message arrays are completely
  partitioned and every check, variable and syndrome loop is unrolled over the
  compile-time graph, so a BP iteration is a short pipeline instead of an
  edge-by-edge walk. The float build stays bit-exact; **synthesise with
  `RLBP_MODE=fixed`** (default `ap_fixed<10,6,AP_TRN,AP_SAT>`, set by `RLBP_LLR="W I"`).

  Results on surface-3 r=3 (16 checks, 55 vars, 94 edges), KV260, 300 MHz, Vitis HLS 2025.1:

  | | serial float (previous) | parallel fixed 10-bit |
  |---|---|---|
  | cycles / BP iteration | 3938 | **13** |
  | worst case (30 iters) | 118,362 cycles (~395 us) | **476 cycles (~1.6 us)** |
  | co-sim avg / max over 1000 shots | 2160 / 43,870 cycles | **86 / 249 cycles** (0.29 / 0.83 us) |
  | estimated clock | 3.64 ns (timing missed) | 2.42 ns (met) |
  | LUT / FF | 4.3k / 2.9k | 33.6k (28%) / 7.9k (3%) |
  | LER (golden 2.20%) | 2.20% (bit-exact) | 2.30% |

  LLR widths from 16 down to 8 bits gave the same LER (2.3–2.4%) on these
  vectors. Of the ~86 average cycles, ~75 are the serial `syndrome` load and
  `posterior` write over `ap_memory`.

  Next steps:
  1. Check the real LUT count after Vivado implementation (HLS estimates are pessimistic).
     Surface-5 is ~5x larger; if it does not fit fully parallel, use partial unrolling
     (process checks in groups) instead.
  2. Wider I/O: an `ap_uint<NUM_CHECKS>` syndrome word and packed outputs remove the
     ~75 cycles of array transfer.
  3. Move OSD-0 into hardware (bitset Gaussian elimination with `ap_uint<NUM_CHECKS>`)
     or run it on the Zynq PS.
  4. AXI interfaces (`s_axilite` control, `axis` syndrome stream from AURA-Link) for
     system integration.
