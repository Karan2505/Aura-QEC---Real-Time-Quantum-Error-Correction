# AURA-QEC — Layer 0 + Layer 1

**Layer 0:** virtual superconducting QPU, real-time control system, and control-system host (QEC compiler, calibration, experiment runner, operations console).
**Layer 1:** the AURA-Link ingress transport gateway.

This is the first two layers of the AURA-QEC real-time error-correction stack
(see the AURA-QEC system architecture). It is a **pre-silicon digital twin and
golden reference model**. Every block runs on a deterministic virtual clock with
picosecond resolution, and the transport also runs over real UDP sockets.
Nanosecond figures are modelled values, not wall-clock measurements.

```
virtual QPU ⇄ control system ⇄ AURA-Link (PHY → parser → timestamp → sequence → FIFO) ⇄ [Layer 2: RPU] → decoder
      ▲              ▲
 calibration     QEC compiler  ←  calibration store         operations console / CLI
```

## Install and verify

```bash
pip install -e ".[test]"        # runtime: numpy, scipy.  tests: pytest, stim, pymatching
aura selftest                   # ~2 s end-to-end sanity check
python -m pytest tests          # 120 tests, ~80 s
```

Python ≥ 3.11.

## Five-minute tour

```bash
aura info                                            # codes (surface, repetition, BB 72/90/108/144/288), profiles
aura compile --code surface:5 --rounds 5 --listing 1 --stim s5.stim
aura run --code surface:3 --rounds 3 --shots 500     # calibrate from scratch, then stream through the full stack
aura run --code surface:3 --branching --calibration oracle   # feed-forward (logical branching) response time
aura link-loopback --shots 200                       # AURA-Link over real UDP sockets
aura console --calibration oracle                    # dashboard on http://127.0.0.1:8765/
python examples/03_link_fault_injection.py           # BER, bursts, fibre pull -> erasures + auto-recovery
aura run --code surface:5 --rounds 5 --decoder rlbp --mode batch --shots 5000 --calibration oracle
aura train-decoder --code surface:5 --rounds 5 --source device --out rlbp.json   # RL-BP policy from device shots
python examples/07_rl_decoder.py                     # RL-BP vs BP+OSD vs MWPM: iterations and LER
```

## Iterative decoding and RL-BP

`--decoder bposd` is min-sum belief propagation on the detector error model with
an OSD-0 fallback (any CSS code, including BB codes, which matching cannot
decode). It never runs more BP iterations than needed:

* a shot with no detection events costs **0 iterations**;
* a shot leaves BP **at the first iteration whose hard decision explains its
  syndrome**, so a shot that converges at iteration 3 costs 3, not `bp_max_iter`.

`--decoder rlbp` adds a **reinforcement-learned halting policy**. After every
iteration it looks at a shot's convergence state (iteration, residual syndrome,
progress, stall) and decides whether to continue or hand the shot to OSD now.
The policy learns to recognise runs that will not converge and to cut them short.
Its reward is the modelled decoder latency (iterations × `bp_iter_ns` + fallback ×
`osd_ns`) plus a large penalty per logical error. It is trained by fitted
Q-iteration, on shots from the DEM or from the device itself
(`aura train-decoder`).

On device shots (12k training / 12k test, `sota`):

| code | decoder | LER | mean BP iterations | p99 |
|---|---|---|---|---|
| surface-5, r=5 | BP+OSD (budget 30) | 1.06 % | 5.56 | 30 |
|  | **RL-BP** | 1.08 % | **2.21 (−60 %)** | **12** |
|  | MWPM (reference) | 1.08 % | – | – |
| BB-72, r=6 | BP+OSD (budget 30) | 35.1 % | 21.9 | 30 |
|  | **RL-BP** | **32.2 %** | **14.0 (−36 %)** | 30 |

See `docs/ARCHITECTURE.md §3a` for the details and the honest comparison with a
hand-tuned fixed iteration cap.

```python
from aura.lab import VirtualLab, LabConfig, PatchConfig
lab = VirtualLab(LabConfig(profile="sota", patches=[PatchConfig(0, "surface:3")]))
lab.calibrate("full")                                   # discovers the hidden device parameters
res = lab.run(lab.experiment(rounds=3), shots=300, decoder="matching", mode="stream")
print(res.summary())
```

## What is inside

| Package | Contents |
|---|---|
| `aura.core` | discrete-event kernel (ps), realtime scheduler, disciplined clocks, reproducible RNG streams, HDR-style telemetry, strict config loading |
| `aura.codes` | CSS codes: repetition, rotated surface (hook-safe schedule), bivariate-bicycle presets; GF(2) algebra |
| `aura.qpu` | device physics and profiles, drift (OU, TLS, cosmic rays), CHP tableau + batched Pauli-frame engine, noise semantics, dispersive readout, Lindblad calibration physics, leakage |
| `aura.control` | AURA-SEQ sequencer ISA, per-channel sequencers, readout cards (discrimination, LLR, leakage flag), results crossbar, syndrome packetizer, feed-forward decision registers |
| `aura.host.compiler` | circuit builder, native lowering, timing, dynamical decoupling, sequencer programs, detectors, detector error model, decoder manifest, Stim export |
| `aura.host.calibration` | 10 calibration experiments, robust fitting, dependency graph (bring-up and maintenance), versioned store |
| `aura.host.console` + `aura.cli` | REST API, live dashboard, `aura` command |
| `aura.link` | AURA-Link v1.1: 64b/66b PHY, frames + CRC-32C, Frame Parser, Arrival Timestamp Tagger, Sequence Tracker (erasures), Ingress Elastic FIFO, credits, time sync, session, management ARQ, UDP transport |
| `aura.decoding` | batched min-sum BP with convergence early exit, OSD-0 fallback, RL-learned halting policy (fitted Q-iteration / Q-learning) |
| `aura.lab` | `VirtualLab`: everything wired together; stream and batch runs |
| `aura.testbed` | decoder stand-ins (null, oracle, PyMatching, BP+OSD, RL-BP) until Layers 2–3 exist; UDP loopback soak test |

## Documentation

* `docs/ARCHITECTURE.md` — how every block works, with a measured timing table
* `docs/AURA_LINK_SPEC.md` — wire protocol and gateway specification (v1.1, with changes vs v1.0)
* `docs/VALIDATION.md` — every correctness claim and the test that proves it; known limitations
* `docs/DECISIONS.md` — design decisions taken, and **decisions that need your input**
* `docs/MIGRATION.md` — mapping from the original Deltaflow test harness

## Interface to Layer 2

The gateway delivers to any object implementing `aura.link.ingress.IngressSink`:

* `on_syndrome(SyndromeRecord)` carries patch, shot, round, bits, soft LLRs, leakage flags, sequence number, origin/arrival/ready timestamps, and a late flag;
* `on_erasure(ErasureRecord)` carries a lost round with its reason and (shot, round);
* `on_session(event, info)` delivers the decoder manifest (codes, round layouts, detectors, observables, reference parities, DEM) at session start.

Corrections go back through `IngressGateway.send_correction(CorrectionRecord)`.

## Provenance

AURA-QEC and AURA-Link are independent designs. They do not reuse Riverlane
trademarks (Deltaflow, QECi) or any Riverlane specification text. Published
work is cited where it informed a model (Caune et al., Nat. Commun. 2026;
Bravyi et al., Nature 2024; McEwen et al., Nat. Phys. 2022; Kelly et al.,
calibration graphs, 2018).
