%title: OAM-BP: Oscillation-Aware Learned Memory for Real-Time BP Decoding of Quantum LDPC Codes on FPGA
%subtitle: Draft paper: method, decoding results and FPGA implementation
%status: Working draft  ·  23 November 2026  ·  not for distribution
%running: OAM-BP  ·  Draft paper

> Belief-propagation (BP) decoders are attractive for real-time quantum error correction because they map to fully parallel, fixed-latency hardware, but on quantum LDPC codes they fail by oscillating. Memory BP and Relay-BP damp this with a memory term whose strength is either fixed or drawn at random, and recover accuracy by running several BP passes. We propose **oscillation-aware memory BP (OAM-BP)**: each variable node chooses its own memory strength at every iteration from a **16-entry table indexed by its local oscillation state** (recent sign flips, current hard decision and decoding phase). The table is learned offline by the cross-entropy method, a policy-search method, so the decoder remains a learned controller of BP while adding only a 4-bit register and a table lookup per variable.
> On the [[72,12,6]] bivariate-bicycle code at calibrated device noise, a single OAM-BP run reaches a logical error rate of **6.40 × 10⁻²** with a 200-iteration worst case and 15.6 mean iterations. That is significantly better than the best fixed memory strength (7.13 × 10⁻², paired McNemar p = 5 × 10⁻⁵), matches Relay-BP-1 and a four-member memory-BP ensemble within error bars, and needs about half the ensemble's iterations. Because a single table's remaining errors are mostly converged-but-wrong solutions, we add a second, **check-aware 48-state table** that is no better on its own (p = 0.50) but fails on different shots. Running both tables and keeping the lower-weight solution (**OAM pair**) lowers the error rate to **5.69 × 10⁻²** (p = 7 × 10⁻¹⁰ against single OAM-BP), and to 5.88 × 10⁻² when each run is capped at 100 iterations, the same 200-iteration total worst case. That beats a four-member memory-BP ensemble (6.16 × 10⁻²) at about half its iterations. On surface codes the same table, without retraining, keeps the error rate of BP+OSD-0 while halving the OSD fallback rate. We implement OAM-BP in a fully parallel HLS decoder on an AMD Kria KV260: the surface-3 decoder closes timing at 300 MHz post-route, adds 2.5% LUTs over the baseline, keeps 13 cycles per BP iteration and lowers the mean decode latency from 67 ns to 47 ns. A two-core OAM pair kernel also closes timing at 300 MHz post-route, with 21% of the device's LUTs and 14 cycles per iteration.

## 1. Introduction

Fault-tolerant quantum computers must decode syndromes faster than they are produced. On superconducting hardware a syndrome round takes about a microsecond, so the decoder needs a bounded, microsecond-scale worst-case latency, not only a good average. Quantum LDPC codes such as the bivariate-bicycle (BB) codes promise much lower qubit overhead than the surface code, but their decoders are harder to make fast. Minimum-weight matching does not apply, and BP alone fails often because the code's degeneracy and short cycles make the messages oscillate instead of converging.

The standard remedy, BP followed by ordered-statistics decoding (BP+OSD), adds a Gaussian-elimination stage whose worst case grows with the code size. On BB-72 our model of it reaches hundreds of microseconds. Recent work instead keeps BP and changes its dynamics. Memory BP mixes the previous marginal into the next prior, and Relay-BP chains several memory-BP runs with randomly disordered per-variable memory strengths. These methods remove OSD but pay for accuracy with more BP iterations.

This work started from a learned-control view of BP decoding: an agent that decides, from cheap features, when to stop BP. That reinforcement-learning (RL) halting controller works in hardware but only ties a simple stall rule (Appendix A). The study log in Appendix A shows the lesson: **controlling BP at the level of whole shots is too coarse, because the failure happens at individual oscillating variables.** OAM-BP moves the learned controller to where the oscillation happens.

**Contributions.**

1. **OAM-BP**, a memory-BP decoder in which each variable's memory strength is set at every iteration by a learned table indexed by that variable's local oscillation state. As far as we can find, no prior decoder learns a state-indexed per-variable memory policy.
2. **OAM pair**, two OAM-BP runs with different learned tables (16-state, and a 48-state table that also sees the number of unsatisfied neighbouring checks) combined by lowest solution weight. It is the most accurate OSD-free decoder we tested on BB-72, and in hardware the two runs execute side by side, so the latency is that of a single run.
3. **A policy-search training procedure** (cross-entropy method) that learns the 16-entry table from simulated syndromes in minutes, with a penalty on iterations.
4. **An evaluation on BB-72 and surface codes** with paired significance tests on shared test shots, compared against fixed memory, a hand-designed table, a memory-BP ensemble, Relay-BP-1 and BP+OSD-0.
5. **A fully parallel FPGA implementation** with on-chip OSD-0 fallback, verified bit-for-bit in C simulation and by C/RTL co-simulation, with post-route results on a Kria KV260. We also report an HLS synthesis pitfall we hit and its fix.

## 2. Background

### 2.1 Decoding problem

A detector error model gives a parity-check matrix H (C checks, V error mechanisms with priors pⱼ) and an observable matrix L. Given a syndrome s, the decoder outputs ê with Hê = s, and the shot fails when Lê ≠ Le. We use circuit-level detector error models from a calibrated device-noise profile; λ scales every mechanism's probability (λ = 1 is the calibrated device).

### 2.2 Min-sum BP and its failure mode

Normalised min-sum BP passes log-likelihood messages on the Tanner graph. Each variable's marginal is its prior LLR Λⱼ(0) = log((1 − pⱼ)/pⱼ) plus the sum of incoming check messages. On BB codes we observe that failing shots are dominated by **period-2 oscillation**: a few variables flip their hard decision every iteration and BP never satisfies the syndrome.

### 2.3 Memory BP and Relay-BP

Memory BP replaces the fixed prior by a mix of the prior and the previous marginal:

$$ Λⱼ(t) = (1 − γ) Λⱼ(0) + γ Mⱼ(t − 1)

A positive γ adds inertia that damps oscillation; too much inertia freezes wrong decisions. Relay-BP draws γⱼ per variable at random from an interval, and chains several such runs ("legs"), keeping the lowest-weight solution. In our tests, the best fixed γ differs from shot to shot. An oracle that picks the best of six γ values per shot reaches 3.8 × 10⁻² on BB-72, about half the error rate of the best single γ, so the choice of γ has large headroom.

## 3. OAM-BP: oscillation-aware learned memory

### 3.1 Local state

Each variable j keeps a 4-bit shift register hⱼ of its last four hard-decision changes (1 = the sign of Mⱼ flipped from the previous iteration). At iteration t its state is

$$ sⱼ(t) = 4 · min(popcount(hⱼ), 3) + 2 · [Mⱼ(t) < 0] + [t > 8]

giving 16 states: recent flip count (0, 1, 2, 3+), current hard decision, and early or late phase.

### 3.2 Update rule

The memory strength is read from a learned table Γ and used for the next iteration's prior:

$$ γⱼ(t) = Γ[sⱼ(t)],   Λⱼ(t + 1) = Λⱼ(0) + γⱼ(t) · (Mⱼ(t) − Λⱼ(0))

The rest of BP is unchanged: min-sum check update, early exit on a satisfied syndrome, and an optional OSD-0 fallback on the averaged posterior of the last two iterations. With Γ ≡ 0 the decoder is plain BP, and with Γ ≡ γ it is memory BP, so OAM-BP strictly generalises both. It is deterministic, which matters for hardware verification and reproducibility, unlike Relay-BP's random disorder.

{{fig:arch}}

### 3.3 Learning Γ by policy search

We treat Γ as the parameters of a per-variable control policy and learn it with the cross-entropy method. Each generation:

- samples 24 candidate tables from a Gaussian;
- scores each on 3,000 training shots by logical error rate plus 10⁻⁴ × mean iterations (cap 100);
- refits the Gaussian to the best 25%.

Candidates are clipped to γ ∈ [−0.25, 0.7]; we run eight generations on BB-72 at λ = 1 on a laptop CPU. Training and test shots are always disjoint. The learned table is shown in Figure 3. It is not monotone in the flip count, which a hand-designed rule would be; the hand-set table (γ = 0.1, 0.25, 0.5 for 0, 1, 2+ flips) performs clearly worse (Table 1).

{{fig:gamma}}

### 3.4 Check-aware state and the OAM pair

At λ = 1 only about 1% of BB-72 shots end without satisfying the syndrome, yet the logical error rate of a single OAM-BP run is 6.4%. Most failures are therefore **converged-but-wrong**: BP finds a valid correction in the wrong logical class. A richer per-variable state cannot detect such shots, but a second decoder with *different* dynamics can find a different, lower-weight solution.

- **Check-aware table.** We extend the state with the number of unsatisfied checks next to the variable (0, 1, 2+), giving 48 states: sⱼ'(t) = 3 · sⱼ(t) + min(uⱼ(t), 2). This is the signal used by bit-flipping decoders; in hardware it is a 2-bit saturating popcount over syndrome-check bits the decoder already computes. The table is initialised from the 16-state table and refined by the same cross-entropy method (10 generations, 28 candidates, 4,000 training shots).
- **OAM pair.** Both tables decode the shot. Among the converged runs we return the solution with the lowest weight Σⱼ Λⱼ(0) · x̂ⱼ (the prior LLR weight); if neither converges we return the first run's hard decision. The two runs are independent, so in hardware they run in parallel on the same syndrome.

## 4. Hardware architecture

The decoder is a fully parallel HLS kernel (Vitis HLS 2025.1) generated from the code's detector error model by an exporter that also emits the Γ table and test vectors. Its main features:

- **Datapath.** All check nodes and all variable nodes are updated in parallel every iteration. LLRs use 10-bit fixed point (ap_fixed<10,6>, truncation, saturation).
- **I/O.** The syndrome and outputs are packed into single AXI-Lite registers.
- **Control.** The iteration loop runs sequentially with early exit on convergence and the learned RL halting table from earlier work.
- **OSD-0 fallback.** Implemented on chip: it ranks the variables with V parallel comparators, then walks the columns in posterior order, reading them from a column ROM, and does incremental GF(2) elimination.

**OAM additions per variable.** Each variable gets:

- a 4-bit flip-history register and a 1-bit previous sign;
- a 2-bit popcount saturating at 3;
- a read port on a shared 16 × 9-bit Γ ROM;
- one multiply-add for the memory mix.

The mix is written as Λ(0) + γ·(M − Λ(0)), which needs one multiplier per variable (mapped to DSP slices). It is mathematically identical to the (1 − γ)·Λ(0) + γ·M form.

**An HLS pitfall.** With the (1 − γ)·Λ(0) + γ·M form, Vitis HLS 2025.1 mapped the expression onto a DSP pre-adder that takes the constant 1 as a 1-bit operand. The resulting RTL disagreed with the C model, and only C/RTL co-simulation caught it: co-simulation measured 4.8% logical error against 2.0% in C. Rewriting the expression as above gives the same exact value, and co-simulation then passes. We also found that HLS turns a by-reference output array into an output-only port, so state that must persist across iterations is kept local. We report both as practical lessons for HLS decoders.

**OAM pair in hardware.** The pair instantiates two OAM-BP cores on the same syndrome register. The second core differs only in a 48-entry Γ ROM addressed with two more state bits, the unsatisfied-check count per variable (a 2-bit saturating popcount over the check bits already computed for the convergence test). Each core accumulates its solution weight Σ Λⱼ(0)·x̂ⱼ with an adder tree over the hard decisions when it converges, and a single comparator selects the output. The latency is that of the slower core, not the sum. In the HLS code both cores sit in one iteration loop and update in lockstep; a core that has stopped holds its state, and the loop ends when both have stopped. We built the pair on surface-3 (Section 6.1).

## 5. Evaluation

### 5.1 Setup

- **Codes.** BB-72 [[72,12,6]] with 6 syndrome rounds (252 checks, 2,232 error mechanisms, 8,244 edges) is the main target. Surface codes d = 3 (16 checks, 55 mechanisms) and d = 5 (72, 301) test transfer and hardware.
- **Noise.** Circuit-level detector error models from a calibrated device profile, scaled by λ.
- **Statistics.** Error bars are 95% Wilson intervals. Every decoder at a given λ decodes the same test shots, so we use the exact McNemar test for paired comparisons.
- **BP settings.** BB-72 decoders use min-sum without normalisation (α = 1) and a 200-iteration cap unless stated. Surface-code decoders use the hardware settings: α = 0.75, 30 iterations, then OSD-0.

### 5.2 BB-72 at device noise

**Table 1.** BB-72, λ = 1, 10,000 test shots, 200-iteration worst case. Mean / p99 iterations count all BP iterations spent on a shot (summed over runs; the OAM pair's two runs execute in parallel in hardware). The OAM pair with cap 100 per run has the same 200-iteration total worst case as the other rows.

| Decoder | LER | 95% CI | Mean it. | p99 it. | BP runs |
|---|---|---|---|---|---|
| BP+OSD-0, averaged OSD input (30 it.) | 8.04e-2 | – | – | 30 | 1 + OSD |
| Memory BP, fixed γ = 0.25 | 7.37e-2 | 6.87–7.90e-2 | 13.2 | 113 | 1 |
| Memory BP, fixed γ = 0.125 | 7.13e-2 | 6.64–7.65e-2 | 20.3 | 200 | 1 |
| OAM, hand-set Γ | 7.58e-2 | 7.08–8.12e-2 | 11.7 | 73 | 1 |
| **OAM-BP, learned Γ** | **6.40e-2** | **5.94–6.90e-2** | **15.6** | 200 | **1** |
| Relay-BP-1 (repeat runs) | 6.34–6.70e-2 | – | 18.5 | 200 | 1 relay |
| Ensemble: OAM + 3 fixed γ, agree-stop | 6.31e-2 | 5.85–6.80e-2 | 32.5 | 188 | 2.15 |
| Ensemble: OAM + 3 fixed γ, all 4 | 6.16e-2 | 5.71–6.65e-2 | 55.3 | 188 | 4 |
| OAM-BP, check-aware 48-state Γ | 6.28e-2 | 5.82–6.77e-2 | 14.7 | 126 | 1 |
| OAM pair, second run only if first fails | 6.15e-2 | 5.70–6.64e-2 | 16.8 | 200 | 1.01 |
| **OAM pair, cap 100 per run (200 total)** | **5.88e-2** | **5.44–6.36e-2** | 28.2 | 183 | 2 |
| **OAM pair, cap 200 per run** | **5.69e-2** | **5.25–6.16e-2** | 30.3 | 262 | 2 |

{{fig:tradeoff}}

**Table 2.** Paired McNemar tests on the same 10,000 shots at λ = 1. "Only A wrong" counts shots that decoder A fails and decoder B decodes correctly.

| Comparison (A vs B) | Only A wrong | Only B wrong | p-value | Verdict |
|---|---|---|---|---|
| OAM-BP vs fixed γ = 0.25 | 141 | 238 | 7.2e-7 | OAM-BP better |
| OAM-BP vs fixed γ = 0.125 | 122 | 195 | 4.9e-5 | OAM-BP better |
| 4-member ensemble vs OAM-BP | 101 | 125 | 0.13 | no significant difference |
| 48-state check-aware vs 16-state OAM-BP | 128 | 140 | 0.50 | no significant difference |
| OAM pair (cap 200 each) vs OAM-BP | 32 | 103 | 6.8e-10 | **OAM pair better** |
| OAM pair (cap 100 each) vs OAM-BP (cap 100) | 29 | 115 | 2.6e-13 | **OAM pair better** |

A single learned OAM-BP run removes 10% of the errors of the best fixed memory strength, with 23% fewer mean iterations. It is statistically indistinguishable from an ensemble that spends 3.5 times the iterations.

The check-aware table alone is no better than the 16-state table, but the two disagree on 268 of the 10,000 shots. The minimum-weight rule picks the right one in most of them: the OAM pair fixes 103 errors of single OAM-BP and introduces 32, an 11% reduction in logical error rate. Running the second table only when the first fails recovers just 25 shots, which confirms that the gain comes from converged-but-wrong shots. The OAM pair is also below the four-member ensemble (5.69 against 6.16 × 10⁻²) with 45% fewer total iterations, and needs two runs instead of four.

### 5.3 Noise generalisation and budget

**Table 3.** BB-72 across noise; the table is trained at λ = 1 only. 10,000 shots per row.

| λ | Cap | Fixed γ = 0.125 | Fixed γ = 0.25 | OAM hand-set | **OAM-BP learned** | 4-member ensemble | **OAM pair** |
|---|---|---|---|---|---|---|---|
| 0.5 | 100 | 6.70e-3 | 7.50e-3 | 6.70e-3 | **6.50e-3** | – | – |
| 0.5 | 200 | 5.90e-3 | 7.50e-3 | 6.60e-3 | 6.20e-3 | 5.50e-3 | 5.50e-3 |
| 0.7 | 100 | 2.24e-2 | 2.21e-2 | 2.53e-2 | **2.02e-2** | – | – |
| 0.7 | 200 | 1.94e-2 | 2.16e-2 | 2.50e-2 | 1.93e-2 | 1.71e-2 | 1.75e-2 |
| 1.0 | 100 | 7.83e-2 | 7.54e-2 | 7.72e-2 | 6.74e-2 | – | **5.88e-2** |
| 1.0 | 200 | 7.13e-2 | 7.37e-2 | 7.58e-2 | 6.40e-2 | 6.16e-2 | **5.69e-2** |

At lower noise OAM-BP is never significantly worse than the best fixed γ (λ = 0.7: p = 1.0 against γ = 0.125 and p = 0.03 better than γ = 0.25). The gain grows with noise, where oscillation dominates. The four-member ensemble is significantly better only at λ = 0.7 (p = 0.02), at about four times the iterations. The OAM pair (cap = per-run cap) is significantly better than single OAM-BP at λ = 1 and at λ = 0.7 (p = 0.005; 10 against 28 discordant shots). At λ = 0.5 it ties the ensemble for the lowest error rate, but with only 13 discordant shots the difference to single OAM-BP is not significant (p = 0.09). The check-aware table was also trained at λ = 1 only.

{{fig:lambda}}

### 5.4 Surface codes (transfer, no retraining)

**Table 4.** Surface codes with the BB-72-trained Γ and hardware settings. 100,000 shots (d = 3) and 50,000 shots (d = 5) per point. Fallback = share of shots that need OSD-0.

| Code | λ | MWPM | BP+OSD-0 | OAM-BP+OSD-0 | Fallback BP+OSD → OAM | Mean it. BP+OSD → OAM |
|---|---|---|---|---|---|---|
| d = 3 | 0.5 | 2.38e-3 | 2.39e-3 | 2.61e-3 | 0.63% → 0.30% | 0.42 → 0.33 |
| d = 3 | 1 | 8.31e-3 | 8.62e-3 | 8.84e-3 | 2.14% → 1.14% | 1.06 → 0.79 |
| d = 3 | 2 | 3.15e-2 | 3.47e-2 | 3.38e-2 | 6.49% → 4.40% | 2.65 → 2.11 |
| d = 5 | 0.5 | 9.80e-4 | 1.06e-3 | 9.80e-4 | 2.73% → 1.33% | 1.64 → 1.29 |
| d = 5 | 1 | 5.56e-3 | 5.72e-3 | 5.96e-3 | 9.72% → 5.17% | 4.19 → 3.07 |
| d = 5 | 2 | 3.86e-2 | 4.03e-2 | 4.06e-2 | 29.7% → 18.9% | 10.4 → 7.97 |

On surface codes BP+OSD-0 is already close to MWPM, and OAM-BP keeps that accuracy: every pair of intervals overlaps. The practical gain is latency. **OAM-BP halves the share of shots that need the slow OSD stage** (35–53% fewer) and cuts mean iterations by 20–30%. This happens even though Γ was trained on a different code family.

**The OAM pair on surface codes.** With the same settings, the pair does not improve the surface-code error rate: on d = 3 it makes exactly the same errors as single OAM-BP at every λ, and on d = 5 the differences are within noise (p ≥ 0.24). Small surface codes rarely have the converged-but-wrong failures that the pair fixes on BB-72. The pair only lowers the OSD fallback rate further (d = 5, λ = 1: 5.17% → 4.39%; λ = 2: 18.9% → 15.8%). Its accuracy gain is a property of the qLDPC code, and the surface-3 build in Section 6 measures its hardware cost.

{{fig:surface}}

## 6. FPGA implementation results

### 6.1 Measured builds

All three builds target the Kria KV260 (XCK26-SFVC784-2LV-C) at a 3.333 ns clock (300 MHz), go through Vivado place and route, and are verified on 1,000 test vectors. The C model matches the Python golden model (fixed-point logical error within 0.5 percentage points), and C/RTL co-simulation passes.

**Table 5.** Build configuration.

| Item | RL-BP baseline | OAM-BP | OAM pair |
|---|---|---|---|
| Code / graph | surface-3, 16 checks, 55 vars, 94 edges | same | same |
| Arithmetic | 10-bit LLR ap_fixed<10,6>, α = 0.75 | same LLRs, α = 1 (memory replaces normalisation); Γ in 12-bit fixed | as OAM-BP |
| BP schedule | flooding, fully parallel, max 30 it. | same | two cores in lockstep |
| Stopping | converge / learned RL halting table | same | same, per core |
| OSD input | mean of last two posteriors | same | core 0, only if neither core converges |
| Memory | none | per-variable, learned 16-state Γ | core 0: 16-state Γ; core 1: 48-state check-aware Γ |
| Output | BP or OSD-0 | BP or OSD-0 | lower-weight converged solution, else OSD-0 |

**Table 6.** Post-implementation resources and timing (Vivado 2025.1). Utilisation in % of the XCK26.

| Metric | RL-BP baseline | OAM-BP | Change | OAM pair |
|---|---|---|---|---|
| LUT | 11,698 (10.0%) | 11,991 (10.2%) | +293 (+2.5%) | 24,866 (21.2%) |
| FF | 9,172 (3.9%) | 10,857 (4.6%) | +1,685 | 23,833 (10.2%) |
| DSP | 2 (0.2%) | 57 (4.6%) | +55 (one per variable) | 113 (9.1%) |
| BRAM | 1 (0.7%) | 1 (0.7%) | 0 | 1 (0.7%) |
| CLB | 2,356 | 2,458 | +102 | 5,202 |
| Clock period, post-synthesis | 2.801 ns | 2.837 ns | +0.036 ns | 2.838 ns |
| Clock period, post-route | 3.213 ns | 3.243 ns | +0.030 ns | 3.318 ns |
| Achievable clock (post-route) | 311 MHz | 308 MHz | timing met at 300 MHz | 301 MHz, met |

**Table 7.** Latency and throughput (C/RTL co-simulation, 1,000 vectors, 300 MHz).

| Metric | RL-BP baseline | OAM-BP | OAM pair |
|---|---|---|---|
| Cycles per BP iteration | 13 | 13 | 14 |
| Latency, min (cycles / ns) | 5 / 17 | 5 / 17 | 5 / 17 |
| Latency, mean (cycles / ns) | 20 / 67 | **14 / 47** | 22 / 73 |
| Latency, max (cycles / ns) | 380 / 1,267 | **309 / 1,030** | 401 / 1,337 |
| Initiation interval, mean (cycles) | 21 | 15 | 23 |
| Throughput, mean (decodes / µs) | 14.3 | **20.0** | 13.0 |
| Vector-set logical error (kernel / golden) | – / 2.20% | 1.90% / 2.00% | 2.00% / 1.90% |
| OSD fallback rate on the vectors | 4.3% | 1.2% | 0.9% |

{{fig:fpga}}

OAM-BP costs 293 LUTs, 1,685 flip-flops and 55 DSP slices on surface-3. It leaves the per-iteration latency unchanged at 13 cycles and the clock at 300 MHz. Because fewer shots fall back to OSD-0, mean latency drops by 30% and worst-case latency by 19%.

**The OAM pair build.** The two-core kernel passes C simulation bit-exactly in floating point against the Python model of the pair (all 1,000 vectors) and within tolerance in fixed point, and C/RTL co-simulation passes. It costs 2.07 × the LUTs and 1.98 × the DSPs of the single-core OAM-BP, as expected for two cores plus a weight adder tree and comparator, and still closes timing at 300 MHz post-route (3.318 ns, with less slack than the single core). The cores run in parallel: one iteration takes 14 cycles instead of 13. Mean latency rises from 14 to 22 cycles (73 ns), most likely because the shared iteration loop, the weight comparison and the wider initialisation add fixed cycles to every non-trivial shot, which dominate on surface-3 where most shots need under one iteration. On surface-3 the pair brings no accuracy gain (Section 5.4); the build establishes that the BB-72 accuracy gain costs 2 × logic, one extra cycle per iteration and no change in clock frequency.

### 6.2 Scaling: surface-5 and BB-72 (modelled)

A fully parallel surface-5 build (566 edges) did not fit the 7 GB of memory of our build machine during HLS scheduling. We therefore report modelled numbers for surface-5 and BB-72, from a cost model calibrated on the measured surface-3 kernel:

- about 96 LUT per edge lane;
- 13 cycles per fully parallel iteration;
- 2⌈E/P⌉ + 13 cycles per iteration with P edge lanes.

The OAM overhead comes from Table 6: about 5.3 LUT, 31 FF and 1 DSP per variable lane, plus 15 bits of state per variable.

**Table 8.** Modelled cost (not built). Latency at 300 MHz; mean latency uses the measured mean iterations at λ = 1 from Tables 1 and 4.

| Design | Lanes P | BP LUT (model) | OAM extra | Device class | Cycles / it. | Worst case (200 it.) | Mean, OAM-BP |
|---|---|---|---|---|---|---|---|
| Surface-5, fully parallel | 566 | ≈ 54k | +1.6k LUT, +301 DSP | KV260 (XCK26) | 13 | 30 it.: 1.3 µs + OSD | 3.07 it.: 0.13 µs + OSD |
| BB-72, fully parallel | 8,244 | ≈ 910k | +12k LUT, +2.2k DSP | VU9P class | 13 | 8.7 µs | 0.68 µs |
| BB-72, partially parallel | 2,048 | ≈ 226k | +3k LUT, +555 DSP | VU9P class | 23 | 15.3 µs | 1.20 µs |
| BB-72, partially parallel | 512 | ≈ 57k | +0.7k LUT, +139 DSP, 1 BRAM | KV260 (XCK26) | 47 | 31.3 µs | 2.44 µs |

**Table 9.** BB-72 decoders at λ = 1 under the same model. OSD-0 cost is given for two models: naive (bit-serial, as built) and optimistic (single-pass column reduction).

| Decoder | LER | Worst case, P = 8,244 | Worst case, P = 512 | Mean it. |
|---|---|---|---|---|
| BP+OSD-0, averaged input | 8.04e-2 | 403 µs / 40 µs | 407 µs / 44 µs | – |
| Relay-BP-1, cap 200 | 6.70e-2 | 8.7 µs | 31.3 µs | 18.5 |
| **OAM-BP, cap 200** | **6.40e-2** | **8.7 µs** | **31.3 µs** | **15.6** |
| Ensemble, agree-stop | 6.31e-2 | 8.7 µs | 31.3 µs | 32.5 |
| **OAM pair, cap 100 per run, two cores in parallel** | **5.88e-2** | **4.7 µs** (2 × LUT) | **16.0 µs** (2 × LUT) | 28.2 (sum) |

At an equal iteration cap all OSD-free decoders share the same worst case, and they differ in accuracy and mean latency. OAM-BP has the lowest mean cost at Relay-BP-level accuracy and needs no random number generation. The ensemble's small accuracy edge costs twice the mean latency. The OAM pair changes the trade-off (its row uses 14 cycles per iteration, as measured on the surface-3 pair build, plus one extra cycle in the partially parallel case): with two cores side by side and 100 iterations per core, it is the most accurate decoder in the table and has half the worst-case latency of the single-run decoders, at twice the logic. At P = 512 two cores need about 115k LUT, more than a KV260 can route in practice, so for BB-72 the pair targets a larger device.

## 7. Related work

**Table 10.** Positioning. Entries summarise each method's mechanism; reported FPGA figures from other works use different codes, devices and noise models and are not compared numerically here.

| Method | Memory / damping | Per-variable | Learned | Needs OSD | Deterministic |
|---|---|---|---|---|---|
| BP+OSD | none | – | no | yes | yes |
| Memory BP (fixed γ) | prior–marginal mix | no | no | no | yes |
| Relay-BP (arXiv:2506.01779) | prior–marginal mix | yes, random | no | no | no |
| Neural BP (e.g. arXiv:1811.07835, 2308.08208) | learned message weights | by edge / iteration | yes | optional | yes |
| BP with flip-based signals (e.g. arXiv:0803.1090, 2507.00254) | uses sign-flip information | yes | no | varies | varies |
| Dithered BP (arXiv:2611.07144) | perturbed messages | yes | no | no | no |
| **OAM-BP (this work)** | **prior–marginal mix** | **yes, state-driven** | **yes (16 values)** | **no (optional)** | **yes** |
| **OAM pair (this work)** | **two learned tables, min-weight** | **yes, state- and check-driven** | **yes (16 + 48 values)** | **no** | **yes** |

Neural BP learns weights indexed by position (edge and iteration), so its parameter count grows with the code and the iteration count. OAM-BP learns 16 numbers indexed by a local dynamical state, so the same table transfers across codes (Section 5.4). Flip-based methods use oscillation signals but with hand-designed rules. Relay-BP uses per-variable memory but chooses it at random. All references are to be verified against their final versions before submission.

## 8. Limitations and next steps

- **Hardware scale.** The surface-5 and BB-72 hardware numbers are modelled, not built. A partially parallel BB-72 kernel, and a fully parallel surface-5 build on a machine with more memory, are the next hardware steps.
- **Statistics.** BB-72 results use 10,000 test shots per point. The headline comparisons are significant, but finer ones (for example λ = 0.5) need about 50,000 shots.
- **Ablations still to run:** which state features matter (flip count only, adding the hard decision, adding the phase), table size, and a sweep of the training λ.
- **Larger codes.** BB-144 [[144,12,12]] has only been probed in software.
- **OAM pair hardware.** The pair is built and verified on surface-3, where it brings no accuracy gain; its BB-72 hardware cost is modelled from that build. A partially parallel BB-72 pair kernel is the step that would show the accuracy gain in hardware.
- **More learned tables.** Two tables trained independently already give an 11% gain; training further tables explicitly for diversity (for example, scoring candidates on the shots the existing tables get wrong) is a natural next step. A sequential relay of OAM legs with warm starts gave no gain in training (about 6.5 × 10⁻²), so we did not pursue it.
- **Training noise.** The cross-entropy method scores candidates on 4,000 shots, where the sampling error (about ±0.4 percentage points) is as large as the differences it tries to resolve. Larger or paired training sets should help.

## 9. Conclusion

Moving learned control from the shot level to the variable level turns a controller that only tied simple rules into one that changes the decoder's dynamics. OAM-BP, a 16-entry learned memory table indexed by local oscillation state, makes a single BP run on BB-72 as accurate as Relay-BP-1 or a four-member ensemble, at half the ensemble's iterations and with no randomness. On surface codes it halves the OSD fallback rate. The hardware cost is a few registers and one multiply per variable, at unchanged iteration latency and 300 MHz on a KV260. Two learned tables that fail on different shots, combined by minimum weight, cut the BB-72 error rate by a further 11% (5.69 × 10⁻², p = 7 × 10⁻¹⁰) and beat the four-member ensemble with two parallel runs: OSD-free BP decoding benefits from learned diversity as well as learned damping.

## Appendix A. Study log: how we arrived at OAM-BP

**Table A1.** Earlier experiments (details in the companion study draft). Each row is a tested idea and its outcome.

| Step | Idea tested | Outcome |
|---|---|---|
| 1 | Tabular Q-learning halting policy (1,320 states) for BP+OSD | Ties a stall-K rule on surface codes; implemented in hardware |
| 2 | Stopping rules on BB-72 | Halting after 8 stalled iterations beats run-to-budget and the RL policy |
| 3 | OSD-0 input: final vs averaged posteriors | Mean of last two iterations cuts BB-72 LER by 39–54% (adopted, also in hardware) |
| 4 | Relay-BP-1 re-implementation | Beats BP+OSD-0 on BB-72 without OSD, given ~100+ iterations |
| 5 | Relay-BP under a hard budget, + OSD hybrid | OSD dominates the worst case; OSD-free designs preferred |
| 6 | RL-controlled restarts (fitted Q-iteration) | No gain over fixed restarts |
| 7 | Memory-BP γ ensemble, min-weight selection | Oracle headroom 3.8e-2; ensemble ties Relay-BP-1 |
| 8 | RL early exit for the ensemble | Ties a simple agree-stop rule |
| 9 | **OAM-BP: learned per-variable memory** | **Single run matches ensemble and Relay-BP-1 at half the ensemble's cost** |
| 10 | Check-aware 48-state Γ; OAM-Relay legs | 48-state alone ties 16-state (p = 0.50); relay legs give no gain in training |
| 11 | **OAM pair: 16 + 48-state tables, min-weight** | **5.69e-2 at λ = 1, p = 7e-10 against single OAM-BP; best OSD-free decoder** |

## Appendix B. Reproducibility

- **Software.** The decoder library has MinSumBP with `memory_table=` (verified identical, shot for shot, to the experiment implementation) and 128 unit tests.
- **Experiments:** `oam_bp.py` (training and Table 3), `oam_ensemble.py` (Tables 1–2), `surface_oam.py` (Table 4), `oam2.py` (check-aware table and relay legs; `oam2_unsat_L1_S1.csv`), `oam_pair.py` (OAM pair in Tables 1–3; `oam_pair_cap200.csv`, `oam_pair_cap100.csv`), `surface_pair.py` (pair on surface codes; `surface_pair.csv`), `cost_model.py` (Tables 8–9).
- **Hardware.** `export_hls.py --memory-table` generates the kernel configuration and vectors; adding `--memory-table2` with the 48-state table builds the OAM pair (`RLBP_PAIR`). The surface-3 OAM builds use `--alpha 1.0`. `run_builds.ps1` runs C simulation, synthesis, co-simulation and Vivado implementation and prints the numbers in Tables 6–7.
- **Surface-5 build on a larger machine:** `export_hls.py --code surface:5 --memory-table results/oam_gamma_bb72.json`, then `run_builds.ps1 s5_oam`. Allow 16 GB or more of RAM.
