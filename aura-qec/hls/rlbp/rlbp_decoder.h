// RL-BP decoder kernel for Vitis HLS.
//
// Normalised min-sum belief propagation on the detector error model with
//   * zero-syndrome fast path (0 iterations),
//   * convergence early exit (leave at the first iteration whose hard
//     decision explains the syndrome),
//   * a reinforcement-learned halting policy (HALT_TABLE lookup) that hands
//     hopeless shots to the OSD fallback early.
//
// Port of aura.decoding.bp.MinSumBP + aura.decoding.policy.QHaltingPolicy.
// One shot per call. The OSD-0 fallback stays outside the kernel (testbench /
// host CPU for now): the kernel returns the posterior LLRs it needs.
#ifndef RLBP_DECODER_H
#define RLBP_DECODER_H

#include "rlbp_graph.h"

// ---- LLR arithmetic --------------------------------------------------------
// Default: float  -> bit-exact against the Python golden model.
// -DRLBP_FIXED    -> ap_fixed (much cheaper on the FPGA; compare LER, not bits).
#ifdef RLBP_FIXED
#include <ap_fixed.h>
// Width/integer bits are overridable (-DRLBP_LLR_W=.. -DRLBP_LLR_I=..). Saturation
// is required (sums of LLRs must not wrap); truncation instead of rounding saves
// an adder per operation.
// Default 10 bits: range +-32, step 1/16. On surface:3 every width from 16 down
// to 8 bits gave the same LER (within 0.2%) on the golden vectors.
#ifndef RLBP_LLR_W
#define RLBP_LLR_W 10
#endif
#ifndef RLBP_LLR_I
#define RLBP_LLR_I 6
#endif
#ifndef RLBP_LLR_Q
#define RLBP_LLR_Q AP_TRN
#endif
typedef ap_fixed<RLBP_LLR_W, RLBP_LLR_I, RLBP_LLR_Q, AP_SAT> llr_t;
#define RLBP_LLR_INF llr_t(4096.0)                // saturates to the largest value
#else
typedef float llr_t;
#define RLBP_LLR_INF __builtin_huge_valf()
#endif

// exit reasons (same codes as aura.decoding.bp)
enum {
    EXIT_TRIVIAL   = 0,   // empty syndrome
    EXIT_CONVERGED = 1,   // hard decision explains the syndrome
    EXIT_HALTED    = 2,   // RL policy stopped BP -> OSD
    EXIT_MAX_ITER  = 3    // budget exhausted (WCET) -> OSD
};

// Top-level function (synthesised).
//   syndrome[c]   detection event on check c (0/1)
//   posterior[v]  LLRs at exit (input to the OSD fallback)
//   hard[v]       BP hard decision at exit
//   obs_bp        logical prediction from the hard decision (bit k = observable k)
//   iterations    BP iterations spent
//   exit_reason   EXIT_*; OSD is required unless TRIVIAL or CONVERGED
void rlbp_decode(const unsigned char syndrome[RLBP_NUM_CHECKS],
                 llr_t posterior[RLBP_NUM_VARS],
                 unsigned char hard[RLBP_NUM_VARS],
                 unsigned int *obs_bp,
                 unsigned char *iterations,
                 unsigned char *exit_reason);

// Policy state encoding (shared with the testbench for unit checks).
unsigned short rlbp_state(int t, int unsat, int unsat0, int stall);

#endif
