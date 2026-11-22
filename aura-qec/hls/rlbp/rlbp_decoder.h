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
typedef ap_fixed<16, 8, AP_RND, AP_SAT> llr_t;   // range +-128, step 1/256
#define RLBP_LLR_INF llr_t(127.99)
#else
#include <limits>
typedef float llr_t;
#define RLBP_LLR_INF std::numeric_limits<float>::infinity()
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
