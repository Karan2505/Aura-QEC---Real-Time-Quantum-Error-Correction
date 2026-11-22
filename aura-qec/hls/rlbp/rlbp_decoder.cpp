// RL-BP decoder kernel (synthesisable). See rlbp_decoder.h.
//
// Architecture: the Tanner graph is a compile-time constant (rlbp_graph.h), so
// every per-check / per-variable loop is fully unrolled over the graph and the
// message arrays are completely partitioned into registers. All CSR lookups
// (CHK_PTR, E_VAR, VAR_EDGE, ...) then fold to constants and one BP iteration
// becomes a short fixed pipeline of compare / add stages instead of a
// sequential walk over the edges.
//
// Arithmetic order is unchanged from the sequential reference: edges of a
// check are scanned in order (same min1/min2/argmin tie-breaking) and each
// variable sums its edges in ascending order (the Python model's summation
// order), so the float build stays bit-identical to aura.decoding.
//
// Synthesise with RLBP_MODE=fixed: in float mode the unrolled datapath needs
// one floating-point adder / comparator per edge.
#include "rlbp_decoder.h"

// ---- RL policy state: aura.decoding.policy.encode_state --------------------
//   iteration bucket : 1..8, 9-12, 13-20, 21+                 (11)
//   residual unsat   : 1, 2, 3-4, 5-8, 9-16, 17+               (6)
//   unsat / unsat0   : <.34, <.67, <1, >=1                     (4)  integer compare, exact
//   stall            : 0, 1, 2-3, 4-7, 8+                      (5)
unsigned short rlbp_state(int t, int unsat, int unsat0, int stall) {
#pragma HLS INLINE
    int tb = t >= 21 ? 10 : t >= 13 ? 9 : t >= 9 ? 8 : t - 1;
    int u = unsat < 1 ? 1 : unsat;
    int wb = u >= 17 ? 5 : u >= 9 ? 4 : u >= 5 ? 3 : u >= 3 ? 2 : u - 1;
    int u0 = unsat0 < 1 ? 1 : unsat0;
    int rb = (100 * unsat >= 34 * u0) + (100 * unsat >= 67 * u0) + (unsat >= u0);
    int sb = stall >= 8 ? 4 : stall >= 4 ? 3 : stall >= 2 ? 2 : stall;
    return (unsigned short)(((tb * 6 + wb) * 4 + rb) * 5 + sb);
}

// prior LLRs as constants of type llr_t (no float->fixed conversion in hardware)
static const llr_t PRIOR[RLBP_NUM_VARS] = RLBP_PRIOR_INIT;

static llr_t llr_abs(llr_t x) {
#pragma HLS INLINE
    return x < 0 ? llr_t(-x) : x;
}

// ---- BP iterations with early exit and RL halting (runs only for a non-empty,
// explainable syndrome). Returns the exit reason; *t_exit = iterations spent.
static unsigned char bp_iterate(const bool s[RLBP_NUM_CHECKS], int w0,
                                llr_t tot[RLBP_NUM_VARS], bool hd[RLBP_NUM_VARS], int *t_exit) {
    llr_t m_vc[RLBP_NUM_EDGES];      // variable -> check messages
    llr_t m_cv[RLBP_NUM_EDGES];      // check -> variable messages
#pragma HLS ARRAY_PARTITION variable = m_vc complete
#pragma HLS ARRAY_PARTITION variable = m_cv complete
    const llr_t alpha = RLBP_ALPHA;
    int best = w0, stall = 0;
    unsigned char reason = EXIT_MAX_ITER;
    *t_exit = RLBP_MAX_ITER;

INIT_E:
    for (int e = 0; e < RLBP_NUM_EDGES; e++) {
#pragma HLS UNROLL
        m_vc[e] = PRIOR[E_VAR[e]];
    }

ITER:
    for (int t = 1; t <= RLBP_MAX_ITER; t++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = RLBP_MAX_ITER avg = 3
        // ---- check-node update: normalised min-sum, min1/min2 (all checks in parallel)
    CHECK:
        for (int c = 0; c < RLBP_NUM_CHECKS; c++) {
#pragma HLS UNROLL
            const int e0 = CHK_PTR[c], e1 = CHK_PTR[c + 1];
            llr_t min1 = RLBP_LLR_INF, min2 = RLBP_LLR_INF;
            unsigned char arg = 255;      // position (0..deg-1) of the smallest |message|
            bool par = s[c];
        CHECK_MIN:
            for (int d = 0; d < RLBP_MAX_CHK_DEG; d++) {
#pragma HLS UNROLL
                const int e = e0 + d;
                if (e < e1) {
                    llr_t x = m_vc[e];
                    llr_t mag = llr_abs(x);
                    par ^= (x < 0);
                    if (mag < min1) { min2 = min1; min1 = mag; arg = d; }
                    else if (mag < min2) { min2 = mag; }
                }
            }
        CHECK_OUT:
            for (int d = 0; d < RLBP_MAX_CHK_DEG; d++) {
#pragma HLS UNROLL
                const int e = e0 + d;
                if (e < e1) {
                    llr_t ext = (d == arg) ? min2 : min1;
                    llr_t msg = ext * alpha;
                    bool flip = par ^ (m_vc[e] < 0);
                    m_cv[e] = flip ? llr_t(-msg) : msg;
                }
            }
        }
        // ---- variable-node update + hard decision (all variables in parallel)
    VAR:
        for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
            const int p0 = VAR_PTR[v], p1 = VAR_PTR[v + 1];
            llr_t acc = 0;
        VAR_SUM:
            for (int d = 0; d < RLBP_MAX_VAR_DEG; d++) {
#pragma HLS UNROLL
                if (p0 + d < p1) acc += m_cv[VAR_EDGE[p0 + d]];
            }
            llr_t tv = acc + PRIOR[v];
            tot[v] = tv;
            hd[v] = tv < 0;
        VAR_OUT:
            for (int d = 0; d < RLBP_MAX_VAR_DEG; d++) {
#pragma HLS UNROLL
                if (p0 + d < p1) {
                    const int e = VAR_EDGE[p0 + d];
                    m_vc[e] = tv - m_cv[e];
                }
            }
        }
        // ---- syndrome check: parity of each check, then popcount ------------
        int unsat = 0;
    SYN:
        for (int c = 0; c < RLBP_NUM_CHECKS; c++) {
#pragma HLS UNROLL
            bool p = s[c];
        SYN_E:
            for (int d = 0; d < RLBP_MAX_CHK_DEG; d++) {
#pragma HLS UNROLL
                if (CHK_PTR[c] + d < CHK_PTR[c + 1]) p ^= hd[E_VAR[CHK_PTR[c] + d]];
            }
            unsat += p;
        }
        if (unsat < best) { best = unsat; stall = 0; } else { stall++; }

        // ---- exit decision: converged / RL halt / budget ---------------------
        bool done = unsat == 0;
        bool halted = !done && t >= RLBP_MIN_ITER && HALT_TABLE[rlbp_state(t, unsat, w0, stall)];
        if (done || halted || t == RLBP_MAX_ITER) {
            reason = done ? EXIT_CONVERGED : halted ? EXIT_HALTED : EXIT_MAX_ITER;
            *t_exit = t;
            break;
        }
    }
    return reason;
}

void rlbp_decode(const unsigned char syndrome[RLBP_NUM_CHECKS],
                 llr_t posterior[RLBP_NUM_VARS],
                 unsigned char hard[RLBP_NUM_VARS],
                 unsigned int *obs_bp,
                 unsigned char *iterations,
                 unsigned char *exit_reason) {
    // ---- interfaces ---------------------------------------------------------
    // Default block-level ap_ctrl_hs + ap_memory arrays: simplest for C/RTL
    // co-simulation. For a Zynq/Alveo system, switch to AXI, e.g.:
    //   #pragma HLS INTERFACE mode=s_axilite port=return
    //   #pragma HLS INTERFACE mode=s_axilite port=obs_bp   (etc. for scalars)
    //   #pragma HLS INTERFACE mode=bram  port=syndrome     (or m_axi / axis)

    llr_t tot[RLBP_NUM_VARS];        // posterior LLRs (written to the port once, at exit)
    bool hd[RLBP_NUM_VARS];          // hard decision
    bool s[RLBP_NUM_CHECKS];
#pragma HLS ARRAY_PARTITION variable = tot complete
#pragma HLS ARRAY_PARTITION variable = hd complete
#pragma HLS ARRAY_PARTITION variable = s complete

    // ---- syndrome weight; bits on checks no mechanism touches are unexplainable
    int w0 = 0;
    bool unexplainable = false;
LOAD_S:
    for (int c = 0; c < RLBP_NUM_CHECKS; c++) {
#pragma HLS PIPELINE II = 1
        s[c] = syndrome[c] & 1;
        w0 += s[c];
        if (s[c] && CHK_PTR[c] == CHK_PTR[c + 1]) unexplainable = true;
    }
INIT_V:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
        tot[v] = PRIOR[v];
        hd[v] = false;
    }
    // trivial: 0 iterations, converged.  unexplainable: 0 iterations, needs OSD.
    int t_exit = 0;
    unsigned char reason = w0 == 0 ? EXIT_TRIVIAL : EXIT_MAX_ITER;
    if (w0 != 0 && !unexplainable) reason = bp_iterate(s, w0, tot, hd, &t_exit);

    // ---- outputs: one write per output port -----------------------------------
    unsigned int ob = 0;
OUT_V:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS PIPELINE II = 1
        posterior[v] = tot[v];
        hard[v] = hd[v];
        if (hd[v]) ob ^= VAR_OBS[v];
    }
    *obs_bp = ob;
    *iterations = (unsigned char)t_exit;
    *exit_reason = reason;
}
