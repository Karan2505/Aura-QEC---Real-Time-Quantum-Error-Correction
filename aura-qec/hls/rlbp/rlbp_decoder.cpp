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
#if RLBP_MEM
static const mem_t MEM_TABLE[16] = RLBP_MEM_TABLE_INIT;
#endif

static llr_t llr_abs(llr_t x) {
#pragma HLS INLINE
    return x < 0 ? llr_t(-x) : x;
}

// mean of two LLRs; float: same operation order as numpy ((a + b) * 0.5f, bit-exact);
// fixed: one extra bit so the sum cannot saturate before halving
static llr_t llr_avg(llr_t a, llr_t b) {
#pragma HLS INLINE
#ifdef RLBP_FIXED
    ap_fixed<RLBP_LLR_W + 1, RLBP_LLR_I + 1, RLBP_LLR_Q, AP_SAT> s = a;
    s += b;
    return llr_t(s >> 1);
#else
    return (a + b) * 0.5f;
#endif
}

// ---- BP iterations with early exit and RL halting (runs only for a non-empty,
// explainable syndrome). Returns the exit reason; *t_exit = iterations spent.
static unsigned char bp_iterate(const bool s[RLBP_NUM_CHECKS], int w0,
                                llr_t tot[RLBP_NUM_VARS], bool hd[RLBP_NUM_VARS], int *t_exit) {
    llr_t m_vc[RLBP_NUM_EDGES];      // variable -> check messages
    llr_t m_cv[RLBP_NUM_EDGES];      // check -> variable messages
#pragma HLS ARRAY_PARTITION variable = m_vc complete
#pragma HLS ARRAY_PARTITION variable = m_cv complete
#if RLBP_POST_AVG == 2
    llr_t prev[RLBP_NUM_VARS];       // previous iteration's posterior
#pragma HLS ARRAY_PARTITION variable = prev complete
#endif
#if RLBP_MEM
    llr_t lam[RLBP_NUM_VARS];        // memory-mixed prior (OAM-BP)
    ap_uint<4> hist[RLBP_NUM_VARS];  // last 4 hard-decision flips
    bool hprev[RLBP_NUM_VARS];       // previous hard decision (local: hd is an output-only port in RTL)
#pragma HLS ARRAY_PARTITION variable = lam complete
#pragma HLS ARRAY_PARTITION variable = hist complete
#pragma HLS ARRAY_PARTITION variable = hprev complete
INIT_MEM:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
        lam[v] = PRIOR[v];
        hist[v] = 0;
        hprev[v] = false;
    }
#endif
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
        // Each iteration depends on the previous one; overlapping them (auto-pipelining) only lets a
        // speculative iteration t+1 run past a converged exit and corrupt the state, so keep it sequential.
#pragma HLS PIPELINE off
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
#if RLBP_MEM
            llr_t tv = acc + lam[v];
            {   // OAM-BP: per-variable memory strength from the local oscillation state
                const bool nh = tv < 0;
                const ap_uint<4> h = (ap_uint<4>)((hist[v] << 1) | ap_uint<1>(nh != hprev[v]));
                hist[v] = h;
                hprev[v] = nh;
                const int pc = (int)h[0] + (int)h[1] + (int)h[2] + (int)h[3];
                const int st = ((pc > 3 ? 3 : pc) * 2 + (nh ? 1 : 0)) * 2 + (t > RLBP_MEM_SPLIT ? 1 : 0);
                const mem_t gm = MEM_TABLE[st];
                // (1-g)*P + g*tv written as P + g*(tv-P): same exact value, and avoids the
                // (1 - g) pre-adder form, which HLS 2025.1 mapped to RTL that disagrees with C
                lam[v] = llr_t(PRIOR[v] + gm * (tv - PRIOR[v]));
            }
#else
            llr_t tv = acc + PRIOR[v];
#endif
#if RLBP_POST_AVG == 2
            prev[v] = (t == 1) ? tv : tot[v];      // t = 1: no previous iteration, mean = current
#endif
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
#if RLBP_POST_AVG == 2
POST_AVG:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
        tot[v] = llr_avg(prev[v], tot[v]);
    }
#endif
    return reason;
}

// ---- OSD-0 fallback: aura.decoding.osd.osd0 ---------------------------------
// Insert check-matrix columns in posterior order (most likely flipped first,
// ties by index = stable sort) into an incremental GF(2) basis; stop as soon as
// the syndrome lies in the span. Unsolvable syndrome -> predict no flip.
//   1. rank[v] = position of v in the stable ascending sort of post  (V cycles,
//      V comparators in parallel); order[rank[v]] = v                (V cycles);
//   2. walk r = 0..V-1: read the column of order[r] from the VAR_COL ROM, reduce
//      it against the pivots top-down, add it as a new pivot, then reduce the
//      syndrome; stop when the syndrome is cleared.
static syn_t var_col(int v) {
#pragma HLS INLINE
    syn_t c = 0;
    for (int w = 0; w < RLBP_COL_WORDS; w++) {
#pragma HLS UNROLL
        const int lo = 32 * w, hi = (32 * w + 31 < RLBP_NUM_CHECKS - 1) ? 32 * w + 31 : RLBP_NUM_CHECKS - 1;
        c.range(hi, lo) = VAR_COL[v * RLBP_COL_WORDS + w];
    }
    return c;
}

static unsigned int osd0(syn_t syn, const llr_t post[RLBP_NUM_VARS]) {
    unsigned short rank[RLBP_NUM_VARS];        // 16 bits: graphs have more than 255 variables
#pragma HLS ARRAY_PARTITION variable = rank complete
    unsigned short order[RLBP_NUM_VARS];         // order[r] = variable of rank r (small RAM)
OSD_RANK0:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
        rank[v] = 0;
    }
OSD_RANK:
    for (int u = 0; u < RLBP_NUM_VARS; u++) {
#pragma HLS PIPELINE II = 1
        const llr_t pu = post[u];
        for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
            if (pu < post[v] || (pu == post[v] && u < v)) rank[v]++;
        }
    }
OSD_PERM:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS PIPELINE II = 1
        order[rank[v]] = (unsigned short)v;
    }

    syn_t piv[RLBP_NUM_CHECKS];                // pivot vectors, indexed by their top bit
    unsigned int pobs[RLBP_NUM_CHECKS];        // observables of each pivot
#pragma HLS ARRAY_PARTITION variable = piv complete
#pragma HLS ARRAY_PARTITION variable = pobs complete
    syn_t has = 0;                             // which pivot slots are filled
    syn_t s = syn;
    unsigned int s_obs = 0, result = 0;
OSD_WALK:
    for (int r = 0; r < RLBP_NUM_VARS; r++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = RLBP_NUM_VARS avg = 20
        const int j = order[r];
        syn_t v = var_col(j);
        unsigned int vo = VAR_OBS[j];
        bool stop = false;
        int hb = -1;
    OSD_RED_V:
        for (int b = RLBP_NUM_CHECKS - 1; b >= 0; b--) {
#pragma HLS UNROLL
            if (!stop && v[b]) {
                if (has[b]) { v ^= piv[b]; vo ^= pobs[b]; }
                else { stop = true; hb = b; }
            }
        }
        if (hb < 0) continue;                  // dependent on more reliable columns
    OSD_INS:
        for (int b = 0; b < RLBP_NUM_CHECKS; b++) {
#pragma HLS UNROLL
            if (b == hb) { piv[b] = v; pobs[b] = vo; has[b] = 1; }
        }
        bool st = false;
    OSD_RED_S:
        for (int b = RLBP_NUM_CHECKS - 1; b >= 0; b--) {
#pragma HLS UNROLL
            if (!st && s[b]) {
                if (has[b]) { s ^= piv[b]; s_obs ^= pobs[b]; }
                else st = true;
            }
        }
        if (s == 0) { result = s_obs; break; }
    }
    return result;
}

void rlbp_decode(syn_t syndrome,
                 vbits_t *hard,
                 unsigned int *obs_bp,
                 unsigned int *obs,
                 unsigned char *iterations,
                 unsigned char *exit_reason) {
#ifdef RLBP_AXI
#pragma HLS INTERFACE mode = s_axilite port = syndrome
#pragma HLS INTERFACE mode = s_axilite port = hard
#pragma HLS INTERFACE mode = s_axilite port = obs_bp
#pragma HLS INTERFACE mode = s_axilite port = obs
#pragma HLS INTERFACE mode = s_axilite port = iterations
#pragma HLS INTERFACE mode = s_axilite port = exit_reason
#pragma HLS INTERFACE mode = s_axilite port = return
#endif
    llr_t tot[RLBP_NUM_VARS];        // posterior LLRs at exit
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
#pragma HLS UNROLL
        s[c] = syndrome[c];
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

    // ---- packed outputs
    vbits_t hb = 0;
    unsigned int ob = 0;
OUT_V:
    for (int v = 0; v < RLBP_NUM_VARS; v++) {
#pragma HLS UNROLL
        hb[v] = hd[v];
        if (hd[v]) ob ^= VAR_OBS[v];
    }
    unsigned int fin = ob;
    if (reason == EXIT_HALTED || reason == EXIT_MAX_ITER) fin = osd0(syndrome, tot);
    *hard = hb;
    *obs_bp = ob;
    *obs = fin;
    *iterations = (unsigned char)t_exit;
    *exit_reason = reason;
}
