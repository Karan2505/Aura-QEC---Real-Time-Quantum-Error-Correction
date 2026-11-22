// C/RTL co-simulation testbench for rlbp_decode.
//
// Replays vectors.txt (written by export_hls.py from the Python golden model)
// and checks, per shot:
//   exit reason, BP iterations, BP logical prediction   -- exact (float build)
//   final logical prediction after OSD-0 fallback        -- exact (float build)
// then prints LER / mean & p99 iterations / fallback rate for the kernel.
//
// Returns 0 on success (Vitis HLS treats non-zero as a failed simulation).
// With -DRLBP_FIXED bit-exactness is not expected: the test passes if the
// kernel's LER is within RLBP_FIXED_LER_TOL of the golden LER.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rlbp_decoder.h"

#ifndef RLBP_FIXED_LER_TOL
#define RLBP_FIXED_LER_TOL 0.005
#endif

// ---------------------------------------------------------------- OSD-0
// Port of aura.decoding.osd.osd0: insert check-matrix columns in posterior
// order (most likely flipped first) into an incremental GF(2) basis; stop as
// soon as the syndrome is in the span.
struct Bits {
    std::vector<uint64_t> w;
    explicit Bits(int n = RLBP_NUM_CHECKS) : w((n + 63) / 64, 0) {}
    bool any() const { for (auto x : w) if (x) return true; return false; }
    int top() const {                       // index of the highest set bit, -1 if none
        for (int i = (int)w.size() - 1; i >= 0; i--)
            if (w[i]) return i * 64 + 63 - __builtin_clzll(w[i]);
        return -1;
    }
    void set(int i) { w[i / 64] |= 1ull << (i % 64); }
    void x(const Bits &o) { for (size_t i = 0; i < w.size(); i++) w[i] ^= o.w[i]; }
};

static unsigned osd0(const unsigned char syn[RLBP_NUM_CHECKS], const float post[RLBP_NUM_VARS]) {
    Bits target;
    for (int c = 0; c < RLBP_NUM_CHECKS; c++) if (syn[c]) target.set(c);
    if (!target.any()) return 0;
    std::vector<int> order(RLBP_NUM_VARS);
    for (int v = 0; v < RLBP_NUM_VARS; v++) order[v] = v;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return post[a] < post[b]; });
    std::vector<Bits> piv_vec(RLBP_NUM_CHECKS);
    std::vector<unsigned> piv_obs(RLBP_NUM_CHECKS, 0);
    std::vector<bool> has(RLBP_NUM_CHECKS, false);
    Bits s = target;
    unsigned s_obs = 0;
    for (int j : order) {
        Bits v;
        for (int k = VAR_PTR[j]; k < VAR_PTR[j + 1]; k++) v.set(E_CHK[VAR_EDGE[k]]);
        unsigned vo = VAR_OBS[j];
        int hb;
        while ((hb = v.top()) >= 0 && has[hb]) { v.x(piv_vec[hb]); vo ^= piv_obs[hb]; }
        if (hb < 0) continue;                     // dependent on more reliable columns
        piv_vec[hb] = v; piv_obs[hb] = vo; has[hb] = true;
        int hs;
        while ((hs = s.top()) >= 0 && has[hs]) { s.x(piv_vec[hs]); s_obs ^= piv_obs[hs]; }
        if (hs < 0) return s_obs;
    }
    return 0;                                     // unsolvable: predict no flip (as Python)
}

static unsigned parse_bits(const std::string &b) {   // "0110" -> bit k = char k
    unsigned x = 0;
    for (size_t k = 0; k < b.size(); k++) if (b[k] == '1') x |= 1u << k;
    return x;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "vectors.txt";
    std::ifstream f(path);
    if (!f) { std::printf("cannot open %s\n", path); return 2; }
    std::string line;
    do { std::getline(f, line); } while (f && (line.empty() || line[0] == '#'));
    int n, nc, nk;
    if (std::sscanf(line.c_str(), "%d %d %d", &n, &nc, &nk) != 3 || nc != RLBP_NUM_CHECKS || nk != RLBP_NUM_OBS) {
        std::printf("vectors.txt does not match rlbp_graph.h (regenerate both with export_hls.py)\n");
        return 2;
    }

    int bad_exit = 0, bad_iter = 0, bad_bp = 0, bad_final = 0;
    int err_kernel = 0, err_golden = 0, fallbacks = 0, halts = 0;
    long total_it = 0;
    std::vector<int> its;
    for (int i = 0; i < n; i++) {
        std::string sb, tob, bpob, fob;
        int g_exit, g_it;
        if (!(f >> sb >> tob >> g_exit >> g_it >> bpob >> fob)) { std::printf("truncated vectors at %d\n", i); return 2; }
        unsigned char syn[RLBP_NUM_CHECKS];
        for (int c = 0; c < RLBP_NUM_CHECKS; c++) syn[c] = sb[c] == '1';

        llr_t post[RLBP_NUM_VARS];
        unsigned char hard[RLBP_NUM_VARS];
        unsigned int obs = 0;
        unsigned char it = 0, ex = 0;
        rlbp_decode(syn, post, hard, &obs, &it, &ex);

        unsigned final_obs = obs;
        if (ex == EXIT_HALTED || ex == EXIT_MAX_ITER) {
            float pf[RLBP_NUM_VARS];
            for (int v = 0; v < RLBP_NUM_VARS; v++) pf[v] = (float)post[v];
            final_obs = osd0(syn, pf);
            fallbacks++;
        }
        halts += ex == EXIT_HALTED;
        total_it += it;
        its.push_back(it);
        unsigned truth = parse_bits(tob);
        err_kernel += final_obs != truth;
        err_golden += parse_bits(fob) != truth;

        bool e1 = ex != g_exit, e2 = it != g_it, e3 = obs != parse_bits(bpob), e4 = final_obs != parse_bits(fob);
        bad_exit += e1; bad_iter += e2; bad_bp += e3; bad_final += e4;
        if ((e1 || e2 || e3 || e4) && bad_exit + bad_iter + bad_bp + bad_final <= 20)
            std::printf("shot %d: exit %d/%d iters %d/%d bp_obs %u/%u final %u/%u (kernel/golden)\n", i, ex, g_exit,
                        it, g_it, obs, parse_bits(bpob), final_obs, parse_bits(fob));
    }
    std::sort(its.begin(), its.end());
    double p99 = 0;                               // numpy.percentile default (linear interpolation)
    if (!its.empty()) {
        double pos = 0.99 * (its.size() - 1);
        size_t lo = (size_t)pos, hi = std::min(lo + 1, its.size() - 1);
        p99 = its[lo] + (pos - lo) * (its[hi] - its[lo]);
    }
    double ler_k = (double)err_kernel / n, ler_g = (double)err_golden / n;

    std::printf("\nRL-BP kernel, %d shots (%d checks, %d vars, %d edges)\n", n, RLBP_NUM_CHECKS, RLBP_NUM_VARS,
                RLBP_NUM_EDGES);
    std::printf("  LER kernel %.4f   golden %.4f\n", ler_k, ler_g);
    std::printf("  mean iterations %.3f   p99 %.0f   fallback %.3f   halted %.3f\n", (double)total_it / n, p99,
                (double)fallbacks / n, (double)halts / n);
    std::printf("  mismatches vs golden: exit %d  iterations %d  bp_obs %d  final %d\n", bad_exit, bad_iter, bad_bp,
                bad_final);
#ifdef RLBP_FIXED
    bool pass = std::fabs(ler_k - ler_g) <= RLBP_FIXED_LER_TOL;
    std::printf("%s (fixed-point: LER within %.3f of golden)\n", pass ? "PASS" : "FAIL", RLBP_FIXED_LER_TOL);
#else
    bool pass = bad_exit + bad_iter + bad_bp + bad_final == 0;
    std::printf("%s (float: bit-exact against the Python model)\n", pass ? "PASS" : "FAIL");
#endif
    return pass ? 0 : 1;
}
