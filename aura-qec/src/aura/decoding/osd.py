"""
Ordered-statistics decoding, order 0 (OSD-0) -- the fallback for shots BP
does not resolve.

Mechanisms are ranked by BP posterior reliability (most likely to have
occurred first).  Columns of the check matrix are inserted in that order into
an incremental GF(2) basis (Python ints as bitsets over detectors, so a row
operation is a single XOR of a machine-word sequence).  Insertion stops as soon
as the syndrome lies in the span of the inserted columns: the solution
supported on the most reliable information set is unique, so stopping there is
exact OSD-0 and usually touches only a small prefix of the columns.
"""

from __future__ import annotations

import numpy as np

from .bp import TannerGraph


def osd0(graph: TannerGraph, syndrome: np.ndarray, posterior: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Return (observable flips (K,), number of columns examined, solved)."""
    target = 0
    for c in np.nonzero(syndrome)[0]:
        target |= 1 << int(c)
    k = graph.num_observables
    if target == 0:
        return np.zeros(k, np.uint8), 0, True
    order = np.argsort(posterior, kind="stable")       # most negative LLR = most likely flipped
    cols, cobs = graph.col_bits, graph.col_obs
    piv_vec: dict[int, int] = {}                        # leading bit -> reduced column combination
    piv_obs: dict[int, int] = {}                        # leading bit -> observable parity of that combination
    s, s_obs = target, 0
    examined = 0
    for j in order:
        examined += 1
        v, vo = cols[j], cobs[j]
        while v:
            hb = v.bit_length() - 1
            pv = piv_vec.get(hb)
            if pv is None:
                break
            v ^= pv
            vo ^= piv_obs[hb]
        if not v:
            continue                                     # dependent on more reliable columns
        hb = v.bit_length() - 1
        piv_vec[hb] = v
        piv_obs[hb] = vo
        while s:
            hs = s.bit_length() - 1
            pv = piv_vec.get(hs)
            if pv is None:
                break
            s ^= pv
            s_obs ^= piv_obs[hs]
        if not s:
            out = np.array([(s_obs >> i) & 1 for i in range(k)], np.uint8)
            return out, examined, True
    return np.zeros(k, np.uint8), examined, False
