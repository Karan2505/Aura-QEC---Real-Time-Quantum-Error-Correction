"""RL-BP: belief propagation whose iteration budget is decided per shot by a
reinforcement-learned halting policy.

For each code: train the policy on shots from the virtual device itself
(fitted Q-iteration), then compare on held-out shots against plain BP+OSD
with a fixed iteration budget and against MWPM.  Iterations are the decoder's
compute cost; the policy's reward is the modelled decoder latency.
"""

import numpy as np

from aura.decoding import MinSumBP, RewardModel, TannerGraph, evaluate, shot_episodes, train_q_policy
from aura.lab import LabConfig, PatchConfig, VirtualLab

TRAIN, TEST = 8000, 8000

for code, rounds in (("surface:5", 5), ("bb:72", 3)):
    lab = VirtualLab(LabConfig(seed=3, patches=[PatchConfig(0, code)], cosmic_rays=False))
    lab.calibrate("oracle")
    ce = lab.compile(lab.experiment(rounds=rounds))
    pm = ce.manifest.patches[0]

    # shots from the device (its real noise, incl. leakage/crosstalk the DEM omits)
    lab.qpu.upload_settings(ce.settings)
    lab.qpu.load_program(ce.qpu_program)
    iq = lab.qpu.run_batch(TRAIN + TEST, t_start_ps=lab.sim.now)["iq"]
    bits = np.zeros((len(iq), ce.num_acq), np.uint8)
    for a, q in enumerate(ce.lowered.acq_qubit):
        bits[:, a] = ce.discriminators[q].classify(iq[:, a])[0]
    keep = [i for i, b in enumerate(pm.det_basis) if b == pm.basis]
    det, obs = pm.detection_events(bits)[:, keep], pm.observable_flips(bits)

    bp = MinSumBP(TannerGraph(pm.dem.restrict(keep)), max_iter=30)
    rm = RewardModel(iter_cost=1.0, osd_cost=20.0, fail_cost=2000.0)
    policy, rep = train_q_policy(bp, shot_episodes(det[:TRAIN], obs[:TRAIN]), episodes=TRAIN, reward=rm)
    print(f"\n=== {code}, {rounds} rounds: {rep.summary().splitlines()[0]}")

    ts, to = det[TRAIN:], obs[TRAIN:]
    rows = [("BP+OSD (30 it)", evaluate(bp, ts, to, None, rm)), ("RL-BP", evaluate(bp, ts, to, policy, rm))]
    for name, r in rows:
        print(f"  {name:15s} LER {r['ler']:.4f}  mean iterations {r['mean_iterations']:5.2f}  "
              f"p99 {r['p99_iterations']:4.0f}  OSD fallback {r['fallback_rate']:.3f}")
    base, rl = rows[0][1], rows[1][1]
    print(f"  -> RL-BP runs {100 * (1 - rl['total_iterations'] / base['total_iterations']):.0f} % fewer BP iterations")
    if code.startswith("surface"):
        r = lab.run(lab.experiment(rounds=rounds), shots=TEST, decoder="matching", mode="batch").patches[0]
        print(f"  {'MWPM':15s} LER {r.ler:.4f}  (reference; not applicable to BB codes)")
