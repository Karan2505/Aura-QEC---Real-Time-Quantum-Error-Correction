"""
Iterative decoding for the AURA-QEC decoder side (Layer 3 reference models).

* ``TannerGraph`` / ``MinSumBP`` -- batched min-sum belief propagation on the
  compiler's detector error model, with convergence early exit;
* ``osd0``                       -- ordered-statistics fallback for unresolved shots;
* ``QHaltingPolicy`` / ``train_q_policy`` -- reinforcement-learned halting
  policy ("RL-BP") that stops hopeless BP runs early.
"""

from .bp import (EXIT_CONVERGED, EXIT_HALTED, EXIT_MAX_ITER, EXIT_TRIVIAL, BPResult, IterFeatures, MinSumBP,
                 TannerGraph)
from .osd import osd0
from .policy import (QHaltingPolicy, RewardModel, TrainReport, Trajectories, collect_trajectories, dem_episodes,
                     evaluate, finish_with_fallback, fitted_q_iteration, shot_episodes, train_q_policy)

__all__ = ["TannerGraph", "MinSumBP", "BPResult", "IterFeatures", "osd0", "QHaltingPolicy", "RewardModel",
           "TrainReport", "Trajectories", "collect_trajectories", "fitted_q_iteration", "train_q_policy",
           "dem_episodes", "shot_episodes", "evaluate", "finish_with_fallback",
           "EXIT_TRIVIAL", "EXIT_CONVERGED", "EXIT_HALTED", "EXIT_MAX_ITER"]
