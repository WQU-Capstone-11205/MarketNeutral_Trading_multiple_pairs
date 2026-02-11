import numpy as np
import itertools
import random
from typing import Dict, List, Any

from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from util.running_mean_std import RunningMeanStd
from tuning.hyperparameter_tuner import hyperparameter_tuner

class BOCPD_tuner(hyperparameter_tuner):
    default_bocpd_space = {
        "hazard": [5, 10, 15, 20, 50, 100, 250],
        "mu": [0, 1, 2],
        "kappa": [0.1, 0.3, 0.5, 1.0, 5.0, 10.0],
        "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
        "beta": [0.1, 0.3, 0.5, 0.8, 1.0, 5.0, 10.0]
    }

    best_bocpd_params = {
        "hazard": 20,
        "mu": 0,
        "kappa": 0.3,
        "alpha": 1.0,
        "beta": 0.8
    }

    def __init__(self, custom_bocpd_space: Dict[str, List[Any]]=None):
        """
        Args:
            bocpd_space: dict of BOCPD hyperparameter ranges
        """
        self.bocpd_space = BOCPD_tuner.default_bocpd_space.copy()
        if custom_bocpd_space:
            self.bocpd_space.update(custom_bocpd_space)
        self.best_bocpd_params = BOCPD_tuner.best_bocpd_params.copy()

    def tune(self, spreads, penalty_lambda=0.5):
        results = []
        pairs = list(spreads.keys())

        for cfg in super().sample_from_space(self.bocpd_space, n = 5):
            fold_scores = []
            
            for train, val in super().walk_forward_splits(spreads, 500, 125, 125):
                bocpd_models = {
                    p: BOCPD(
                        ConstantHazard(cfg["hazard"]),
                        StudentT(mu=cfg["mu"],kappa=cfg["kappa"],alpha=cfg["alpha"],beta=cfg["beta"])
                    )
                    for p in pairs
                }


                rms = {p: RunningMeanStd() for p in pairs}
                cp_probs = {p: [] for p in pairs}

                # -------------------------
                # TRAIN: adapt posterior
                # -------------------------
                for p in pairs:
                    for x in train[p]:
                        rms[p].update([x])
                        bocpd_models[p].update(rms[p].normalize(x))

                # -------------------------
                # VALIDATION: score CPs
                # -------------------------
                for p in pairs:
                    for x in val[p]:
                        cp, _ = bocpd_models[p].update(rms[p].normalize(x))
                        cp_probs[p].append(cp)

                # -------------------------
                # PAIR-WISE SCORES
                # -------------------------
                pair_scores = []
                for p in pairs:
                    if len(cp_probs[p]) < 5:
                        continue

                    p95 = np.percentile(cp_probs[p], 95)
                    mean_cp = np.mean(cp_probs[p])
                    score_p = p95 - penalty_lambda * mean_cp
                    pair_scores.append(score_p)

                if pair_scores:
                    fold_scores.append(np.median(pair_scores))

            if fold_scores:
                results.append((cfg, np.mean(fold_scores)))
                print(f'BOCPD tuning: {cfg} :: {round(np.mean(fold_scores),3)}')

        self.best_bocpd_params = max(results, key=lambda x: x[1])[0]
        print(f'BOCPD tuning complete: {self.best_bocpd_params}')
        return self.best_bocpd_params
