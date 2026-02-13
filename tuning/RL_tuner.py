import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from typing import Dict, List, Any

from train.train_loop_rl import train_loop_rl
from backtest.evaluate_loop_rl import evaluate_loop_rl
from util.running_mean_std import RunningMeanStd
from tuning.hyperparameter_tuner import hyperparameter_tuner

class RL_tuner(hyperparameter_tuner):
    default_rl_space = {
        "state_dim": [12],
        "action_dim": [1],
        "hidden_dim": [64, 128, 256, 512],
        "actor_l2": [1e-6, 1e-5, 1e-4, 5e-4, 1e-3],
        "lr": [1e-4, 5e-5, 1e-5, 5e-6],
        "gamma": [0.95, 0.99, 1.0],
        "action_l2": [0.001, 0.01, 0.1, 0.2],
        "cp_weight": [0.05, 0.08, 0.10, 0.15, 0.2],
        "var_penalty": [1e-5, 5e-4, 1e-4, 5e-3],
        "var_window": [5, 10, 20, 50, 100],
        "dd_penalty": [0.001, 0.005, 0.01, 0.10, 0.25, 0.5],
        "dd_threshold": [0.05, 0.10, 0.2, 0.3, 0.5, 0.8],
        "actor_lr": [1e-3, 1e-4, 5e-5, 1e-5, 0],
        "tau": [0.001, 0.005, 0.05]
    }

    default_joint_space = {
        "state_window": [25, 50, 100],
        "base_action_sigma": [0.01, 0.1, 0.3],
        "wt_multplier": [1.5, 1.8, 2.0],
        "buffer_size_updates": [16, 64, 128, 256],
        "sample_batch_size": [8, 16, 64, 128],
        "transaction_cost": [0.001, 0.01, 0.1],
        "tc_scale": [0.2, 0.3, 0.5, 0.8, 1.0],
        "exploration_alpha": [2.0, 5.0, 6.5, 10.0],
        "update_every": [10, 20, 50]
    }

    best_rl_params = {
        "state_dim": 16,
        "action_dim": 1,
        "hidden_dim": 64,
        "actor_l2": 1e-5,
        "lr": 1e-5,
        "gamma": 0.99,
        "action_l2": 0.01,
        "cp_weight": 0.08,
        "var_penalty": 1e-5,
        "var_window": 20,
        "dd_penalty": 0.01,
        "dd_threshold": 0.10,
        "actor_lr": 1e-3,
        "tau": 0.001
    }

    best_joint_params = {
        "state_window": 25,
        "base_action_sigma": 0.3,
        "wt_multplier": 2.0,
        "buffer_size_updates": 256,
        "sample_batch_size": 64,
        "transaction_cost": 0.001,
        "tc_scale": 1.0,
        "exploration_alpha": 6.5,
        "update_every": 20
    }

    def __init__(self, custom_rl_space: Dict[str, List[Any]]=None,
                 custom_joint_space: Dict[str, List[Any]]=None, seed=42):
        """
        Args:
            custom_rl_space: dict of RL hyperparameter ranges
            custom_joint_space: dict for joint tuning of BOCPD, VAE, and RL
        """
        super().__init__(seed)
        self.rl_space = RL_tuner.default_rl_space.copy()
        if custom_rl_space:
            self.rl_space.update(custom_rl_space)
        self.joint_space = RL_tuner.default_joint_space.copy()
        if custom_joint_space:
            self.joint_space.update(custom_joint_space)
        self.best_rl_params = RL_tuner.best_rl_params.copy()
        self.best_joint_params = RL_tuner.best_joint_params.copy()

    def tune(self, spreads, bocpd_cfg, vae_cfg, n_trials=5): #25
        best_score = -np.inf
        best_cfg = None

        for rl_cfg in super().sample_from_space(self.rl_space, n_trials):
            for joint_cfg in super().sample_from_space(self.joint_space, n_trials): #3
                scores = []

                for train, val in super().walk_forward_splits(spreads, 500, 125, 125):
                    train_loop_rl(
                        spreads=train,
                        bocpd_params=bocpd_cfg,
                        vae_params=vae_cfg,
                        rl_params=rl_cfg,
                        joint_params=joint_cfg,
                        mode="tune",
                        num_epochs=1
                    )

                    metrics, _ = evaluate_loop_rl(
                        spreads=val,
                        bocpd_params=bocpd_cfg,
                        vae_params=vae_cfg,
                        rl_params=rl_cfg,
                        joint_params=joint_cfg
                    )
                    scores.append(metrics["sharpe_ratio"])

                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_cfg = (rl_cfg, joint_cfg)

                print(f'RL and Joint CFG = {best_cfg} :: score = {round(score,3)}')

        print(f'RL and Joint CFG = {best_cfg} :: score = {round(score,3)}')
        return best_cfg
