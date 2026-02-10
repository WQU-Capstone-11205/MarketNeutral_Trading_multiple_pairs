import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from typing import Dict, List, Any

from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from util.running_mean_std import RunningMeanStd
from tuning.hyperparameter_tuner import hyperparameter_tuner

class VAE_tuner(hyperparameter_tuner):
    default_vae_space = {
        "input_dim": [2],
        "latent_dim": [12],
        "hidden_dim": [32, 64, 128, 256],
        "lr": [1e-2, 1e-3, 1e-4],
        "vae_seq_len": [1, 5, 25],
        "kl_wt": [0.0005, 0.0001, 0.005, 0.001, 0.01]
    }

    best_vae_params = {
        "input_dim": 2,
        "latent_dim": 12,
        "hidden_dim": 128,
        "lr": 1e-3,
        "vae_seq_len": 1,
        "kl_wt": 0.001
    }

    def __init__(self, custom_vae_space: Dict[str, List[Any]]=None):
        """
        Args:
            vae_space: dict of VAE hyperparameter ranges
        """
        self.vae_space = VAE_tuner.default_vae_space.copy()
        if custom_vae_space:
            self.vae_space.update(custom_vae_space)
        self.best_vae_params = VAE_tuner.best_vae_params.copy()

    def train_vae_only(
        self,
        train_spread: dict,
        val_spread: dict,
        bocpd_cfg: dict,
        vae_cfg: dict,
        num_epochs: int = 1, #5
        device: str = "cpu"
    ):
        """
        Train ONLY the VAE using BOCPD-generated change probabilities.
        RL is completely excluded.

        Returns
        -------
        float
            Mean validation VAE loss
        """

        train_pairs = list(train_spread.keys())

        # ----------------------------
        # Initialize BOCPD PER PAIR
        # ----------------------------
        bocpd_models_train = {
            p: BOCPD(
                ConstantHazard(bocpd_cfg["hazard"]),
                StudentT(
                    mu=bocpd_cfg["mu"],
                    kappa=bocpd_cfg["kappa"],
                    alpha=bocpd_cfg["alpha"],
                    beta=bocpd_cfg["beta"]
                )
            )
            for p in train_pairs
        }

        # -------------------------------
        # Running normalization (global)
        # -------------------------------
        rms_train = {p: RunningMeanStd() for p in train_pairs}

        # -------------------------------
        # Initialize VAE
        # -------------------------------
        vae = VAEEncoder(
            input_dim=vae_cfg["input_dim"],      # should be 2
            hidden_dim=vae_cfg["hidden_dim"],
            z_dim=vae_cfg["latent_dim"],
            seq_len=vae_cfg["vae_seq_len"]
        ).to(device)

        optimizer = optim.Adam(vae.parameters(), lr=vae_cfg["lr"])

        # ======================================================
        #                   TRAINING LOOP
        # ======================================================
        vae.train()

        for epoch in range(num_epochs):
            # print(f'epoch = {epoch}')
            train_losses = []
            for p, series in train_spread.items():
                data = series.values
                cp_probs = []
                norm_spread_train = []
                # print(f'Pair = {p}')

                for t in range(len(data)):
                    # print(f't = {t}')
                    cur_data = data[t]
                    rms_train[p].update([cur_data])
                    norm_data = rms_train[p].normalize(cur_data)
                    norm_spread_train.append(norm_data)

                    cp, _ = bocpd_models_train[p].update(norm_data)
                    cp_probs.append(cp)

                    # Skip until we have full sequence
                    if t < vae_cfg["vae_seq_len"] - 1:
                        continue

                    # Build sequence
                    seq_norm_data = norm_spread_train[-vae_cfg["vae_seq_len"]:] 
                    seq_cp = cp_probs[-vae_cfg["vae_seq_len"]:]
                    seq_inp = np.stack([seq_norm_data, seq_cp], axis=-1)[None, ...]
                    seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

                    x_hat, mu, logvar, _ = vae(seq_inp_t)
                    loss, _, _ = vae_loss(
                        seq_inp_t,
                        x_hat,
                        mu,
                        logvar,
                        kl_weight=vae_cfg["kl_wt"]
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

            print(f'epoch = {epoch} :: vae loss (train) = {round(float(np.mean(train_losses)), 3)}')

        # ======================================================
        #                   VALIDATION LOOP
        # ======================================================
        vae.eval()
        val_losses = []

        val_pairs = list(val_spread.keys())

        # ----------------------------
        # Initialize BOCPD PER PAIR
        # ----------------------------
        bocpd_models_val = {
            p: BOCPD(
                ConstantHazard(bocpd_cfg["hazard"]),
                StudentT(
                    mu=bocpd_cfg["mu"],
                    kappa=bocpd_cfg["kappa"],
                    alpha=bocpd_cfg["alpha"],
                    beta=bocpd_cfg["beta"]
                )
            )
            for p in val_pairs
        }

        # -------------------------------
        # Running normalization (global)
        # -------------------------------
        rms_val = {p: RunningMeanStd() for p in val_pairs}

        with torch.no_grad():
            for p, series in val_spread.items():
                data = series.values
                cp_probs = []
                norm_spread_val = []

                for t in range(len(data)):
                    cur_data = data[t]
                    rms_val[p].update([cur_data])
                    norm_data = rms_val[p].normalize(cur_data)
                    norm_spread_val.append(norm_data)

                    cp, _ = bocpd_models_val[p].update(norm_data)
                    cp_probs.append(cp)

                    if t < vae_cfg["vae_seq_len"] - 1:
                        continue

                    seq_norm_data = norm_spread_val[-vae_cfg["vae_seq_len"]:] 
                    seq_cp = cp_probs[-vae_cfg["vae_seq_len"]:]
                    seq_inp = np.stack([seq_norm_data, seq_cp], axis=-1)[None, ...]
                    seq_inp_t = torch.tensor(seq_inp, dtype=torch.float32).to(device)

                    x_hat, mu, logvar, _ = vae(seq_inp_t)
                    loss, _, _ = vae_loss(
                        seq_inp_t,
                        x_hat,
                        mu,
                        logvar,
                        kl_weight=vae_cfg["kl_wt"]
                    )

                    val_losses.append(loss.item())

            print(f'vae loss (validation) = {round(float(np.mean(val_losses)), 3)}')

        return float(np.mean(val_losses))

    def tune(self, spreads, bocpd_cfg, n_trials=10): # 20
        best_loss = np.inf
        best_cfg = None

        train_vae_spread, val_vae_spread, test_vae_spread = super().split_tvt(spreads)

        for cfg in super().sample_from_space(self.vae_space, n=n_trials):
            loss = self.train_vae_only(
                train_spread=train_vae_spread,
                val_spread=val_vae_spread,
                bocpd_cfg=bocpd_cfg,
                vae_cfg=cfg
            )

            vae_loss = np.mean(loss)
            if vae_loss < best_loss:
                best_loss = vae_loss
                best_cfg = cfg
            print(f'VAE cfg = {cfg} :: vae loss = {vae_loss}')

        print(f'VAE tuning complete: {best_cfg}')
        return best_cfg
