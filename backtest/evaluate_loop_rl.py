# ============================================================
#  EVALUATION LOOP
# ============================================================
import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import random
from collections import defaultdict

from metrics.stats import compute_max_drawdown as max_drawdown
from util.running_mean_std import RunningMeanStd
# from util.seed_random import seed_random
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.models_io import load_RLmodels

def evaluate_loop_rl(
    spreads: dict,   # {pair_name: pd.Series}
    bocpd_params,
    vae_params,
    rl_params,
    joint_params,
    use_trained_rms=False, # For OOS this should be True
    load_dir="checkpoints",
    device="cpu",
    stop_loss_threshold=-0.02,
    stop_loss_penalty=0.001,
    seed: int = 42,
    use_bocpd=True,        # For Ablations
    use_vae=True           # For Ablations
):
    # seed_random(seed, device=device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pairs = list(spreads.keys())
    n_pairs = len(pairs)

    # ----------------------------
    # Align data
    # ----------------------------
    min_len = min(len(spreads[p]) for p in pairs)
    data = {p: spreads[p].values[:min_len] for p in pairs}

    # ----------------------------
    # Models (already trained)
    # ----------------------------
    state_window = joint_params["state_window"]

    encoder = VAEEncoder(
        input_dim=2,
        hidden_dim=vae_params["hidden_dim"],
        z_dim=vae_params["latent_dim"],
        seq_len=vae_params["vae_seq_len"]
    ).to(device)

    actor = Actor(
        state_dim=state_window,
        z_dim=rl_params["state_dim"],
        hidden_dim=rl_params["hidden_dim"],
        action_dim=1
    ).to(device)

    critic = Critic(
        state_dim=state_window,
        z_dim=rl_params["state_dim"],
        hidden_dim=rl_params["hidden_dim"],
        action_dim=1
    ).to(device)

    vae_opt = torch.optim.Adam(encoder.parameters(), lr=vae_params['lr'])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=rl_params['lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=rl_params['lr'])

    bocpd_cfg, meta = load_RLmodels(load_dir, actor, critic, encoder,
                                    actor_opt, critic_opt, vae_opt,
                                    device)

    actor.eval()
    encoder.eval()

    # ----------------------------
    # BOCPD per pair
    # ----------------------------
    # Extract the hazard rate from the loaded dictionary
    bocpd_hazard_default=bocpd_params['hazard']
    bocpd_hazard = bocpd_cfg.get("bocpd_hazard", bocpd_hazard_default)

    bocpd_models = {
        p: BOCPD(
            ConstantHazard(bocpd_hazard),
            StudentT(
                mu=bocpd_params["mu"],
                kappa=bocpd_params["kappa"],
                alpha=bocpd_params["alpha"],
                beta=bocpd_params["beta"]
            )
        )
        for p in pairs
    }

    rms = {p: RunningMeanStd() for p in pairs}
    if use_trained_rms==True:
      for p in pairs:
        file_name = f'rms_stats_{p}.npz'
        rms_stats = np.load(os.path.join(load_dir, file_name))
        # --- Assign back to rms object ---
        rms[p].mean = rms_stats["mean"]
        rms[p].var = rms_stats["var"]

    cp_weight = rl_params.get("cp_weight", 1.0)           # multiplies reward when change_prob high
    var_penalty = rl_params.get("var_penalty", 0.25)      # penalty * recent variance
    dd_penalty = rl_params.get("dd_penalty", 2.0)         # penalty multiplier for drawdown exceed
    dd_thr = rl_params.get("dd_threshold", 0.10)         # acceptable drawdown before penalty
    var_window = rl_params.get("var_window", 20)         # window for recent var
    tc_scale = joint_params.get("tc_scale", 0.3)         # scale for transaction cost (reduce penalty) # CHANGED

    # ----------------------------
    # State containers
    # ----------------------------
    state_returns = {p: [0.0] * state_window for p in pairs}
    cp_probs = {p: [] for p in pairs}
    rt_mle = {p: [] for p in pairs}
    norm_rets = {p: [] for p in pairs}
    cumulative_pnl = {p: 0.0 for p in pairs}
    raw_cum_pnl = {p: 0.0 for p in pairs}
    peak_raw_pnl = {p: 0.0 for p in pairs}
    raw_pnl = {p: [] for p in pairs}
    all_recons = {p: [] for p in pairs}
    recon_probs = {p: [] for p in pairs}
    last_action = np.zeros(n_pairs)
    portfolio_pnl = []
    results = defaultdict(dict)
    stop_loss_count = 0

    # ----------------------------
    # Evaluation loop
    # ----------------------------
    for t in range(min_len - 1):
        # reseed at each step to ensure deterministic behavior
        step_seed = seed + t + 2000
        if device.startswith("cuda") and torch.cuda.is_available():
            gen = torch.Generator(device='cuda')
        else:
            gen = torch.Generator(device='cpu')
        gen.manual_seed(step_seed)

        np.random.seed(step_seed)
        random.seed(step_seed)
        torch.manual_seed(step_seed)

        action = np.zeros(n_pairs)
        reward = np.zeros(n_pairs)
        pnl = np.zeros(n_pairs)

        step_pnl = 0.0
        for i, p in enumerate(pairs):
            cur_ret = data[p][t]
    
            norm_ret = rms[p].normalize(cur_ret)
            if use_bocpd:
                cp_prob, cp_flag = bocpd_models[p].update(float(norm_ret))
            else:
                cp_prob, cp_flag = 0.0, 0

            cp_probs[p].append(cp_prob)
            norm_rets[p].append(norm_ret)

            state_returns[p].append(norm_ret)
            state_returns[p] = state_returns[p][-state_window:]

            # ----------------------------
            # BUILD STATE VECTOR
            # ----------------------------
            state_t = torch.tensor(state_returns[p], dtype=torch.float32)[None, :].to(device)

            # ----------------------------
            # VAE INPUT
            # ----------------------------
            if not use_vae:
                z_detach = torch.zeros(1, rl_params["state_dim"], device=device)
                action_mean2 = actor(state_t, z_detach) # z_detach is (1, z_dim) from VAE output
            else:
                if len(norm_rets[p]) < vae_params["vae_seq_len"]:
                    z_detach = torch.zeros(1, rl_params["state_dim"], device=device)
                    action_mean2 = actor(state_t, z_detach) # z_detach is (1, z_dim) from VAE output
                else:
                    # ----------------------------
                    # VAE forward (NO BACKPROP)
                    # ----------------------------
                    vae_inp = np.stack(
                        [
                            norm_rets[p][-vae_params["vae_seq_len"]:],
                            cp_probs[p][-vae_params["vae_seq_len"]:]
                        ],
                        axis=-1
                    )[None, ...]
    
                    vae_inp_t = torch.tensor(vae_inp, dtype=torch.float32).to(device)
    
                    with torch.no_grad():
                        x_hat, mu, _, z_t  = encoder(vae_inp_t)
    
                        #-------------- VAE recon starts------------------------
                        recon_np = x_hat.detach().cpu().numpy()[0, :, 0]  # seq_len values
                        recon_cp = x_hat.detach().cpu().numpy()[0, :, 1]
                        # take last timestep reconstruction (corresponds to current idx)
                        recon_last_norm = recon_np[-1]
                        recon_last_denorm = recon_last_norm * math.sqrt(rms[p].var) + rms[p].mean
                        all_recons[p].append(recon_last_denorm)
                        recon_cp_last_norm = recon_cp[-1]
                        recon_probs[p].append(recon_cp_last_norm)
                        #-------------- VAE recon ends------------------------
                        # # Fix: Pass action_mu_tensor directly to torch.tanh
                        mu_detach = mu.detach()
                        action_scale = 1 #0.3
                        action_mean2 = actor(state_t, mu_detach) * action_scale
    
            action_mean = action_mean2
            action[i] = action_mean.detach().cpu().numpy().squeeze().item()
            
            # ----------------------------
            # REWARD (MARKET-NEUTRAL)
            # ----------------------------
            raw_reward = -action[i] * (rms[p].normalize(data[p][t + 1]) - rms[p].normalize(data[p][t]))
            pnl[i] = -action[i] * (data[p][t + 1] - data[p][t])
            raw_cum_pnl[p] += pnl[i]
            raw_pnl[p].append(raw_reward)

            # CHANGED: compute recent rolling variance for variance penalty
            if len(raw_pnl[p]) >= 2:
                recent_window = max(1, min(var_window, len(raw_pnl[p])))
                rolling_var = float(np.var(raw_pnl[p][-recent_window:]))
            else:
                rolling_var = 1e-8  # fallback

            if use_bocpd:
                # CHANGED: apply cp-weighting, variance penalty, drawdown penalty, and scale transaction cost
                # cp amplification
                reward[i] = raw_reward * (1.0 + cp_weight * cp_prob)  # CHANGED: amplify reward when CP high
            else:
                reward[i] = raw_reward

            # drawdown bookkeeping BEFORE applying stop-loss
            # raw_cum_pnl currently tracks normalized rewards sum
            # update peak for drawdown calc
            if raw_cum_pnl[p] > peak_raw_pnl[p]:
                peak_raw_pnl[p] = raw_cum_pnl[p]

            # compute drawdown as fraction of peak (safe denom)
            if peak_raw_pnl[p] > 1e-8:
                cur_dd = (peak_raw_pnl[p] - raw_cum_pnl[p]) / abs(peak_raw_pnl[p])
            else:
                cur_dd = 0.0

            # variance penalty (reduces reward when recent variance high)
            reward[i] = reward[i] - (var_penalty * rolling_var)

            # drawdown penalty (only applied when exceeding threshold)
            if cur_dd > dd_thr:
                reward[i] = reward[i] - dd_penalty * (cur_dd - dd_thr)

            # transaction cost: proportional to change in action magnitude
            tc = joint_params.get('transaction_cost', 0.0)
            trans_cost = tc * float(np.abs(action[i] - last_action[i]).sum())  # sum if vector action
            # CHANGED: scale down effective tc to avoid over-penalizing turnover
            reward[i] = reward[i] - (tc_scale * trans_cost)
            pnl[i] = pnl[i] - (tc_scale * trans_cost)

            # ----------------------------
            # STOP-LOSS
            # ----------------------------

            action_l2 = rl_params.get('action_l2', 0.1)
            reward[i] -= action_l2 * (action[i] ** 2)

            step_pnl += pnl[i]
            cumulative_pnl[p] += pnl[i]

            # STOP-LOSS CHECK
            stop_triggered = False
            if cumulative_pnl[p] <= stop_loss_threshold:
                reward[i] -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                action[i] = 0.0                         # force close position
                stop_triggered = True
                stop_loss_count += 1
                cumulative_pnl[p] = 0.0

            last_action[i] = action[i]

            # ----------------------------------
            # Store per-pair metrics
            # ----------------------------------
            results[p].setdefault("pnl", []).append(raw_reward)
            results[p].setdefault("position", []).append(action[i])
            results[p].setdefault("cp_prob", []).append(cp_prob)

        # portfolio_pnl.append(float(step_pnl/n_pairs))
        portfolio_pnl.append(step_pnl)

    # ===============================
    # Compute metrics
    # ===============================
    for p in pairs:
        change_probs, rt_mle, cp_flags = bocpd_models[p].results
        results["change_probs"][p] = change_probs
        results["rt_mle"][p] = rt_mle

    portfolio_pnl = np.array(portfolio_pnl)

    sharpe_ratio = (np.mean(portfolio_pnl) / (np.std(portfolio_pnl) + 1e-8)) * np.sqrt(252)

    print(f'Evaluation loop: Sharpe Ratio = {round(sharpe_ratio,3)}')
    metrics = {
        "portfolio_pnl": portfolio_pnl,
        "cumulative_pnl": np.cumsum(portfolio_pnl),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown(portfolio_pnl),
        "recons": all_recons
    }

    return metrics, results
