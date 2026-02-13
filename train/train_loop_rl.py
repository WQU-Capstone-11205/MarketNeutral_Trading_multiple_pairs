# train_loop.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json
import matplotlib.pyplot as plt
import math
import random
from collections import deque, namedtuple
from tqdm import trange
import pandas as pd
import copy

from util.running_mean_std import RunningMeanStd
from structural_break.bocpd import BOCPD
from structural_break.hazard import ConstantHazard
from structural_break.distribution import StudentT
from ml_dl_models.rnn_vae import VAEEncoder, vae_loss
from ml_dl_models.actor_critic import Actor
from ml_dl_models.actor_critic import Critic
from util.weighted_replay_buffer import WeightedReplayBuffer
from util.eval_strategy import evaluate_strategy
from util.models_io import save_RLmodels
# from util.seed_random import seed_random
from util.soft_update import soft_update
from structural_break.MultiPairBOCPD import MultiPairBOCPD


def train_loop_rl(
    spreads: dict,   # {pair_name: pd.Series}
    bocpd_params,
    vae_params,
    rl_params,
    joint_params,
    mode='train', #'train' | 'tune'
    num_epochs=20,
    save_dir="checkpoints",
    total_steps=10000,
    device="cpu",
    stop_loss_threshold=-0.02,
    stop_loss_penalty=0.001,
    seed=42,
    use_bocpd=True,        # For Ablations
    use_vae=True           # For Ablations
):
    """
    Multi-pair RL training loop with:
    - One BOCPD per pair
    - Shared VAE + Actor-Critic
    """
    # seed_random(seed, device=device)
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)

    pairs = list(spreads.keys())
    n_pairs = len(pairs)

    # ----------------------------
    # Align and stack data
    # ----------------------------
    min_len = min(len(spreads[p]) for p in pairs)
    data = {
        p: spreads[p].values[:min_len]
        for p in pairs
    }

    # ----------------------------
    # BOCPD PER PAIR
    # ----------------------------
    bocpd_models = {
        p: BOCPD(
            ConstantHazard(bocpd_params["hazard"]),
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

    # ----------------------------
    # MODELS (shared)
    # ----------------------------
    state_window = joint_params["state_window"]
    state_dim = state_window # * n_pairs

    actor = Actor(
        state_dim=state_dim,
        z_dim=rl_params["state_dim"],
        hidden_dim=rl_params["hidden_dim"],
        action_dim=1
    ).to(device)

    critic = Critic(
        state_dim=state_dim,
        z_dim=rl_params["state_dim"],
        hidden_dim=rl_params["hidden_dim"],
        action_dim=1
    ).to(device)

    encoder = VAEEncoder(
        input_dim=2,  # [return, cp_prob]
        hidden_dim=vae_params["hidden_dim"],
        z_dim=vae_params["latent_dim"],
        seq_len=vae_params["vae_seq_len"]
    ).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=rl_params["lr"])
    critic_opt = optim.Adam(critic.parameters(), lr=rl_params["lr"])
    vae_opt = optim.Adam(encoder.parameters(), lr=vae_params["lr"])

    target_actor = copy.deepcopy(actor).eval()
    target_critic = copy.deepcopy(critic).eval()

    buffers = {p: WeightedReplayBuffer(capacity=20000) for p in pairs}# if mode == "train" else None

    gamma = rl_params.get("gamma", 0.99)
    base_action_sigma = joint_params["base_action_sigma"]

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    if mode == 'tune':
        epochs = 1
    else:
        epochs = num_epochs
        # EARLY STOPPING PARAMETERS
        patience = rl_params.get("patience", 5)
        min_delta = rl_params.get("min_delta", 1e-4)
        es_counter = 0
        best_train_sharpe = -np.inf
        stopped_early = False

    vae_update_every = joint_params.get("update_every", 10)
    # CHANGED: reward-shaping hyperparams (tunable via rl_params or joint_params)
    cp_weight = rl_params.get("cp_weight", 1.0)           # multiplies reward when change_prob high
    var_penalty = rl_params.get("var_penalty", 0.25)      # penalty * recent variance
    dd_penalty = rl_params.get("dd_penalty", 2.0)         # penalty multiplier for drawdown exceed
    dd_thr = rl_params.get("dd_threshold", 0.10)         # acceptable drawdown before penalty
    var_window = rl_params.get("var_window", 20)         # window for recent var
    tc_scale = joint_params.get("tc_scale", 0.3)         # scale for transaction cost (reduce penalty) # CHANGED
    exploration_alpha = joint_params.get("exploration_alpha", 10.0)  # CHANGED: was 5.0 before

    for epoch in range(epochs):
        epoch_seed = seed + epoch
        # seed_random(seed + epoch, device=device)

        T = min(total_steps, min_len - 1)

        state = {p: [0.0] * state_window for p in pairs}
        cumulative_pnl = {p: 0.0 for p in pairs}
        raw_cum_pnl = {p: 0.0 for p in pairs}
        peak_raw_pnl = {p: 0.0 for p in pairs}
        cp_probs = {p: [] for p in pairs}
        vae_state = {p: [] for p in pairs}
        raw_pnl = {p: [] for p in pairs}
        stop_loss_count = 0
        total_recon, total_kl = 0, 0
        total_policy_loss, total_policy_cnt = 0.0, 0
        portfolio_pnl = []
        total_actions = {p: [] for p in pairs}
        total_mus = {p: [] for p in pairs}
        epoch_actions = {p: [] for p in pairs}
        epoch_mus = {p: [] for p in pairs}
        last_action = np.zeros(n_pairs)
        vae_loss_accum = 0.0
        vae_update_count = 0

        # Freeze VAE after warmup
        if mode == "train" and epoch >= 2:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        for t in range(T):
            # reseed per-step for any sampling/noise used during step
            step_seed = epoch_seed + (t * 100) + 1000
            # PyTorch generator for randn-like draws (per-device)
            if device.startswith("cuda") and torch.cuda.is_available():
                gen = torch.Generator(device='cuda')
            else:
                gen = torch.Generator(device='cpu')
            gen.manual_seed(step_seed)

            # keep numpy and python random deterministic for any sampling inside this step (e.g., buffer pushes)
            np.random.seed(step_seed)
            random.seed(step_seed)
            torch.manual_seed(step_seed)  # ensures CPU-side rng deterministic for code that uses torch.randn()

            action = np.zeros(n_pairs)
            reward = np.zeros(n_pairs)
            pnl = np.zeros(n_pairs)

            step_pnl = 0.0
            # ----------------------------
            # PER-PAIR UPDATE
            # ----------------------------
            for i, p in enumerate(pairs):
                raw_reward = 0.0
                cur_data = data[p][t]
                rms[p].update([cur_data])
                norm_ret = rms[p].normalize(cur_data)

                if use_bocpd:
                    cp_prob, cp_flag = bocpd_models[p].update(float(norm_ret))
                else:
                    cp_prob, cp_flag = 0.0, 0

                cp_probs[p].append(cp_prob)
                vae_state[p].append(norm_ret)

                state[p].append(norm_ret)
                state[p] = state[p][-state_window:]

                # ----------------------------
                # BUILD STATE VECTOR
                # ----------------------------
                state_t = torch.tensor(state[p], dtype=torch.float32)[None, :].to(device)

                # ----------------------------
                # VAE INPUT
                # ----------------------------
                if not use_vae:
                    z_detach = torch.zeros(1, rl_params["state_dim"], device=device)
                    action_mean2 = actor(state_t, z_detach)
                else:
                    if len(vae_state[p]) < vae_params["vae_seq_len"]:
                        z_detach = torch.zeros(1, rl_params["state_dim"], device=device)
                        action_mean2 = actor(state_t, z_detach) # z_detach is (1, z_dim) from VAE output
                    else:
                        vae_inp = np.stack(
                            [
                                vae_state[p][-vae_params["vae_seq_len"]:] ,
                                cp_probs[p][-vae_params["vae_seq_len"]:]
                            ],
                            axis=-1
                        )[None, ...]
                        vae_inp_t = torch.tensor(vae_inp, dtype=torch.float32).to(device)
    
                        x_hat, mu, logvar, z = encoder(vae_inp_t)
                        vae_loss_val, recon_loss, kl_loss = vae_loss(
                            vae_inp_t, x_hat, mu, logvar, kl_weight=vae_params["kl_wt"]
                        )
                        vae_loss_accum += vae_loss_val
                        vae_update_count += 1
    
                        total_recon += recon_loss
                        total_kl += kl_loss
    
                        # ----------------------------
                        # ACTION SELECTION
                        # ----------------------------
                        action_scale = 1.0 # 0.3
                        with torch.no_grad():
                            z_detach = mu.detach()
                            action_mean2 = actor(state_t, z_detach) * action_scale # z_detach is (1, z_dim) from VAE output

                # action_mean = torch.clamp(action_mean2, -0.7, 0.7)
                action_mean = action_mean2

                if mode == "train":
                    if use_bocpd:
                        # exploration scale increases with change_prob
                        # alpha=5 scaling
                        noise_scale=base_action_sigma * (1.0 + exploration_alpha * np.clip(cp_prob, 0.0, 0.8))
                    else:
                        noise_scale=base_action_sigma
                    noise = np.random.normal(scale=noise_scale)
                else:
                    noise = 0.0

                action[i] = action_mean.detach().cpu().numpy().squeeze().item() + noise
                epoch_actions[p].append(action_mean.item())
                # epoch_mus[p].append(mu.detach().cpu())
                epoch_mus[p].append(z_detach.cpu())

                # ----------------------------
                # REWARD (MARKET-NEUTRAL)
                # ----------------------------
                raw_reward = -action[i] * (rms[p].normalize(data[p][t + 1]) - rms[p].normalize(data[p][t])) # -ve sign prefix to action[i], is new change
                raw_pnl[p].append(raw_reward)

                pnl[i] = -action[i] * (data[p][t + 1] - data[p][t])
                raw_cum_pnl[p] += pnl[i]

                # CHANGED: compute recent rolling variance for variance penalty
                if len(raw_pnl[p]) >= 2:
                    recent_window = max(1, min(var_window, len(raw_pnl[p])))
                    rolling_var = float(np.var(raw_pnl[p][-recent_window:]))
                else:
                    rolling_var = 1e-8  # fallback

                # CHANGED: apply cp-weighting, variance penalty, drawdown penalty, and scale transaction cost
                # cp amplification
                if use_bocpd:
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
                tc = joint_params.get("transaction_cost", 0.0)
                trans_cost = tc * float(np.abs(action[i] - last_action[i]).sum())  # sum if vector action
                # CHANGED: scale down effective tc to avoid over-penalizing turnover
                reward[i] = reward[i] - (tc_scale * trans_cost)

                # For risk control under regime uncertainty
                action_l2 = rl_params.get("action_l2", 0.1)
                reward[i] -= action_l2 * (action[i] ** 2) ## new

                step_pnl += pnl[i]
                cumulative_pnl[p] += pnl[i]
                total_actions[p].append(action[i])
                # mu_detach = mu.detach().cpu().numpy().squeeze()
                mu_mean_detach = z_detach.mean()
                total_mus[p].append(mu_mean_detach) # Removed .item() here

                # STOP-LOSS CHECK
                executed_action = action[i]
                stop_triggered = False
                if cumulative_pnl[p] <= stop_loss_threshold:
                    reward[i] -= abs(stop_loss_penalty)   # penalize hitting stop-loss
                    action[i] = 0.0                         # force close position
                    stop_triggered = True
                    cumulative_pnl[p] = 0.0
                    stop_loss_count += 1

                last_action[i] = action[i]

                # ----------------------------
                # STORE TRANSITION
                # ----------------------------
                norm_next_state = rms[p].normalize(data[p][t + 1])
                next_state_vec = state[p][1:] + [norm_next_state]

                if mode == 'train':
                    if use_vae:
                        z_to_store = z_detach.cpu().numpy().astype(np.float32)
                    else:
                        z_to_store = np.zeros((1, rl_params["state_dim"]), dtype=np.float32)
                    buffers[p].push(
                        state_t.cpu().numpy().astype(np.float32),
                        np.array([executed_action], dtype=np.float32),
                        reward[i],
                        np.array(next_state_vec).astype(np.float32),
                        stop_triggered,
                        1.0,
                        z_to_store
                    )
                    # if change_prob large, upweight recent transitions
                    if cp_flag == 1:
                        buffers[p].upweight_recent(window=200, multiplier=joint_params["wt_multplier"])

            # ----------------------------
            # Update VAE periodically
            # ----------------------------
            if (
                use_vae
                and t % vae_update_every == 0
                and vae_update_count > 0
                and encoder.training
            ):
                vae_opt.zero_grad()
                (vae_loss_accum / vae_update_count).backward()
                vae_opt.step()

            vae_loss_accum = 0.0
            vae_update_count = 0

            # ----------------------------
            # UPDATE POLICY
            # ----------------------------
            if all(buffers[p].size() >= joint_params["buffer_size_updates"] for p in pairs) and t % 8 == 0:
                # reseed numpy/random before sampling to ensure consistent sampled indices
                np.random.seed(step_seed + 123456789)
                random.seed(step_seed + 123456789)

                batch_size = joint_params["sample_batch_size"]
                for p in pairs:
                    if buffers[p].size() < batch_size:
                        continue
                    batch = buffers[p].sample(batch_size)

                    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32).to(device)
                    actions = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32).to(device)
                    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(-1).to(device)
                    next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32).to(device)
                    dones = torch.tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(-1).to(device)
                    z_batch = torch.tensor(np.stack([b.seq_inp for b in batch]), dtype=torch.float32).to(device)
                    z_batch = z_batch.view(z_batch.size(0), -1)
                    values = critic(states.squeeze(1), z_batch, actions).to(device)

                    with torch.no_grad():
                        next_actions = target_actor(next_states.squeeze(1), z_batch)
                        next_values = target_critic(next_states.squeeze(1), z_batch, next_actions)
                        targets = rewards + gamma * (1 - dones) * next_values

                    critic_loss = nn.MSELoss()(values, targets)
                    # print(f'TD error = {round(critic_loss.item(),6)}')
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    pred_actions = actor(states.squeeze(1), z_batch)
                    actor_loss = -critic(states.squeeze(1), z_batch, pred_actions).mean()

                    # CHANGED: optionally add small L2 regularization on policy parameters to avoid collapse (if desired)
                    # It brings numerical stability when latent states shift
                    if rl_params.get('actor_l2', 0.0) > 0.0:
                        l2_reg = 0.0
                        for param in actor.parameters():
                            l2_reg += (param**2).sum()
                        actor_loss = actor_loss + rl_params['actor_l2'] * l2_reg

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()
                    total_policy_loss += actor_loss.item()
                    total_policy_cnt += 1

                    # Update target networks
                    soft_update(target_actor, actor, tau=rl_params['tau'])
                    soft_update(target_critic, critic, tau=rl_params['tau'])

            portfolio_pnl.append(float(step_pnl/n_pairs))

        avg_recon = total_recon / (T * n_pairs)
        avg_kl = total_kl / (T * n_pairs)

        if mode == 'train':
            avg_policy = total_policy_loss / total_policy_cnt
            print(f"Epoch {epoch:03d} | recon loss = {avg_recon:.3f} | kl loss = {avg_kl:.3f} | policy loss = {avg_policy:.3f} | Cumulative PnL = {sum(cumulative_pnl.values()):.3f}")

            # ============================================================
            #   Save models for best Sharpe ratio
            # ============================================================
            portfolio_pnl = np.array(portfolio_pnl)
            sharpe_ratio = (np.mean(portfolio_pnl) / (np.std(portfolio_pnl) + 1e-8)) * np.sqrt(252)


            # --- save best checkpoint ---
            if sharpe_ratio > best_train_sharpe + min_delta:
                best_train_sharpe = sharpe_ratio
                es_counter = 0  # reset patience counter
                meta = {"epoch": epoch, "recon loss": (round(avg_recon,3)), "kl loss": (round(avg_kl,3)), "policy loss" : (round(avg_policy,3))}
                bocpd_cfg = {"bocpd_hazard": bocpd_params["hazard"]}
                save_RLmodels(save_dir, actor, critic, encoder,
                                actor_opt, critic_opt, vae_opt,
                                bocpd_cfg, meta)
                print(f"Saved best models at epoch {epoch:03d} (Sharpe={sharpe_ratio:.3f})")
            else:
                es_counter += 1
                print(f"No improvement. Early stopping patience counter = {es_counter}/{patience}")
                if es_counter >= patience:
                    print(f"EARLY STOPPING TRIGGERED at epoch {epoch}")
                    stopped_early = True
                    break
        else:
            print(f"Epoch {epoch:03d} | recon loss = {avg_recon:.3f} | kl loss = {avg_kl:.3f} | Cumulative PnL = {sum(cumulative_pnl.values()):.3f}")
            meta = {"epoch": epoch, "recon loss": (round(avg_recon,3)), "kl loss": (round(avg_kl,3))}
            bocpd_cfg = {"bocpd_hazard": bocpd_params["hazard"]}
            save_RLmodels(save_dir, actor, critic, encoder,
                            actor_opt, critic_opt, vae_opt,
                            bocpd_cfg, meta)
            print(f"Saved models (Tune) at epoch {epoch:03d}")


    for p in pairs:
        file_name = f'rms_stats_{p}.npz'
        np.savez(os.path.join(save_dir, file_name), mean=rms[p].mean, var=rms[p].var)

    if mode == 'train':
        if stopped_early:
            print("Multi-pair RL training stopped early due to no improvement in Sharpe.")
        else:
            print("Multi-pair RL training completed all epochs.")
    else:
        print("Multi-pair RL training for tuning complete.")
        
# # Optional: quick test
# if __name__ == "__main__":
#     dummy_series = pd.Series(np.random.randn(2000),
#                              index=pd.date_range("2020-01-01", periods=2000))
#     actor, critic, encoder, bocpd = train_loop_rl(dummy_series)
#     print("train_loop ran successfully!")
