import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# Actor-Critic networks (continuous actions)
# -------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_dim=128, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # action in [-1,1]
        )

    def forward(self, state, z):
        if state.dim() > 2:
            state = state.view(state.size(0), -1)  # flatten time dimension if needed
        x = torch.cat([state, z], dim=-1)
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc_state = nn.Linear(state_dim + z_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, z, action):
        h = torch.cat([state, z], dim=-1)
        h = F.relu(self.fc_state(h))
        h = torch.cat([h, action], dim=-1)
        h = F.relu(self.fc_action(h))
        q = self.fc_out(h)
        return q
