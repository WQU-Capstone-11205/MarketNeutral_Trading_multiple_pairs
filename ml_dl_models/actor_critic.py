import torch
import torch.nn as nn

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
    def __init__(self, state_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        return self.net(x)
