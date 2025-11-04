import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Simple MLP Q-Network
    """
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)