import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        action = self.fc(state)
        return action

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        # self.fcs = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     # nn.BatchNorm1d(512),
        #     nn.ReLU()
        # )
        # self.fca = nn.Sequential(
        #     nn.Linear(action_dim + 64, 64),
        #     # nn.BatchNorm1d(256),
        #     nn.ReLU()
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(64, 1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        # x = self.fcs(state)
        action_value = self.fc(torch.cat([state, action], dim=-1))
        # action_value = self.out(x)
        return action_value

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

