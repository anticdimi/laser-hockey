import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')
import torch
import torch.nn as nn
from torch.distributions import Normal
from base.network import Feedforward


class CriticNetwork(Feedforward):
    def __init__(self, input_dims, n_actions, learning_rate, device, hidden_sizes=[256, 256]):
        super().__init__(
            input_size=input_dims[0] + n_actions,
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.q = torch.nn.Linear(hidden_sizes[-1], 1)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.loss = nn.MSELoss()
        self.activations = [torch.nn.ReLU() for _ in self.layers]

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = super().forward(x)

        q = self.q(x)
        return q


class ValueNetwork(Feedforward):
    def __init__(self, input_dims, learning_rate, device, hidden_sizes=[256, 256]):
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.v = torch.nn.Linear(hidden_sizes[-1], 1)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.loss = nn.MSELoss()

    def forward(self, state):
        value = super().forward(state)
        value = self.v(value)

        return value


class ActorNetwork(Feedforward):
    def __init__(self, input_dims, max_action, learning_rate, device,
                 n_actions, hidden_sizes=[256, 256], reparam_noise=1e-6):
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.reparam_noise = reparam_noise
        self.max_action = max_action

        self.mu = torch.nn.Linear(hidden_sizes[-1], n_actions)
        self.sigma = torch.nn.Linear(hidden_sizes[-1], n_actions)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.loss = nn.MSELoss()
        self.activations = [torch.nn.ReLU() for _ in self.layers]

    def forward(self, state):
        prob = super().forward(state)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample(self, state, reparam=True):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)

        actions = probs.rsample() if reparam else probs.sample()

        # From appendix
        action = torch.tanh(actions) * torch.tensor(self.max_action)[:actions.shape[0]].to(self.device)
        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum()

        return action, log_probs
