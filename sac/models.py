import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
import torch
import torch.nn as nn
from torch.distributions import Normal
from base.network import Feedforward


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, n_actions, learning_rate, device, hidden_sizes=[256, 256]):
        super(CriticNetwork, self).__init__()
        self.device = device

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs[0] + n_actions, hidden_sizes[0])
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = nn.Linear(hidden_sizes[1], 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs[0] + n_actions, hidden_sizes[0])
        self.linear5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear6 = nn.Linear(hidden_sizes[1], 1)

        self.apply(weights_init_)

        if device.type == 'cuda':
            self.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = torch.relu(self.linear1(xu))
        x1 = torch.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.relu(self.linear4(xu))
        x2 = torch.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class ActorNetwork(Feedforward):
    def __init__(self, input_dims, max_action, learning_rate, device,
                 n_actions, action_space=None, hidden_sizes=[256, 256], reparam_noise=1e-6):
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.reparam_noise = reparam_noise
        self.max_action = max_action
        self.action_space = action_space

        self.mu = torch.nn.Linear(hidden_sizes[-1], n_actions)
        self.sigma = torch.nn.Linear(hidden_sizes[-1], n_actions)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.loss = nn.MSELoss()
        self.activations = [torch.nn.ReLU() for _ in self.layers]

        if self.action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.min) / 2.
            ).to(self.device)

            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            ).to(self.device)
        else:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)

    def forward(self, state):
        prob = super().forward(state)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample(self, state):
        mu, sigma = self.forward(state)
        sigma = sigma.exp()
        normal = Normal(mu, sigma)

        # For reparametrization trick
        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum()
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu
