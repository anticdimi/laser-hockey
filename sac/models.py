import sys

sys.path.insert(0, '.')
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from base.network import Feedforward


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, learning_rate, device, lr_milestones=[1000, 2000], lr_factor=0.5, loss='l2', hidden_sizes=[256, 256]):
        super(CriticNetwork, self).__init__()
        self.device = device
        layer_sizes = [input_dim[0] + n_actions] + hidden_sizes + [1]

        # Q1 architecture
        self.q1_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

        # Q2 architecture
        self.q2_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

        self.apply(weights_init_)

        if device.type == 'cuda':
            self.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )

        if loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f'Unkown loss function name: {loss}')

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = xu
        for l in self.q1_layers[:-1]:
            x1 = F.relu(l(x1))
        x1 = self.q1_layers[-1](x1)

        x2 = xu
        for l in self.q2_layers[:-1]:
            x2 = F.relu(l(x2))
        x2 = self.q2_layers[-1](x2)

        return x1, x2


class ActorNetwork(Feedforward):
    def __init__(self, input_dims, learning_rate, device, lr_milestones=[1000, 2000], lr_factor=0.5,
                 action_space=None, hidden_sizes=[256, 256], reparam_noise=1e-6):
        super().__init__(
            input_size=input_dims[0],
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device
        )

        self.reparam_noise = reparam_noise
        self.action_space = action_space
        n_actions = action_space.shape[0]

        self.mu = nn.Linear(hidden_sizes[-1], n_actions)
        self.sigma = nn.Linear(hidden_sizes[-1], n_actions)

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )

        if self.action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.
            ).to(self.device)

            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            ).to(self.device)
        else:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)

    def forward(self, state):
        prob = state
        for layer in self.layers:
            prob = F.relu(layer(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample(self, state):
        mu, sigma = self.forward(state)
        sigma = sigma.exp()
        normal = Normal(mu, sigma)

        # For the reparametrization trick
        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.reparam_noise)
        log_prob = log_prob.sum()
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu
