import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import torch
import numpy as np
from base.network import Feedforward


class QFeedforward(torch.nn.Module):
    """
    The QFeedforward class implements Dueling architecture of DQN.

    Parameters
    ----------
    input_size : int
        The variable specifies the input shape of the network.
    hidden_sizes: list
        The variable specifies the width of the hidden layers.
    device: torch.device
        The variable specifies on which device the network is evaluated.
    dueling: bool
        The variable specifies whether or not the architecture should implement a Dueling DQN.
    """

    def __init__(self, input_size, hidden_sizes, output_size, device, dueling):
        super(QFeedforward, self).__init__()
        self.dueling = dueling
        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.fc = torch.nn.Linear(input_size, 512)

        if dueling:
            self.pre_A = torch.nn.Linear(512, 512)
            self.pre_V = torch.nn.Linear(512, 512)

            self.A = torch.nn.Linear(512, output_size)
            self.V = torch.nn.Linear(512, 1)
        else:
            self.pre_Q = torch.nn.Linear(256, 128)
            self.Q = torch.nn.Linear(128, output_size)

    def forward(self, x):
        if self.device.type == 'cuda' and x.device.type != 'cuda':
            x = x.to(self.device)

        x = torch.nn.functional.relu(self.fc(x))

        if self.dueling:
            pre_A = torch.nn.functional.relu(self.pre_A(x))
            pre_V = torch.nn.functional.relu(self.pre_V(x))

            A = self.A(pre_A)
            V = self.V(pre_V)

            Q = torch.add(V, (A - A.mean(dim=-1, keepdim=True)))
        else:
            pre_Q = torch.nn.functional.relu(self.pre_Q(x))
            Q = self.Q(pre_Q)

        return Q

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class QFunction(QFeedforward):
    """
    The QFunction class implements Dueling architecture of DQN.

    Parameters
    ----------
    obs_dim : int
        The variable specifies the size of the observation vector.
    action_dim: int
        The variable specifies the size of the action vector.
    hidden_sizes: list
        The variable specifies the width of the hidden layers.
    device: torch.device
        The variable specifies on which device the network is evaluated.
    dueling: bool
        The variable specifies whether or not the architecture should implement a Dueling DQN.
    learning_rate: float
        The variable specifies the learning rate for neural net.
    lr_factor: float
        The variable specifies the learning rate scaling factor for neural net.
    lr_milestones: Iterable
        The variable specifies
    """

    def __init__(self, obs_dim, action_dim, device, hidden_sizes, dueling, learning_rate, lr_factor, lr_milestones):
        super().__init__(
            input_size=obs_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            device=device,
            dueling=dueling
        )

        self.learning_rate = learning_rate
        self.lr_milestones = lr_milestones
        self.lr_factor = lr_factor
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=self.lr_factor
        )
        self.loss = torch.nn.SmoothL1Loss(reduction='none')

    def fit(self, observations, actions, targets, weights):
        weights = torch.from_numpy(weights).to(self.device).float()
        targets = torch.from_numpy(targets).to(self.device).float()
        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)
        weighted_loss = loss * weights
        mean_weighted_loss = weighted_loss.mean()
        self.optimizer.zero_grad()
        mean_weighted_loss.backward()
        # for param in self.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return mean_weighted_loss.item(), pred.detach().numpy()

    def Q_value(self, observations, actions):
        # compute the Q value for the give actions
        pred = self.forward(torch.from_numpy(observations).to(self.device).float())
        return torch.gather(pred, 1, torch.from_numpy(actions).to(self.device).long())

    def maxQ(self, observations):
        # compute the maximal Q-value
        return np.max(self.predict(observations), axis=-1)

    def greedyAction(self, observations):
        # this computes the greedy action
        return np.argmax(self.predict(observations), axis=-1)
