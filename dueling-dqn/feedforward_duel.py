import torch
import numpy as np


class FeedforwardDuel(torch.nn.Module):
    """
    The FeedforwardDuel class implements Dueling architecture of DQN.

    Parameters
    ----------
    input_size : int
        The variable specifies the input shape of the network.
    hidden_sizes: list
        The variable specifies the width of the hidden layers.
    device: str
        The variable specifies on which device the network is evaluated.
    """

    def __init__(self, input_size, hidden_sizes, output_size, device):
        super(FeedforwardDuel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.Tanh() for l in self.layers]
        self.A = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.V = torch.nn.Linear(self.hidden_sizes[-1], 1)
        self.device = device

    def forward(self, x):
        if self.device.type == 'cuda' and x.device.type != 'cuda':
            x = x.to(self.device)

        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        A = self.A(x)
        V = self.V(x)

        Q = torch.add(V, (A - A.mean(dim=-1, keepdim=True)))
        return Q

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class QFunction(FeedforwardDuel):
    """
    The FeedforwardDuel class implements Dueling architecture of DQN.

    Parameters
    ----------
    observation_dim : int
        The variable specifies the size of the observation vector.
    action_dim: int
        The variable specifies the size of the action vector.
    hidden_sizes: list
        The variable specifies the width of the hidden layers.
    learning_rate: float
        The variable specifies the learning rate for neural net.
    """

    def __init__(self, observation_dim, action_dim, device, hidden_sizes, learning_rate):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            device=device,
        )
        if device.type == 'cuda':
            self.cuda()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):
        targets = torch.from_numpy(targets).to(self.device).float()
        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

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

    def halve_learning_rate(self):
        # Method to halve learning rate (leading to better convergence)
        self.learning_rate /= 2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
