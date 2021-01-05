import torch
import numpy as np


class Feedforward(torch.nn.Module):
    """
    The Feedforward class implements Dueling architecture of DQN.

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
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dueling = dueling
        self.device = device
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.Tanh() for l in self.layers]

        if dueling:
            self.A = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
            self.V = torch.nn.Linear(self.hidden_sizes[-1], 1)
        else:
            self.Q = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        if self.device.type == 'cuda' and x.device.type != 'cuda':
            x = x.to(self.device)

        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        if self.dueling:
            A = self.A(x)
            V = self.V(x)
            Q = torch.add(V, (A - A.mean(dim=-1, keepdim=True)))
        else:
            Q = self.Q(x)

        return Q

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class QFunction(Feedforward):
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
    device: torch.device
        The variable specifies on which device the network is evaluated.
    dueling: bool
        The variable specifies whether or not the architecture should implement a Dueling DQN.
    learning_rate: float
        The variable specifies the learning rate for neural net.
    lr_milestones: Iterable
        The variable specifies
    """

    def __init__(self, observation_dim, action_dim, device, hidden_sizes, dueling, learning_rate, lr_milestones):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            device=device,
            dueling=dueling
        )
        if device.type == 'cuda':
            self.cuda()
        self.learning_rate = learning_rate
        self.lr_milestones = lr_milestones
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=0.000001)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.5
        )
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
