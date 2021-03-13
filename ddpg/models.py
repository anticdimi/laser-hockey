import torch
import numpy as np


class Actor(torch.nn.Module):
    def __init__(self, num_inputs, n_actions, device, learning_rate, lr_milestones, lr_factor=0.5,
                 hidden_sizes=[256, 256]):
        super(Actor, self).__init__()

        self.num_inputs = num_inputs
        self.n_actions = n_actions / 2
        layer_sizes = [num_inputs[0]] + hidden_sizes + [4]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            x = x.cpu()
        x = x.reshape(-1, self.num_inputs[0])
        x = torch.FloatTensor(x).to(self.device)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = torch.tanh(self.layers[-1](x))
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class Critic(torch.nn.Module):
    def __init__(self, num_inputs, n_actions, device, learning_rate, lr_milestones, lr_factor=0.5,
                 hidden_sizes=[256, 256]):
        super(Critic, self).__init__()

        self.num_inputs = num_inputs
        self.n_actions = n_actions

        layer_sizes = [num_inputs[0] + 4] + hidden_sizes + [1]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])

        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )
        self.loss = torch.nn.MSELoss()

    def forward(self, state, action):
        if not isinstance(state, np.ndarray):
            state = state.cpu()

        if not isinstance(action, np.ndarray):
            action = action.cpu()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        x = torch.cat([state, action], 1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class TwinCritic(torch.nn.Module):
    def __init__(self, num_inputs, n_actions, device, learning_rate, lr_milestones, lr_factor=0.5,
                 hidden_sizes=[256, 256, 256]):
        super(TwinCritic, self).__init__()

        self.num_inputs = num_inputs
        self.n_actions = n_actions

        layer_sizes = [num_inputs[0] + 4] + hidden_sizes + [1]
        self.layers1 = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layers2 = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=lr_factor
        )
        self.loss = torch.nn.MSELoss()

    def Q1(self, state, action):
        if not isinstance(action, np.ndarray):
            action = action.cpu()
        state = torch.FloatTensor(state.cpu()).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        x1 = torch.cat([state, action], 1)
        for layer in self.layers1[:-1]:
            x1 = torch.relu(layer(x1))
        x1 = self.layers1[-1](x1)
        return x1

    def forward(self, state, action):
        if not isinstance(action, np.ndarray):
            action = action.cpu()

        x1 = self.Q1(state, action)

        state = torch.FloatTensor(state.cpu()).to(self.device)
        action = torch.FloatTensor(action.cpu()).to(self.device)
        x2 = torch.cat([state, action], 1)
        for layer in self.layers2[:-1]:
            x2 = torch.relu(layer(x2))
        x2 = self.layers2[-1](x2)
        return x1, x2

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()
