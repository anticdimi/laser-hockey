import torch
import numpy as np


class Actor(torch.nn.Module):
    def __init__(self, num_inputs, n_actions, device, learning_rate, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()

        self.num_inputs = num_inputs
        self.n_actions = n_actions

        self.linear1 = torch.nn.Linear(num_inputs[0], hidden_sizes[0])
        self.linear2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = torch.nn.Linear(hidden_sizes[1], n_actions)

        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            x = x.cpu()
        x = x.reshape(-1, self.num_inputs[0])
        x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))  # torch.softmax(self.linear3(x),dim=-1)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()


class Critic(torch.nn.Module):
    def __init__(self, num_inputs, n_actions, device, learning_rate, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()

        self.num_inputs = num_inputs
        self.n_actions = n_actions

        self.linear1 = torch.nn.Linear(num_inputs[0] + n_actions, hidden_sizes[0])
        self.linear2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.linear3 = torch.nn.Linear(hidden_sizes[1], 1)

        self.device = device
        if device.type == 'cuda':
            self.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)

        self.loss = torch.nn.MSELoss()

    def forward(self, state, action):
        if not isinstance(state, np.ndarray):
            state = state.cpu()

        if not isinstance(action, np.ndarray):
            action = action.cpu()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        x = torch.cat([state, action], 1)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()
