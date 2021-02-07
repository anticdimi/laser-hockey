import torch
import numpy as np


class Feedforward(torch.nn.Module):
    """
    The Feedforward class implements a base class for feedforward neural network.

    Parameters
    ----------
    input_size : int
        The variable specifies the input shape of the network.
    hidden_sizes: list
        The variable specifies the width of the hidden layers.
    device: torch.device
        The variable specifies on which device the network is evaluated.
    """

    def __init__(self, input_size, hidden_sizes, output_size, device):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device
        if device.type == 'cuda':
            self.cuda()

        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.ReLU() for l in self.layers]

    def forward(self, x):
        if self.device.type == 'cuda' and x.device.type != 'cuda':
            x = x.to(self.device)

        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32)).to(self.device)).cpu().numpy()
