import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, dim, out_dim, n_hidden_layers=3):
        super(MLP, self).__init__()
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FreqEncoding(nn.Module):
    def __init__(self, n_freqs=64, max_freq=10):
        super().__init__()
        self.max_freq = max_freq
        self.n_freqs = n_freqs
        freqs = torch.arange(1, n_freqs + 1) / n_freqs * max_freq
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x_enc = x.unsqueeze(1) if x.ndim == 2 else x
        x_enc = torch.einsum("b c d, k -> b c d k", x_enc, self.freqs)
        x_enc = x_enc.view(*x.shape[:-1], -1)
        return x_enc
