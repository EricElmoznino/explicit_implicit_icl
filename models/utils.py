from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return self.model(x)
