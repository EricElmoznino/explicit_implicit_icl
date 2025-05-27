import torch
from torch import nn
from torch.nn.functional import one_hot

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
    

class Embedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        pass


class AlchemyEmbedding(Embedding):
    QUERY_MASK = 1337

    def __init__(self, dim: int) -> None:
        super().__init__(dim)

        assert (
            dim % 3
        ) == 0, (
            "The embedding dimension must be divisible by 3 because we concatenate [stone, potion, stone']"
        )
        dim_ = dim // 3
        

        self.dim = dim

        self.stone_w = nn.Linear(13, dim_, bias=False)
        self.pot_w = nn.Linear(6, dim_, bias=False)
        self.start_pe = nn.Parameter(data=torch.rand(dim_))
        self.pot_pe = nn.Parameter(data=torch.rand(dim_))
        self.end_pe = nn.Parameter(data=torch.rand(dim_))
        self.query_e = nn.Parameter(data=torch.rand(dim_))


    def forward(self, batch, return_query_idx=False):
        """
        Args:
            batch (_type_): Size [N, L, 9]

        Returns:
            _type_: _description_
        """
        
        # Find the index of the query with the QUERY_MASK indentifier. Replace the mask with [0,0,0,0] for the embedding to work
        query_idx = (
            torch.all(batch[:, :, -4:] == AlchemyEmbedding.QUERY_MASK, dim=-1).int().argmax(-1)
        )
        batch[torch.arange(batch.shape[0], device=batch.device), query_idx, -4:] = 0

        # Split the transitions into (stone, reward, pot, stone, reward)
        s, r, p, s_, r_ = torch.split(batch, [3, 1, 1, 3, 1], dim=-1)

        # Turn everything to onehots
        s, r, p, s_, r_ = (
            one_hot(s, 3).flatten(2).float(),
            one_hot(r, 4).flatten(2).float(),
            one_hot(p, 6).flatten(2).float(),
            one_hot(s_, 3).flatten(2).float(),
            one_hot(r_, 4).flatten(2).float(),
        )

        # Cat reward and stone
        s, p, s_ = (torch.cat((s, r), dim=-1), p, torch.cat((s_, r_), dim=-1))

        # Affine transform
        s, p, s_ = (
            self.stone_w(s) + self.start_pe,
            self.pot_w(p) + self.pot_pe,
            self.stone_w(s_) + self.end_pe,
        )
        
        # Modify the query stone with a special embedding
        s_[torch.arange(s_.shape[0], device=s_.device), query_idx] = (
            self.query_e + self.end_pe
        )
        
        out = torch.cat([s, p, s_], dim=-1)

        if return_query_idx:
            return out, query_idx
        else:
            return out
