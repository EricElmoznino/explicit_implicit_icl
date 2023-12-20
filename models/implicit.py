from abc import ABC, abstractmethod
import torch
from torch import nn
import numpy as np


class ImplicitModel(ABC, nn.Module):
    def forward(self, x_c, y_c, x_q) -> tuple[torch.Tensor, None]:
        return self.yq_given_d(x_c, y_c, x_q), None

    @abstractmethod
    def yq_given_d(self, x_c, y_c, x_q) -> torch.Tensor:
        pass


class TransformerImplicit(ImplicitModel):
    def __init__(
        self,
        x_dim,
        y_dim,
        n_features,
        n_heads,
        n_hidden,
        n_layers,
        freq_enc: bool = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_features = n_features

        self.value_embedding = nn.Linear(x_dim + y_dim, n_features)
        self.query_embedding = nn.Parameter(
            torch.randn(n_features) / np.sqrt(n_features)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        self.prediction_head = nn.Linear(n_features, self.y_dim)

        self.init_weights()

    def init_weights(self):
        # Xavier uniform init for the transformer
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def yq_given_d(self, x_c, y_c, x_q):
        # x_c: (bsz, c_len, x_dim)
        # y_c: (bsz, c_len, y_dim)
        # x_q: (bsz, q_len, x_dim)

        bsz, c_len, _ = x_c.shape
        _, q_len, _ = x_q.shape

        xy_c = torch.cat([x_c, y_c], dim=-1)
        xy_u = torch.cat(
            [x_q, torch.zeros(bsz, q_len, self.y_dim).to(x_q.device)], dim=-1
        )
        xy = torch.cat([xy_c, xy_u], dim=1)

        xy = self.value_embedding(xy)
        xy[:, -q_len:] += self.query_embedding.view(1, 1, self.n_features)

        src_mask = torch.zeros(c_len + q_len, c_len + q_len).bool().to(xy.device)
        src_mask[:, -q_len:] = True

        y_q = self.encoder(xy, mask=src_mask)[:, -q_len:]
        y_q = self.prediction_head(y_q)
        return y_q
