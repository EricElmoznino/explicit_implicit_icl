from abc import ABC, abstractmethod
import torch
from torch import nn
from models.utils import MLP
import numpy as np


class ExplicitModel(ABC, nn.Module):
    def forward(self, x_c, y_c, x_q) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z_given_d(x_c, y_c)
        y_q = self.yq_given_z(z, x_q)
        return y_q, z

    @abstractmethod
    def z_given_d(self, x_c, y_c) -> torch.Tensor:
        pass

    @abstractmethod
    def yq_given_z(self, z, x_q) -> torch.Tensor:
        pass


####################################################
################# Explicit Models ##################
####################################################


class ExplicitModelWith(ExplicitModel):
    def __init__(self, context_model, prediction_model):
        super().__init__()
        self.context_model = context_model
        self.prediction_model = prediction_model

    def z_given_d(self, x_c, y_c):
        return self.context_model(x_c, y_c)

    def yq_given_z(self, z, x_q):
        return self.prediction_model(z, x_q)


####################################################
################## Context Models ##################
####################################################


class TransformerContext(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        n_features,
        n_heads,
        n_hidden,
        n_layers,
        z_dim=None,
        dropout=0.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = n_features if z_dim is None else z_dim
        self.n_features = n_features

        self.value_embedding = nn.Linear(x_dim + y_dim, n_features)
        self.context_embedding = nn.Parameter(torch.randn(n_features))
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        if self.z_dim != n_features:
            self.z_encoder = nn.Linear(n_features, z_dim)

        self.init_weights()

    def init_weights(self):
        # Xavier uniform init for the transformer
        for p in self.context_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_c, y_c):
        xy_c = x_c if y_c is None else torch.cat([x_c, y_c], dim=-1)
        xy_c = self.value_embedding(xy_c)
        c_token = (
            self.context_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(xy_c.shape[0], -1, -1)
        )
        z = torch.cat([c_token, xy_c], dim=1)
        mask = torch.zeros(
            xy_c.shape[1] + 1,
            xy_c.shape[1] + 1,
            dtype=torch.bool,
            device=z.device,
        )
        mask[:, 0] = True
        z = self.context_encoder(z, mask=mask)[:, 0]
        if self.z_dim != self.n_features:
            z = self.z_encoder(z)
        return z


####################################################
################ Prediction Models #################
####################################################


class TransformerPrediction(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        n_features,
        n_heads,
        n_hidden,
        n_layers,
        z_dim=None,
        dropout=0.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = n_features if z_dim is None else z_dim
        self.n_features = n_features

        self.value_embedding = nn.Linear(x_dim, n_features)
        if self.z_dim != n_features:
            self.context_embedding = nn.Linear(z_dim, n_features)
        else:
            self.context_embedding = None
        self.prediction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.prediction_head = nn.Linear(n_features, self.y_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.prediction_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x_q):
        _, q_len, _ = x_q.shape
        x_q = x_q.to(torch.float32)
        z = z.unsqueeze(1)
        z = self.context_embedding(z) if self.context_embedding else z
        x_q = self.value_embedding(x_q)
        src_mask = (1 - torch.eye(1 + q_len)).bool().to(x_q.device)
        src_mask[:, 0] = False
        pred_input = torch.cat([z, x_q], dim=1)
        y_q = self.prediction_encoder(pred_input, mask=src_mask)[:, -q_len:]
        y_q = self.prediction_head(y_q)
        return y_q


class MLPPrediction(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim):
        super().__init__()
        self.mlp = MLP(x_dim + z_dim, hidden_dim, y_dim)

    def forward(self, z, x_q):
        z = z.unsqueeze(1).repeat(1, x_q.shape[1], 1)
        x_q = torch.cat([z, x_q], dim=-1)
        y_q = self.mlp(x_q)
        return y_q


####################################################
############## Task-specific Models ################
####################################################


class RavenTransformerContext(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_hidden,
        n_layers,
        z_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = dim if z_dim is None else z_dim
        self.context_embedding = nn.Parameter(torch.randn(dim))
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        if self.z_dim != dim:
            self.z_encoder = nn.Linear(dim, z_dim)
        self.init_weights()

    def init_weights(self):
        # Xavier uniform init for the transformer
        for p in self.context_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_c, y_c):
        assert y_c is None
        c_token = (
            self.context_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(x_c.shape[0], -1, -1)
        )
        z = torch.cat([c_token, x_c], dim=1)
        z = self.context_encoder(z)[:, 0]
        if self.z_dim != self.dim:
            z = self.z_encoder(z)
        return z


class RavenTransformerPrediction(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_hidden,
        n_layers,
        z_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = dim if z_dim is None else z_dim
        if self.z_dim != dim:
            self.context_embedding = nn.Linear(z_dim, dim)
        else:
            self.context_embedding = None
        self.prediction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.init_weights()

    def init_weights(self):
        for p in self.prediction_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x_q):
        z = z.unsqueeze(1)
        z = self.context_embedding(z) if self.context_embedding else z
        pred_input = torch.cat([z, x_q], dim=1)
        y_q = self.prediction_encoder(pred_input)[:, -1:]
        return y_q


class RavenMLPPrediction(nn.Module):
    def __init__(self, dim, z_dim, hidden_dim):
        super().__init__()
        self.mlp = MLP(dim * 2 + z_dim, hidden_dim, dim)

    def forward(self, z, x_q):
        x_q = x_q.view(x_q.shape[0], 2 * x_q.shape[-1])
        x_q = torch.cat([z, x_q], dim=-1)
        y_q = self.mlp(x_q)
        return y_q.unsqueeze(1)


class RavenKnownPrediction(nn.Module):
    rule_applications = [
        lambda x: x[:, 0],
        lambda x: x[:, 1] - 2,
        lambda x: x[:, 1] - 1,
        lambda x: x[:, 1] + 1,
        lambda x: x[:, 1] + 2,
        lambda x: x[:, 0] - x[:, 1],
        lambda x: x[:, 0] + x[:, 1],
        lambda x: x.min(dim=-1).values,
        lambda x: x.max(dim=-1).values,
        lambda x: x[:, 1] + 2,
        lambda x: x[:, 1] + 1,
        lambda x: x[:, 1] - 2,
        lambda x: x[:, 1] - 1,
    ]

    def __init__(
        self,
        z_dim,
    ):
        super().__init__()
        self.z_dim = z_dim
        if z_dim != 4 * 13:
            self.rule_encoder = nn.Linear(z_dim, 4 * 13)
        else:
            self.rule_encoder = nn.Identity()

    def forward(self, z, x_q):
        rule_probs = self.rule_encoder(z).view(-1, 4, 13)
        rule_probs = torch.softmax(rule_probs, dim=-1)
        y_q = torch.stack(
            [
                torch.stack([f(x_q[:, :, i]) for i in range(4)], dim=-1)
                for f in self.rule_applications
            ],
            dim=-1,
        )
        y_q *= rule_probs
        y_q = y_q.sum(dim=-1)
        return y_q.unsqueeze(1)


class LinRegPrediction(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        if z_dim != (x_dim + 1) * y_dim:
            self.w_encoder = nn.Linear(z_dim, (x_dim + 1) * y_dim)
        else:
            self.w_encoder = None

    def forward(self, z, x_q):
        x_q = torch.cat([x_q, torch.ones_like(x_q[..., :1])], dim=-1)
        if self.w_encoder:
            w = self.w_encoder(z)
        else:
            w = z
        w = w.reshape(-1, self.x_dim + 1, self.y_dim)
        y_q = x_q @ w
        return y_q


class AffinePrediction(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        if z_dim != (x_dim + 1) * y_dim:
            self.w_encoder = nn.Linear(z_dim, (x_dim + 1) * y_dim)
        else:
            self.w_encoder = None

    def forward(self, z, x_q):
        x_q = torch.cat([x_q, torch.ones_like(x_q[..., :1])], dim=-1)
        if self.w_encoder:
            w = self.w_encoder(z)
        else:
            w = z
        w = w.reshape(-1, self.x_dim + 1, self.y_dim)
        y_q = x_q @ w
        return y_q


class SinRegPrediction(nn.Module):
    def __init__(self, x_dim, z_dim, n_freq, fixed_freq):
        super().__init__()
        self.fixed_freq = fixed_freq
        self.n_freq = n_freq
        self.x_dim = x_dim
        target_zdim = x_dim * n_freq if fixed_freq else 2 * x_dim * n_freq
        if z_dim != target_zdim:
            self.z_encoder = nn.Linear(z_dim, target_zdim)
        else:
            self.z_encoder = None
        self.freqs = None

    def set_freqs(self, freqs):
        assert self.fixed_freq
        self.freqs = freqs

    def forward(self, z, x_q):
        if self.fixed_freq:
            assert self.freqs is not None
            amplitudes = self.z_encoder(z) if self.z_encoder else z
            amplitudes = amplitudes.view(-1, self.x_dim, self.n_freq)
            freqs = self.freqs.expand(z.shape[0], -1, -1)
        else:
            amplitudes, freqs = torch.split(
                self.z_encoder(z) if self.z_encoder else z,
                self.x_dim * self.n_freq,
                dim=-1,
            )
            amplitudes, freqs = (
                amplitudes.view(-1, self.x_dim, self.n_freq),
                freqs.view(-1, self.x_dim, self.n_freq),
            )
        x = torch.sin(torch.einsum("bqd,bdf->bqdf", x_q, freqs))
        y = torch.einsum("bqdf,bdf->bq", x, amplitudes)
        y = y.unsqueeze(-1)
        return y


class ScrambledTransformerPrediction(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        z_dim,
        n_features,
        n_heads,
        n_hidden,
        n_layers,
        cross_attention=False,
        dropout=0.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.permutation = np.arange(z_dim)
        np.random.choice(self.permutation)

        self.n_features = n_features
        self.cross_attention = cross_attention

        self.value_embedding = nn.Linear(x_dim, n_features)
        if z_dim != n_features:
            self.context_embedding = nn.Linear(z_dim, n_features)
        else:
            self.context_embedding = None
        self.prediction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.prediction_head = nn.Linear(n_features, 1)

        self.init_weights()

    def init_weights(self):
        for p in self.prediction_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x_q):
        z = z[..., self.permutation]
        bsz, q_len, _ = x_q.shape
        z = z.unsqueeze(1)
        z = self.context_embedding(z) if self.context_embedding else z
        x_q = self.value_embedding(x_q)

        src_mask = (1 - torch.eye(1 + q_len)).bool().to(x_q.device)
        src_mask[:, 0] = False

        pred_input = torch.cat([z, x_q], dim=1)
        y_q = self.prediction_encoder(pred_input, mask=src_mask)[:, -q_len:]
        y_q = self.prediction_head(y_q)
        return y_q
