from abc import ABC, abstractmethod
import torch
from torch import nn
from models.utils import MLP, FreqEncoding


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
        cross_attention=False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = n_features if z_dim is None else z_dim
        self.n_features = n_features
        self.cross_attention = cross_attention

        self.value_embedding = nn.Linear(x_dim + y_dim, n_features)
        self.context_embedding = nn.Parameter(torch.randn(n_features))
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_features,
                nhead=n_heads,
                dim_feedforward=n_hidden,
                dropout=0.0,
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
        xy_c = torch.cat([x_c, y_c], dim=-1)
        xy_c = self.value_embedding(xy_c)
        c_token = (
            self.context_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(xy_c.shape[0], -1, -1)
        )
        z = torch.cat([c_token, xy_c], dim=1)
        if self.cross_attention:
            mask = torch.zeros(
                xy_c.shape[1] + 1,
                xy_c.shape[1] + 1,
                dtype=torch.bool,
                device=z.device,
            )
            mask[1:, :1] = True
        else:
            mask = None
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
        cross_attention=False,
        freq_enc: bool = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = n_features if z_dim is None else z_dim
        self.n_features = n_features
        self.cross_attention = cross_attention

        input_dim = x_dim * 64 if freq_enc else x_dim
        self.freq_enc = FreqEncoding(64) if freq_enc else nn.Identity()
        self.value_embedding = nn.Linear(input_dim, n_features)
        if self.z_dim != n_features:
            self.context_embedding = nn.Linear(z_dim, n_features)
        else:
            self.context_embedding = None
        self.prediction_encoder = nn.TransformerEncoder(
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
        for p in self.prediction_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x_q):
        z = self.context_embedding(z) if self.context_embedding else z
        x_q = self.value_embedding(self.freq_enc(x_q))
        pred_input = torch.stack([z, x_q], dim=1)
        if self.cross_attention:
            mask = torch.BoolTensor(
                [[False, True], [False, False]],
                device=pred_input.device,
            )
        else:
            mask = None
        y_q = self.prediction_encoder(pred_input, mask=mask)[:, -1]
        y_q = self.prediction_head(y_q)
        return y_q


class MLPPrediction(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim, freq_enc: bool = False):
        super().__init__()
        input_dim = x_dim * 64 if freq_enc else x_dim
        self.freq_enc = FreqEncoding(64) if freq_enc else nn.Identity()
        self.mlp = MLP(input_dim + z_dim, hidden_dim, y_dim)

    def forward(self, z, x_q):
        x_q = self.freq_enc(x_q)
        x_q = torch.cat([z, x_q], dim=-1)
        y_q = self.mlp(x_q)
        return y_q


class HyperMLPPrediction(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dims, freq_enc: bool = False):
        super().__init__()
        input_dim = x_dim * 64 if freq_enc else x_dim
        self.freq_enc = FreqEncoding(64) if freq_enc else nn.Identity()
        self.layer_sizes = [input_dim] + hidden_dims + [y_dim]
        self.layer_weight_shapes = list(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        )
        self.total_params = sum(
            [inp * out + out for inp, out in self.layer_weight_shapes]
        )
        assert z_dim == self.total_params

    def forward(self, z, x_q):
        x = self.freq_enc(x_q)
        i = 0
        for layer_idx, w_shape in enumerate(self.layer_weight_shapes):
            w_size, b_size = w_shape[0] * w_shape[1], w_shape[1]
            w = z[:, i : i + w_size].reshape(-1, *w_shape)
            b = z[:, i + w_size : i + w_size + b_size]
            x = torch.bmm(x.unsqueeze(1), w).squeeze(1) + b
            if layer_idx < len(self.layer_weight_shapes) - 1:
                x = torch.relu(x)
            i += w_size + b_size
        return x


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
        x_q = torch.cat([x_q, torch.ones_like(x_q[:, :1])], dim=-1)
        if self.w_encoder:
            w = self.w_encoder(z)
        else:
            w = z
        w = w.reshape(-1, self.x_dim + 1, self.y_dim)
        y_q = (x_q.unsqueeze(1) @ w).squeeze(1)
        return y_q


class PolyRegPrediction(nn.Module):
    def __init__(self, order, z_dim):
        super().__init__()
        self.order = order
        if z_dim != order + 1:
            self.w_encoder = nn.Linear(z_dim, order + 1)
        else:
            self.w_encoder = None

    def forward(self, z, x_q):
        x_q = torch.stack([x_q**i for i in range(self.order + 1)], dim=-1)
        if self.w_encoder:
            w = self.w_encoder(z)
        else:
            w = z
        w = w.unsqueeze(-1)
        y_q = torch.bmm(x_q, w).squeeze(1)
        return y_q


class SinRegPrediction(nn.Module):
    def __init__(self, freqs, z_dim):
        super().__init__()
        self.freqs = torch.FloatTensor(freqs)
        if z_dim != len(freqs):
            self.amplitudes_encoder = nn.Linear(z_dim, len(freqs))
        else:
            self.amplitudes_encoder = None

    def forward(self, z, x_q):
        x = torch.cat([torch.sin(x_q * f) for f in self.freqs], dim=-1)
        amplitudes = self.amplitudes_encoder(z) if self.amplitudes_encoder else z
        y = (x * amplitudes).sum(dim=-1, keepdim=True)
        return y
