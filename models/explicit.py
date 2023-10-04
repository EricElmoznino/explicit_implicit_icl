from abc import ABC, abstractmethod
import torch
from torch import nn
from models.utils import MLP


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
        cross_attention=False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
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
        return z


####################################################
################ Prediction Models #################
####################################################


class TransformerPrediction(nn.Module):
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
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
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
                dropout=0.0,
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
        z = self.context_embedding(z) if self.context_embedding else z
        x_q = self.value_embedding(x_q)
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
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim):
        super().__init__()
        self.mlp = MLP(x_dim + z_dim, hidden_dim, y_dim)

    def forward(self, z, x_q):
        x_q = torch.cat([z, x_q], dim=-1)
        y_q = self.mlp(x_q)
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
        x_q = torch.cat([x_q, torch.ones_like(x_q[:, :1])], dim=-1)
        if self.w_encoder:
            w = self.w_encoder(z)
        else:
            w = z
        w = w.reshape(-1, self.x_dim + 1, self.y_dim)
        y_q = (x_q.unsqueeze(1) @ w).squeeze(1)
        return y_q
