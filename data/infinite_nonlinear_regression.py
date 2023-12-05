import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from lightning import LightningDataModule
import warnings
from data.utils import BatchedLinear

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")

def init_weights(m, std=1.):
    if isinstance(m, BatchedLinear):
        torch.nn.init.normal_(m.weight, std=std)
        torch.nn.init.normal_(m.bias, std=std)

class InfiniteNonlinearRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        activation: str = "relu",
        layers: int = 1,
        hidden_dim: int = 32,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 1000,
        noise: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_data = InfiniteNonlinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            activation=self.hparams.activation,
            layers=self.hparams.layers,
            hidden_dim=self.hparams.hidden_dim,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.train_size,
            noise=self.hparams.noise,
        )
        self.val_data = InfiniteNonlinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            activation=self.hparams.activation,
            layers=self.hparams.layers,
            hidden_dim=self.hparams.hidden_dim,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.val_size,
            noise=self.hparams.noise,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None)

class InfiniteNonlinear(IterDataPipe):
    def __init__(
        self,
        data_size: int,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        activation: str = "relu",
        layers: int = 1,
        hidden_dim: int = 32,
        batch_size: int = 128,
        noise: float = 0.5,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_context = min_context
        self.max_context = max_context
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.batch_size = batch_size
        self.noise = noise

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.x_dist = torch.distributions.normal.Normal(0., 1.)

        self.model = self.get_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def get_model(self):
        layers = [BatchedLinear(self.x_dim, self.hidden_dim, self.batch_size), self.activation]
        for _ in range(self.layers - 1):
            layers.append(BatchedLinear(self.hidden_dim, self.hidden_dim, self.batch_size))
            layers.append(self.activation)
        layers.append(BatchedLinear(self.hidden_dim, self.y_dim, self.batch_size))
        model = torch.nn.Sequential(*layers)
        return model

    @torch.inference_mode()
    def get_parameters(self):
        w = []
        for name, param in self.model.named_parameters():
            w.append(param.view(self.batch_size, -1))
        w = torch.cat(w, dim=-1)
        return w

    @torch.inference_mode()
    def function(self, x):
        # x: (bsz, n_samples, x_dim)
        y = self.model(x)
        return y

    def get_batch(self, n_context=None, indices=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        
        self.model.apply(init_weights)
        w = self.get_parameters().view(self.batch_size, -1)

        x = self.x_dist.rsample((self.batch_size, 2 * n_context, self.x_dim))
        if torch.cuda.is_available():
            x = x.cuda()
        y = self.function(x)
        y = y + self.noise * torch.randn_like(y)

        x_c, y_c = x[:, :n_context], y[:, :n_context]
        x_q, y_q = x[:, n_context:], y[:, n_context:]
        return (x_c, y_c), (x_q, y_q), w

    def __len__(self) -> int:
        return self.data_size // self.batch_size

    def __iter__(self) -> torch.Tensor:
        for _ in range(len(self)):
            yield self.get_batch()