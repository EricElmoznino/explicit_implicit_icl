import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


class FiniteLinearDataModule(LightningDataModule):
    def __init__(
        self,
        x_dim=1,
        y_dim=1,
        min_context=3,
        max_context=20,
        batch_size=128,
        train_size=128,
        val_size=128,
        noise=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_data = FiniteLinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.train_size,
            noise=self.hparams.noise,
        )
        self.val_data = FiniteLinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.val_size,
            noise=self.hparams.noise,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None)


class FiniteLinear(IterDataPipe):
    def __init__(
        self,
        x_dim=1,
        y_dim=1,
        min_context=3,
        max_context=10,
        batch_size=128,
        data_size=1000,
        noise=0.0,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.data_size = data_size
        self.noise = noise
        self.x_dist = torch.distributions.uniform.Uniform(-1.0, 1.0)
        self.ws = torch.randn(data_size, x_dim + 1, y_dim)
        self.ws[:, -1] *= 0.1  # Smaller y-intercepts

    @torch.inference_mode()
    def function(self, x, w):
        # x: (bsz, n_samples, x_dim)
        # w: (bsz, x_dim + 1, y_dim)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        x = x.unsqueeze(-1)
        w = w.unsqueeze(1)
        y = (x * w).sum(dim=-2)
        return y

    def get_batch(self, n_context=None, indices=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        if indices is None:
            indices = np.random.choice(
                range(self.data_size), self.batch_size, replace=False
            )
        w = self.ws[indices]
        x = self.x_dist.sample((w.shape[0], n_context + 1, self.x_dim))
        y = self.function(x, w)
        y_noise = y + self.noise * torch.randn_like(y)
        x_c, y_c = x[:, :n_context], y_noise[:, :n_context]
        x_q, y_q = x[:, n_context], y[:, n_context]
        return (x_c, y_c), (x_q, y_q), w

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()
