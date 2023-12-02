import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


class PolynomialRegressionDataModule(LightningDataModule):
    def __init__(
        self,
        order: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 1000,
        noise: float = 0.5,
        ood: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_data = PolynomialRegression(
            order=self.hparams.order,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.train_size,
            noise=self.hparams.noise,
            ood=False,
        )
        self.val_data = PolynomialRegression(
            order=self.hparams.order,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.val_size,
            noise=self.hparams.noise,
            ood=self.hparams.ood,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None)


class PolynomialRegression(IterDataPipe):
    def __init__(
        self,
        data_size: int,
        order: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        noise: float = 0.5,
        ood: bool = False,
    ) -> None:
        super().__init__()

        self.order = order
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.data_size = data_size
        self.noise = noise
        self.ood = ood

        std = torch.linspace(1, 1 / order**2, order)
        std = torch.cat([0.1 * torch.ones(1), std])  # Smaller y-intercepts
        self.w_dist = torch.distributions.normal.Normal(torch.zeros(order + 1), std)
        self.ws_fixed = self.w_dist.rsample((100,))  # For visualization purposes

    def sample_x(self, n_samples, n_context):
        x_c = torch.randn(n_samples, n_context)
        if self.ood:
            x_q_mean = 3 if np.random.random() > 0.5 else -3
            x_q = x_q_mean + torch.randn(n_samples) * 0.1
        else:
            x_q = x_c[:, 0] + torch.randn(n_samples) * 0.1
        return x_c, x_q

    @torch.inference_mode()
    def function(self, x, w):
        # x: (bsz, n_samples) or (bsz,)
        # w: (bsz, order + 1)
        x = torch.stack([x**i for i in range(self.order + 1)], dim=-1)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        w = w.unsqueeze(-1)
        y = torch.bmm(x, w).squeeze(-1)
        if y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y

    def get_batch(self, n_context=None, indices=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        if indices is None:
            w = self.w_dist.rsample((self.batch_size,))
        else:
            w = self.ws_fixed[indices]
        x_c, x_q = self.sample_x(w.shape[0], n_context)
        y_c, y_q = self.function(x_c, w), self.function(x_q, w)
        y_c = y_c + self.noise * torch.randn_like(y_c)
        x_c, y_c, x_q, y_q = (
            x_c.unsqueeze(-1),
            y_c.unsqueeze(-1),
            x_q.unsqueeze(-1),
            y_q.unsqueeze(-1),
        )
        return (x_c, y_c), (x_q, y_q), w

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()
