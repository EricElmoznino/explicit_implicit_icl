import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")

class InfiniteLinearClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 1000,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_data = InfiniteLinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.train_size,
            temperature=self.hparams.temperature,
        )
        self.val_data = InfiniteLinear(
            x_dim=self.hparams.x_dim,
            y_dim=self.hparams.y_dim,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            batch_size=self.hparams.batch_size,
            data_size=self.hparams.val_size,
            temperature=self.hparams.temperature,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None)

class InfiniteLinear(IterDataPipe):
    def __init__(
        self,
        data_size: int,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.temperature = temperature
        self.x_dist = torch.distributions.normal.Normal(0., 1.)
        self.w_dist = torch.distributions.normal.Normal(0., 1.)

    @torch.inference_mode()
    def function(self, x, w):
        # x: (bsz, n_samples, x_dim)
        # w: (bsz, x_dim + 1, y_dim)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        y = torch.bmm(x, w)
        return y

    def get_batch(self, n_context=None, indices=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        
        w = self.w_dist.rsample((self.batch_size, self.x_dim + 1, self.y_dim))
        x = self.x_dist.rsample((self.batch_size, 2 * n_context, self.x_dim))
        y = self.function(x, w)
        y = torch.distributions.categorical.Categorical(logits=y / self.temperature).sample().unsqueeze(-1)

        x_c, y_c = x[:, :n_context], y[:, :n_context]
        x_q, y_q = x[:, n_context:], y[:, n_context:]
        return (x_c, y_c), (x_q, y_q), w

    def __len__(self) -> int:
        return self.data_size // self.batch_size

    def __iter__(self) -> torch.Tensor:
        for _ in range(len(self)):
            yield self.get_batch()