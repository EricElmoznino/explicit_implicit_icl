from abc import ABC, abstractmethod
from typing import Literal, Any, Iterable
import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


class RegressionDataModule(LightningDataModule):
    RegressionKind = Literal["polynomial", "sinusoidal"]

    def __init__(
        self,
        kind: RegressionKind,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 1000,
        noise: float = 0.5,
        ood_train: bool = False,
        ood_val: bool = True,
        kind_kwargs: dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        RegressionDatasetCls = {
            "polynomial": PolynomialRegressionDataset,
            "sinusoidal": SinusoidalRegressionDataset,
        }[kind]
        self.train_data = RegressionDatasetCls(
            min_context=min_context,
            max_context=max_context,
            batch_size=batch_size,
            data_size=train_size,
            noise=noise,
            ood=ood_train,
            **kind_kwargs,
        )
        self.val_data = RegressionDatasetCls(
            min_context=min_context,
            max_context=max_context,
            batch_size=batch_size,
            data_size=val_size,
            noise=noise,
            ood=ood_val,
            **kind_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None)


class RegressionDataset(ABC, IterDataPipe):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        data_size: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        noise: float = 0.5,
        ood: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.data_size = data_size
        self.noise = noise
        self.ood = ood
        self.params_fixed = self.sample_function_params(100)

    def sample_x(self, n_samples, n_context):
        x_c = torch.randn(n_samples, n_context, self.x_dim)
        if self.ood:
            x_q_mean = 3 if np.random.random() > 0.5 else -3
            x_q = x_q_mean + torch.randn(n_samples, self.x_dim) * 0.1
        else:
            x_q = x_c[:, 0] + torch.randn(n_samples, self.x_dim) * 0.1
        return x_c, x_q.unsqueeze(1)

    @abstractmethod
    def sample_function_params(self, n_samples) -> FloatTensor:
        pass

    @abstractmethod
    @torch.inference_mode()
    def function(self, x, params) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # params: (bsz, ...) parameters of the function
        # returns y: (bsz, n_samples, y_dim)
        pass

    def get_batch(self, n_context=None, indices=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        if indices is None:
            params = self.sample_function_params(self.batch_size)
        else:
            params = self.params_fixed[indices]
        x_c, x_q = self.sample_x(params.shape[0], n_context)
        y_c, y_q = self.function(x_c, params), self.function(x_q, params)
        y_c = y_c + self.noise * torch.randn_like(y_c)
        x_q, y_q = x_q.squeeze(1), y_q.squeeze(1)
        return (x_c, y_c), (x_q, y_q), params

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()


class PolynomialRegressionDataset(RegressionDataset):
    def __init__(
        self,
        order: int,
        **kwargs,
    ) -> None:
        self.order = order
        std = torch.linspace(1, 1 / order**2, order)
        std = torch.cat([0.1 * torch.ones(1), std])  # Smaller y-intercepts
        self.w_dist = torch.distributions.normal.Normal(torch.zeros(order + 1), std)
        super().__init__(x_dim=1, y_dim=1, **kwargs)

    def sample_function_params(self, n_samples) -> FloatTensor:
        # Polynomial regression weights
        return self.w_dist.rsample((n_samples,))

    @torch.inference_mode()
    def function(self, x, params) -> FloatTensor:
        x = torch.cat([x**i for i in range(self.order + 1)], dim=-1)
        params = params.unsqueeze(-1)
        y = torch.bmm(x, params)
        return y


class SinusoidalRegressionDataset(RegressionDataset):
    def __init__(
        self,
        freqs: list[float],
        max_amplitudes: list[float],
        **kwargs,
    ) -> None:
        self.freqs = torch.FloatTensor(freqs)
        self.amplitude_dist = torch.distributions.uniform.Uniform(
            -torch.FloatTensor(max_amplitudes), torch.FloatTensor(max_amplitudes)
        )
        super().__init__(x_dim=1, y_dim=1, **kwargs)

    def sample_function_params(self, n_samples) -> FloatTensor:
        return self.amplitude_dist.sample((n_samples,))

    @torch.inference_mode()
    def function(self, x, params) -> FloatTensor:
        x = torch.cat([torch.sin(x * f) for f in self.freqs], dim=-1)
        y = (x * params.unsqueeze(1)).sum(dim=-1, keepdim=True)
        return y
