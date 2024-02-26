from abc import ABC, abstractmethod
import pickle
from typing import Literal, Any, Iterable

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe

from lightning import LightningDataModule
from lightning.pytorch.utilities.seed import isolate_rng
import warnings


from data.utils import *
from data.utils_fastfood import FastfoodWrapper
from models.utils import MLP

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


def init_weights(m, std=1.0):
    if isinstance(m, BatchedLinear):
        torch.nn.init.normal_(m.weight, std=std)
        torch.nn.init.normal_(m.bias, std=std)


class RegressionDataModule(LightningDataModule):
    RegressionKind = Literal[
        "polynomial", "sinusoid", "linear", "mlp", "low_rank_mlp", "gp", "hh"
    ]

    def __init__(
        self,
        kind: RegressionKind,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 100,
        noise: float = 0.5,
        context_style: str = "same",
        ood_styles: tuple[str] | None = ["far", "wide"],
        kind_kwargs: dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        RegressionDatasetCls = {
            "polynomial": PolynomialRegressionDataset,
            "sinusoid": SinusoidalRegressionDataset,
            "linear": LinearRegressionDataset,
            "mlp": MLPRegressionDataset,
            "low_rank_mlp": MLPLowRankRegressionDataset,
            "gp": GPRegressionDataset,
            "hh": HHRegressionDataset,
        }[kind]
        self.train_data = RegressionDatasetCls(
            x_dim=x_dim,
            y_dim=y_dim,
            min_context=min_context,
            max_context=max_context,
            batch_size=batch_size,
            data_size=train_size,
            noise=noise,
            context_style=context_style,
            **kind_kwargs,
        )
        self.val_data = {
            "iid": RegressionDatasetCls(
                x_dim=x_dim,
                y_dim=y_dim,
                min_context=min_context,
                max_context=max_context,
                batch_size=val_size,
                data_size=val_size,
                noise=noise,
                context_style=context_style,
                finite=True,
                ood=False,
                **kind_kwargs,
            )
        }
        if ood_styles is not None:
            for style in ood_styles:
                self.val_data[style] = RegressionDatasetCls(
                    x_dim=x_dim,
                    y_dim=y_dim,
                    min_context=min_context,
                    max_context=max_context,
                    batch_size=val_size,
                    data_size=val_size,
                    noise=noise,
                    context_style=context_style,
                    finite=True,
                    ood=True,
                    ood_style=style,
                    **kind_kwargs,
                )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return [DataLoader(v, batch_size=None) for v in self.val_data.values()]


class RegressionDataset(ABC, IterDataPipe):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        data_size: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        noise: float = 0.0,
        ood: bool = False,
        context_style: str = "same",
        ood_style: str = "far",
        finite: bool = False,
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
        self.context_style = context_style
        self.ood_style = ood_style
        self.finite = finite

    def generate_finite_data(self):
        with isolate_rng():
            torch.manual_seed(0)
            self.fixed_x_c = torch.randn(self.data_size, self.max_context, self.x_dim)
            self.fixed_x_q = torch.randn(self.data_size, self.max_context, self.x_dim)
            if self.ood:
                if self.ood_style == "wide":
                    self.fixed_x_q *= 3.0
                elif self.ood_style == "far":
                    direction = torch.randn_like(self.fixed_x_q)
                    self.fixed_x_q = (
                        self.fixed_x_q * 0.1
                        + 3.0 * direction / direction.norm(dim=-1, keepdim=True)
                    )
            self.fixed_params = self.sample_function_params()
            self.fixed_y_c = self.function(self.fixed_x_c, self.fixed_params)
            self.fixed_y_q = self.function(self.fixed_x_q, self.fixed_params)
            self.fixed_y_c += self.noise * torch.randn_like(self.fixed_y_c)
            self.fixed_y_q += self.noise * torch.randn_like(self.fixed_y_q)

    def sample_finite_batch(self, n_context, return_vis=False):
        x_c = self.fixed_x_c[: self.batch_size, :n_context]
        x_q = self.fixed_x_q[: self.batch_size, :n_context]
        y_c = self.fixed_y_c[: self.batch_size, :n_context]
        y_q = self.fixed_y_q[: self.batch_size, :n_context]
        if self.fixed_params is not None:
            params = self.fixed_params[: self.batch_size]
            return (x_c, y_c), (x_q, y_q), params
        else:
            if return_vis:
                return (
                    (x_c, y_c),
                    (x_q, y_q),
                    None,
                    (
                        self.fixed_x_vis[: self.batch_size],
                        self.fixed_y_vis[: self.batch_size],
                    ),
                )
            else:
                return (x_c, y_c), (x_q, y_q), None

    def sample_x(self, n_context):
        x_c = torch.randn(self.batch_size, n_context, self.x_dim)
        if self.context_style == "same":
            x_q = torch.randn(self.batch_size, n_context, self.x_dim)
        elif self.context_style == "near":
            x_q = x_c + 0.1 * torch.randn_like(x_c)
        else:
            raise ValueError("Invalid context style")
        return x_c, x_q

    @abstractmethod
    def sample_function_params(self) -> FloatTensor:
        pass

    def function_params(self) -> FloatTensor:
        return self.sample_function_params()

    @abstractmethod
    def function(self, x, params, noise=None) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # params: (bsz, ...) parameters of the function
        # returns y: (bsz, n_samples, y_dim)
        pass

    def get_batch(self, n_context=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)

        if self.finite:
            n_context = (self.min_context + self.max_context) // 2
            return self.sample_finite_batch(n_context)

        x_c, x_q = self.sample_x(n_context)
        params = self.function_params()
        y_c, y_q = self.function(x_c, params), self.function(x_q, params)
        y_c += self.noise * torch.randn_like(y_c)
        y_q += self.noise * torch.randn_like(y_q)
        return (x_c, y_c), (x_q, y_q), params

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()


class LinearRegressionDataset(RegressionDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_params = (self.x_dim + 1) * self.y_dim
        if self.finite:
            self.generate_finite_data()

    def sample_function_params(self) -> FloatTensor:
        # Linear regression weights
        return torch.randn(self.batch_size, self.x_dim + 1, self.y_dim)

    def function(self, x, w) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # w: (bsz, x_dim + 1, y_dim)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        y = torch.bmm(x, w)
        return y


class MLPRegressionDataset(RegressionDataset):
    def __init__(
        self,
        activation: str = "relu",
        layers: int = 1,
        hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.layers = layers
        self.hidden_dim = hidden_dim

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()

        self.get_model()
        if self.finite:
            self.generate_finite_data()
        self.n_params = (
            sum(p.numel() for p in self.model.parameters()) // self.batch_size
        )

    def get_model(self):
        layers = [
            BatchedLinear(self.x_dim, self.hidden_dim, self.batch_size),
            self.activation,
        ]
        for _ in range(self.layers - 1):
            layers.append(
                BatchedLinear(self.hidden_dim, self.hidden_dim, self.batch_size)
            )
            layers.append(self.activation)
        layers.append(BatchedLinear(self.hidden_dim, self.y_dim, self.batch_size))
        self.model = torch.nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def get_parameters(self):
        w = []
        for name, param in self.model.named_parameters():
            w.append(param.view(self.batch_size, -1))
        w = torch.cat(w, dim=-1)
        return w

    def sample_function_params(self) -> FloatTensor:
        # Linear regression weights
        if self.model is None:
            self.get_model()
        self.model.apply(init_weights)
        return self.get_parameters().view(self.batch_size, -1)

    def function(self, x, w=None, noise=None):
        # x: (bsz, n_samples, x_dim)
        if torch.cuda.is_available():
            x = x.cuda()

        y = self.model(x)
        return y


class MLPLowRankRegressionDataset(RegressionDataset):
    def __init__(
        self,
        low_dim: int,
        layers: int = 1,
        hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.low_dim = low_dim
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.model = FastfoodWrapper(
            model=MLP(self.x_dim, self.hidden_dim, self.y_dim, layers),
            low_dim=low_dim,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if self.finite:
            self.generate_finite_data()
        self.n_params = low_dim

    def sample_function_params(self) -> FloatTensor:
        return torch.randn(self.batch_size, self.low_dim)

    def function(self, x, params, noise=None):
        # x: (bsz, n_samples, x_dim)
        if torch.cuda.is_available():
            x = x.cuda()
        y = []
        for i in range(x.shape[0]):
            y.append(self.model(x[i], params[i]))
        y = torch.stack(y, dim=0)
        return y.detach()


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
        super().__init__(**kwargs)

    def sample_function_params(self, n_samples) -> FloatTensor:
        # Polynomial regression weights
        return self.w_dist.rsample((n_samples,))

    def function(self, x, params) -> FloatTensor:
        x = torch.cat([x**i for i in range(self.order + 1)], dim=-1)
        params = params.unsqueeze(-1)
        y = torch.bmm(x, params)
        return y


class SinusoidalRegressionDataset(RegressionDataset):
    def __init__(
        self,
        fixed_freq: bool = True,
        n_freq: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_freq = n_freq
        self.fixed_freq = fixed_freq

        if fixed_freq:
            with isolate_rng():
                torch.manual_seed(1)
                self.freqs = torch.rand(self.x_dim, n_freq).unsqueeze(0) * 5

        self.n_params = n_freq * self.x_dim
        if not fixed_freq:
            self.n_params *= 2

        if self.finite:
            self.generate_finite_data()

    def sample_function_params(self) -> FloatTensor:
        amplitudes = (torch.rand(self.batch_size, self.x_dim, self.n_freq) - 0.5) * 2
        if self.fixed_freq:
            return amplitudes
        else:
            freqs = torch.rand(self.batch_size, self.x_dim, self.n_freq) * 5
            return torch.cat([amplitudes, freqs], dim=-1)

    def function(self, x, params) -> FloatTensor:
        if self.fixed_freq:
            freq = self.freqs.to(x.device)
            amplitudes = params
        else:
            amplitudes = params[:, :, : self.n_freq]
            freq = params[:, :, self.n_freq :]

        x = torch.sin(x.unsqueeze(-1) * freq.unsqueeze(1))
        y = (x * amplitudes.unsqueeze(1)).sum(dim=-1).sum(dim=-1, keepdim=True)
        return y


class GPRegressionDataset(RegressionDataset):
    def __init__(
        self,
        kernel: str = "RBF",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert self.x_dim == 1
        self.n_params = None

        if kernel == "RBF":
            self.kernel = RBFKernel()
        elif kernel == "Matern":
            self.kernel = Matern52Kernel()
        elif kernel == "Periodic":
            self.kernel = PeriodicKernel()

        if self.finite:
            self.generate_finite_data()

    def sample_function_params(self):
        return None

    def function(self, x, params=None) -> FloatTensor:
        cov = self.kernel(x)
        mean = torch.zeros(x.shape[:2], device=x.device)
        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)
        return y

    def generate_finite_data(self):
        with isolate_rng():
            torch.manual_seed(0)
            self.fixed_x = torch.randn(self.data_size, 2 * self.max_context, self.x_dim)
            if self.ood:
                if self.ood_style == "wide":
                    self.fixed_x[:, self.max_context :] *= 3.0
                elif self.ood_style == "far":
                    direction = torch.randn_like(self.fixed_x[:, self.max_context :])
                    self.fixed_x[:, self.max_context :] = self.fixed_x[
                        :, self.max_context :
                    ] * 0.1 + 3.0 * direction / direction.norm(dim=-1, keepdim=True)
            self.fixed_x_vis = (
                torch.linspace(self.fixed_x.min(), self.fixed_x.max(), 100)
                .view(1, 100, self.x_dim)
                .repeat(self.data_size, 1, 1)
            )
            x_temp = torch.cat([self.fixed_x, self.fixed_x_vis], dim=1)
            y_temp = self.function(x_temp)
            self.fixed_y, self.fixed_y_vis = (
                y_temp[:, : 2 * self.max_context],
                y_temp[:, 2 * self.max_context :],
            )

            self.fixed_x_c = self.fixed_x[:, : self.max_context]
            self.fixed_x_q = self.fixed_x[:, self.max_context :]
            self.fixed_y_c = self.fixed_y[:, : self.max_context]
            self.fixed_y_q = self.fixed_y[:, self.max_context :]
            self.fixed_params = None

    def get_batch(self, n_context=None, return_vis=False):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)

        if self.finite:
            n_context = (self.min_context + self.max_context) // 2
            return self.sample_finite_batch(n_context, return_vis)

        x_c, x_q = self.sample_x(n_context)
        x = torch.cat([x_c, x_q], dim=1)
        if return_vis:
            x_vis = (
                torch.linspace(x.min(), x.max(), 100)
                .view(1, 100, x_c.shape[-1])
                .repeat(x_c.shape[0], 1, 1)
            )
            x = torch.cat([x, x_vis], dim=1)
        y = self.function(x)
        if return_vis:
            y, y_vis = y[:, : 2 * n_context], y[:, 2 * n_context :]
        y_c, y_q = y[:, :n_context], y[:, n_context:]
        if return_vis:
            return (x_c, y_c), (x_q, y_q), None, (x_vis, y_vis)
        return (x_c, y_c), (x_q, y_q), None


class HHRegressionDataset(RegressionDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.x_dim == 1
        self.n_params = 2
        with open("data/hh_data.pkl", "rb") as f:
            self.data, self.params_list = pickle.load(f).values()
        self.params_list = torch.stack(
            [
                torch.stack(list(self.params_list[i].values()))
                for i in range(len(self.params_list))
            ]
        )
        self.simulation_timesteps = self.data.shape[1]
        self.x_points = torch.linspace(0, 1000, 1000)
        self.x_points_dict = {
            self.x_points[i].item(): i for i in (range(len(self.x_points)))
        }

        if self.finite:
            self.generate_finite_data()

    def sample_x(self, n_context):
        x_c = (
            torch.zeros((self.batch_size, n_context, self.x_dim)).uniform_(0, 250).int()
        )
        # x_c = (torch.randn(self.batch_size, n_context, self.x_dim) * 200).int().abs()
        if self.context_style == "same":
            x_q = x_c = (
                torch.zeros((self.batch_size, n_context, self.x_dim))
                .uniform_(0, 250)
                .int()
            )
        elif self.context_style == "near":
            x_q = (x_c + 10.0 * torch.randn(x_c.size())).int()
        else:
            raise ValueError("Invalid context style")

        x_c, x_q = x_c.clamp(min=0, max=self.simulation_timesteps - 1), x_q.clamp(
            min=0, max=self.simulation_timesteps - 1
        )

        x_c = self.x_points[x_c]
        x_q = self.x_points[x_q]
        return x_c, x_q

    def sample_function_params(self):
        # Uniform over [0,40]^2
        return self.params_list[
            torch.randperm(len(self.params_list))[: self.batch_size]
        ]

    def function(self, x: torch.Tensor, params) -> FloatTensor:
        # Duration can be bigger than self.simulation_timesteps
        x_id = x.clone().to("cpu")
        x_id.apply_(self.x_points_dict.get)
        x_id = x_id.to(int).to(params.device)
        params_id = torch.stack(
            [
                torch.all(self.params_list == p.to("cpu"), dim=1).int().argmax()
                for p in params
            ]
        )
        return self.data[params_id[:, None, None].to("cpu"), x_id.to("cpu")].to(
            x.device
        )

    def generate_finite_data(self):

        with isolate_rng():
            torch.manual_seed(0)
            # self.fixed_x_c = (torch.randn(self.data_size, self.max_context, self.x_dim) * 200).int().abs()
            # self.fixed_x_q = (torch.randn(self.data_size, self.max_context, self.x_dim) * 200).int().abs()
            self.fixed_x_c = (
                torch.zeros((self.data_size, self.max_context, self.x_dim))
                .uniform_(0, 250)
                .int()
            )
            self.fixed_x_q = (
                torch.zeros((self.data_size, self.max_context, self.x_dim))
                .uniform_(0, 250)
                .int()
            )

            if self.ood:
                if self.ood_style == "wide":
                    self.fixed_x_q *= 3
                elif self.ood_style == "far":
                    self.fixed_x_q = (self.fixed_x_q * 0.2 + 750).to(int)
            self.fixed_x_c = self.fixed_x_c.clamp(
                min=0, max=self.simulation_timesteps - 1
            ).to(int)
            self.fixed_x_q = self.fixed_x_q.clamp(
                min=0, max=self.simulation_timesteps - 1
            ).to(int)

            self.fixed_x_c = self.x_points[self.fixed_x_c]
            self.fixed_x_q = self.x_points[self.fixed_x_q]

            self.fixed_params = self.sample_function_params()
            self.fixed_y_c = self.function(self.fixed_x_c, self.fixed_params)
            self.fixed_y_q = self.function(self.fixed_x_q, self.fixed_params)
            self.fixed_y_c += self.noise * torch.randn_like(self.fixed_y_c)
            self.fixed_y_q += self.noise * torch.randn_like(self.fixed_y_q)
