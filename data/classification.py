from abc import ABC, abstractmethod
from typing import Literal, Any, Iterable

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe

from lightning import LightningDataModule
from lightning.pytorch.utilities.seed import isolate_rng
import warnings

from data.utils import BatchedLinear

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")

def init_weights(m, std=1.):
    if isinstance(m, BatchedLinear):
        torch.nn.init.normal_(m.weight, std=std)
        torch.nn.init.normal_(m.bias, std=std)

class ClassificationDataModule(LightningDataModule):
    ClassificationKind = Literal["linear", "mlp"]

    def __init__(
        self,
        kind: ClassificationKind,
        x_dim: int,
        y_dim: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        train_size: int = 10000,
        val_size: int = 100,
        temperature: float = 0.1,
        kind_kwargs: dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        ClassificationDatasetCls = {
            "linear": LinearClassificationDataset,
            "mlp": MLPClassificationDataset,
        }[kind]
        self.train_data = ClassificationDatasetCls(
            x_dim=x_dim,
            y_dim=y_dim,
            min_context=min_context,
            max_context=max_context,
            batch_size=batch_size,
            data_size=train_size,
            temperature=temperature,
            **kind_kwargs,
        )
        self.val_data = ClassificationDatasetCls(
            x_dim=x_dim,
            y_dim=y_dim,
            min_context=min_context,
            max_context=max_context,
            batch_size=val_size,
            data_size=val_size,
            temperature=temperature,
            finite=True,
            **kind_kwargs,
        )

        self.ood_val_data = ClassificationDatasetCls(
            x_dim=x_dim,
            y_dim=y_dim,
            min_context=min_context,
            max_context=max_context,
            batch_size=val_size,
            data_size=val_size,
            temperature=temperature,
            finite=True,
            ood=True,
            **kind_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return [DataLoader(self.val_data, batch_size=None), DataLoader(self.ood_val_data, batch_size=None)]

class ClassificationDataset(ABC, IterDataPipe):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        data_size: int,
        min_context: int,
        max_context: int,
        batch_size: int = 128,
        temperature: float = 0.1,
        ood: bool = False,
        finite: bool = False
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.data_size = data_size
        self.temperature = temperature
        self.ood = ood
        self.finite = finite

    def generate_finite_data(self):
        with isolate_rng():
            torch.manual_seed(0)
            self.fixed_x_c = torch.randn(self.data_size, self.max_context, self.x_dim)
            self.fixed_x_q = torch.randn(self.data_size, self.max_context, self.x_dim)
            if self.ood:
                self.fixed_x_q *= 5.
            self.fixed_params = self.sample_function_params()
            self.fixed_y_c = self.function(self.fixed_x_c, self.fixed_params)
            self.fixed_y_q = self.function(self.fixed_x_q, self.fixed_params)

    def sample_finite_batch(self, n_context):
        x_c = self.fixed_x_c[:self.batch_size, :n_context]
        x_q = self.fixed_x_q[:self.batch_size, :n_context]
        y_c = self.fixed_y_c[:self.batch_size, :n_context]
        y_q = self.fixed_y_q[:self.batch_size, :n_context]
        params = self.fixed_params[:self.batch_size]
        return (x_c, y_c), (x_q, y_q), params

    def sample_x(self, n_context):
        x_c = torch.randn(self.batch_size, n_context, self.x_dim)
        x_q = torch.randn(self.batch_size, n_context, self.x_dim)
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
        return (x_c, y_c), (x_q, y_q), params

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()

class LinearClassificationDataset(ClassificationDataset):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_params = (self.x_dim + 1) * self.y_dim
        if self.finite:
            self.generate_finite_data()

    def sample_function_params(self) -> FloatTensor:
        # Linear Classification weights
        return torch.randn(self.batch_size, self.x_dim + 1, self.y_dim)

    def function(self, x, w, noise=None) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # w: (bsz, x_dim + 1, y_dim)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        y = torch.bmm(x, w)
        y = torch.distributions.categorical.Categorical(logits=y / self.temperature).sample()
        return y

class MLPClassificationDataset(ClassificationDataset):
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

        if self.finite:
            self.generate_finite_data()
        self.n_params = sum(p.numel() for p in self.model.parameters()) // self.batch_size

    def get_model(self):
        layers = [BatchedLinear(self.x_dim, self.hidden_dim, self.batch_size), self.activation]
        for _ in range(self.layers - 1):
            layers.append(BatchedLinear(self.hidden_dim, self.hidden_dim, self.batch_size))
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
        # Linear Classification weights
        if self.model is None:
            self.get_model()
        self.model.apply(init_weights)
        return self.get_parameters().view(self.batch_size, -1)

    def function(self, x, w=None, noise=None):
        # x: (bsz, n_samples, x_dim)
        if torch.cuda.is_available():
            x = x.cuda()

        y = self.model(x)
        y = torch.distributions.categorical.Categorical(logits=y / self.temperature).sample()
        return y