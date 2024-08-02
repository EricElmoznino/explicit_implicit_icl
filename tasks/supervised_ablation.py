import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from models.explicit import ExplicitModelWith
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


class SupervisedAblation(LightningModule):
    def __init__(self, model: ExplicitModelWith, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.context_model = model.context_model
        self.prediction_model = model.prediction_model
        assert self.context_model.z_dim == self.prediction_model.z_dim

    def forward(self, x_c, y_c, x_q, w):
        z = self.context_model(x_c, y_c)
        y_q_pred = self.prediction_model(w.view_as(z), x_q)
        with torch.no_grad():
            y_q_from_z = x_q @ z.view_as(w)
        return z.view_as(w), y_q_from_z, y_q_pred

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        z, y_q_from_z, y_q_pred = self.forward(x_c, y_c, x_q, w)
        z_loss = torch.nn.functional.mse_loss(z, w)
        y_q_from_z_loss = torch.nn.functional.mse_loss(y_q_from_z, y_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        loss = z_loss + y_q_loss
        self.log("train/z_loss", z_loss, on_step=False, on_epoch=True)
        self.log("train/yq_fromz_loss", y_q_from_z_loss, on_step=False, on_epoch=True)
        self.log("train/yq_loss", y_q_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (x_c, y_c), (x_q, y_q), w = batch
        z, y_q_from_z, y_q_pred = self.forward(x_c, y_c, x_q, w)
        z_loss = torch.nn.functional.mse_loss(z, w)
        y_q_from_z_loss = torch.nn.functional.mse_loss(y_q_from_z, y_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        val_style = list(self.trainer.datamodule.val_data.keys())[dataloader_idx]
        self.log(
            f"val/{val_style}_z_loss",
            z_loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{val_style}_yq_fromz_loss",
            y_q_from_z_loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{val_style}_yq_loss",
            y_q_loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        param_groups = [
            {"params": self.context_model.parameters()},
            {"params": self.prediction_model.parameters()},
        ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)


class SupervisedAblationDataModule(LightningDataModule):
    def __init__(
        self,
        min_context: int = 16,
        max_context: int = 128,
        batch_size: int = 128,
        train_size: int = 1000,
        val_size: int = 1000,
        noise: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_data = LinearRegressionDataPipe(
            min_context=min_context,
            max_context=max_context,
            batch_size=batch_size,
            data_size=train_size,
            noise=noise,
            w_ood_style="none",
            pred_ood_style="none",
        )
        self.val_data = {
            "iid": LinearRegressionDataPipe(
                min_context=min_context,
                max_context=max_context,
                batch_size=batch_size,
                data_size=val_size,
                noise=noise,
                w_ood_style="none",
                pred_ood_style="none",
            ),
            "rangeW": LinearRegressionDataPipe(
                min_context=min_context,
                max_context=max_context,
                batch_size=batch_size,
                data_size=val_size,
                noise=noise,
                w_ood_style="range",
                pred_ood_style="none",
            ),
            "compW": LinearRegressionDataPipe(
                min_context=min_context,
                max_context=max_context,
                batch_size=batch_size,
                data_size=val_size,
                noise=noise,
                w_ood_style="composition",
                pred_ood_style="none",
            ),
            "rangeP": LinearRegressionDataPipe(
                min_context=min_context,
                max_context=max_context,
                batch_size=batch_size,
                data_size=val_size,
                noise=noise,
                w_ood_style="none",
                pred_ood_style="range",
            ),
            "compP": LinearRegressionDataPipe(
                min_context=min_context,
                max_context=max_context,
                batch_size=batch_size,
                data_size=val_size,
                noise=noise,
                w_ood_style="none",
                pred_ood_style="composition",
            ),
        }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=None)

    def val_dataloader(self):
        return [DataLoader(v, batch_size=None) for v in self.val_data.values()]


class LinearRegressionDataPipe(IterDataPipe):
    def __init__(
        self,
        min_context: int,
        max_context: int,
        batch_size: int,
        data_size: int,
        noise: float,
        w_ood_style: str,
        pred_ood_style: str,
    ):
        super().__init__()
        assert w_ood_style in ["none", "range", "composition"]
        assert pred_ood_style in ["none", "range", "composition"]
        self.min_context = min_context
        self.max_context = max_context
        self.batch_size = batch_size
        self.data_size = data_size
        self.noise = noise
        self.w_ood_style = w_ood_style
        self.pred_ood_style = pred_ood_style

    def sample_three_quadrants(self, n_context):
        a = torch.rand(self.batch_size, n_context, 2)
        quadrant = torch.randint(0, 3, (self.batch_size, n_context))
        a[quadrant == 1, [0]] = -1
        a[quadrant == 2, [1]] = -1
        return a

    def sample_x(self, n_context):
        x_c = self.sample_three_quadrants(n_context)
        if self.pred_ood_style == "none":
            x_q = self.sample_three_quadrants(n_context)
        elif self.pred_ood_style == "range":
            x_q = self.sample_three_quadrants(n_context) + 1
        elif self.pred_ood_style == "composition":
            x_q = torch.rand(self.batch_size, n_context, 2) * -1
        else:
            raise ValueError("Invalid pred_ood style")
        return x_c, x_q

    def sample_w(self):
        if self.w_ood_style == "none":
            w = self.sample_three_quadrants(1)
        elif self.w_ood_style == "range":
            w = self.sample_three_quadrants(1) + 1
        elif self.w_ood_style == "composition":
            w = torch.rand(self.batch_size, 1, 2) * -1
        else:
            raise ValueError("Invalid w_ood style")
        w = w.transpose(1, 2)
        return w

    def function(self, x, w):
        return torch.bmm(x, w)

    def get_batch(self, n_context=None):
        if n_context is None:
            n_context = np.random.randint(self.min_context, self.max_context + 1)
        x_c, x_q = self.sample_x(n_context)
        w = self.sample_w()
        y_c, y_q = self.function(x_c, w), self.function(x_q, w)
        y_c += self.noise * torch.randn_like(y_c)
        y_q += self.noise * torch.randn_like(y_q)
        return (x_c, y_c), (x_q, y_q), w

    def __len__(self):
        return self.data_size // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.get_batch()
