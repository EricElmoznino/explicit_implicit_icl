import torch
from pytorch_lightning import LightningModule
from matplotlib import pyplot as plt
from models.implicit import ImplicitModel
from models.explicit import (
    ExplicitModel,
    ExplicitModelWith,
    KnownLatent,
    MLPLowRankPrediction,
)
from data.classification import MLPLowRankClassificationDataset
from tasks.utils import fig2img, make_grid
import numpy as np


class ClassificationICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        lr: float = 1e-4,
        no_plot: bool = False,
        aux_loss: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.aux_loss = aux_loss
        self.w_predictor = None
        self.aux_predictor = None
        if isinstance(model, ExplicitModelWith) and isinstance(
            model.context_model, KnownLatent
        ):
            self.known_z = True
        else:
            self.known_z = False

    def forward(self, x_c, y_c, x_q, w):
        y_c = torch.nn.functional.one_hot(
            y_c, self.trainer.datamodule.train_data.y_dim
        ).float()
        if self.known_z:
            self.model.context_model.set_z(w)
        y_q_pred, z = self.model(x_c, y_c, x_q)
        return y_q_pred, z

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        bsz, q_len, _ = x_q.shape
        y_q_pred, z = self.forward(x_c, y_c, x_q, w)
        y_q_loss = torch.nn.functional.cross_entropy(
            y_q_pred.view(bsz * q_len, -1), y_q.reshape(-1)
        )
        y_q_acc = (y_q_pred.argmax(dim=-1) == y_q).float().mean(dim=1).mean(dim=0)
        loss = y_q_loss
        if z is not None:
            w_pred = self.w_predictor(z.detach()).view(*w.shape)
            w_loss = torch.nn.functional.mse_loss(w_pred, w)
            loss += w_loss

            if self.aux_loss:
                aux_pred = self.aux_predictor(z).view(*w.shape)
                aux_loss = torch.nn.functional.mse_loss(aux_pred, w)
                loss += aux_loss

            self.log("train/w_MSE", w_loss, on_step=False, on_epoch=True)

        self.log("train/CE", y_q_loss, on_step=False, on_epoch=True)
        self.log("train/Accuracy", y_q_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (x_c, y_c), (x_q, y_q), w = batch
        bsz, q_len, _ = x_q.shape
        y_q_pred, z = self.forward(x_c, y_c, x_q, w)
        y_q_loss = torch.nn.functional.cross_entropy(
            y_q_pred.view(bsz * q_len, -1), y_q.reshape(-1)
        )
        y_q_acc = (y_q_pred.argmax(dim=-1) == y_q).float().mean(dim=1).mean(dim=0)
        val_style = list(self.trainer.datamodule.val_data.keys())[dataloader_idx]
        if z is not None:
            w_pred = self.w_predictor(z).view(*w.shape)
            w_loss = torch.nn.functional.mse_loss(w_pred, w)
            self.log(
                f"val/{val_style}_w_loss",
                w_loss,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        self.log(
            f"val/{val_style}_CE",
            y_q_loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val/{val_style}_Accuracy",
            y_q_acc,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def on_train_epoch_start(self):
        if self.trainer.current_epoch % 10 == 0:
            self.eval()
            self.plot_model("train")
            for val_style in self.trainer.datamodule.val_data.keys():
                self.plot_model(val_style)
            self.train()

    @torch.inference_mode()
    def plot_model(self, stage, n_examples=4):
        if isinstance(self.model, ImplicitModel):
            x_dim, y_dim = self.model.x_dim, self.model.y_dim
        elif isinstance(self.model, ExplicitModelWith):
            x_dim, y_dim = (
                self.model.prediction_model.x_dim,
                self.model.prediction_model.y_dim,
            )
        if self.hparams.no_plot or x_dim != 2 or y_dim != 2 or self.logger is None:
            return
        if stage == "train":
            dataset = self.trainer.datamodule.train_data
        else:
            dataset = self.trainer.datamodule.val_data[stage]
            stage = f"val_{stage}"

        (x_c, y_c), (x_q, y_q), w = dataset.get_batch(n_context=dataset.max_context)
        (xx, yy), grid = make_grid(x_q)
        levels = np.linspace(0.0, 1.0, 10)

        x_c, y_c, w = x_c.to(self.device), y_c.to(self.device), w.to(self.device)
        x_c, y_c, w = x_c[:n_examples], y_c[:n_examples], w[:n_examples]

        x_reshaped = grid.unsqueeze(0).repeat(n_examples, 1, 1).to(self.device)
        ypred, _ = self.forward(x_c, y_c, x_reshaped, w)
        ypred = torch.softmax(ypred, dim=-1)

        x_c, y_c, ypred = x_c.cpu(), y_c.cpu(), ypred.cpu()
        x_q, y_q = x_q.cpu(), y_q.cpu()

        fig, axs = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
        for i, ax in enumerate(axs):
            ax.contourf(
                xx,
                yy,
                ypred[i, :, 0].view(xx.shape),
                cmap="RdBu",
                levels=levels,
                vmax=1.0,
                vmin=0.0,
                label="Model",
            )
            # ax.scatter(x_c[i, :, 0], x_c[i, :, 1], c=y_c[i, :], marker='.', label="Context", s=10)
            ax.scatter(x_q[i, :, 0], x_q[i, :, 1], c=y_q[i, :], marker="*", s=20)
            # ax.legend(loc="upper left")
        fig.tight_layout()

        self.logger.log_image(f"examples/{stage}", [fig2img(fig)])

    def configure_optimizers(self):
        if isinstance(self.model, ExplicitModelWith):
            if self.aux_loss:
                self.aux_predictor = torch.nn.Linear(
                    self.model.context_model.z_dim,
                    self.trainer.datamodule.train_data.n_params,
                ).to(self.device)
            self.w_predictor = torch.nn.Linear(
                self.model.context_model.z_dim,
                self.trainer.datamodule.train_data.n_params,
            ).to(self.device)

        param_groups = [{"params": self.model.parameters()}]
        if self.w_predictor is not None:
            param_groups += [
                {
                    "params": self.w_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]

        if self.aux_predictor is not None:
            param_groups += [
                {
                    "params": self.aux_predictor.parameters(),
                    "lr": self.hparams.lr,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)

    def on_validation_start(self):
        # If we're using a known prediction model with some sampled hyperparameters,
        # we need to get them from the dataset
        data = self.trainer.datamodule.train_data
        if (
            isinstance(self.model, ExplicitModelWith)
            and isinstance(self.model.prediction_model, MLPLowRankPrediction)
            and isinstance(data, MLPLowRankClassificationDataset)
        ):
            self.model.prediction_model.set_model(data.model.to(self.device))
