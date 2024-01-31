import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith, SinRegPrediction
from data.regression import SinusoidalRegressionDataset, GPRegressionDataset, HHRegressionDataset
from tasks.utils import fig2img


class RegressionICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        lr: float = 1e-4,
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.w_predictor = None

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None and self.w_predictor is not None:
            w_pred = self.w_predictor(z.detach()).view(*w.shape)
            w_loss = torch.nn.functional.mse_loss(w_pred, w.detach())
            loss = y_q_loss + w_loss
            self.log("train/w_MSE", w_loss, on_step=False, on_epoch=True)
        else:
            loss = y_q_loss
        self.log("train/MSE", y_q_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (x_c, y_c), (x_q, y_q), w = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
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
            f"val/{val_style}_MSE",
            y_q_loss,
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
                self.model.context_model.x_dim,
                self.model.context_model.y_dim,
            )
        if x_dim > 1 or y_dim > 1 or self.logger is None:
            return
        if stage == "train":
            dataset = self.trainer.datamodule.train_data
        else:
            dataset = self.trainer.datamodule.val_data[stage]
            stage = f"val_{stage}"

        if isinstance(dataset, GPRegressionDataset):
            (x_c, y_c), (x_q, y_q), w, (x, y) = dataset.get_batch(
                n_context=dataset.max_context, return_vis=True
            )
            x = x[0]
            x, y = x.to(self.device), y.to(self.device)
        elif isinstance(dataset, HHRegressionDataset):
            (x_c, y_c), (x_q, y_q), w = dataset.get_batch(n_context=dataset.max_context)
            x = dataset.x_points.to(self.device)
            y = dataset.function(x.view(1, -1, 1).repeat(x_c.shape[0], 1, 1), w)
        else:
            (x_c, y_c), (x_q, y_q), w = dataset.get_batch(n_context=dataset.max_context)
            w = w.to(self.device)
            x = torch.linspace(x_q.min(), x_q.max(), 100).to(self.device)
            y = dataset.function(x.view(1, -1, 1).repeat(x_c.shape[0], 1, 1), w)

        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        x_c, y_c, y = x_c[:n_examples], y_c[:n_examples], y[:n_examples]

        x_reshaped = x.view(1, -1, 1).repeat(n_examples, 1, 1).to(self.device)
        ypred, _ = self.model(x_c, y_c, x_reshaped)

        x_q, y_q, x_c, x, y_c, ypred, y = (
            x_q.cpu(),
            y_q.cpu(),
            x_c.cpu(),
            x.cpu(),
            y_c.cpu(),
            ypred.cpu(),
            y.cpu(),
        )

        fig, axs = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
        for i, ax in enumerate(axs):
            ax.plot(x, y[i, :, 0], label="True", c="black")
            ax.plot(x, ypred[i, :, 0], label="Model", c="green")
            ax.scatter(x_c[i, :, 0], y_c[i, :, 0], label="Context", s=10, c="blue")
            ax.scatter(x_q[i, :, 0], y_q[i, :, 0], label="Query", s=10, c="red")
            ax.set(xlabel="x", ylabel="y")
            ax.legend(loc="upper left")
            if isinstance(dataset, HHRegressionDataset):
                ax.set_ylim([-100,50])
        fig.tight_layout()

        self.logger.log_image(f"examples/{stage}", [fig2img(fig)])

    def configure_optimizers(self):
        if self.trainer.datamodule.val_data["iid"].fixed_params is not None:
            if isinstance(self.model, ExplicitModelWith):
                self.w_predictor = torch.nn.Linear(
                    self.model.context_model.n_features,
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
        return torch.optim.Adam(param_groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def on_validation_start(self):
        # If we're using a known sinusoidal prediction model with fixed frequencies,
        # we need to get the ground-truth frequencies from the dataset
        data = self.trainer.datamodule.train_data
        if (
            isinstance(self.model, ExplicitModelWith)
            and isinstance(self.model.prediction_model, SinRegPrediction)
            and isinstance(data, SinusoidalRegressionDataset)
            and data.fixed_freq
        ):
            self.model.prediction_model.set_freqs(data.freqs.to(self.device))
