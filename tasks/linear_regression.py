import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith, TransformerContext
from tasks.utils import fig2img


class LinearRegressionICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model

        if isinstance(model, ExplicitModelWith):
            if isinstance(model.context_model, TransformerContext):
                self.w_predictor = torch.nn.Linear(
                    model.context_model.n_features,
                    (model.context_model.x_dim + 1) * model.context_model.y_dim,
                )
            else:
                raise ValueError(
                    f"Unknown context model: {type(model.context_model)}. Must specify how to get it's z shape to build the W predictor."
                )
        else:
            self.w_predictor = None

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None:
            w_pred = self.w_predictor(z.detach()).view(*w.shape)
            w_loss = torch.nn.functional.mse_loss(w_pred, w)
            loss = y_q_loss + w_loss
            self.log("train/w_MSE", w_loss, on_step=False, on_epoch=True)
        else:
            loss = y_q_loss
        self.log("train/MSE", y_q_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None:
            w_pred = self.w_predictor(z).view(*w.shape)
            w_loss = torch.nn.functional.mse_loss(w_pred, w)
            self.log("val/w_loss", w_loss, on_step=False, on_epoch=True)
        self.log("val/MSE", y_q_loss, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if self.trainer.current_epoch % 10 == 0:
            self.eval()
            self.plot_model("train")
            self.plot_model("val")
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
        elif stage == "val":
            dataset = self.trainer.datamodule.val_data
        else:
            raise ValueError(f"Invalid dataset: {dataset}")

        (x_c, y_c), _, w = dataset.get_batch(
            n_context=dataset.max_context, indices=range(n_examples)
        )
        x_c, y_c, w = x_c.to(self.device), y_c.to(self.device), w.to(self.device)
        ypred = []
        x_qs = torch.linspace(-1, 1, 50)
        for x_q in x_qs:
            x_q = x_q * torch.ones(n_examples, 1, device=self.device)
            y_q_pred, _ = self.model(x_c, y_c, x_q)
            ypred.append(y_q_pred.squeeze(-1))
        ypred = torch.stack(ypred, dim=1)

        x = x_qs.numpy()
        ypred = ypred.cpu().numpy()
        w = w.cpu().numpy()[..., -1]

        fig, axs = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
        for i, ax in enumerate(axs):
            ax.plot([-1, 1], [-w[i, 0] + w[i, 1], w[i, 0] + w[i, 1]], label="True")
            ax.plot(x, ypred[i], label="Model")
            ax.set(xlabel="x", ylabel="y", ylim=(-2, 2))
            ax.legend(loc="upper left")
        fig.tight_layout()

        self.logger.log_image(f"examples/{stage}", [fig2img(fig)])

    def configure_optimizers(self):
        param_groups = [{"params": self.model.parameters()}]
        if self.w_predictor is not None:
            param_groups += [
                {
                    "params": self.w_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)
