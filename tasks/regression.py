import torch
from lightning import LightningModule
from einops import repeat
from matplotlib import pyplot as plt
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith, TransformerContext
from tasks.utils import fig2img


class RegressionICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        param_dim: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

        if isinstance(model, ExplicitModelWith):
            if isinstance(model.context_model, TransformerContext):
                self.param_predictor = torch.nn.Linear(
                    model.context_model.z_dim, param_dim
                )
            else:
                raise ValueError(
                    f"Unknown context model: {type(model.context_model)}. \
                    Must specify how to get its z shape to build the parameter predictor."
                )
        else:
            self.param_predictor = None

    def forward(self, x_c, y_c, x_q):
        return self.model(x_c, y_c, x_q)

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), params = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None:
            params_pred = self.param_predictor(z.detach())
            params_loss = torch.nn.functional.mse_loss(params_pred, params)
            loss = y_q_loss + params_loss
            self.log("train/params_MSE", params_loss, on_step=False, on_epoch=True)
        else:
            loss = y_q_loss
        self.log("train/MSE", y_q_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), params = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None:
            params_pred = self.param_predictor(z)
            params_loss = torch.nn.functional.mse_loss(params_pred, params)
            self.log("val/params_MSE", params_loss, on_step=False, on_epoch=True)
        self.log("val/MSE", y_q_loss, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if isinstance(self.model, ImplicitModel):
            x_dim, y_dim = self.model.x_dim, self.model.y_dim
        elif isinstance(self.model, ExplicitModelWith):
            x_dim, y_dim = (
                self.model.context_model.x_dim,
                self.model.context_model.y_dim,
            )
        if (
            self.trainer.current_epoch % 10 == 0
            and x_dim == 1
            and y_dim == 1
            and self.logger is not None
        ):
            self.eval()
            self.plot_model()
            self.train()

    @torch.inference_mode()
    def plot_model(self, n_examples=4):
        dataset = self.trainer.datamodule.train_data
        (x_c, y_c), _, params = dataset.get_batch(
            n_context=dataset.max_context, indices=range(n_examples)
        )
        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        ypred = []
        x = torch.linspace(-4, 4, 100)
        y = dataset.function(repeat(x, "c -> b c 1", b=n_examples), params).squeeze(-1)
        for x_q in x:
            x_q = x_q * torch.ones(n_examples, 1, device=self.device)
            y_q_pred, _ = self.model(x_c, y_c, x_q)
            ypred.append(y_q_pred.squeeze(-1))
        ypred = torch.stack(ypred, dim=1).cpu()
        x_c, y_c = x_c.cpu(), y_c.cpu()

        fig, axs = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
        for i, ax in enumerate(axs):
            ax.plot(x, y[i], label="True")
            ax.plot(x, ypred[i], label="Model")
            ax.scatter(x_c[i], y_c[i])
            ax.set(xlabel="x", ylabel="y")
            ax.legend(loc="upper left")
        fig.tight_layout()

        self.logger.log_image(f"examples", [fig2img(fig)])

    def configure_optimizers(self):
        param_groups = [{"params": self.model.parameters()}]
        if self.param_predictor is not None:
            param_groups += [
                {
                    "params": self.param_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)
