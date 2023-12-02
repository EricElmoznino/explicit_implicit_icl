import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith, TransformerContext
from tasks.utils import fig2img


class PolynomialRegressionICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        order: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model

        if isinstance(model, ExplicitModelWith):
            if isinstance(model.context_model, TransformerContext):
                self.w_predictor = torch.nn.Linear(
                    model.context_model.n_features, order + 1
                )
            else:
                raise ValueError(
                    f"Unknown context model: {type(model.context_model)}. Must specify how to get its z shape to build the W predictor."
                )
        else:
            self.w_predictor = None

    def training_step(self, batch, batch_idx):
        (x_c, y_c), (x_q, y_q), w = batch
        y_q_pred, z = self.model(x_c, y_c, x_q)
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)
        if z is not None:
            w_pred = self.w_predictor(z.detach())
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
            w_pred = self.w_predictor(z)
            w_loss = torch.nn.functional.mse_loss(w_pred, w)
            self.log("val/w_loss", w_loss, on_step=False, on_epoch=True)
        self.log("val/MSE", y_q_loss, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if self.trainer.current_epoch % 10 == 0 and self.logger is not None:
            self.eval()
            self.plot_model()
            self.train()

    @torch.inference_mode()
    def plot_model(self, n_examples=4):
        dataset = self.trainer.datamodule.train_data
        (x_c, y_c), _, w = dataset.get_batch(
            n_context=dataset.max_context, indices=range(n_examples)
        )
        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        ypred = []
        x = torch.linspace(-4, 4, 100)
        y = dataset.function(x.unsqueeze(0).expand(n_examples, -1), w)
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
        if self.w_predictor is not None:
            param_groups += [
                {
                    "params": self.w_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)
