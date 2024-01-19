import torch
from torchmetrics.regression import R2Score
from torchmetrics.classification import MultilabelExactMatch
from lightning import LightningModule
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith
from data.gene_crispr import GeneCrisprDataModule


class GeneCrisprICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.ptb_predictor = None

        # Metrics initialized at the start of training so that
        # we can infer the number of targets from the datamodule
        self.train_r2 = R2Score()
        self.train_nonzero_r2 = R2Score()
        self.val_r2 = R2Score()
        self.val_nonzero_r2 = R2Score()
        self.train_ptb_id_accuracy: MultilabelExactMatch | None = None
        self.val_ptb_id_accuracy: MultilabelExactMatch | None = None

    def training_step(self, batch, batch_idx):
        x_c, x_q, y_q, q_mask, ptb_id = batch
        y_q_pred, z = self.model(x_c, None, x_q)

        y_q, y_q_pred = y_q[q_mask], y_q_pred[q_mask]
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)

        if z is not None and self.ptb_predictor is not None:
            ptb_id_pred = self.ptb_predictor(z.detach())
            ptb_id_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ptb_id_pred, ptb_id
            )
            loss = y_q_loss + ptb_id_loss
            self.train_ptb_id_accuracy(ptb_id_pred, ptb_id.int())
            self.log("train/ptb_id_loss", ptb_id_loss, on_step=False, on_epoch=True)
            self.log(
                "train/ptb_id_acc",
                self.train_ptb_id_accuracy,
                on_step=False,
                on_epoch=True,
            )
        else:
            loss = y_q_loss

        self.train_r2(y_q_pred, y_q)
        self.train_nonzero_r2(y_q_pred[y_q != 0], y_q[y_q != 0])
        self.log("train/MSE", y_q_loss, on_step=False, on_epoch=True)
        self.log("train/R2", self.train_r2, on_step=False, on_epoch=True)
        self.log(
            "train/R2_nonzero", self.train_nonzero_r2, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x_c, x_q, y_q, q_mask, ptb_id = batch
        y_q_pred, z = self.model(x_c, None, x_q)

        y_q, y_q_pred = y_q[q_mask], y_q_pred[q_mask]
        y_q_loss = torch.nn.functional.mse_loss(y_q_pred, y_q)

        if z is not None and self.ptb_predictor is not None:
            ptb_id_pred = self.ptb_predictor(z.detach())
            ptb_id_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ptb_id_pred, ptb_id
            )
            self.val_ptb_id_accuracy(ptb_id_pred, ptb_id.int())
            self.log("val/ptb_id_loss", ptb_id_loss, on_step=False, on_epoch=True)
            self.log(
                "val/ptb_id_acc", self.val_ptb_id_accuracy, on_step=False, on_epoch=True
            )

        self.val_r2(y_q_pred, y_q)
        self.val_nonzero_r2(y_q_pred[y_q != 0], y_q[y_q != 0])
        self.log("val/MSE", y_q_loss, on_step=False, on_epoch=True)
        self.log("val/R2", self.val_r2, on_step=False, on_epoch=True)
        self.log("val/R2_nonzero", self.val_nonzero_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if isinstance(self.model, ExplicitModelWith):
            self.ptb_predictor = torch.nn.Linear(
                self.model.context_model.n_features,
                self.trainer.datamodule.n_ptb_targets,
            ).to(self.device)
            dm: GeneCrisprDataModule = self.trainer.datamodule
            self.train_ptb_id_accuracy = MultilabelExactMatch(dm.n_ptb_targets).to(
                self.device
            )
            self.val_ptb_id_accuracy = MultilabelExactMatch(dm.n_ptb_targets).to(
                self.device
            )

        param_groups = [{"params": self.model.parameters()}]
        if self.ptb_predictor is not None:
            param_groups += [
                {
                    "params": self.ptb_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)
