import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassExactMatch
from lightning import LightningModule
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith
from data.raven import RavenDataModule


class RavenICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        embedding_dim: int = 256,
        use_query_pos_encodings: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.context_pos_encodings = nn.Parameter(torch.randn(6, embedding_dim))
        if use_query_pos_encodings:
            self.query_pos_encodings = nn.Parameter(torch.randn(2, embedding_dim))
        else:
            self.query_pos_encodings = None
        self.embedding: nn.Linear | None = None
        self.rule_predictor: nn.Linear = None

        self.num_attributes: int | None = None
        self.num_values: int | None = None
        self.num_rules: int | None = None

        self.train_accuracy = MulticlassAccuracy()
        self.val_accuracy = MulticlassAccuracy()
        self.train_rule_accuracy: MulticlassExactMatch | None = None
        self.val_rule_accuracy: MulticlassExactMatch | None = None

    def forward(self, x_c, x_q) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        x_c, x_q = (
            F.one_hot(x_c, num_classes=self.num_values).float(),
            F.one_hot(x_q, num_classes=self.num_values).float(),
        )
        x_c, x_q = x_c.flatten(start_dim=2), x_q.flatten(start_dim=2)
        x_c, x_q = self.embedding(x_c), self.embedding(x_q)
        x_c += self.context_pos_encodings.unsqueeze(0)
        if self.query_pos_encodings is not None:
            x_q += self.query_pos_encodings.unsqueeze(0)
        y_q_embedding, z = self.model(x_c, None, x_q.unsqueeze(1))
        y_q_embedding = y_q_embedding.squeeze(1)
        return y_q_embedding, z

    def compare_to_candidates(self, y_q_embedding, y_q_candidates) -> torch.FloatTensor:
        y_q_candidates = F.one_hot(y_q_candidates, num_classes=self.num_values).float()
        y_q_candidates = y_q_candidates.flatten(start_dim=2)
        y_q_candidates = self.embedding(y_q_candidates)
        sim = torch.einsum("bd,bmd->bm", y_q_embedding, y_q_candidates)
        return sim

    def predict_rule(self, z) -> torch.FloatTensor:
        rule_pred = self.rule_predictor(z.detach())
        rule_pred = rule_pred.view(-1, self.num_attributes, self.num_rules)
        rule_pred = rule_pred.permute(0, 2, 1)
        return rule_pred

    def training_step(self, batch, batch_idx):
        x_c, x_q, y_q_candidates, y_q_label, rule = batch
        y_q_embedding, z = self(x_c, x_q)
        y_q_pred = self.compare_to_candidates(y_q_embedding, y_q_candidates)
        y_q_loss = torch.nn.functional.cross_entropy(y_q_pred, y_q_label)

        if z is not None and self.rule_predictor is not None:
            rule_pred = self.predict_rule(z)
            rule_loss = torch.nn.functional.cross_entropy(rule_pred, rule)
            loss = y_q_loss + rule_loss
            self.train_rule_accuracy(rule_pred, rule)
            self.log("train/rule_loss", rule_loss, on_step=False, on_epoch=True)
            self.log(
                "train/rule_accuracy",
                self.train_rule_accuracy,
                on_step=False,
                on_epoch=True,
            )
        else:
            loss = y_q_loss

        self.train_accuracy(y_q_pred, y_q_label)
        self.log("train/loss", y_q_loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        setting = "iid" if dataloader_idx == 0 else "ood"

        x_c, x_q, y_q_candidates, y_q_label, rule = batch
        y_q_embedding, z = self(x_c, x_q)
        y_q_pred = self.compare_to_candidates(y_q_embedding, y_q_candidates)
        y_q_loss = torch.nn.functional.cross_entropy(y_q_pred, y_q_label)

        if z is not None and self.rule_predictor is not None:
            rule_pred = self.predict_rule(z)
            rule_loss = torch.nn.functional.cross_entropy(rule_pred, rule)
            self.val_rule_accuracy(rule_pred, rule)
            self.log(
                f"val_{setting}/rule_loss", rule_loss, on_step=False, on_epoch=True
            )
            self.log(
                f"val_{setting}/rule_accuracy",
                self.val_rule_accuracy,
                on_step=False,
                on_epoch=True,
            )

        self.val_accuracy(y_q_pred, y_q_label)
        self.log(f"val_{setting}/loss", y_q_loss, on_step=False, on_epoch=True)
        self.log(
            f"val_{setting}/accuracy", self.val_accuracy, on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        dm: RavenDataModule = self.trainer.datamodule
        self.num_attributes = dm.num_attributes
        self.num_values = dm.num_values
        self.num_rules = dm.num_rules

        self.train_rule_accuracy = MulticlassExactMatch(num_classes=self.num_rules)
        self.val_rule_accuracy = MulticlassExactMatch(num_classes=self.num_rules)

        self.embedding = torch.nn.Linear(
            self.num_attributes * self.num_values,
            self.hparams.embedding_dim,
            bias=False,
        ).to(self.device)

        if isinstance(self.model, ExplicitModelWith):
            self.rule_predictor = torch.nn.Linear(
                self.model.context_model.z_dim,
                self.num_attributes * self.num_rules,
            ).to(self.device)

        param_groups = [
            {
                "params": (
                    list(self.model.parameters()) + list(self.embedding.parameters()),
                )
            }
        ]
        if self.rule_predictor is not None:
            param_groups += [
                {
                    "params": self.rule_predictor.parameters(),
                    "lr": self.hparams.lr * 10,
                }
            ]
        return torch.optim.Adam(param_groups, lr=self.hparams.lr)
