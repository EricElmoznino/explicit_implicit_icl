import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassExactMatch
from pytorch_lightning import LightningModule
from models.implicit import ImplicitModel
from models.explicit import ExplicitModel, ExplicitModelWith, RavenKnownPrediction
from data.raven import RavenDataModule


class RavenICL(LightningModule):
    def __init__(
        self,
        model: ImplicitModel | ExplicitModel,
        embedding_dim: int = 256,
        use_query_pos_encodings: bool = True,
        lr: float = 1e-4,
        backprop_rule_pred: bool = False,
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

        self.n_attributes: int | None = None
        self.n_rules: int | None = None

        self.train_accuracy = MulticlassAccuracy(num_classes=8)
        self.val_accuracy = MulticlassAccuracy(num_classes=8)
        self.train_rule_accuracy: MulticlassExactMatch | None = None
        self.val_rule_accuracy: MulticlassExactMatch | None = None

        if isinstance(self.model, ExplicitModelWith) and isinstance(
            self.model.prediction_model, RavenKnownPrediction
        ):
            self.encode_query = False
        else:
            self.encode_query = True

    def forward(self, x_c, x_q) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        x_c = self.embedding(x_c)
        x_c += self.context_pos_encodings.unsqueeze(0)
        if self.encode_query:
            x_q = self.embedding(x_q)
            if self.query_pos_encodings is not None:
                x_q += self.query_pos_encodings.unsqueeze(0)
        y_q_embedding, z = self.model(x_c, None, x_q)
        y_q_embedding = y_q_embedding
        return y_q_embedding, z

    def compare_to_candidates(self, y_q_embedding, y_q_candidates) -> torch.FloatTensor:
        if self.encode_query:
            y_q_candidates = self.embedding(y_q_candidates)
        sim = -torch.cdist(y_q_embedding, y_q_candidates).squeeze(1)
        return sim

    def predict_rule(self, z) -> torch.FloatTensor:
        if not self.hparams.backprop_rule_pred:
            z = z.detach()
        rule_pred = self.rule_predictor(z)
        rule_pred = rule_pred.view(-1, self.n_rules, self.n_attributes)
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
                f"val_{setting}/rule_loss",
                rule_loss,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val_{setting}/rule_accuracy",
                self.val_rule_accuracy,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        self.val_accuracy(y_q_pred, y_q_label)
        self.log(
            f"val_{setting}/loss",
            y_q_loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"val_{setting}/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def configure_optimizers(self):
        dm: RavenDataModule = self.trainer.datamodule
        self.n_attributes = dm.n_attributes
        self.n_rules = dm.n_rules

        self.train_rule_accuracy = MulticlassExactMatch(num_classes=self.n_rules).to(
            self.device
        )
        self.val_rule_accuracy = MulticlassExactMatch(num_classes=self.n_rules).to(
            self.device
        )

        self.embedding = torch.nn.Linear(
            self.n_attributes, self.hparams.embedding_dim
        ).to(self.device)

        if isinstance(self.model, ExplicitModelWith):
            self.rule_predictor = torch.nn.Linear(
                self.model.context_model.z_dim,
                self.n_attributes * self.n_rules,
            ).to(self.device)

        params = (
            list(self.model.parameters())
            + list(self.embedding.parameters())
            + [self.context_pos_encodings]
        )
        if self.query_pos_encodings is not None:
            params += [self.query_pos_encodings]
        if self.rule_predictor is not None:
            params += [self.rule_predictor.parameters()]
        return torch.optim.Adam(params, lr=self.hparams.lr)
