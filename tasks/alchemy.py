import torch
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Any, Iterator, Union
import torch.nn as nn
from dm_alchemy.types.stones_and_potions import *
import os
from data.alchemy import nodes_to_state, state_to_nodes
import wandb
from torch.utils.data import (
    Subset,
    Sampler,
    DataLoader,
    WeightedRandomSampler,
    RandomSampler,
    BatchSampler
)
from models.utils import AlchemyEmbedding

class AlchemyICL(LightningModule):
    def __init__(
        self,
        seq_model: nn.Module, # Should have a simple explicit/implicit model here which uses a <data.utils.AlchemyEmbedding> embedding
        save_model: bool,
        lr: float
    ):
        super().__init__()

        # Sequence model
        self.seq_model = seq_model
        self.lr = lr

        self.save_model = save_model
        if save_model:
            os.mkdir(os.path.join(wandb.run.dir, "models/"))

        self.automatic_optimization = False

    def on_train_epoch_end(self) -> None:
        if self.save_model & (wandb.run != None):
            dir = os.path.join(
                wandb.run.dir, f"models/epoch={self.trainer.current_epoch}"
            )
            torch.save(self.seq_model, dir)
            wandb.save(dir)
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.seq_model.parameters(), lr=self.lr)

    def training_step(self, batch: Any, batch_idx: Any):
        data, query_idx = batch
        N = data.size(0)
        opt = self.optimizers()
        opt.zero_grad()

        # Compute target and logits for every node
        targets = data[torch.arange(N), query_idx, -4:].clone()

        data[torch.arange(N), query_idx, -4:] = AlchemyEmbedding.QUERY_MASK
        logits = self.seq_model(data)[torch.arange(N), query_idx]

        # Equivalent of np.unravel_multi_index
        loss = nn.functional.cross_entropy(
            input=logits, target=nodes_to_state(targets)
        )
        self.manual_backward(loss)
        opt.step()

        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: Any):
        data, query_idx = batch
        N = data.size(0)

        # Compute target and logits for every node
        targets = data[torch.arange(N), query_idx, -4:].clone()
        data[torch.arange(N), query_idx, -4:] = AlchemyEmbedding.QUERY_MASK
        logits = self.seq_model(data)[torch.arange(N), query_idx]

        loss = nn.functional.cross_entropy(
            input=logits, target=nodes_to_state(targets)
        )
        pred = state_to_nodes(torch.softmax(logits, dim=-1).argmax(-1))

        correct = (pred == targets).all(dim=-1)
        acc = sum(correct) / len(correct)
        start_stone = data[torch.arange(N), query_idx, :4]
        trivial_pred = (pred == start_stone).all(dim=-1)
        trivial_ratio = sum(trivial_pred) / len(trivial_pred)
        non_trivial_correct = torch.logical_and(~trivial_pred, correct)
        non_trivial_correct_ratio = sum(non_trivial_correct) / sum(~trivial_pred)

        self.log("val/trivial_ratio", trivial_ratio)
        self.log("val/non_trivial_correct_ratio", non_trivial_correct_ratio)
        self.log("val/loss", loss)
        self.log("val/acc", acc)

    def on_validation_start(self) -> None:
        self.trainer.datamodule.val_data.dataset.fixed_context_lenght = True
    def on_validation_end(self) -> None:
        self.trainer.datamodule.val_data.dataset.fixed_context_lenght = False