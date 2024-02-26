from typing import Literal

import numpy as np
import torch
from torch import FloatTensor, BoolTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
import scanpy as sc

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# To make splits: https://stackoverflow.com/a/76176344


class GeneCrisprDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        contexts_per_ptb: int = 50,
        n_context: int = 40,
        n_queries: int = 5,
        query_dim_pct: float = 0.5,
        perturb_type: str = "both",
        include_control: bool = True,
        batch_size: int = 128,
        train_size: float = 0.8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Load the data and store attributes
        samples, ptb_ids = load_norman2019(data_path, perturb_type, include_control)
        self.n_ptbs = len(samples)
        self.n_ptb_targets = ptb_ids.shape[1]
        self.dim = samples[0].shape[1]
        self.query_dim_pct = query_dim_pct

        # Re-order the data so that the validation set only consists of
        # paired perturbations (so that there are no truly unseen perturbations)
        ptb_ordering = ptb_ids.sum(axis=1).argsort()
        samples = [samples[i] for i in ptb_ordering]
        ptb_ids = ptb_ids[ptb_ordering]

        # Create the PyTorch train/val datasets
        self.train_data = GeneCrisprDataset(
            samples=samples[: int(len(samples) * train_size)],
            ptb_ids=ptb_ids[: int(len(samples) * train_size)],
            contexts_per_ptb=contexts_per_ptb,
            n_context=n_context,
            n_queries=n_queries,
            query_dim_pct=query_dim_pct,
            fixed=False,
        )
        self.val_data = GeneCrisprDataset(
            samples=samples[int(len(samples) * train_size) :],
            ptb_ids=ptb_ids[int(len(samples) * train_size) :],
            contexts_per_ptb=contexts_per_ptb,
            n_context=n_context,
            n_queries=n_queries,
            query_dim_pct=query_dim_pct,
            fixed=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class GeneCrisprDataset(MapDataPipe):
    # Much has been borrowed from:
    # https://github.com/uhlerlab/discrepancy_vae/blob/master/src/dataset.py

    PerturbType = Literal["single", "double", "both"]

    def __init__(
        self,
        samples: list[np.ndarray],
        ptb_ids: np.ndarray,
        contexts_per_ptb: int,
        n_context: int,
        n_queries: int = 5,
        query_dim_pct: float = 0.5,
        fixed: bool = False,
    ) -> None:
        super().__init__()
        self.n_ptbs = len(samples)
        self.dim = samples[0].shape[1]
        self.samples = samples
        self.ptb_ids = ptb_ids
        self.contexts_per_ptb = contexts_per_ptb
        self.n_context = n_context
        self.n_queries = n_queries
        self.query_dim_pct = query_dim_pct
        self.fixed = fixed
        if fixed:
            (
                self.fixed_query_indices,
                self.fixed_context_indices,
                self.fixed_query_masks,
            ) = self.make_fixed_query_context()

    def make_fixed_query_context(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        query_indices, context_indices, query_masks = [], [], []
        for ptb_samples in self.samples:
            for _ in range(self.contexts_per_ptb):
                c_indices, q_indices, q_mask = self.make_indices_and_masks(ptb_samples)
                query_indices.append(q_indices)
                context_indices.append(c_indices)
                query_masks.append(q_mask)
        return query_indices, context_indices, query_masks

    def make_indices_and_masks(
        self, sample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cell_indices = np.arange(sample.shape[0])
        np.random.shuffle(cell_indices)
        q_indices, c_indices = (
            cell_indices[: self.n_queries],
            cell_indices[self.n_queries :],
        )
        c_indices = np.random.choice(c_indices, size=self.n_context, replace=True)
        q_masks = np.zeros((self.n_queries, self.dim), dtype=bool)
        for i in range(self.n_queries):
            q_masks[
                i,
                np.random.choice(
                    self.dim, size=int(self.dim * self.query_dim_pct), replace=False
                ),
            ] = True
        return c_indices, q_indices, q_masks

    def __len__(self) -> int:
        return self.n_ptbs * self.contexts_per_ptb

    def __getitem__(
        self, item
    ) -> tuple[FloatTensor, FloatTensor, FloatTensor, BoolTensor, FloatTensor]:
        ptb_item = item // self.contexts_per_ptb
        x = self.samples[ptb_item]
        ptb_id = self.ptb_ids[ptb_item]

        if self.fixed:
            context_indices, query_indices, query_masks = (
                self.fixed_context_indices[item],
                self.fixed_query_indices[item],
                self.fixed_query_masks[item],
            )
        else:
            context_indices, query_indices, query_masks = self.make_indices_and_masks(x)

        x_c, y_q = (x[context_indices].toarray(), x[query_indices].toarray())
        x_q = np.where(query_masks, -1, y_q)  # Mask queried values using a -1

        x_c, x_q, y_q, q_mask, ptb_id = (
            torch.from_numpy(x_c),
            torch.from_numpy(x_q),
            torch.from_numpy(y_q),
            torch.from_numpy(query_masks),
            torch.from_numpy(ptb_id),
        )
        return x_c, x_q, y_q, q_mask, ptb_id


def load_norman2019(
    data_path: str,
    perturb_type: str = "both",
    include_control: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    # Read the raw data
    adata = sc.read_h5ad(data_path)
    if not include_control:
        adata = adata[adata.obs["guide_ids"] != ""]
    if perturb_type == "single":
        adata = adata[(~adata.obs["guide_ids"].str.contains(","))]
    elif perturb_type == "double":
        adata = adata[adata.obs["guide_ids"].str.contains(",")]
    samples = adata.X  # (n_cells X n_genes)
    ptb_targets = list(
        set().union(
            *[set(i.split(",")) for i in adata.obs["guide_ids"].value_counts().index]
        )
    )
    ptb_targets.remove("")
    ptb_ids = map_ptb_features(
        ptb_targets, adata.obs["guide_ids"].values
    )  # (n_cells X n_guides)

    # Group data by perturbation
    unique_ptb_ids = np.unique(ptb_ids, axis=0)
    ptb_samples = []
    for ptb_id in unique_ptb_ids:
        ptb_samples.append(samples[np.all(ptb_ids == ptb_id, axis=1)])

    del adata
    return ptb_samples, unique_ptb_ids


def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[
            [all_ptb_targets.index(i) for i in id.split(",") if i in all_ptb_targets]
        ] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)
