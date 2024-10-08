from typing import Literal
import pickle
import os
import numpy as np
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings


warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")


class RavenDataModule(LightningDataModule):
    OODSetting = Literal["inpo", "expo-l1", "expo-l2"]
    NValues = Literal[20, 30, 40, 80]

    def __init__(
        self,
        data_dir: str,
        n_values: NValues = 40,
        setting: OODSetting = "inpo",
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_values: int = n_values
        self.n_attributes: int | None = None
        self.n_rules: int | None = None

    def setup(self, stage: str) -> None:
        folder = {
            "inpo": "l2_inpo",
            "expo-l1": "l1_expo",
            "expo-l2": "l2_expo",
        }[self.hparams.setting]
        folder = os.path.join(self.hparams.data_dir, f"{folder}_{self.n_values}")
        self.train_data = RavenDataset(os.path.join(folder, "train_visual.pkl"))
        self.val_iid_data = RavenDataset(os.path.join(folder, "validation_visual.pkl"))
        self.val_ood_data = RavenDataset(os.path.join(folder, "test_visual.pkl"))
        self.n_attributes = self.train_data.num_attributes
        self.n_rules = self.train_data.num_rules

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.val_iid_data,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
            ),
            DataLoader(
                self.val_ood_data,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
            ),
        ]


class RavenDataset(MapDataPipe):
    rule_codes = [
        ("constant", 0),
        ("progression", -2),
        ("progression", -1),
        ("progression", 1),
        ("progression", 2),
        ("arithmetic", -1),
        ("arithmetic", 1),
        ("comparison", -1),  # MIN
        ("comparison", 1),  # MAX
        ("varprogression", 1),  # +1, +2
        ("varprogression", 2),  # +2, +1
        ("varprogression", -1),  # -1, -2
        ("varprogression", -2),  # -2, -1
    ]

    def __init__(self, data_path: str) -> None:
        super().__init__()
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        def process_datapoint(
            datapoint: dict,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
            label = datapoint["label"]
            context = datapoint["symbol"][:6][:, 0, :]
            query = datapoint["symbol"][6:8][:, 0, :]
            candidates = datapoint["symbol"][8:][:, 0, :]
            rule = np.array([self.rule_codes.index(r[1:]) for r in datapoint["rules"]])
            return context, query, candidates, label, rule

        context, query, candidates, label, rule = [], [], [], [], []
        for datapoint in data:
            c, q, cs, l, r = process_datapoint(datapoint)
            context.append(c)
            query.append(q)
            candidates.append(cs)
            label.append(l)
            rule.append(r)
        self.context, self.query, self.candidates, self.label, self.rule = (
            torch.from_numpy(np.stack(context)).float(),
            torch.from_numpy(np.stack(query)).float(),
            torch.from_numpy(np.stack(candidates)).float(),
            torch.from_numpy(np.stack(label)),
            torch.from_numpy(np.stack(rule)),
        )

        self.num_attributes = self.context.shape[-1]
        self.num_values = self.context.max() + 1
        self.num_rules = len(self.rule_codes)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(
        self, index
    ) -> tuple[FloatTensor, FloatTensor, FloatTensor, int, LongTensor]:
        return (
            self.context[index],
            self.query[index],
            self.candidates[index],
            self.label[index],
            self.rule[index],
        )
