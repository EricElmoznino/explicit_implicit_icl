from dm_alchemy.types import graphs
from dm_alchemy.types.stones_and_potions import *
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types.graphs import Graph
import numpy as np
import torch
import frozendict
from tqdm import tqdm
from itertools import product
import os
import os
from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import gdown
import random
from pytorch_lightning import LightningDataModule
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Any, Iterator, Union
from models.utils import AlchemyEmbedding
import torch.nn as nn
from dm_alchemy.types.stones_and_potions import *
import os
import wandb
from torch.utils.data import (
    Subset,
    Sampler,
    DataLoader,
    WeightedRandomSampler,
    RandomSampler,
    BatchSampler
)

N_ENV = 167424

POTION_COLOUR_AT_PERCEIVED_POTION = frozendict.frozendict(
    {
        PerceivedPotion(0, 1): "green",
        PerceivedPotion(0, -1): "red",
        PerceivedPotion(1, 1): "yellow",
        PerceivedPotion(1, -1): "orange",
        PerceivedPotion(2, 1): "turquoise",
        PerceivedPotion(2, -1): "pink",
    }
)

COLOUR_NAME_AT_COORD = frozendict.frozendict({-1: "purple", 0: "blurple", 1: "blue"})

ROUNDNESS_NAME_AT_COORD = frozendict.frozendict(
    {-1: "pointy", 0: "somewhat pointy", 1: "round"}
)

SIZE_NAME_AT_COORD = frozendict.frozendict({-1: "small", 0: "medium", 1: "large"})


def latents_to_env(nodes: torch.Tensor):
    return (
        nodes[..., 0] * (48 * 8 * 109)
        + nodes[..., 1] * (8 * 109)
        + nodes[..., 2] * (109)
        + nodes[..., 3]
    )


def nodes_to_state(nodes: torch.Tensor):
    return (
        nodes[..., 0] * (3 * 3 * 4)
        + nodes[..., 1] * (3 * 4)
        + nodes[..., 2] * (4)
        + nodes[..., 3]
    )


def state_to_nodes(states: torch.Tensor):
    states_ = states.clone()  # To make it not in-place

    out = []
    for dim in reversed((3, 3, 3, 4)):
        out.append(states_ % dim)
        states_ = states_ // dim
    return torch.stack(tuple(reversed(out))).T


def stone_to_str(stone: PerceivedStone):
    colour = COLOUR_NAME_AT_COORD[stone.perceived_coords[0]]
    size = SIZE_NAME_AT_COORD[stone.perceived_coords[1]]
    roundness = ROUNDNESS_NAME_AT_COORD[stone.perceived_coords[2]]

    return colour + ", " + size + ", " + roundness


def potion_to_str(self, potion: PerceivedPotion):
    return POTION_COLOUR_AT_PERCEIVED_POTION[potion]


def coord_to_index(states: torch.Tensor):
    """
    Converts alchemy samples using coordinates (e.g. [-1,0,1] for potions)
    to those using indices (e.g. [0,1,2] for potions)

    Args:
        states (torch.Tensor): Tensor of shape [..., 9] containing alchemy samples
    """
    # Not inplace
    states = states.clone()

    # fmt: off
    states[torch.kron(states[...,3] == -3, torch.eye(9)[3].bool()).view_as(states)] = 0
    states[torch.kron(states[...,3] == 1, torch.eye(9)[3].bool()).view_as(states)] = 2
    states[torch.kron(states[...,3] == -1, torch.eye(9)[3].bool()).view_as(states)] = 1
    states[torch.kron(states[...,8] == -3, torch.eye(9)[8].bool()).view_as(states)] = 0
    states[torch.kron(states[...,8] == 1, torch.eye(9)[8].bool()).view_as(states)] = 2
    states[torch.kron(states[...,8] == -1, torch.eye(9)[8].bool()).view_as(states)] = 1
    states[...,:3] += 1
    states[...,-4:-1] += 1
    # fmt: on

    return states


class AlchemyDataset(Dataset):
    """
    Dataset of Alchemy environements. An element is a tensor of shape [context_len + 1, 7] containing
    <context_len> samples from the environment plus one that the model will need to predict
    """

    def __init__(
        self,
        context_len: Union[int, Tuple[int, int]],
        non_trivial_query: bool,
    ) -> None:
        super().__init__()

        self.context_len = context_len
        self.non_trivial_query = non_trivial_query
        self.setup()

    def setup(self):
        print("Setting up dataset...", end="")

        # Download dataset if not already there
        if not os.path.isfile(os.path.join(os.getcwd(), "transitions.pt")):
            print("Downloading...", end="")
            url = "https://drive.google.com/uc?id=16L0mdMHvNxxltGJKEy9VKNVPk7dbOvIz"
            output = "transitions.pt"
            gdown.download(url, output, quiet=False)
        else:
            print("Already downloaded...", end="")
        self.full_data = torch.load("transitions.pt")
        print("Done")

        self.fixed_context_lenght = False

    def __len__(self):
        return len(self.full_data)

    def __getitem__(
        self, i: int, context_len: int = None, return_idx: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """
        Takes index i in the dataset and randomly select context and target (if not specifed)
        """
        if context_len == None:
            context_len = self.context_len
        if self.fixed_context_lenght == True:
            context_len = self.context_len[1]

        if isinstance(context_len, tuple):
            max_l = context_len[1] + 1
            l = random.randint(context_len[0], context_len[1])
        else:
            max_l = context_len + 1
            l = context_len

        data = torch.zeros(size=(max_l, 9), dtype=int)

        # Select index of query
        if self.non_trivial_query:
            non_trivial_idx = torch.arange(self.full_data.shape[1])[
                (self.full_data[i, :, :4] != self.full_data[i, :, -4:]).any(-1)
            ]
            query_idx = non_trivial_idx[
                torch.randint(0, len(non_trivial_idx) - 1, (1,))
            ]
        else:
            query_idx = torch.randint(0, self.full_data.shape[1], (1,))

        # Select the context indices
        context_idx = torch.randperm(self.full_data.shape[1])
        context_idx = context_idx[context_idx != query_idx]
        context_idx = context_idx[:l]
        data[: l + 1] = torch.cat(
            (self.full_data[i][context_idx], self.full_data[i][query_idx])
        )

        if return_idx:
            return data, l, torch.cat([context_idx, query_idx])

        return data, l



class AlchemyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        context_len: Union[int, Tuple[int, int]],
        non_trivial_query: bool,
        holdout_val: bool,
        val_size: int,
        num_env: int,
    ) -> None:
        """
        Setup the dataloaders

        Args:
            batch_size (int): Batch size
            context_len (Union[int, Tuple[int, int]]): Number of transitions in the context. Can be an interval.
            non_trivial_query (bool): Always give non-trivial queries.
            holdout_val (bool): Non-overlapping train and val.
            val_size (int): Size of validation set.
            train_mult (int): How many samples from every environment
            num_env (int): How many environment to put
            bias (float): Propotion of each batch that is the same element
        """
        super().__init__()

        self.batch_size = batch_size
        self.context_len = context_len
        self.non_trivial_query = non_trivial_query
        self.holdout_val = holdout_val
        self.val_size = val_size
        self.num_env = num_env

    def setup(self, stage: str):
        full_data = AlchemyDataset(
            context_len=self.context_len, non_trivial_query=self.non_trivial_query
        )

        if self.num_env != -1:
            full_data = Subset(
                full_data, torch.randperm(len(full_data))[: self.num_env]
            )

        if self.holdout_val:
            train_idx, val_idx = torch.split(
                torch.randperm(len(full_data)),
                [len(full_data) - self.val_size, self.val_size],
            )

            # Save heldout indices
            if wandb.run != None:
                dir = os.path.join(wandb.run.dir, "heldout-idx")
                torch.save(val_idx, dir)

            train_data = Subset(full_data, train_idx)
            val_data = Subset(full_data, val_idx)
        else:
            train_data = full_data
            val_data = Subset(
                full_data, torch.randperm(len(full_data))[: self.val_size]
            )

        self.train_data = train_data
        self.val_data = val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,)
    