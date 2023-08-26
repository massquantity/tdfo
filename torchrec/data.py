from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_set: pd.DataFrame):
        self.train_set = train_set

    def __len__(self) -> int:
        return len(self.train_set)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        row = self.train_set.iloc[index]
        return torch.LongTensor(row["train_interactions"]), torch.LongTensor(row["labels"])


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, eval_set: pd.DataFrame):
        self.eval_set = eval_set

    def __len__(self) -> int:
        return len(self.eval_set)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        row = self.eval_set.iloc[index]
        return torch.LongTensor(row["eval_seqs"]), torch.LongTensor(row["candidate_items"])


class Bert4RecDataLoader:
    def __init__(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame,
        train_batch_size: int,
        eval_batch_size: int,
    ):
        self.train_data = train_data
        self.eval_data = eval_data
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def get_pytorch_dataloaders(self, rank: int, world_size: int):
        train_loader = self._get_train_loader(rank, world_size)
        val_loader = self._get_eval_loader(rank, world_size)
        return train_loader, val_loader

    def _get_train_loader(self, rank: int, world_size: int):
        sampler = DistributedSampler(
            TrainDataset(self.train_data),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        dataloader = DataLoader(
            TrainDataset(self.train_data),
            batch_size=self.train_batch_size,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader

    def _get_eval_loader(self, rank: int, world_size: int):
        sampler = DistributedSampler(
            TrainDataset(self.eval_data),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(
            EvalDataset(self.eval_data),
            batch_size=self.eval_batch_size,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader
