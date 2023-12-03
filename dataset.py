import torch
from torch.utils.data import Dataset
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
import os
from glob import glob
from typing import List
from tqdm import tqdm
import numpy as np


class TinyStories(Dataset):
    def __init__(self, data_path: str, train: bool, train_size: float = 0.95, limit: int = None):
        all_inputs = np.load(data_path)
        train_size = int(train_size * all_inputs.shape[0])
        if train:
            self.input_ids = all_inputs[:train_size]
        else:
            self.input_ids = all_inputs[train_size:]
        if limit is not None:
            self.input_ids = self.input_ids[:limit]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]).long()
