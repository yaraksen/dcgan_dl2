import torch
from torch.utils.data import Dataset
import os
from glob import glob
from typing import List
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


# class KittyDataset(Dataset):
#     def __init__(self, data_path: str, train: bool, train_size: float = 0.95, limit: int = None):
#         all_files = Path(data_path).glob("/**/*.jpg")
#         train_size = int(train_size * len(all_files))

#         if train:
#             self.all_files = all_files[:train_size]
#         else:
#             self.all_files = all_files[train_size:]

#         if limit is not None:
#             self.all_files = self.all_files[:limit]

#     def __len__(self):
#         return len(self.all_files)

#     def __getitem__(self, idx):
#         image = np.array(Image.open(self.all_files[idx]))
#         assert image.dim() == 3
#         return torch.tensor(image)


class KittyDataset(Dataset):
    def __init__(self, data_path: str, limit: int = None):
        self.all_files = list(Path(data_path).glob("*.jpg"))

        if limit is not None:
            self.all_files = self.all_files[:limit]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.all_files[idx]))
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
