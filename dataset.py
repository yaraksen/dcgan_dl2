import torch
from torch.utils.data import Dataset
import os
from glob import glob
from typing import List
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


class KittyDataset(Dataset):
    def __init__(self, data_path: str, limit: int = None):
        self.all_files = list(Path(data_path).glob("*.jpg"))

        if limit is not None:
            self.all_files = self.all_files[:limit]
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.all_files[idx]))
        return image
    

class TestKittyDataset(Dataset):
    def __init__(self, images: torch.Tensor):
        self.all_images = images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        return {"images": self.all_images[idx]}
