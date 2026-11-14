"""Training data: images/<id>.png + steering.csv written by recorder.DataRecorder."""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .config import DATA_DIR
from .preprocessing import to_tensor


class SteeringDataset(Dataset):
    def __init__(self, data_dir: str = DATA_DIR):
        self.img_dir = os.path.join(data_dir, "images")
        self.df = pd.read_csv(os.path.join(data_dir, "steering.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rgb = np.array(Image.open(os.path.join(self.img_dir, row["image"])).convert("RGB"))
        return to_tensor(rgb), torch.tensor([row["steering"]], dtype=torch.float32)


def make_loaders(data_dir: str = DATA_DIR, batch_size: int = 64, val_fraction: float = 0.2,
                 num_workers: int = 0, seed: int = 0):
    """Random train/val split -> (train_loader, val_loader).
    num_workers=0 is safest on Windows notebooks; raise it if multiprocessing works for you."""
    dataset = SteeringDataset(data_dir)
    n_val = int(val_fraction * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    print(f"{len(train_set)} train / {len(val_set)} val samples")
    return train_loader, val_loader
