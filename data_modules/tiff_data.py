import glob
import os
import random

import numpy as np
import tifffile
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


EPS = 1e-9


class RingArtifactTIFFDataset(Dataset):
    """
    数据集目录结构示例：

    train/
    ├── label/
    │   ├── 001-something-0001.tiff
    │   ├── ...
    │   └── 114-other_thing-1919.tiff
    └── noisy/
        ├── 001-something-0001-000.tiff
        ├── ...
        ├── 001-something-0001-072.tiff  # 一个label可以对应多个noisy
        ├── 001-something-0002-000.tiff
        └── ...
    """
    def __init__(self, data_dir: str, stage: str = "train"):
        self.label_dir = os.path.join(data_dir, "label")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.stage = stage

        self.data_list = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.data_list[index])

        # randomly pick a noisy image
        name, _ = os.path.splitext(os.path.basename(label_path))
        pattern = os.path.join(self.noisy_dir, f"{name}*.tiff")
        noisy_path = random.choice(glob.glob(pattern))

        noisy_data = tifffile.imread(noisy_path)
        label_data = tifffile.imread(label_path)

        # normalize to [0, 1] using noise data
        minimum, maximum = noisy_data.min(), noisy_data.max()
        noisy_data = (noisy_data - minimum) / (maximum - minimum + EPS)

        minimum, maximum = label_data.min(), label_data.max()
        label_data = (label_data - minimum) / (maximum - minimum + EPS)

        # random crop from (720, 256) to (256, 256) when training
        # when validating or testing, use the whole image
        if self.stage == "train":
            rand = random.randint(0, 720 - 256)
            noisy_data = noisy_data[rand:rand + 256, :]
            label_data = label_data[rand:rand+256, :]

        # expand dimension to (1, 256, 256)
        noisy_data = np.expand_dims(noisy_data, axis=0)
        label_data = np.expand_dims(label_data, axis=0)

        noisy_tensor = torch.from_numpy(noisy_data).float()
        label_tensor = torch.from_numpy(label_data).float()

        return noisy_tensor, label_tensor


class RingArtifactTIFFDataModule(LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = RingArtifactTIFFDataset(train_dir, stage="train")
        self.val_dataset = RingArtifactTIFFDataset(val_dir, stage="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, shuffle=True, drop_last=True,
            num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, shuffle=False, drop_last=False,
            num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=self.pin_memory
        )
