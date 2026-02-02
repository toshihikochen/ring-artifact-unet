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
    │   ├── 001-something-0002.tiff
    │   ├── ...
    │   └── 114-other_thing-1919.tiff
    └── noisy/
        ├── 001-something-0001-000.tiff
        ├── ...
        ├── 001-something-0001-072.tiff  # 一个label可以对应多个noisy（添加后缀）
        ├── 001-something-0002.tiff      # 也可以一个label对应一个noisy（文件名相同）
        └── ...
    """
    def __init__(self, data_dir: str, stage: str = "train"):
        self.label_dir = os.path.join(data_dir, "label")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.stage = stage

        self.label_to_noisy = []
        self.noisy_to_label = []

        for label_filename in os.listdir(self.label_dir):
            name, ext = os.path.splitext(label_filename)
            noisy_pattern = os.path.join(self.noisy_dir, f"{name}*{ext}")

            label_path = os.path.join(self.label_dir, label_filename)
            noisy_paths = glob.glob(noisy_pattern)

            self.label_to_noisy.append((label_path, noisy_paths))
            for noisy_path in noisy_paths:
                self.noisy_to_label.append((noisy_path, label_path))


    def __len__(self):
        if self.stage == "train":
            return len(self.label_to_noisy)
        return len(self.noisy_to_label)

    def __getitem__(self, index):
        if self.stage == "train":  # randomly choose one noisy corresponding to the label while training
            label_path, noisy_paths = self.label_to_noisy[index]
            noisy_path = random.choice(noisy_paths)
        else:  # pick all noisy corresponding to the label while validating or testing
            noisy_path, label_path = self.noisy_to_label[index]

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
