import os
import random

import numpy as np
import tifffile
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


EPS = 1e-9


class RingArtifactTIFFDataset(Dataset):
    def __init__(self, data_dir: str, stage: str = "train"):
        self.label_dir = os.path.join(data_dir, "target")
        self.noise_dir = os.path.join(data_dir, "input")
        self.stage = stage

        self.data_list = []
        for filename in os.listdir(self.label_dir):
            noise_path = os.path.join(self.noise_dir, filename)
            if os.path.exists(noise_path):
                self.data_list.append(filename)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        noise_path = os.path.join(self.noise_dir, self.data_list[index])
        label_path = os.path.join(self.label_dir, self.data_list[index])

        noise_data = tifffile.imread(noise_path)
        label_data = tifffile.imread(label_path)

        # normalize to [0, 1] using noise data
        minimum, maximum = noise_data.min(), noise_data.max()
        noise_data = (noise_data - minimum) / (maximum - minimum + EPS)

        minimum, maximum = label_data.min(), label_data.max()
        label_data = (label_data - minimum) / (maximum - minimum + EPS)

        # random crop from (720, 256) to (256, 256) when training
        if self.stage == "train":
            rand = random.randint(0, 720 - 256)
            noise_data = noise_data[rand:rand+256, :]
            label_data = label_data[rand:rand+256, :]

        # expand dimension
        noise_data = np.expand_dims(noise_data, axis=0)
        label_data = np.expand_dims(label_data, axis=0)

        noise_tensor = torch.from_numpy(noise_data).float()
        label_tensor = torch.from_numpy(label_data).float()

        return noise_tensor, label_tensor


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
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False
        )
