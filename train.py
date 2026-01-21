import random

import lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models.dense_unet import RingArtifactDenseUNet
from data_modules.tiff_data import RingArtifactTIFFDataModule

random.seed(42)
torch.manual_seed(42)
pl.seed_everything(42)


def main():
    data_module = RingArtifactTIFFDataModule(train_dir="data/tiff/train", val_dir="data/tiff/val", batch_size=16)
    model = RingArtifactDenseUNet()
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=100,
        logger=[
            CSVLogger(save_dir="lightning_logs", name="denseunet"),
            TensorBoardLogger(save_dir="lightning_logs", name="denseunet"),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
