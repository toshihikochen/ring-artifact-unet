from typing import Literal

import lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from data_modules.tiff_data import RingArtifactTIFFDataModule
from models import get_model

"""
"vanilla_unet": 基础 U-Net
"dense_unet": 密集连接 U-Net
"dense_unet_gradloss": 带图像梯度损失的密集 U-Net
"residual_dense_unet": 残差密集 U-Net（带图像梯度损失）
"fpn_residual_dense_unet": 特征金字塔残差密集 U-Net（带图像梯度损失）
"""
MODEL_NAME = "vanilla_unet"

TRAIN_DIR = "data/my_tiff/train"
VAL_DIR = "data/my_tiff/val"

BATCH_SIZE = 4
BATCH_ACCUMULATION = 2
RANDOM_SEED = 42
EPOCHS = 200
NUM_WORKERS = 4
PIN_MEMORY = True


pl.seed_everything(RANDOM_SEED)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # for Ampere or higher gpus
    torch.set_float32_matmul_precision("high")


def main():
    data_module = RingArtifactTIFFDataModule(
        train_dir=TRAIN_DIR, val_dir=VAL_DIR,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    model = get_model(MODEL_NAME)()
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        accumulate_grad_batches=BATCH_ACCUMULATION,
        max_epochs=EPOCHS,
        deterministic=True,
        logger=[
            CSVLogger(save_dir="lightning_logs", name=MODEL_NAME),
            TensorBoardLogger(save_dir="tensorboard_logs", name=MODEL_NAME),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
