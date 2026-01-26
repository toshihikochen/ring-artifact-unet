import random

import lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models.fpn_residual_dense_unet_gradloss import RingArtifactFPNResidualDenseUNet
from data_modules.tiff_data import RingArtifactTIFFDataModule

random.seed(42)
torch.manual_seed(42)
pl.seed_everything(42)

torch.set_float32_matmul_precision("high")


def main():
    data_module = RingArtifactTIFFDataModule(train_dir="data/tiff/train", val_dir="data/tiff/val", batch_size=4)
    model = RingArtifactFPNResidualDenseUNet()
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        accumulate_grad_batches=2,
        max_epochs=200,
        logger=[
            CSVLogger(save_dir="lightning_logs", name="fpn_residual_denseunet_gradloss"),
            TensorBoardLogger(save_dir="lightning_logs", name="fpn_residual_denseunet_gradloss"),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
