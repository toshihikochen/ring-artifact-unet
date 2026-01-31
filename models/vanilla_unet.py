"""
Vanilla UNet - 最简单的UNet
"""
import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder部分
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck(x)

        # decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # final
        output = self.final_conv(x)

        return output


class RingArtifactVanillaUNet(pl.LightningModule):
    def __init__(self, learning_rate: float = 3e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = UNet(in_channels=1, out_channels=1)
        self.criterion = L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noise, label = batch
        pred = self.model(noise)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noise, label = batch
        pred = self.model(noise)
        loss = self.criterion(pred, label)
        psnr, ssim = self.psnr(pred, label), self.ssim(pred, label)
        self.log_dict({"val_loss": loss, "val_psnr": psnr, "val_ssim": ssim}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        noise, label = batch
        pred = self.model(noise)
        loss = self.criterion(pred, label)
        psnr, ssim = self.psnr(pred, label), self.ssim(pred, label)
        self.log_dict({"test_loss": loss, "test_psnr": psnr, "test_ssim": ssim}, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx=None):
        noise = batch
        pred = self.model(noise)
        pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
