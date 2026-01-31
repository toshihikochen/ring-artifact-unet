"""
DenseUNet + GradLoss
"""
import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import image_gradients


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.down(x)


class DenseEncoderBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, out_channels):
        super().__init__()
        self.dense = DenseBlock(in_channels, growth_rate, num_layers)
        self.trans = TransitionDown(in_channels + growth_rate * num_layers, out_channels)

    def forward(self, x):
        x = self.dense(x)
        skip = x
        x = self.trans(x)
        return x, skip


class DenseDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, num_layers, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dense = DenseBlock(out_channels + skip_channels, growth_rate, num_layers)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        return x


class DenseUNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 growth_rate=16, num_layers=4):
        super().__init__()

        g, L = growth_rate, num_layers

        # encoder base channels
        c1, c2, c3, c4 = 64, 128, 256, 512

        # encoder
        self.enc1 = DenseEncoderBlock(in_channels, g, L, c1)
        self.enc2 = DenseEncoderBlock(c1, g, L, c2)
        self.enc3 = DenseEncoderBlock(c2, g, L, c3)
        self.enc4 = DenseEncoderBlock(c3, g, L, c4)

        # channel math
        skip1_c = in_channels + g * L
        skip2_c = c1 + g * L
        skip3_c = c2 + g * L
        skip4_c = c3 + g * L

        bottleneck_in = c4
        bottleneck_out = bottleneck_in + g * L

        # bottleneck
        self.bottleneck = DenseBlock(bottleneck_in, g, L)

        # decoder
        self.dec4 = DenseDecoderBlock(bottleneck_out, skip4_c, g, L, c4)
        dec4_out = c4 + skip4_c + g * L

        self.dec3 = DenseDecoderBlock(dec4_out, skip3_c, g, L, c3)
        dec3_out = c3 + skip3_c + g * L

        self.dec2 = DenseDecoderBlock(dec3_out, skip2_c, g, L, c2)
        dec2_out = c2 + skip2_c + g * L

        self.dec1 = DenseDecoderBlock(dec2_out, skip1_c, g, L, c1)
        dec1_out = c1 + skip1_c + g * L

        self.final_conv = nn.Conv2d(dec1_out, out_channels, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.final_conv(x)


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, label):
        pred_dy, pred_dx = image_gradients(pred)
        label_dy, label_dx = image_gradients(label)
        loss = self.criterion(pred_dy, label_dy) + self.criterion(pred_dx, label_dx)

        return loss


class Criterion(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.l1loss = nn.L1Loss()
        self.grad_loss = GradLoss()

    def forward(self, pred, label):
        loss = self.l1loss(pred, label) + self.alpha * self.grad_loss(pred, label)
        return loss


class RingArtifactDenseUNetPlusGradLoss(pl.LightningModule):
    def __init__(self, learning_rate: float = 3e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = DenseUNet(in_channels=1, out_channels=1)
        self.criterion = Criterion()
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
