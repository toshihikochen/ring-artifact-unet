"""
Feature Pyramid Network based Residual DenseUNet + GradLoss
"""
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio


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


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.dense_block = nn.Sequential(*layers)
        self.residual_conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.dense_block(x) + self.residual_conv(x)


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
        self.dense = ResidualDenseBlock(in_channels, growth_rate, num_layers)
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
        self.dense = ResidualDenseBlock(out_channels + skip_channels, growth_rate, num_layers)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dense(x)
        return x


class FPNResidualDenseUNet(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            growth_rate=32,
            encoder_layers_config=(6, 12, 24, 16),
            decoder_layers_config=(16, 12, 8, 4),
            bottleneck_layers=8
    ):
        super().__init__()

        g = growth_rate

        # encoder base channels
        c1, c2, c3, c4 = 64, 128, 256, 512

        assert len(encoder_layers_config) == 4, "encoder_layers_config 应该包含4个值"
        assert len(decoder_layers_config) == 4, "decoder_layers_config 应该包含4个值"

        self.enc1 = DenseEncoderBlock(in_channels, g, encoder_layers_config[0], c1)
        self.enc2 = DenseEncoderBlock(c1, g, encoder_layers_config[1], c2)
        self.enc3 = DenseEncoderBlock(c2, g, encoder_layers_config[2], c3)
        self.enc4 = DenseEncoderBlock(c3, g, encoder_layers_config[3], c4)

        skip1_c = in_channels + g * encoder_layers_config[0]
        skip2_c = c1 + g * encoder_layers_config[1]
        skip3_c = c2 + g * encoder_layers_config[2]
        skip4_c = c3 + g * encoder_layers_config[3]

        bottleneck_in = c4
        bottleneck_out = bottleneck_in + g * bottleneck_layers

        self.bottleneck = ResidualDenseBlock(bottleneck_in, g, bottleneck_layers)

        self.dec4 = DenseDecoderBlock(bottleneck_out, skip4_c, g, decoder_layers_config[0], c4)
        dec4_out = c4 + skip4_c + g * decoder_layers_config[0]
        self.final_conv_4 = nn.Conv2d(dec4_out, out_channels, 1)

        self.dec3 = DenseDecoderBlock(dec4_out, skip3_c, g, decoder_layers_config[1], c3)
        dec3_out = c3 + skip3_c + g * decoder_layers_config[1]
        self.final_conv_3 = nn.Conv2d(dec3_out, out_channels, 1)

        self.dec2 = DenseDecoderBlock(dec3_out, skip2_c, g, decoder_layers_config[2], c2)
        dec2_out = c2 + skip2_c + g * decoder_layers_config[2]
        self.final_conv_2 = nn.Conv2d(dec2_out, out_channels, 1)

        self.dec1 = DenseDecoderBlock(dec2_out, skip1_c, g, decoder_layers_config[3], c1)
        dec1_out = c1 + skip1_c + g * decoder_layers_config[3]
        self.final_conv_1 = nn.Conv2d(dec1_out, out_channels, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        out4 = self.final_conv_4(x)

        x = self.dec3(x, s3)
        out3 = self.final_conv_3(x)

        x = self.dec2(x, s2)
        out2 = self.final_conv_2(x)

        x = self.dec1(x, s1)
        out1 = self.final_conv_1(x)

        return out1, out2, out3, out4


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, label):
        grad_pred_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        grad_pred_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

        grad_label_x = label[:, :, :, :-1] - label[:, :, :, 1:]
        grad_label_y = label[:, :, :-1, :] - label[:, :, 1:, :]

        loss = self.criterion(grad_pred_x, grad_label_x) + self.criterion(grad_pred_y, grad_label_y)

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


class FPNCriterion(nn.Module):
    def __init__(self, *weights, alpha: float = 0.5):
        super().__init__()
        self.criterion = Criterion(alpha)
        self.weights = weights

    def forward(self, *preds, label):
        loss = 0
        for i, weight in enumerate(self.weights):
            reshaped_label = F.interpolate(label, preds[i].shape[2:], mode="nearest")
            loss += weight * self.criterion(preds[i], reshaped_label)
        return loss


class RingArtifactFPNResidualDenseUNet(pl.LightningModule):
    def __init__(self, learning_rate: float = 3e-4):
        super().__init__()
        self.model = FPNResidualDenseUNet(in_channels=1, out_channels=1)
        self.criterion = FPNCriterion(1.0, 0.6, 0.4, 0.1, alpha=0.5)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noise, label = batch
        pred = self.model(noise)
        loss = self.criterion(*pred, label=label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noise, label = batch
        pred = self.model(noise)
        loss = self.criterion(*pred, label=label)
        psnr = self.psnr(pred[0], label)
        self.log_dict({"val_loss": loss, "val_psnr": psnr}, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
