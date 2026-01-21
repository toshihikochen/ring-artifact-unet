from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn as nn
from torchmetrics.functional.image import peak_signal_noise_ratio

from models.dense_unet import RingArtifactDenseUNet


NOISE_PATH = "data/tiff/test/input/Cube_sino00034.tif"
LABEL_PATH = "data/tiff/test/target/Cube_sino00034.tif"
CHECKPOINT_PATH = "lightning_logs/denseunet/version_0/checkpoints/epoch=99-step=3100.ckpt"
DEVICE = "cuda"
EPS = 1e-12

model = RingArtifactDenseUNet.load_from_checkpoint(CHECKPOINT_PATH)
model.to(DEVICE)
model.eval()


def inference_one_image(model: nn.Module, noise_data: np.ndarray) -> np.ndarray:
    """ 推理一张图片

    Args:
        model: 模型
        noise_data: 噪声

    Returns:
        预测结果
    """
    # normalize noise to [0, 1]
    maximum, minimum = noise_data.max(), noise_data.min()
    noise_data = (noise_data - minimum) / (maximum - minimum + EPS)

    noise_data = np.expand_dims(noise_data, axis=0)
    noise_tensor = torch.from_numpy(noise_data).to(DEVICE)

    with torch.no_grad():
        noise_tensor = noise_tensor.unsqueeze(0)
        pred = model(noise_tensor)
        pred = pred[0]

    pred = pred.squeeze(0).cpu().numpy()

    return pred


def visualize(inputs: np.ndarray, pred: np.ndarray, label: Optional[np.ndarray] = None):
    """ 可视化结果

    Args:
        inputs: 输入图片
        pred: 预测结果
        label: 标签
    """

    maximum, minimum = inputs.max(), inputs.min()
    inputs = (inputs - minimum) / (maximum - minimum + EPS)

    # maximum, minimum = pred.max(), pred.min()
    # pred = (pred - minimum) / (maximum - minimum + EPS)

    maximum, minimum = label.max(), label.min()
    label = (label - minimum) / (maximum - minimum + EPS)

    if label is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].matshow(inputs)
        axes[0].set_title("Input (Noise)")
        axes[0].axis('off')

        axes[1].matshow(pred)
        axes[1].set_title("Prediction")
        axes[1].axis('off')

        axes[2].matshow(label)
        axes[2].set_title("Label (GT)")
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].matshow(inputs)
        axes[0].set_title("Input (Noise)")
        axes[0].axis('off')

        axes[1].matshow(pred)
        axes[1].set_title("Prediction")
        axes[1].axis('off')

    psnr = peak_signal_noise_ratio(torch.from_numpy(pred), torch.from_numpy(label), data_range=1.0)
    print(f"PSNR: {psnr:.4f}")

    plt.tight_layout()
    plt.show()


def main():
    noise_data = tifffile.imread(NOISE_PATH)
    pred_data = inference_one_image(model, noise_data)

    if LABEL_PATH is not None:
        label_data = tifffile.imread(LABEL_PATH)
    else:
        label_data = None
    visualize(noise_data, pred_data, label_data)


if __name__ == "__main__":
    main()
