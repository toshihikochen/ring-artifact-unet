import importlib
from typing import Any

from lightning import LightningModule

from models.vanilla_unet import RingArtifactVanillaUNet
from models.dense_unet import RingArtifactDenseUNet
from models.dense_unet_gradloss import RingArtifactDenseUNetPlusGradLoss
from models.residual_dense_unet import RingArtifactResidualDenseUNet
from models.fpn_residual_dense_unet import RingArtifactFPNResidualDenseUNet


__all__ = [
    "get_model",
    "RingArtifactVanillaUNet",
    "RingArtifactDenseUNet",
    "RingArtifactDenseUNetPlusGradLoss",
    "RingArtifactResidualDenseUNet",
    "RingArtifactFPNResidualDenseUNet"
]


def get_model(model_name: str):
    """
    根据模型名称创建对应的 Lightning 模型

    Args:
        model_name: 模型名称，支持以下选项:
            - "vanilla_unet": 基础 U-Net
            - "dense_unet": 密集连接 U-Net
            - "dense_unet_gradloss": 带梯度损失的密集 U-Net
            - "residual_dense_unet": 残差密集 U-Net
            - "fpn_residual_dense_unet": 特征金字塔残差密集 U-Net

    Returns:
        model: Lightning 模型实例
    """
    model_mapping = {
        "vanilla_unet": ("models.vanilla_unet", "RingArtifactVanillaUNet"),
        "dense_unet": ("models.dense_unet", "RingArtifactDenseUNet"),
        "dense_unet_gradloss": ("models.dense_unet_gradloss", "RingArtifactDenseUNetPlusGradLoss"),
        "residual_dense_unet": ("models.residual_dense_unet", "RingArtifactResidualDenseUNet"),
        "fpn_residual_dense_unet": ("models.fpn_residual_dense_unet", "RingArtifactFPNResidualDenseUNet"),
    }

    if model_name not in model_mapping:
        available_models = ", ".join(model_mapping.keys())
        raise ValueError(f"{model_name} is not a valid model name. Available models: {available_models}")

    module_path, class_name = model_mapping[model_name]

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    return model_class