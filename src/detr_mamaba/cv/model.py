"""Model factory utilities for the computer vision branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from torchvision.models.detection import detr_resnet50
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "torchvision is required to instantiate the DETR model."
    ) from exc


@dataclass
class DetrConfig:
    num_classes: int
    pretrained: bool = True
    aux_loss: bool = True
    trainable_backbone_layers: Optional[int] = None
    dilation: bool = False


def create_forward_detr(config: DetrConfig) -> nn.Module:
    """Create a DETR model adapted to the soybean disease dataset.

    The implementation relies on the official torchvision DETR reference. It supports
    re-initialising the classification head to match the desired number of classes while
    keeping the weights of the backbone when ``pretrained`` is ``True``.
    """

    model = detr_resnet50(
        weights="DEFAULT" if config.pretrained else None,
        weights_backbone="DEFAULT" if config.pretrained else None,
        num_classes=config.num_classes,
        aux_loss=config.aux_loss,
        trainable_backbone_layers=config.trainable_backbone_layers,
        dilation=config.dilation,
    )

    # torchvision's DETR expects the background class to be accounted for in the last
    # logit. By default ``num_classes`` already includes the background token.
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze the convolutional backbone parameters of DETR."""

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for param in backbone.parameters():
        param.requires_grad = False


def create_transform(image_size: tuple[int, int] = (800, 800)) -> GeneralizedRCNNTransform:
    """Create the preprocessing transform used by DETR."""

    min_size, max_size = image_size
    transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return transform
