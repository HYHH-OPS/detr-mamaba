"""Augmentation utilities for soybean leaf images."""
from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch


def random_flip(image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    if random.random() < 0.5:
        width, _ = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = target["boxes"].clone()
        x_min = width - boxes[:, 2]
        x_max = width - boxes[:, 0]
        boxes[:, 0] = x_min
        boxes[:, 2] = x_max
        target = {**target, "boxes": boxes}
    return image, target


def color_jitter(image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    if random.random() < 0.8:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    return image, target


def compose(*augmentations):
    def wrapper(image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        for augmentation in augmentations:
            image, target = augmentation(image, target)
        return image, target

    return wrapper
