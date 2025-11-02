"""Data loading utilities for soybean leaf disease detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _xywhn_to_xyxy(
    boxes: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Convert normalized YOLO style boxes into absolute xyxy boxes.

    Args:
        boxes: Array shaped ``(N, 4)`` in ``(x_center, y_center, width, height)`` order,
            normalized in ``[0, 1]``.
        image_size: Image size as ``(width, height)`` in pixels.

    Returns:
        Array of shape ``(N, 4)`` with ``(x_min, y_min, x_max, y_max)`` boxes in pixels.
    """

    if boxes.size == 0:
        return boxes

    img_w, img_h = image_size
    x_c, y_c, w, h = boxes.T

    x1 = (x_c - w / 2.0) * img_w
    y1 = (y_c - h / 2.0) * img_h
    x2 = (x_c + w / 2.0) * img_w
    y2 = (y_c + h / 2.0) * img_h
    return np.stack([x1, y1, x2, y2], axis=1)


@dataclass
class DetectionSample:
    """Container returned by :class:`SoyDiseaseDetectionDataset`.

    Attributes:
        image: Tensor in ``C x H x W`` format.
        target: Dictionary with keys compatible with DETR: ``boxes`` (``N x 4``),
            ``labels`` (``N``), ``image_id`` (``1``), ``area`` (``N``) and ``iscrowd`` (``N``).
        path: Path to the image file, useful for diagnostics.
    """

    image: torch.Tensor
    target: Dict[str, torch.Tensor]
    path: Path


class SoyDiseaseDetectionDataset(Dataset[DetectionSample]):
    """Dataset loader for YOLO-style bounding box annotations.

    The dataset is organised with two folders ``images/`` and ``labels/`` containing
    image files (``*.jpg``, ``*.png`` ...) and annotation files (``*.txt``) respectively.
    Each label file should contain one row per bounding box following the format::

        <class_id> <x_center> <y_center> <width> <height>

    Coordinates are expected to be normalised in the ``[0, 1]`` range. The class id is
    zero-based and will be mapped to the ``class_mapping`` provided at initialisation.
    """

    def __init__(
        self,
        root: Path | str,
        transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        augmentation: Optional[Callable[[Image.Image, Dict[str, torch.Tensor]], Tuple[Image.Image, Dict[str, torch.Tensor]]]] = None,
        class_mapping: Optional[Dict[int, int]] = None,
        image_extensions: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.transforms = transforms
        self.augmentation = augmentation
        self.class_mapping = class_mapping or {}
        self.image_extensions = tuple(image_extensions or (".jpg", ".jpeg", ".png", ".bmp"))

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in self.image_extensions
        )
        if not self.image_paths:
            raise RuntimeError(f"No image files found in {self.image_dir}")

    def _load_targets(self, image_path: Path, image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([hash(image_path.name) % (2**31 - 1)]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        raw = np.loadtxt(label_path, ndmin=2)
        if raw.shape[1] < 5:
            raise ValueError(f"Label file {label_path} has invalid format")

        class_ids = raw[:, 0].astype(int)
        boxes_xywh = raw[:, 1:5].astype(np.float32)
        boxes = _xywhn_to_xyxy(boxes_xywh, image_size)
        mapped_labels = np.array([self.class_mapping.get(int(cls), int(cls)) for cls in class_ids], dtype=np.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(mapped_labels, dtype=torch.int64),
            "image_id": torch.tensor([hash(image_path.name) % (2**31 - 1)]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DetectionSample:
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            target = self._load_targets(image_path, (width, height))

            if self.augmentation is not None:
                img, target = self.augmentation(img, target)

            if self.transforms is not None:
                tensor = self.transforms(img)
            else:
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return DetectionSample(image=tensor, target=target, path=image_path)


def collate_fn(batch: List[DetectionSample]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[Path]]:
    images = [sample.image for sample in batch]
    targets = [sample.target for sample in batch]
    paths = [sample.path for sample in batch]
    return images, targets, paths
