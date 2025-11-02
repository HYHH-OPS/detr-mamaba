"""Evaluation helpers for the CV branch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .data import SoyDiseaseDetectionDataset, collate_fn
from .model import create_forward_detr, DetrConfig


@dataclass
class EvaluationResult:
    losses: List[float]
    metrics: dict
    confusion_matrix: torch.Tensor | None = None
    class_names: Sequence[str] | None = None


def evaluate_model(
    checkpoint_path: Path,
    data_root: Path,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> EvaluationResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config: DetrConfig = checkpoint.get("config", DetrConfig(num_classes=len(class_names) + 1))
    model = create_forward_detr(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = SoyDiseaseDetectionDataset(data_root)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    metric_records: List[dict] = []
    losses: List[float] = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses.append(float(sum(loss for loss in loss_dict.values())))
            outputs = model(images)
            metric_records.append(
                _prepare_metrics(outputs, targets, iou_threshold, score_threshold, num_classes=len(class_names))
            )

    aggregated = _aggregate_metrics(metric_records)
    confusion = aggregated.pop("confusion_matrix", None)

    return EvaluationResult(losses=losses, metrics=aggregated, confusion_matrix=confusion, class_names=class_names)


def _prepare_metrics(outputs, targets, iou_threshold, score_threshold, num_classes: int):
    preds = outputs[0]
    t = targets[0]
    scores = preds["scores"].detach().cpu()
    labels = preds["labels"].detach().cpu()
    boxes = preds["boxes"].detach().cpu()

    keep = scores >= score_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    target_labels = t["labels"].detach().cpu()

    if boxes.numel() == 0 or target_labels.numel() == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": torch.zeros((num_classes, num_classes), dtype=torch.int64),
        }

    ious = _box_iou(boxes, t["boxes"].detach().cpu())
    assigned = ious >= iou_threshold

    tp = assigned.sum().item()
    fp = len(scores) - tp
    fn = len(target_labels) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for pred_label, target_label in zip(labels[:tp], target_labels[:tp]):
        if target_label < num_classes and pred_label < num_classes:
            cm[target_label.long(), pred_label.long()] += 1

    return {"precision": precision, "recall": recall, "confusion_matrix": cm}


def _aggregate_metrics(records: Iterable[dict]) -> dict:
    aggregate = {}
    confusion = None
    for record in records:
        for key, value in record.items():
            if key == "confusion_matrix":
                if confusion is None:
                    confusion = value.clone()
                else:
                    confusion += value
            else:
                aggregate.setdefault(key, []).append(value)
    if confusion is not None:
        aggregate["confusion_matrix"] = confusion
    for key, values in aggregate.items():
        if key == "confusion_matrix":
            continue
        aggregate[key] = float(sum(values) / len(values))
    return aggregate


def plot_confusion_matrix(result: EvaluationResult, save_path: Path | None = None) -> None:
    if result.confusion_matrix is None or result.class_names is None:
        raise ValueError("Confusion matrix requires class names and confusion data")

    matrix = result.confusion_matrix.numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=result.class_names, yticklabels=result.class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Soybean Disease Detection Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union
