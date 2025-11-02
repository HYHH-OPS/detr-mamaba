"""Evaluation utilities for the multimodal QA system."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .model import MultimodalFusionModel


@dataclass
class MultimodalEvaluationResult:
    history: List[Dict[str, float]]
    confusion: torch.Tensor
    answer_space: List[str]


def evaluate_multimodal_model(
    model: MultimodalFusionModel,
    dataloader,
    answer_space: List[str],
    device: torch.device,
    save_heatmap: Path | None = None,
) -> MultimodalEvaluationResult:
    model.eval()
    confusion = torch.zeros((len(answer_space), len(answer_space)), dtype=torch.int32, device=device)
    history: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = [img.to(device) for img in batch["images"]]
            text = {key: value.to(device) for key, value in batch["text"].items()}
            labels = batch["labels"].to(device)
            logits = model(images, text)
            preds = logits.argmax(dim=1)
            for true, pred in zip(labels, preds):
                confusion[true.long(), pred.long()] += 1
            accuracy = (preds == labels).float().mean().item()
            history.append({"accuracy": accuracy})

    confusion_cpu = confusion.cpu()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_cpu.numpy(), annot=True, fmt="d", xticklabels=answer_space, yticklabels=answer_space, ax=ax)
    ax.set_xlabel("Predicted answer")
    ax.set_ylabel("True answer")
    ax.set_title("Multimodal QA Confusion Matrix")
    plt.tight_layout()

    if save_heatmap:
        fig.savefig(save_heatmap)
    else:
        plt.show()

    return MultimodalEvaluationResult(history=history, confusion=confusion_cpu, answer_space=answer_space)
