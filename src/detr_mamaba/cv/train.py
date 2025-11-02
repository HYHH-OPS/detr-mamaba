"""Training utilities for DETR on the soybean disease dataset."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import SoyDiseaseDetectionDataset, collate_fn
from .augmentation import compose, random_flip, color_jitter
from .model import DetrConfig, create_forward_detr

try:  # pragma: no cover - optional dependency
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:  # pragma: no cover - degrade gracefully
    MeanAveragePrecision = None  # type: ignore


@dataclass
class TrainingConfig:
    train_root: Path
    val_root: Optional[Path] = None
    num_classes: int = 4  # DETR expects ``num_classes`` to include the background class
    batch_size: int = 2
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    lr_warmup_steps: int = 500
    max_norm: float = 0.1
    gradient_accumulation: int = 1
    amp: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_every: int = 10
    use_augmentation: bool = True

    def __post_init__(self) -> None:
        self.train_root = Path(self.train_root)
        if self.val_root is not None:
            self.val_root = Path(self.val_root)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingArtifacts:
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoints: List[Path] = field(default_factory=list)


def _create_dataloaders(config: TrainingConfig) -> Dict[str, DataLoader]:
    augmentation = compose(random_flip, color_jitter) if config.use_augmentation else None
    datasets = {
        "train": SoyDiseaseDetectionDataset(config.train_root, augmentation=augmentation),
    }
    if config.val_root:
        datasets["val"] = SoyDiseaseDetectionDataset(config.val_root)

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
    return loaders


def train_detr(config: TrainingConfig, model_config: Optional[DetrConfig] = None) -> TrainingArtifacts:
    """Run DETR training and validation loops."""

    device = torch.device(config.device)
    model_config = model_config or DetrConfig(num_classes=config.num_classes)
    model = create_forward_detr(model_config)
    model.to(device)

    dataloaders = _create_dataloaders(config)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    history: List[Dict[str, float]] = []
    global_step = 0

    metric = MeanAveragePrecision() if MeanAveragePrecision is not None else None

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.epochs):
        model.train()
        train_loader = dataloaders["train"]
        num_batches = len(train_loader)
        running_loss = 0.0
        for step, (images, targets, _) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                losses = model(images, targets)
                total_loss = sum(loss for loss in losses.values())

            scaler.scale(total_loss / config.gradient_accumulation).backward()
            if (step + 1) % config.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += total_loss.item()
            global_step += 1

            if config.log_every and (step + 1) % config.log_every == 0:
                avg_loss = running_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} Step {step+1}: loss={avg_loss:.4f}")

        if num_batches % config.gradient_accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        lr_scheduler.step()

        epoch_metrics: Dict[str, float] = {"epoch": float(epoch + 1), "train_loss": running_loss / max(1, num_batches)}

        if "val" in dataloaders:
            model.eval()
            val_loss = 0.0
            metric.reset() if metric else None
            with torch.no_grad():
                for images, targets, _ in dataloaders["val"]:
                    images = list(image.to(device) for image in images)
                    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                    outputs = model(images)
                    loss_dict = model(images, targets)
                    val_loss += sum(loss for loss in loss_dict.values()).item()

                    if metric is not None:
                        preds = [
                            {
                                "boxes": out["boxes"].detach().cpu(),
                                "scores": out["scores"].detach().cpu(),
                                "labels": out["labels"].detach().cpu(),
                            }
                            for out in outputs
                        ]
                        metric.update(preds, targets)

            num_batches = len(dataloaders["val"])
            epoch_metrics["val_loss"] = val_loss / max(1, num_batches)
            if metric is not None:
                epoch_metrics.update({f"val_{k}": float(v) for k, v in metric.compute().items()})

        history.append(epoch_metrics)

        ckpt_path = config.checkpoint_dir / f"detr_epoch_{epoch+1:03d}.pt"
        torch.save({"model": model.state_dict(), "config": model_config}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    return TrainingArtifacts(history=history, checkpoints=list(config.checkpoint_dir.glob("detr_epoch_*.pt")))
