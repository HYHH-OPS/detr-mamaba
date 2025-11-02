"""Training loop for the multimodal fusion model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from .model import MultimodalFusionModel, MultimodalConfig
from .data import MultimodalSoyDataset, multimodal_collate_fn


@dataclass
class MultimodalTrainingConfig:
    image_root: Path
    text_csv: Path
    tokenizer: Any = None
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    checkpoint_dir: Path = Path("checkpoints_multimodal")
    detr_checkpoint: Optional[Path] = None
    mamba_checkpoint: Optional[Path] = None

    def __post_init__(self) -> None:
        self.image_root = Path(self.image_root)
        self.text_csv = Path(self.text_csv)
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for multimodal training")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.detr_checkpoint is not None:
            self.detr_checkpoint = Path(self.detr_checkpoint)
        if self.mamba_checkpoint is not None:
            self.mamba_checkpoint = Path(self.mamba_checkpoint)


@dataclass
class MultimodalTrainingArtifacts:
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoints: List[Path] = field(default_factory=list)


def train_multimodal_model(config: MultimodalTrainingConfig, model_config: MultimodalConfig) -> MultimodalTrainingArtifacts:
    dataset = MultimodalSoyDataset(config.image_root, config.text_csv, tokenizer=config.tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
    )

    device = torch.device(config.device)
    model = MultimodalFusionModel(model_config).to(device)
    if config.detr_checkpoint:
        detr_ckpt = torch.load(config.detr_checkpoint, map_location=device)
        model.detr.load_state_dict(detr_ckpt["model"])
    if config.mamba_checkpoint:
        mamba_ckpt = torch.load(config.mamba_checkpoint, map_location=device)
        model.mamba.load_state_dict(mamba_ckpt["model"])

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = CrossEntropyLoss()

    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            images = [img.to(device) for img in batch["images"]]
            text = {key: value.to(device) for key, value in batch["text"].items()}
            labels = batch["labels"].to(device)

            logits = model(images, text)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / max(1, len(loader))
        accuracy = total_correct / max(1, total_samples)
        history.append({"epoch": float(epoch + 1), "train_loss": avg_loss, "train_accuracy": accuracy})

        ckpt_path = config.checkpoint_dir / f"multimodal_epoch_{epoch+1:03d}.pt"
        torch.save({"model": model.state_dict(), "config": model_config}, ckpt_path)
        print(f"Saved multimodal checkpoint to {ckpt_path}")

    return MultimodalTrainingArtifacts(history=history, checkpoints=list(config.checkpoint_dir.glob("multimodal_epoch_*.pt")))
