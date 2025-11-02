"""Training loop for the Mamba-based NLP classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from .data import SoyDiseaseTextDataset, LABELS
from .model import MambaClassifier, MambaConfig


@dataclass
class NLPTrainingConfig:
    train_csv: Path
    val_csv: Optional[Path] = None
    tokenizer: Any = None
    vocab_size: int = 32000
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    checkpoint_dir: Path = Path("checkpoints_nlp")

    def __post_init__(self) -> None:
        self.train_csv = Path(self.train_csv)
        if self.val_csv is not None:
            self.val_csv = Path(self.val_csv)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class NLPTrainingArtifacts:
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoints: List[Path] = field(default_factory=list)


def train_mamba_classifier(config: NLPTrainingConfig, model_config: Optional[MambaConfig] = None) -> NLPTrainingArtifacts:
    if config.tokenizer is None:
        raise ValueError("A tokenizer compatible with the training data must be provided.")

    model_config = model_config or MambaConfig(vocab_size=config.vocab_size, num_classes=len(LABELS))
    dataset = SoyDiseaseTextDataset(config.train_csv, tokenizer=config.tokenizer, max_length=config.max_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    if config.val_csv:
        val_dataset = SoyDiseaseTextDataset(config.val_csv, tokenizer=config.tokenizer, max_length=config.max_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    else:
        val_loader = None

    device = torch.device(config.device)
    model = MambaClassifier(model_config).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1)
    criterion = CrossEntropyLoss()

    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(1, len(loader))

        epoch_record: Dict[str, float] = {"epoch": float(epoch + 1), "train_loss": avg_train_loss}

        if val_loader is not None:
            model.eval()
            total_val = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**batch)
                    loss = criterion(logits, batch["labels"])
                    total_val += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += batch["labels"].size(0)
            avg_val_loss = total_val / max(1, len(val_loader))
            accuracy = correct / max(1, total)
            scheduler.step(avg_val_loss)
            epoch_record.update({"val_loss": avg_val_loss, "val_accuracy": accuracy})

        history.append(epoch_record)

        ckpt_path = config.checkpoint_dir / f"mamba_epoch_{epoch+1:03d}.pt"
        torch.save({"model": model.state_dict(), "config": model_config}, ckpt_path)
        print(f"Saved NLP checkpoint to {ckpt_path}")

    return NLPTrainingArtifacts(history=history, checkpoints=list(config.checkpoint_dir.glob("mamba_epoch_*.pt")))
