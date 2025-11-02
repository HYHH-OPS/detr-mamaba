"""Mamba-based text classification model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:  # pragma: no cover - optional heavy dependency
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover
    Mamba = None  # type: ignore


@dataclass
class MambaConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 2
    num_classes: int = 3
    dropout: float = 0.1


class MambaClassifier(nn.Module):
    """Thin wrapper around the ``mamba_ssm`` implementation for classification."""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for the MambaClassifier. Install it via `pip install mamba-ssm`."
            )
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Mamba(d_model=config.hidden_size) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            x = x * mask
            pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits
