"""Multimodal fusion between DETR features and Mamba text embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..cv.model import create_forward_detr, DetrConfig
from ..nlp.model import MambaClassifier, MambaConfig


@dataclass
class MultimodalConfig:
    detr_config: DetrConfig
    mamba_config: MambaConfig
    fusion_hidden: int = 512
    num_answers: int = 5


class MultimodalFusionModel(nn.Module):
    """Joint model for soybean disease visual QA."""

    def __init__(self, config: MultimodalConfig) -> None:
        super().__init__()
        self.detr = create_forward_detr(config.detr_config)
        self.mamba = MambaClassifier(config.mamba_config)

        # Freeze DETR by default – fine-tuning can be enabled by toggling ``requires_grad``
        self.detr.eval()
        for param in self.detr.parameters():
            param.requires_grad = False

        visual_dim = self.detr.class_embed.out_features
        text_dim = config.mamba_config.hidden_size

        self.visual_proj = nn.Linear(visual_dim, config.fusion_hidden)
        self.text_proj = nn.Linear(text_dim, config.fusion_hidden)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.answer_head = nn.Linear(config.fusion_hidden, config.num_answers)

    def train(self, mode: bool = True) -> "MultimodalFusionModel":  # type: ignore[override]
        super().train(mode)
        # Keep DETR frozen even when the fusion module is in training mode
        self.detr.eval()
        return self

    def forward(
        self,
        images: list[torch.Tensor],
        text_batch: dict,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = images[0].device if images else torch.device("cpu")
        with torch.no_grad():
            detections = self.detr(images)

        bags = []
        for detection in detections:
            bag = torch.zeros(self.detr.class_embed.out_features, device=device)
            if detection["scores"].numel():
                labels = detection["labels"].to(device)
                scores = detection["scores"].to(device)
                bag.scatter_add_(0, labels, scores)
            bags.append(bag)

        visual_features = self.visual_proj(torch.stack(bags))

        text_logits = self.mamba(**text_batch)
        text_features = self.text_proj(text_logits)

        fused = self.fusion(visual_features + text_features)
        answers = self.answer_head(fused)

        if return_embeddings:
            return answers, visual_features, text_features
        return answers
