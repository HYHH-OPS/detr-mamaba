"""Utilities to align the CV and NLP datasets for multimodal training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..cv.data import SoyDiseaseDetectionDataset
from ..nlp.data import SoyDiseaseTextDataset, LABELS, LABEL_TO_ID


@dataclass
class MultimodalSample:
    image: torch.Tensor
    text: Dict[str, torch.Tensor]
    label: int


class MultimodalSoyDataset(Dataset[MultimodalSample]):
    """Pair vision and language data based on shared labels."""

    def __init__(
        self,
        image_root: Path | str,
        text_csv: Path | str,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.cv_dataset = SoyDiseaseDetectionDataset(image_root)
        self.text_dataset = SoyDiseaseTextDataset(text_csv, tokenizer=tokenizer, max_length=max_length)

        self.label_to_indices: Dict[int, List[int]] = {LABEL_TO_ID[label]: [] for label in LABELS}
        for idx, sample in enumerate(self.text_dataset.samples):
            self.label_to_indices[sample.label].append(idx)

        if len(self.text_dataset) == 0:
            raise RuntimeError("Text dataset cannot be empty for multimodal alignment")

        for label, indices in self.label_to_indices.items():
            if not indices:
                # Guarantee at least one index per label by cycling through the dataset
                self.label_to_indices[label].append(label % len(self.text_dataset))

        self.image_to_label = []
        for idx in range(len(self.cv_dataset)):
            sample = self.cv_dataset[idx]
            if sample.target["labels"].numel() == 0:
                label = LABEL_TO_ID["Healthy"]
            else:
                label = int(sample.target["labels"][0].item())
                if label >= len(LABELS):
                    label = LABEL_TO_ID["Healthy"]
            self.image_to_label.append(label)

    def __len__(self) -> int:
        return len(self.cv_dataset)

    def __getitem__(self, index: int) -> MultimodalSample:
        sample = self.cv_dataset[index]
        label = self.image_to_label[index]
        indices = self.label_to_indices[label]
        text_idx = indices[index % len(indices)]
        text_features = self.text_dataset[text_idx]
        return MultimodalSample(image=sample.image, text=text_features, label=label)


def multimodal_collate_fn(batch: List[MultimodalSample]):
    images = torch.stack([sample.image for sample in batch])
    text = {}
    for key in batch[0].text.keys():
        text[key] = torch.stack([sample.text[key] for sample in batch])
    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)
    return {"images": images, "text": text, "labels": labels}
