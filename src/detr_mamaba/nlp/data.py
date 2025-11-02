"""NLP dataset utilities for soybean disease descriptions."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import Dataset


LABELS = ["Healthy", "Bean_Rust", "Angular_Leaf_Spot"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class TextSample:
    text: str
    label: int


class SoyDiseaseTextDataset(Dataset[TextSample]):
    """A simple CSV-backed dataset for soybean disease textual descriptions.

    The CSV file is expected to contain two columns: ``text`` and ``label``. ``label``
    should correspond to one of ``Healthy``, ``Bean_Rust`` or ``Angular_Leaf_Spot``.
    """

    def __init__(self, csv_path: Path | str, tokenizer, max_length: int = 512) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: List[TextSample] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_str = row["label"].strip()
                if label_str not in LABEL_TO_ID:
                    raise ValueError(f"Unknown label {label_str} in {self.csv_path}")
                self.samples.append(TextSample(text=row["text"], label=LABEL_TO_ID[label_str]))
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.squeeze(0) for key, value in encoded.items()}
        encoded["labels"] = torch.tensor(sample.label, dtype=torch.long)
        return encoded
