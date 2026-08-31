from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class AIGCImageDataset(Dataset):
    """Canonical image dataset. Training datasets explicitly reject organiser demo data."""

    def __init__(self, records: list[dict], transform=None, for_training: bool = True):
        if for_training and any(record.get("split") == "organizer_demo" for record in records):
            raise AssertionError("organizer_demo is evaluation-only and must never be used for training")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = Path(record["path"])
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                value = self.transform(image) if self.transform else image.copy()
        except Exception as error:
            raise RuntimeError(f"Could not open image at {path}: {error}") from error
        return {"image": value, "label": torch.tensor(int(record["label"]), dtype=torch.long),
                "path": str(path), "source_dataset": record["source_dataset"], "generator": record.get("generator")}
