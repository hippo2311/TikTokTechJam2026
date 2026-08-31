from __future__ import annotations

from collections import defaultdict
import warnings

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PairedAIGCImageDataset(Dataset):
    """Return a clean view and an independently augmented view of the same image."""

    def __init__(self, records: list[dict], image_size: int, augmentation=None,
                 normalization_mean=(0.485, 0.456, 0.406), normalization_std=(0.229, 0.224, 0.225),
                 for_training: bool = True):
        if for_training and any(record.get("split") == "organizer_demo" for record in records):
            raise AssertionError("organizer_demo is evaluation-only and must never be used for training")
        self.records, self.image_size, self.augmentation = records, image_size, augmentation
        self.normalization_mean, self.normalization_std = tuple(normalization_mean), tuple(normalization_std)
        self.for_training = for_training
        # A training fallback preserves the sampler's source/class balance
        # when an occasional archive contains a corrupt image. Evaluation is
        # intentionally strict: a bad benchmark image must fail visibly, not
        # silently change its measured population.
        self.group_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
        self.group_positions: dict[int, int] = {}
        for item_index, record in enumerate(records):
            group = (record.get("source_dataset", "unknown"), int(record["label"]))
            self.group_positions[item_index] = len(self.group_indices[group])
            self.group_indices[group].append(item_index)
        self.reported_bad_paths: set[str] = set()

    @staticmethod
    def _to_tensor(image: np.ndarray, image_size: int, mean, std) -> torch.Tensor:
        pil = Image.fromarray(image).convert("RGB")
        pil = TF.resize(pil, image_size)
        pil = TF.center_crop(pil, [image_size, image_size])
        return TF.normalize(TF.to_tensor(pil), mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.records)

    def _read_clean_image(self, record: dict) -> np.ndarray:
        with Image.open(record["path"]) as image:
            return np.asarray(image.convert("RGB"))

    def _replacement_indices(self, index: int):
        record = self.records[index]
        group = (record.get("source_dataset", "unknown"), int(record["label"]))
        members = self.group_indices[group]
        start = self.group_positions[index]
        for offset in range(1, len(members)):
            yield members[(start + offset) % len(members)]

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        try:
            clean = self._read_clean_image(record)
        # Pillow uses SyntaxError for some structurally invalid PNG chunks.
        # Treat it as an unreadable image, not as a training-code failure.
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as error:
            if not self.for_training:
                raise RuntimeError(f"Could not open evaluation image at {record['path']}: {error}") from error
            if record["path"] not in self.reported_bad_paths:
                warnings.warn(
                    f"Skipping unreadable training image {record['path']}; selecting a same-source, same-label replacement: {error}",
                    RuntimeWarning,
                )
                self.reported_bad_paths.add(record["path"])
            for replacement_index in self._replacement_indices(index):
                replacement = self.records[replacement_index]
                try:
                    clean = self._read_clean_image(replacement)
                    record = replacement
                    break
                except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
                    continue
            else:
                raise RuntimeError(
                    f"No readable same-source, same-label replacement exists for {record['path']}"
                ) from error
        augmented = self.augmentation(image=clean)["image"] if self.augmentation else clean.copy()
        return {
            "image": self._to_tensor(clean, self.image_size, self.normalization_mean, self.normalization_std),
            "augmented_image": self._to_tensor(augmented, self.image_size, self.normalization_mean, self.normalization_std),
            "label": torch.tensor(int(record["label"]), dtype=torch.float32),
            "path": record["path"],
            # Metadata only: never passed into the model. It enables a
            # source-wise robustness report without mixing source labels into
            # image features or training targets.
            "source_dataset": record.get("source_dataset", "unknown"),
        }
