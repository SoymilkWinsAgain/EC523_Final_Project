from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_transforms(
    image_size: int,
    train: bool,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=tuple(mean), std=tuple(std)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_image_folder(
    root: str | Path,
    image_size: int,
    train: bool,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> datasets.ImageFolder:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    dataset = datasets.ImageFolder(root=str(root), transform=make_transforms(image_size, train=train, mean=mean, std=std))
    if not dataset.samples:
        raise ValueError(f"No images found in dataset directory: {root}")
    return dataset


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "path" not in record or "identity" not in record:
                raise ValueError(f"Manifest line {line_number} must contain path and identity fields.")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        train: bool,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(self.manifest_path)
        self.transform = make_transforms(image_size, train=train, mean=mean, std=std)

        identities = sorted({str(record["identity"]) for record in self.records})
        self.classes = identities
        self.class_to_idx = {identity: index for index, identity in enumerate(identities)}
        self.samples = [(str(Path(record["path"])), self.class_to_idx[str(record["identity"])]) for record in self.records]
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = load_rgb_image(path)
        return self.transform(image), label


def make_dataset(
    root: str | Path | None,
    manifest: str | Path | None,
    image_size: int,
    train: bool,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> datasets.ImageFolder | ManifestImageDataset:
    if manifest:
        return ManifestImageDataset(manifest, image_size=image_size, train=train, mean=mean, std=std)
    if not root:
        raise ValueError("Either a dataset directory or a manifest path must be provided.")
    return make_image_folder(root, image_size=image_size, train=train, mean=mean, std=std)


class PKBatchSampler(Sampler[list[int]]):
    """Samples P identities and K images per identity for supervised contrastive learning."""

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        samples_per_identity: int,
        batches_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        if identities_per_batch < 1:
            raise ValueError("identities_per_batch must be at least 1")
        if samples_per_identity < 2:
            raise ValueError("samples_per_identity must be at least 2 for contrastive training")

        self.labels = list(labels)
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.seed = seed

        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            by_label[int(label)].append(index)
        if len(by_label) < 2:
            raise ValueError("At least two identities are required for contrastive training")

        self.by_label = dict(by_label)
        self.unique_labels = sorted(self.by_label)
        default_batches = max(1, len(self.labels) // (identities_per_batch * samples_per_identity))
        self.batches_per_epoch = batches_per_epoch or default_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.batches_per_epoch):
            if len(self.unique_labels) >= self.identities_per_batch:
                chosen_labels = rng.sample(self.unique_labels, self.identities_per_batch)
            else:
                chosen_labels = [rng.choice(self.unique_labels) for _ in range(self.identities_per_batch)]

            batch: list[int] = []
            for label in chosen_labels:
                candidates = self.by_label[label]
                if len(candidates) >= self.samples_per_identity:
                    batch.extend(rng.sample(candidates, self.samples_per_identity))
                else:
                    batch.extend(rng.choices(candidates, k=self.samples_per_identity))
            rng.shuffle(batch)
            yield batch
            self.seed += 1

    def __len__(self) -> int:
        return self.batches_per_epoch


def tensor_from_pil(
    image: Image.Image,
    image_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    transform = make_transforms(image_size, train=False, mean=mean, std=std)
    return transform(image.convert("RGB"))


def count_images_by_identity(root: str | Path) -> dict[str, int]:
    root = Path(root)
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        count = sum(1 for path in identity_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if count:
            counts[identity_dir.name] = count
    return counts
