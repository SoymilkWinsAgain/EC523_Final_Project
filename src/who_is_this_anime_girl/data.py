from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_image_folder(root: str | Path, image_size: int, train: bool) -> datasets.ImageFolder:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    dataset = datasets.ImageFolder(root=str(root), transform=make_transforms(image_size, train=train))
    if not dataset.samples:
        raise ValueError(f"No images found in dataset directory: {root}")
    return dataset


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


def tensor_from_pil(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = make_transforms(image_size, train=False)
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
