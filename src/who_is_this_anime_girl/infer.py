from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image

from .data import tensor_from_pil
from .model import load_checkpoint
from .utils import resolve_device


def load_metadata(index_dir: str | Path) -> dict[str, Any]:
    with (Path(index_dir) / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_faiss_index(index_dir: str | Path):
    return faiss.read_index(str(Path(index_dir) / "gallery.faiss"))


@torch.no_grad()
def encode_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
    device: torch.device,
) -> np.ndarray:
    tensor = tensor_from_pil(image, image_size=image_size, mean=image_mean, std=image_std).unsqueeze(0).to(device)
    embedding = model(tensor)["embedding"].detach().cpu().numpy().astype("float32")
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-12)
    return embedding


def aggregate_identity_matches(raw_matches: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    best_by_identity: dict[str, dict[str, Any]] = {}
    for match in raw_matches:
        identity = match["identity"]
        if identity not in best_by_identity or match["score"] > best_by_identity[identity]["score"]:
            best_by_identity[identity] = match
    return sorted(best_by_identity.values(), key=lambda item: item["score"], reverse=True)[:top_k]


class CachedGallerySearcher:
    """Keeps the checkpoint, metadata, and FAISS index resident for repeated UI queries."""

    def __init__(self, checkpoint_path: str | Path, index_dir: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index_dir = Path(index_dir)
        self.device_name = device
        self.device = resolve_device(device)
        self.lock = Lock()
        self.metadata: dict[str, Any] = {}
        self.faiss_index = None
        self.model: torch.nn.Module | None = None
        self.reload()

    def reload(self) -> None:
        metadata = load_metadata(self.index_dir)
        faiss_index = load_faiss_index(self.index_dir)
        model, _ = load_checkpoint(self.checkpoint_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        with self.lock:
            self.metadata = metadata
            self.faiss_index = faiss_index
            self.model = model

    def search(self, image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
        with self.lock:
            if self.model is None or self.faiss_index is None:
                raise RuntimeError("Cached gallery searcher is not loaded.")
            if self.faiss_index.ntotal == 0:
                return []
            metadata = self.metadata
            faiss_index = self.faiss_index
            model = self.model

            query = encode_image(
                model,
                image.convert("RGB"),
                image_size=int(metadata["image_size"]),
                image_mean=metadata.get("image_mean", [0.485, 0.456, 0.406]),
                image_std=metadata.get("image_std", [0.229, 0.224, 0.225]),
                device=self.device,
            )
            search_k = min(max(top_k * 5, top_k), faiss_index.ntotal)
            scores, indices = faiss_index.search(query, search_k)

        raw_matches: list[dict[str, Any]] = []
        items = metadata["items"]
        for score, index_value in zip(scores[0], indices[0]):
            if index_value < 0:
                continue
            item = items[int(index_value)]
            raw_matches.append(
                {
                    "identity": item["identity"],
                    "score": float(score),
                    "path": item["path"],
                    "index": int(index_value),
                }
            )
        return aggregate_identity_matches(raw_matches, top_k=top_k)

    def search_bytes(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, Any]]:
        with Image.open(BytesIO(image_bytes)) as image:
            return self.search(image, top_k=top_k)

    def search_file(self, image_path: str | Path, top_k: int = 5) -> list[dict[str, Any]]:
        with Image.open(image_path) as image:
            return self.search(image, top_k=top_k)


def search_image(
    checkpoint_path: str | Path,
    index_dir: str | Path,
    image: Image.Image,
    top_k: int = 5,
    device: str = "auto",
) -> list[dict[str, Any]]:
    index_dir = Path(index_dir)
    metadata = load_metadata(index_dir)
    faiss_index = load_faiss_index(index_dir)
    if faiss_index.ntotal == 0:
        return []

    torch_device = resolve_device(device)
    model, _ = load_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)

    query = encode_image(
        model,
        image.convert("RGB"),
        image_size=int(metadata["image_size"]),
        image_mean=metadata.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=metadata.get("image_std", [0.229, 0.224, 0.225]),
        device=torch_device,
    )
    search_k = min(max(top_k * 5, top_k), faiss_index.ntotal)
    scores, indices = faiss_index.search(query, search_k)

    raw_matches: list[dict[str, Any]] = []
    items = metadata["items"]
    for score, index_value in zip(scores[0], indices[0]):
        if index_value < 0:
            continue
        item = items[int(index_value)]
        raw_matches.append(
            {
                "identity": item["identity"],
                "score": float(score),
                "path": item["path"],
                "index": int(index_value),
            }
        )
    return aggregate_identity_matches(raw_matches, top_k=top_k)


def search_image_file(
    checkpoint_path: str | Path,
    index_dir: str | Path,
    image_path: str | Path,
    top_k: int = 5,
    device: str = "auto",
) -> list[dict[str, Any]]:
    with Image.open(image_path) as image:
        return search_image(checkpoint_path, index_dir, image, top_k=top_k, device=device)


def search_image_bytes(
    checkpoint_path: str | Path,
    index_dir: str | Path,
    image_bytes: bytes,
    top_k: int = 5,
    device: str = "auto",
) -> list[dict[str, Any]]:
    with Image.open(BytesIO(image_bytes)) as image:
        return search_image(checkpoint_path, index_dir, image, top_k=top_k, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query an anime character gallery index.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--index-dir", required=True, help="Directory containing gallery.faiss and metadata.json.")
    parser.add_argument("--image", required=True, help="Query image path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = search_image_file(
        checkpoint_path=args.checkpoint,
        index_dir=args.index_dir,
        image_path=args.image,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(matches, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
