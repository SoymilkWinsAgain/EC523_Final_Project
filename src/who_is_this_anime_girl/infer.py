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
from .devise import TEXT_MODEL_NAME, TextEmbeddingEncoder, load_devise_checkpoint, search_text_with_index
from .model import load_checkpoint
from .utils import resolve_device


def load_metadata(index_dir: str | Path) -> dict[str, Any]:
    with (Path(index_dir) / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_faiss_index(index_dir: str | Path):
    return faiss.read_index(str(Path(index_dir) / "gallery.faiss"))


def load_embedding_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_type") == "devise_frozen_image_transform_v1":
        return load_devise_checkpoint(checkpoint_path, map_location=map_location)
    return load_checkpoint(checkpoint_path, map_location=map_location)


@torch.no_grad()
def encode_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
    device: torch.device,
    embedding_key: str = "embedding",
) -> np.ndarray:
    tensor = tensor_from_pil(image, image_size=image_size, mean=image_mean, std=image_std).unsqueeze(0).to(device)
    outputs = model(tensor)
    selected_key = embedding_key if embedding_key in outputs else "embedding"
    embedding = outputs[selected_key].detach().cpu().numpy().astype("float32")
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-12)
    return embedding


def image_query_embedding_key(metadata: dict[str, Any]) -> str:
    if metadata.get("embedding_space") == "image" or metadata.get("index_type") == "image_embedding_space":
        return "image_embedding"
    return "embedding"


def aggregate_identity_matches(raw_matches: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    best_by_identity: dict[str, dict[str, Any]] = {}
    for match in raw_matches:
        identity = match["identity"]
        if identity not in best_by_identity or match["score"] > best_by_identity[identity]["score"]:
            best_by_identity[identity] = match
    return sorted(best_by_identity.values(), key=lambda item: item["score"], reverse=True)[:top_k]


class CachedGallerySearcher:
    """Keeps the checkpoint, metadata, and FAISS index resident for repeated UI queries."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        index_dir: str | Path,
        device: str = "auto",
        text_index_dir: str | Path | None = None,
        text_model_name: str = TEXT_MODEL_NAME,
        text_embedding_dim: int = 256,
        text_device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index_dir = Path(index_dir)
        self.text_index_dir = Path(text_index_dir) if text_index_dir else self.index_dir
        self.device_name = device
        self.device = resolve_device(device)
        self.text_model_name = text_model_name
        self.text_embedding_dim = text_embedding_dim
        self.text_device = text_device
        self.lock = Lock()
        self.metadata: dict[str, Any] = {}
        self.faiss_index = None
        self.text_metadata: dict[str, Any] = {}
        self.text_faiss_index = None
        self.model: torch.nn.Module | None = None
        self.text_encoder: TextEmbeddingEncoder | None = None
        self.reload()

    def reload(self) -> None:
        metadata = load_metadata(self.index_dir)
        faiss_index = load_faiss_index(self.index_dir)
        text_metadata = load_metadata(self.text_index_dir)
        text_faiss_index = load_faiss_index(self.text_index_dir)
        model, _ = load_embedding_checkpoint(self.checkpoint_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        with self.lock:
            self.metadata = metadata
            self.faiss_index = faiss_index
            self.text_metadata = text_metadata
            self.text_faiss_index = text_faiss_index
            self.model = model
            self.text_encoder = None

    def search(self, image: Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
        with self.lock:
            if self.model is None or self.faiss_index is None:
                raise RuntimeError("Cached gallery searcher is not loaded.")
            if self.faiss_index.ntotal == 0:
                return []
            metadata = self.metadata
            faiss_index = self.faiss_index
            model = self.model
            if metadata.get("index_type") == "devise_text_space":
                raise RuntimeError("Image queries require an image-space index, not a DeViSE text-space index.")

            query = encode_image(
                model,
                image.convert("RGB"),
                image_size=int(metadata["image_size"]),
                image_mean=metadata.get("image_mean", [0.485, 0.456, 0.406]),
                image_std=metadata.get("image_std", [0.229, 0.224, 0.225]),
                device=self.device,
                embedding_key=image_query_embedding_key(metadata),
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

    def _get_text_encoder(self) -> TextEmbeddingEncoder:
        if self.text_encoder is None:
            self.text_encoder = TextEmbeddingEncoder(
                model_name=self.text_model_name,
                embedding_dim=self.text_embedding_dim,
                device=self.text_device,
            )
        return self.text_encoder

    def search_text(self, query: str, top_k: int = 5) -> dict[str, Any]:
        with self.lock:
            if self.text_faiss_index is None:
                raise RuntimeError("Cached gallery searcher is not loaded.")
            if self.text_faiss_index.ntotal == 0:
                return {
                    "mode": "empty",
                    "matched_identities": [],
                    "matched_aliases": [],
                    "visual_query": "",
                    "matches": [],
                }
            text_metadata = self.text_metadata
            identities = [item["identity"] for item in text_metadata["items"]]

        from .devise import find_identity_matches

        matched_identities, _, visual_query = find_identity_matches(query, identities)
        needs_encoder = bool(visual_query) or not matched_identities
        if needs_encoder and self.text_metadata.get("index_type") != "devise_text_space":
            raise RuntimeError("Semantic text queries require a DeViSE text-space index.")
        encoder = self._get_text_encoder() if needs_encoder else None
        with self.lock:
            return search_text_with_index(
                query=query,
                items=self.text_metadata["items"],
                index=self.text_faiss_index,
                text_encoder=encoder,
                top_k=top_k,
            )


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
    if metadata.get("index_type") == "devise_text_space":
        raise RuntimeError("Image queries require an image-space index, not a DeViSE text-space index.")

    torch_device = resolve_device(device)
    model, _ = load_embedding_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)

    query = encode_image(
        model,
        image.convert("RGB"),
        image_size=int(metadata["image_size"]),
        image_mean=metadata.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=metadata.get("image_std", [0.229, 0.224, 0.225]),
        device=torch_device,
        embedding_key=image_query_embedding_key(metadata),
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
