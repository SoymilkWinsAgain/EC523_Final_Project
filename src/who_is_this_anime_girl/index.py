from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from torch.utils.data import DataLoader

from .data import make_image_folder
from .metrics import extract_embeddings
from .model import load_checkpoint
from .utils import resolve_device, write_json


def build_gallery_index(
    checkpoint_path: str | Path,
    gallery_dir: str | Path,
    output_dir: str | Path,
    image_size: int | None = None,
    batch_size: int = 64,
    workers: int = 4,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    gallery_dir = Path(gallery_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)

    run_config = checkpoint.get("run_config", {})
    resolved_image_size = int(image_size or run_config.get("image_size", 224))
    dataset = make_image_folder(gallery_dir, image_size=resolved_image_size, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch_device.type == "cuda")
    embeddings, labels = extract_embeddings(model, loader, torch_device)
    embeddings = embeddings.astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "gallery.faiss"))

    idx_to_class = {index_value: class_name for class_name, index_value in dataset.class_to_idx.items()}
    items = []
    for sample_index, (path, label) in enumerate(dataset.samples):
        items.append(
            {
                "index": sample_index,
                "path": str(Path(path).resolve()),
                "identity": idx_to_class[int(label)],
                "label": int(label),
            }
        )

    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "gallery_dir": str(gallery_dir.resolve()),
        "image_size": resolved_image_size,
        "embedding_dim": int(embeddings.shape[1]),
        "items": items,
        "class_to_idx": dataset.class_to_idx,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS gallery index from an ImageFolder gallery.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--gallery-dir", required=True, help="ImageFolder gallery directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for gallery.faiss and metadata.json.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size from the checkpoint config.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_gallery_index(
        checkpoint_path=args.checkpoint,
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    print(f"Indexed {len(metadata['items'])} images into {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
