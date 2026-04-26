from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc="Extract embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        embeddings = F.normalize(outputs["embedding"], dim=1)
        all_embeddings.append(embeddings.cpu().numpy().astype("float32"))
        all_labels.append(labels.numpy().astype("int64"))
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray, top_k: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have the same length")
    if len(embeddings) < 2:
        return {f"recall@{k}": 0.0 for k in top_k} | {"mrr": 0.0, "valid_queries": 0.0}

    embeddings = embeddings.astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    scores = embeddings @ embeddings.T
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=1)

    recalls = {int(k): [] for k in top_k}
    reciprocal_ranks: list[float] = []

    for query_index, ranked_indices in enumerate(order):
        matches = labels[ranked_indices] == labels[query_index]
        if not np.any(matches):
            continue
        first_rank = int(np.argmax(matches)) + 1
        reciprocal_ranks.append(1.0 / first_rank)
        for k in recalls:
            recalls[k].append(float(np.any(matches[:k])))

    metrics = {f"recall@{k}": float(np.mean(values)) if values else 0.0 for k, values in recalls.items()}
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["valid_queries"] = float(len(reciprocal_ranks))
    return metrics
