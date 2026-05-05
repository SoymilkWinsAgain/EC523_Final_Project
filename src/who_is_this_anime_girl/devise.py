from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import load_manifest, load_rgb_image, make_dataset, make_transforms
from .losses import symmetric_image_text_contrastive_loss
from .metrics import extract_embeddings
from .model import load_checkpoint
from .reporting import save_training_artifacts
from .utils import resolve_device, set_seed, write_json


TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# QUERY_TASK = "Given a text query about anime character appearance, retrieve matching anime images."
QUERY_TASK = "Given a text query describing an anime character's visible appearance, retrieve relevant anime image descriptions"

def preprocess_tag(value: str) -> str:
    value = str(value).replace("_", " ").replace("-", " ")
    value = re.sub(r"[\\/]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def preprocess_query_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ")).strip()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _tag_names(record: dict[str, Any], key: str) -> list[str]:
    tags = record.get("tags") or {}
    value = tags.get(key) if isinstance(tags, dict) else []
    return [str(item) for item in value or [] if item]


TEXT_DOCUMENT_VERSION = "anime_visual_document_v2"


def build_devise_document_text(record: dict[str, Any], include_character: bool = False) -> str:
    parts: list[str] = ["Anime image description."]

    characters = _tag_names(record, "character")
    copyrights = _tag_names(record, "copyright")
    artists = _tag_names(record, "artist")
    general = _tag_names(record, "general")

    if include_character and characters:
        parts.append("Character name: " + ", ".join(preprocess_tag(tag) for tag in characters[:2]) + ".")
    if copyrights:
        parts.append("Series or source: " + ", ".join(preprocess_tag(tag) for tag in copyrights[:8]) + ".")
    if artists:
        parts.append("Artist: " + ", ".join(preprocess_tag(tag) for tag in artists[:2]) + ".")
    if general:
        parts.append("Visible attributes: " + ", ".join(preprocess_tag(tag) for tag in general[:32]) + ".")

    text = " ".join(parts).strip()
    if text != "Anime image description.":
        return text

    original = str(record.get("text") or "")
    return re.sub(r"(^|\.\s*)character:\s*[^.]+\.?\s*", " ", original, flags=re.IGNORECASE).strip() or original

def query_instruction_text(query: str) -> str:
    return f"Instruct: {QUERY_TASK}\nQuery: {preprocess_query_text(query)}"


class TextEmbeddingEncoder:
    def __init__(
        self,
        model_name: str = TEXT_MODEL_NAME,
        embedding_dim: int = 256,
        device: str = "auto",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = resolved_device
        self.model = SentenceTransformer(model_name, device=resolved_device)
        self.model.eval()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        query: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prepared = [query_instruction_text(text) for text in texts] if query else list(texts)
        kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": show_progress_bar,
        }
        try:
            embeddings = self.model.encode(prepared, truncate_dim=self.embedding_dim, **kwargs)
        except TypeError:
            embeddings = self.model.encode(prepared, **kwargs)
            embeddings = embeddings[:, : self.embedding_dim]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
            embeddings = embeddings / norms
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        return embeddings / norms


def precompute_text_embeddings(
    manifest_path: str | Path,
    output_dir: str | Path,
    split: str,
    model_name: str = TEXT_MODEL_NAME,
    embedding_dim: int = 256,
    batch_size: int = 32,
    device: str = "auto",
    include_character: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{split}.jsonl"
    embeddings_path = output_dir / f"{split}_text_embeddings.npy"
    summary_path = output_dir / f"{split}_summary.json"
    source_rows = load_manifest(manifest_path)

    if not force and records_path.exists() and embeddings_path.exists() and summary_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # if embeddings.shape == (len(source_rows), embedding_dim) and summary.get("tag_preprocessing") == "underscore_to_space_v1":
        if embeddings.shape == (len(source_rows), embedding_dim) and summary.get("tag_preprocessing") == TEXT_DOCUMENT_VERSION:
            summary["reused"] = True
            return summary

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for index, row in enumerate(source_rows):
        devise_text = build_devise_document_text(row, include_character=include_character)
        enriched = dict(row)
        enriched["original_text"] = row.get("text", "")
        enriched["devise_text"] = devise_text
        enriched["text_embedding_index"] = index
        rows.append(enriched)
        texts.append(devise_text)

    started = time.time()
    encoder = TextEmbeddingEncoder(model_name=model_name, embedding_dim=embedding_dim, device=device)
    embeddings = encoder.encode(texts, batch_size=batch_size, query=False, show_progress_bar=True)
    np.save(embeddings_path, embeddings.astype("float32"))
    write_jsonl(records_path, rows)
    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "records": str(records_path.resolve()),
        "embeddings": str(embeddings_path.resolve()),
        "rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": model_name,
        "include_character": include_character,
        # "tag_preprocessing": "underscore_to_space_v1",
        "tag_preprocessing": TEXT_DOCUMENT_VERSION,
        "elapsed_sec": time.time() - started,
        "reused": False,
        "norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
    }
    write_json(summary_path, summary)
    return summary


class ImageTextEmbeddingDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        text_embeddings_path: str | Path,
        image_size: int,
        train: bool,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(manifest_path)
        self.text_embeddings = np.load(text_embeddings_path, mmap_mode="r")
        if len(self.records) != int(self.text_embeddings.shape[0]):
            raise ValueError(f"Manifest rows and text embeddings differ: {len(self.records)} vs {self.text_embeddings.shape[0]}")
        identities = sorted({str(record["identity"]) for record in self.records})
        self.classes = identities
        self.class_to_idx = {identity: index for index, identity in enumerate(identities)}
        self.samples = [(str(Path(record["path"])), self.class_to_idx[str(record["identity"])]) for record in self.records]
        self.targets = [label for _, label in self.samples]
        self.transform = make_transforms(image_size=image_size, train=train, mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        image = load_rgb_image(self.samples[index][0])
        text_embedding = torch.from_numpy(np.array(self.text_embeddings[index], dtype="float32", copy=True))
        return self.transform(image), text_embedding, int(self.targets[index]), index


class EmbeddingPairDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_embeddings_path: str | Path,
        text_embeddings_path: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(manifest_path)
        self.image_embeddings = np.load(image_embeddings_path, mmap_mode="r")
        self.text_embeddings = np.load(text_embeddings_path, mmap_mode="r")
        if len(self.records) != int(self.image_embeddings.shape[0]) or len(self.records) != int(self.text_embeddings.shape[0]):
            raise ValueError(
                "Manifest rows, image embeddings, and text embeddings must have the same first dimension: "
                f"{len(self.records)}, {self.image_embeddings.shape}, {self.text_embeddings.shape}"
            )
        identities = sorted({str(record["identity"]) for record in self.records})
        self.classes = identities
        self.class_to_idx = {identity: index for index, identity in enumerate(identities)}
        self.targets = [self.class_to_idx[str(record["identity"])] for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        image_embedding = torch.from_numpy(np.array(self.image_embeddings[index], dtype="float32", copy=True))
        text_embedding = torch.from_numpy(np.array(self.text_embeddings[index], dtype="float32", copy=True))
        return image_embedding, text_embedding, int(self.targets[index]), index


class DeVISETransformation(nn.Module):
    def __init__(self, image_embedding_dim: int = 256, text_embedding_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, text_embedding_dim),
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(F.normalize(image_embeddings, dim=1)), dim=1)


class DeVISEImageModel(nn.Module):
    def __init__(self, image_model: nn.Module, transformation: DeVISETransformation) -> None:
        super().__init__()
        self.image_model = image_model
        self.transformation = transformation
        for parameter in self.image_model.parameters():
            parameter.requires_grad = False
        self.image_model.eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            image_embedding = F.normalize(self.image_model(images)["embedding"], dim=1)
        transformed = self.transformation(image_embedding)
        return {"image_embedding": image_embedding, "embedding": transformed}


def save_devise_checkpoint(
    path: str | Path,
    transformation: DeVISETransformation,
    base_image_checkpoint: str | Path,
    epoch: int,
    metrics: dict[str, float],
    class_to_idx: dict[str, int],
    run_config: dict[str, Any],
    logit_scale: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "devise_frozen_image_transform_v1",
            "epoch": epoch,
            "base_image_checkpoint": str(Path(base_image_checkpoint).resolve()),
            "transform_config": {
                "image_embedding_dim": transformation.image_embedding_dim,
                "text_embedding_dim": transformation.text_embedding_dim,
                "hidden_dim": transformation.hidden_dim,
            },
            "transform_state": transformation.state_dict(),
            "logit_scale": float(logit_scale.detach().exp().cpu()),
            "metrics": metrics,
            "class_to_idx": class_to_idx,
            "run_config": run_config,
        },
        path,
    )


def is_devise_checkpoint(path: str | Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("checkpoint_type") == "devise_frozen_image_transform_v1"


def load_devise_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[DeVISEImageModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("checkpoint_type") != "devise_frozen_image_transform_v1":
        raise ValueError(f"Not a DeVISE transformation checkpoint: {path}")
    base_model, _ = load_checkpoint(checkpoint["base_image_checkpoint"], map_location=map_location)
    transform_config = checkpoint["transform_config"]
    transformation = DeVISETransformation(**transform_config)
    transformation.load_state_dict(checkpoint["transform_state"])
    model = DeVISEImageModel(base_model, transformation)
    model.eval()
    return model, checkpoint


def precompute_image_embeddings(
    image_checkpoint: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    split: str,
    image_size: int = 224,
    image_mean: Sequence[float] = (0.5, 0.5, 0.5),
    image_std: Sequence[float] = (0.5, 0.5, 0.5),
    batch_size: int = 128,
    workers: int = 4,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{split}_image_embeddings.npy"
    summary_path = output_dir / f"{split}_image_summary.json"
    records = load_manifest(manifest_path)
    if not force and embeddings_path.exists() and summary_path.exists():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if embeddings.shape == (len(records), 256) and summary.get("image_checkpoint") == str(Path(image_checkpoint).resolve()):
            summary["reused"] = True
            return summary

    started = time.time()
    torch_device = resolve_device(device)
    model, _ = load_checkpoint(image_checkpoint, map_location=torch_device)
    model.to(torch_device)
    dataset = make_dataset(
        None,
        manifest_path,
        image_size=image_size,
        train=False,
        mean=image_mean,
        std=image_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch_device.type == "cuda")
    embeddings, labels = extract_embeddings(model, loader, torch_device)
    embeddings = embeddings.astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    np.save(embeddings_path, embeddings)
    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "embeddings": str(embeddings_path.resolve()),
        "rows": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "image_checkpoint": str(Path(image_checkpoint).resolve()),
        "elapsed_sec": time.time() - started,
        "reused": False,
        "norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
        "labels": int(len(set(labels.tolist()))),
    }
    write_json(summary_path, summary)
    return summary


def create_devise_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
):
    min_factor = min_lr / lr if lr > 0 else 0.0

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return max(epoch_number / warmup_epochs, min_factor)
        if epochs - warmup_epochs <= 1:
            progress = 1.0
        else:
            progress = (epoch_index - warmup_epochs) / (epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, progress))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(progress * math.pi))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_devise_one_epoch(
    model: torch.nn.Module,
    logit_scale: torch.nn.Parameter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, text_embeddings, _, _ in tqdm(loader, desc="Train DeViSE", leave=False):
        if max_batches is not None and batches >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        text_embeddings = text_embeddings.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            image_embeddings = model(images)["embedding"]
            loss = symmetric_image_text_contrastive_loss(image_embeddings, text_embeddings, logit_scale)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            logit_scale.clamp_(min=math.log(1.0), max=math.log(100.0))
        total_loss += float(loss.detach().cpu())
        batches += 1
    return {
        "loss": total_loss / max(1, batches),
        "image_text_loss": total_loss / max(1, batches),
        "logit_scale": float(logit_scale.detach().exp().cpu()),
    }


def train_transform_one_epoch(
    transformation: DeVISETransformation,
    logit_scale: torch.nn.Parameter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    transformation.train()
    total_loss = 0.0
    batches = 0
    for image_embeddings, text_embeddings, _, _ in tqdm(loader, desc="Train DeVISE Transform", leave=False):
        if max_batches is not None and batches >= max_batches:
            break
        image_embeddings = image_embeddings.to(device, non_blocking=True)
        text_embeddings = text_embeddings.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            transformed = transformation(image_embeddings)
            loss = symmetric_image_text_contrastive_loss(transformed, text_embeddings, logit_scale)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            logit_scale.clamp_(min=math.log(1.0), max=math.log(100.0))
        total_loss += float(loss.detach().cpu())
        batches += 1
    return {
        "loss": total_loss / max(1, batches),
        "image_text_loss": total_loss / max(1, batches),
        "logit_scale": float(logit_scale.detach().exp().cpu()),
    }


@torch.no_grad()
def extract_devise_image_embeddings(
    model: torch.nn.Module,
    dataset: ImageTextEmbeddingDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for images, _, batch_labels, batch_indices in tqdm(loader, desc="Extract DeViSE image embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        output = model(images)["embedding"].detach().cpu().numpy().astype("float32")
        output /= np.linalg.norm(output, axis=1, keepdims=True).clip(min=1e-12)
        embeddings.append(output)
        labels.append(batch_labels.numpy().astype("int64"))
        indices.append(batch_indices.numpy().astype("int64"))
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(indices)


def rank_metrics_from_order(order: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray, top_k: Iterable[int]) -> dict[str, float]:
    recalls = {int(k): [] for k in top_k}
    reciprocal_ranks: list[float] = []
    for query_index, ranked_indices in enumerate(order):
        matches = gallery_labels[ranked_indices] == query_labels[query_index]
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


def evaluate_text_to_image(
    model: torch.nn.Module,
    dataset: ImageTextEmbeddingDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    image_embeddings, image_labels, indices = extract_devise_image_embeddings(model, dataset, batch_size, workers, device)
    text_embeddings = np.array(dataset.text_embeddings, dtype="float32", copy=True)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    scores = text_embeddings @ image_embeddings.T
    order = np.argsort(-scores, axis=1)
    image_ranked = indices[order]
    image_hits = image_ranked == np.arange(len(dataset), dtype=np.int64)[:, None]

    image_recalls: dict[int, list[float]] = {int(k): [] for k in top_k}
    image_rr: list[float] = []
    for row in image_hits:
        if not np.any(row):
            continue
        rank = int(np.argmax(row)) + 1
        image_rr.append(1.0 / rank)
        for k in image_recalls:
            image_recalls[k].append(float(np.any(row[:k])))

    identity_metrics = rank_metrics_from_order(order, np.asarray(dataset.targets, dtype=np.int64), image_labels, top_k=top_k)
    metrics = {
        f"text_to_image_recall@{k}": float(np.mean(values)) if values else 0.0
        for k, values in image_recalls.items()
    }
    metrics["text_to_image_mrr"] = float(np.mean(image_rr)) if image_rr else 0.0
    metrics["text_to_image_valid_queries"] = float(len(image_rr))
    for key, value in identity_metrics.items():
        metrics[f"text_to_identity_{key}"] = value
    return metrics


@torch.no_grad()
def evaluate_embedding_text_to_image(
    transformation: DeVISETransformation,
    dataset: EmbeddingPairDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    transformation.eval()
    image_embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch_image_embeddings, _, batch_labels, batch_indices in tqdm(loader, desc="Transform image embeddings", leave=False):
        batch_image_embeddings = batch_image_embeddings.to(device, non_blocking=True)
        transformed = transformation(batch_image_embeddings).detach().cpu().numpy().astype("float32")
        transformed /= np.linalg.norm(transformed, axis=1, keepdims=True).clip(min=1e-12)
        image_embeddings.append(transformed)
        labels.append(batch_labels.numpy().astype("int64"))
        indices.append(batch_indices.numpy().astype("int64"))
    image_matrix = np.concatenate(image_embeddings)
    image_labels = np.concatenate(labels)
    row_indices = np.concatenate(indices)
    text_embeddings = np.array(dataset.text_embeddings, dtype="float32", copy=True)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    scores = text_embeddings @ image_matrix.T
    order = np.argsort(-scores, axis=1)
    image_ranked = row_indices[order]
    image_hits = image_ranked == np.arange(len(dataset), dtype=np.int64)[:, None]

    image_recalls: dict[int, list[float]] = {int(k): [] for k in top_k}
    image_rr: list[float] = []
    for row in image_hits:
        if not np.any(row):
            continue
        rank = int(np.argmax(row)) + 1
        image_rr.append(1.0 / rank)
        for k in image_recalls:
            image_recalls[k].append(float(np.any(row[:k])))

    identity_metrics = rank_metrics_from_order(order, np.asarray(dataset.targets, dtype=np.int64), image_labels, top_k=top_k)
    metrics = {
        f"text_to_image_recall@{k}": float(np.mean(values)) if values else 0.0
        for k, values in image_recalls.items()
    }
    metrics["text_to_image_mrr"] = float(np.mean(image_rr)) if image_rr else 0.0
    metrics["text_to_image_valid_queries"] = float(len(image_rr))
    for key, value in identity_metrics.items():
        metrics[f"text_to_identity_{key}"] = value
    return metrics


def run_devise_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    train_dataset = EmbeddingPairDataset(
        args.train_manifest,
        args.train_image_embeddings,
        args.train_text_embeddings,
    )
    val_dataset = EmbeddingPairDataset(
        args.val_manifest,
        args.val_image_embeddings,
        args.val_text_embeddings,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    image_embedding_dim = int(train_dataset.image_embeddings.shape[1])
    transformation = DeVISETransformation(
        image_embedding_dim=image_embedding_dim,
        text_embedding_dim=args.embedding_dim,
        hidden_dim=args.projection_hidden_dim,
    ).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / args.temperature), device=device))
    trainable_parameters = list(transformation.parameters()) + [logit_scale]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_devise_scheduler(optimizer, epochs=args.epochs, lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run_config = vars(args).copy()
    run_config.pop("hf_token", None)
    run_config["training_mode"] = "devise_frozen_image_embedding_transform"
    run_config["text_model_name"] = TEXT_MODEL_NAME
    run_config["text_embedding_dim"] = args.embedding_dim
    run_config["image_embedding_dim"] = image_embedding_dim
    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "class_to_idx.json", train_dataset.class_to_idx)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_transform_one_epoch(
            transformation=transformation,
            logit_scale=logit_scale,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=args.amp,
            max_batches=getattr(args, "batches_per_epoch", None),
        )
        val_metrics = evaluate_embedding_text_to_image(
            transformation=transformation,
            dataset=val_dataset,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            device=device,
        )
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
            "lr": current_lr,
        }
        history.append({"epoch": epoch, "metrics": metrics})
        save_training_artifacts(output_dir, history)
        save_devise_checkpoint(
            output_dir / "last.pt",
            transformation,
            args.image_checkpoint,
            epoch,
            metrics,
            train_dataset.class_to_idx,
            run_config,
            logit_scale,
        )
        score = float(val_metrics.get("text_to_identity_recall@1", 0.0))
        if score > best_score:
            best_score = score
            save_devise_checkpoint(
                output_dir / "best.pt",
                transformation,
                args.image_checkpoint,
                epoch,
                metrics,
                train_dataset.class_to_idx,
                run_config,
                logit_scale,
            )
        scheduler.step()
        print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True), flush=True)

    result = {
        "output_dir": str(output_dir.resolve()),
        "elapsed_sec": time.time() - started,
        "best_score": best_score,
        "epochs": args.epochs,
    }
    if torch.cuda.is_available() and device.type == "cuda":
        result["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
    write_json(output_dir / "training_summary.json", result)
    return result


def normalize_keyword(value: str) -> str:
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def identity_aliases(identity: str) -> set[str]:
    raw = str(identity)
    variants = {raw, raw.replace("_", " "), raw.replace("_", "-")}
    stripped = re.sub(r"\([^)]*\)", "", raw).strip("_- ")
    if stripped and stripped != raw:
        variants.update({stripped, stripped.replace("_", " "), stripped.replace("_", "-")})
    normalized = {normalize_keyword(item) for item in variants}
    return {item for item in normalized if len(item) >= 2}


def build_identity_alias_map(identities: Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for identity in sorted({str(item) for item in identities}):
        for alias in identity_aliases(identity):
            mapping.setdefault(alias, []).append(identity)
    return mapping


def find_identity_matches(query: str, identities: Iterable[str]) -> tuple[list[str], list[str], str]:
    normalized_query = normalize_keyword(query)
    alias_map = build_identity_alias_map(identities)
    matched_aliases: list[str] = []
    matched_identities: set[str] = set()
    for alias in sorted(alias_map, key=lambda item: (-len(item), item)):
        if re.search(rf"(^| ){re.escape(alias)}( |$)", normalized_query):
            matched_aliases.append(alias)
            matched_identities.update(alias_map[alias])
    remainder = normalized_query
    for alias in matched_aliases:
        remainder = re.sub(rf"(^| ){re.escape(alias)}( |$)", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    return sorted(matched_identities), matched_aliases, remainder


def metadata_gallery_embeddings(index) -> np.ndarray:
    embeddings = np.zeros((index.ntotal, index.d), dtype="float32")
    index.reconstruct_n(0, index.ntotal, embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    return embeddings


def search_text_with_index(
    query: str,
    items: list[dict[str, Any]],
    index,
    text_encoder: TextEmbeddingEncoder | None,
    top_k: int,
) -> dict[str, Any]:
    identities = [item["identity"] for item in items]
    matched_identities, matched_aliases, visual_query = find_identity_matches(query, identities)
    identity_filter = set(matched_identities)
    candidate_indices = np.arange(len(items), dtype=np.int64)
    if identity_filter:
        candidate_indices = np.array([i for i, item in enumerate(items) if item["identity"] in identity_filter], dtype=np.int64)

    if identity_filter and not visual_query:
        matches = [
            {
                "identity": items[int(index_value)]["identity"],
                "score": 1.0,
                "path": items[int(index_value)]["path"],
                "index": int(index_value),
                "match_mode": "keyword",
            }
            for index_value in candidate_indices[:top_k]
        ]
        return {
            "mode": "keyword",
            "matched_identities": matched_identities,
            "matched_aliases": matched_aliases,
            "visual_query": visual_query,
            "matches": matches,
        }

    if text_encoder is None:
        raise RuntimeError("Text encoder is required for semantic text search.")
    semantic_query = visual_query if identity_filter and visual_query else query
    query_embedding = text_encoder.encode([semantic_query], query=True)

    if identity_filter:
        gallery_embeddings = metadata_gallery_embeddings(index)
        scores = gallery_embeddings[candidate_indices] @ query_embedding[0]
        local_order = np.argsort(-scores)[:top_k]
        matches = []
        for local_index in local_order:
            index_value = int(candidate_indices[int(local_index)])
            item = items[index_value]
            matches.append(
                {
                    "identity": item["identity"],
                    "score": float(scores[int(local_index)]),
                    "path": item["path"],
                    "index": index_value,
                    "match_mode": "keyword+semantic",
                }
            )
        mode = "keyword+semantic"
    else:
        search_k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding.astype("float32"), search_k)
        matches = []
        for score, index_value in zip(scores[0], indices[0]):
            if index_value < 0:
                continue
            item = items[int(index_value)]
            matches.append(
                {
                    "identity": item["identity"],
                    "score": float(score),
                    "path": item["path"],
                    "index": int(index_value),
                    "match_mode": "semantic",
                }
            )
        mode = "semantic"

    return {
        "mode": mode,
        "matched_identities": matched_identities,
        "matched_aliases": matched_aliases,
        "visual_query": visual_query,
        "matches": matches,
    }


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(path: str | Path, rows: list[dict[str, Any]], keys: list[str], ylabel: str, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(keys))
    plt.figure(figsize=(8, 5))
    for offset, key in enumerate(keys):
        values = [float(row.get(key, 0.0)) for row in rows]
        plt.bar(x + (offset - (len(keys) - 1) / 2) * width, values, width=width, label=key)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def evaluate_text_embeddings_against_index(
    text_embeddings_path: str | Path,
    records_path: str | Path,
    index_dir: str | Path,
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, Any]:
    started = time.time()
    top_k = tuple(int(k) for k in top_k)
    index_dir = Path(index_dir)
    text_embeddings = np.load(text_embeddings_path).astype("float32")
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    records = read_jsonl(records_path)
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "gallery.faiss"))
    item_identities = np.array([item["identity"] for item in metadata["items"]])
    query_identities = np.array([record["identity"] for record in records])
    metadata_paths = [str(Path(item["path"]).resolve()) for item in metadata["items"]]
    record_paths = [str(Path(record["path"]).resolve()) for record in records]
    aligned_paths = len(metadata_paths) == len(record_paths) and all(left == right for left, right in zip(metadata_paths, record_paths))
    if aligned_paths:
        target_indices = np.arange(len(records), dtype=np.int64)
    else:
        path_to_index = {path: index_value for index_value, path in enumerate(metadata_paths)}
        target_indices = np.array([path_to_index.get(path, -1) for path in record_paths], dtype=np.int64)
    image_recall_counts = {k: 0 for k in top_k}
    identity_recall_counts = {k: 0 for k in top_k}
    image_rr_sum = 0.0
    identity_rr_sum = 0.0
    image_valid = 0
    identity_valid = 0
    top1_exact_path_matches = 0
    search_batch_size = 512
    for start in range(0, len(records), search_batch_size):
        end = min(start + search_batch_size, len(records))
        _, order = index.search(text_embeddings[start:end], index.ntotal)
        image_matches = order == target_indices[start:end, None]
        identity_matches = item_identities[order] == query_identities[start:end, None]

        top1_exact_path_matches += int(image_matches[:, 0].sum())
        image_any = image_matches.any(axis=1)
        if np.any(image_any):
            image_ranks = np.argmax(image_matches, axis=1)[image_any] + 1
            image_rr_sum += float(np.sum(1.0 / image_ranks))
            image_valid += int(image_any.sum())
        identity_any = identity_matches.any(axis=1)
        if np.any(identity_any):
            identity_ranks = np.argmax(identity_matches, axis=1)[identity_any] + 1
            identity_rr_sum += float(np.sum(1.0 / identity_ranks))
            identity_valid += int(identity_any.sum())
        for k in top_k:
            image_recall_counts[k] += int(image_matches[:, :k].any(axis=1).sum())
            identity_recall_counts[k] += int(identity_matches[:, :k].any(axis=1).sum())

    metrics: dict[str, Any] = {
        "model": "devise_transform_hf_finetune_v2_qwen3_256_top500",
        "valid_queries": len(records),
        "top1_exact_path_matches": top1_exact_path_matches,
        "evaluation_sec": time.time() - started,
        "index_ntotal": int(index.ntotal),
        "embedding_dim": int(index.d),
        "image_valid_queries": image_valid,
        "identity_valid_queries": identity_valid,
    }
    for k in top_k:
        metrics[f"image_recall@{k}"] = image_recall_counts[k] / max(1, len(records))
    metrics["image_mrr"] = image_rr_sum / max(1, image_valid)
    for k in top_k:
        metrics[f"identity_recall@{k}"] = identity_recall_counts[k] / max(1, len(records))
    metrics["identity_mrr"] = identity_rr_sum / max(1, identity_valid)
    return metrics


def build_devise_gallery_index(
    checkpoint_path: str | Path,
    gallery_dir: str | Path | None,
    gallery_manifest: str | Path | None,
    output_dir: str | Path,
    batch_size: int = 128,
    workers: int = 4,
    device: str = "auto",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(device)
    model, checkpoint = load_devise_checkpoint(checkpoint_path, map_location=torch_device)
    model.to(torch_device)
    run_config = checkpoint.get("run_config", {})
    dataset = make_dataset(
        Path(gallery_dir) if gallery_dir else None,
        gallery_manifest,
        image_size=int(run_config.get("image_size", 224)),
        train=False,
        mean=run_config.get("image_mean", [0.5, 0.5, 0.5]),
        std=run_config.get("image_std", [0.5, 0.5, 0.5]),
    )
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
                "path": str(Path(path).absolute()),
                "identity": idx_to_class[int(label)],
                "label": int(label),
            }
        )
    metadata = {
        "index_type": "devise_text_space",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "base_image_checkpoint": checkpoint.get("base_image_checkpoint"),
        "gallery_dir": str(Path(gallery_dir).resolve()) if gallery_dir else None,
        "gallery_manifest": str(Path(gallery_manifest).resolve()) if gallery_manifest else None,
        "image_size": int(run_config.get("image_size", 224)),
        "image_mean": run_config.get("image_mean", [0.5, 0.5, 0.5]),
        "image_std": run_config.get("image_std", [0.5, 0.5, 0.5]),
        "embedding_dim": int(embeddings.shape[1]),
        "items": items,
        "class_to_idx": dataset.class_to_idx,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def keyword_match_metrics(items: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    identities = sorted({item["identity"] for item in items})
    successes = 0
    examples = []
    class DummyIndex:
        ntotal = len(items)
        d = 256

    for identity in identities:
        result = search_text_with_index(identity, items, DummyIndex(), None, top_k=top_k)
        top_identity = result["matches"][0]["identity"] if result["matches"] else None
        ok = top_identity == identity
        successes += int(ok)
        if len(examples) < 10:
            examples.append({"query": identity, "top_identity": top_identity, "ok": ok})
    return {
        "queries": len(identities),
        "recall@1": successes / max(1, len(identities)),
        "examples": examples,
    }
