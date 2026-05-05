from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .data import load_manifest
from .devise import build_devise_document_text
from .losses import symmetric_image_text_contrastive_loss
from .model import apply_linear_lora
from .reporting import save_training_artifacts
from .utils import resolve_device, set_seed, write_json


CHECKPOINT_TYPE = "joint_clip_v1"


@dataclass
class JointClipConfig:
    backend: str
    model_name: str
    pretrained: bool = True
    trust_remote_code: bool = False
    open_clip_pretrained: str | None = None
    train_mode: str = "lora_vision"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    include_character_text: bool = False


def sanitize_model_name(value: str) -> str:
    cleaned = value.replace("hf-hub:", "").replace("/", "_").replace("-", "_").replace(".", "_")
    return "_".join(part for part in cleaned.split("_") if part)


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _set_trainable(module_or_parameter: nn.Module | nn.Parameter | None, trainable: bool) -> None:
    if module_or_parameter is None:
        return
    if isinstance(module_or_parameter, nn.Parameter):
        module_or_parameter.requires_grad = trainable
        return
    for parameter in module_or_parameter.parameters():
        parameter.requires_grad = trainable


def _maybe_apply_lora(module: nn.Module, rank: int, alpha: int, dropout: float, target_modules: list[str] | None) -> int:
    if rank <= 0:
        return 0
    return apply_linear_lora(module, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)


def count_parameters(module: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


class JointImageTextDataset(Dataset):
    def __init__(self, manifest_path: str | Path, include_character_text: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_manifest(self.manifest_path)
        self.texts = [build_devise_document_text(record, include_character=include_character_text) for record in self.records]
        identities = sorted({str(record["identity"]) for record in self.records})
        self.classes = identities
        self.class_to_idx = {identity: index for index, identity in enumerate(identities)}
        self.targets = [self.class_to_idx[str(record["identity"])] for record in self.records]
        self.samples = [(str(Path(record["path"])), self.targets[index]) for index, record in enumerate(self.records)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB").copy()
        return {"image": rgb, "text": self.texts[index], "label": int(label), "index": int(index), "path": path}


class JointBatchCollator:
    def __init__(
        self,
        processor: Any | None = None,
        image_transform: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.processor = processor
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        images = [row["image"] for row in rows]
        texts = [row["text"] for row in rows]
        if self.processor is not None:
            encoded = dict(
                self.processor(
                    images=images,
                    text=texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            )
        else:
            if self.image_transform is None or self.tokenizer is None:
                raise RuntimeError("OpenCLIP collator requires image_transform and tokenizer.")
            encoded = {
                "pixel_values": torch.stack([self.image_transform(image) for image in images]),
                "input_ids": self.tokenizer(texts),
            }
        encoded["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        encoded["indices"] = torch.tensor([row["index"] for row in rows], dtype=torch.long)
        return encoded


class BaseJointClipModel(nn.Module):
    config: JointClipConfig

    def prepare_batch(self, images: Sequence[Image.Image], texts: Sequence[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def create_collator(self) -> JointBatchCollator:
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def image_embedding_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(batch)["image_embedding"]

    def text_embedding_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(batch)["text_embedding"]

    def trainable_parameter_count(self) -> tuple[int, int]:
        counts = count_parameters(self)
        return counts["trainable_parameters"], counts["total_parameters"]


class TransformersJointClipModel(BaseJointClipModel):
    def __init__(self, config: JointClipConfig, hf_token: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code, token=hf_token)
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            token=hf_token,
        )
        self.fallback_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        self.configure_trainability()

    def configure_trainability(self) -> None:
        mode = self.config.train_mode
        _freeze(self.model)
        self.fallback_logit_scale.requires_grad = False
        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = False

        if mode == "frozen":
            return
        if mode == "projection":
            _set_trainable(getattr(self.model, "visual_projection", None), True)
            if hasattr(self.model, "logit_scale"):
                self.model.logit_scale.requires_grad = True
            else:
                self.fallback_logit_scale.requires_grad = True
            return
        if mode == "full_vision":
            _set_trainable(getattr(self.model, "vision_model", None), True)
            _set_trainable(getattr(self.model, "visual_projection", None), True)
            if hasattr(self.model, "logit_scale"):
                self.model.logit_scale.requires_grad = True
            else:
                self.fallback_logit_scale.requires_grad = True
            return
        if mode != "lora_vision":
            raise ValueError("train_mode must be one of: frozen, projection, lora_vision, full_vision")

        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            raise ValueError(f"{self.config.model_name} does not expose a vision_model module for LoRA.")
        target_modules = self.config.lora_target_modules
        if target_modules is None and "siglip" in self.config.model_name.lower():
            target_modules = ["mlp.fc1", "mlp.fc2"]
        _maybe_apply_lora(
            vision_model,
            rank=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )
        _set_trainable(getattr(self.model, "visual_projection", None), True)
        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = True
        else:
            self.fallback_logit_scale.requires_grad = True

    def prepare_batch(self, images: Sequence[Image.Image], texts: Sequence[str]) -> dict[str, torch.Tensor]:
        return dict(
            self.processor(
                images=list(images),
                text=list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        )

    def create_collator(self) -> JointBatchCollator:
        return JointBatchCollator(processor=self.processor)

    def logit_scale(self) -> torch.Tensor:
        value = getattr(self.model, "logit_scale", None)
        return value if value is not None else self.fallback_logit_scale

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_kwargs = {"pixel_values": batch["pixel_values"]}
        text_kwargs = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            text_kwargs["attention_mask"] = batch["attention_mask"]
        image_embedding = self.model.get_image_features(**image_kwargs)
        text_embedding = self.model.get_text_features(**text_kwargs)
        return {
            "image_embedding": F.normalize(image_embedding, dim=1),
            "text_embedding": F.normalize(text_embedding, dim=1),
            "embedding": F.normalize(image_embedding, dim=1),
            "logit_scale": self.logit_scale(),
        }


class OpenClipJointModel(BaseJointClipModel):
    def __init__(self, config: JointClipConfig) -> None:
        super().__init__()
        import open_clip

        self.config = config
        model_name = config.model_name
        pretrained = config.open_clip_pretrained
        if model_name.startswith("hf-hub:") or "/" in model_name:
            hub_id = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
            self.model, self.image_transform = open_clip.create_model_from_pretrained(hub_id)
            self.tokenizer = open_clip.get_tokenizer(hub_id)
        else:
            self.model, _, self.image_transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        self.fallback_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        self.configure_trainability()

    def configure_trainability(self) -> None:
        mode = self.config.train_mode
        _freeze(self.model)
        self.fallback_logit_scale.requires_grad = False
        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = False

        if mode == "frozen":
            return
        if mode == "projection":
            _set_trainable(getattr(self.model, "visual", None).proj if hasattr(getattr(self.model, "visual", None), "proj") else None, True)
        elif mode == "full_vision":
            _set_trainable(getattr(self.model, "visual", None), True)
        elif mode == "lora_vision":
            visual = getattr(self.model, "visual", None)
            if visual is None:
                raise ValueError(f"{self.config.model_name} does not expose a visual module for LoRA.")
            target_modules = self.config.lora_target_modules or ["mlp.c_fc", "mlp.c_proj", "mlp.fc1", "mlp.fc2"]
            _maybe_apply_lora(
                visual,
                rank=self.config.lora_r,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=target_modules,
            )
            _set_trainable(getattr(visual, "proj", None), True)
        else:
            raise ValueError("train_mode must be one of: frozen, projection, lora_vision, full_vision")

        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = True
        else:
            self.fallback_logit_scale.requires_grad = True

    def prepare_batch(self, images: Sequence[Image.Image], texts: Sequence[str]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([self.image_transform(image) for image in images])
        input_ids = self.tokenizer(list(texts))
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def create_collator(self) -> JointBatchCollator:
        return JointBatchCollator(image_transform=self.image_transform, tokenizer=self.tokenizer)

    def logit_scale(self) -> torch.Tensor:
        value = getattr(self.model, "logit_scale", None)
        return value if value is not None else self.fallback_logit_scale

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_embedding = self.model.encode_image(batch["pixel_values"])
        text_embedding = self.model.encode_text(batch["input_ids"])
        return {
            "image_embedding": F.normalize(image_embedding, dim=1),
            "text_embedding": F.normalize(text_embedding, dim=1),
            "embedding": F.normalize(image_embedding, dim=1),
            "logit_scale": self.logit_scale(),
        }


def create_joint_clip_model(config: JointClipConfig, hf_token: str | None = None) -> BaseJointClipModel:
    backend = config.backend.lower()
    if backend in {"hf-transformers-clip", "hf-transformers", "transformers"}:
        return TransformersJointClipModel(config, hf_token=hf_token)
    if backend in {"open-clip", "open_clip"}:
        return OpenClipJointModel(config)
    raise ValueError(f"Unsupported joint CLIP backend: {config.backend}")


def create_joint_clip_model_from_values(
    backend: str,
    model_name: str,
    pretrained: bool = True,
    trust_remote_code: bool = False,
    open_clip_pretrained: str | None = None,
    train_mode: str = "lora_vision",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    include_character_text: bool = False,
    hf_token: str | None = None,
) -> BaseJointClipModel:
    config = JointClipConfig(
        backend=backend,
        model_name=model_name,
        pretrained=pretrained,
        trust_remote_code=trust_remote_code,
        open_clip_pretrained=open_clip_pretrained,
        train_mode=train_mode,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        include_character_text=include_character_text,
    )
    return create_joint_clip_model(config, hf_token=hf_token)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable_names}


def save_joint_clip_checkpoint(
    path: str | Path,
    model: BaseJointClipModel,
    epoch: int,
    metrics: dict[str, float],
    class_to_idx: dict[str, int],
    run_config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": CHECKPOINT_TYPE,
            "epoch": int(epoch),
            "joint_clip_config": asdict(model.config),
            "trainable_state": trainable_state_dict(model),
            "metrics": metrics,
            "class_to_idx": class_to_idx,
            "run_config": run_config,
        },
        path,
    )


def is_joint_clip_checkpoint(path: str | Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("checkpoint_type") == CHECKPOINT_TYPE


def load_joint_clip_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[BaseJointClipModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"Not a joint CLIP checkpoint: {path}")
    config = JointClipConfig(**checkpoint["joint_clip_config"])
    model = create_joint_clip_model(config)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys for {path}: {unexpected}")
    model.to(map_location)
    model.eval()
    checkpoint["missing_keys"] = missing
    return model, checkpoint


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def create_joint_scheduler(
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


def train_joint_clip_one_epoch(
    model: BaseJointClipModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_accum_steps: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_logit_scale = 0.0
    batches = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in tqdm(loader, desc="Train Joint CLIP", leave=False):
        if max_batches is not None and batches >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch)
            loss = symmetric_image_text_contrastive_loss(outputs["image_embedding"], outputs["text_embedding"], outputs["logit_scale"])
            scaled_loss = loss / max(1, grad_accum_steps)
        scaler.scale(scaled_loss).backward()
        if (batches + 1) % max(1, grad_accum_steps) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if hasattr(model, "model") and hasattr(model.model, "logit_scale"):
                    model.model.logit_scale.clamp_(min=math.log(1.0), max=math.log(100.0))
        total_loss += float(loss.detach().cpu())
        total_logit_scale += float(outputs["logit_scale"].detach().exp().cpu())
        batches += 1
    if batches % max(1, grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": total_loss / max(1, batches),
        "image_text_loss": total_loss / max(1, batches),
        "logit_scale": total_logit_scale / max(1, batches),
        "batches": float(batches),
    }


@torch.no_grad()
def extract_joint_embeddings(
    model: BaseJointClipModel,
    dataset: JointImageTextDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=model.create_collator(),
    )
    model.eval()
    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Extract joint embeddings", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        image = F.normalize(outputs["image_embedding"], dim=1).detach().cpu().numpy().astype("float32")
        text = F.normalize(outputs["text_embedding"], dim=1).detach().cpu().numpy().astype("float32")
        image_embeddings.append(image)
        text_embeddings.append(text)
        labels.append(batch["labels"].detach().cpu().numpy().astype("int64"))
        indices.append(batch["indices"].detach().cpu().numpy().astype("int64"))
    return np.concatenate(image_embeddings), np.concatenate(text_embeddings), np.concatenate(labels), np.concatenate(indices)


def text_to_image_metrics(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    labels: np.ndarray,
    row_indices: np.ndarray,
    top_k: Iterable[int] = (1, 5, 10),
    batch_size: int = 512,
) -> dict[str, float]:
    top_k = tuple(int(k) for k in top_k)
    text_embeddings = text_embeddings.astype("float32")
    image_embeddings = image_embeddings.astype("float32")
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    image_recall_counts = {k: 0 for k in top_k}
    identity_recall_counts = {k: 0 for k in top_k}
    image_rr = 0.0
    identity_rr = 0.0
    image_valid = len(text_embeddings)
    identity_valid = 0
    row_to_position = {int(row_index): position for position, row_index in enumerate(row_indices)}
    for start in range(0, len(text_embeddings), batch_size):
        end = min(start + batch_size, len(text_embeddings))
        scores = text_embeddings[start:end] @ image_embeddings.T
        query_rows = row_indices[start:end]
        target_positions = np.array([row_to_position[int(row_index)] for row_index in query_rows], dtype=np.int64)
        exact_scores = scores[np.arange(end - start), target_positions]
        exact_ranks = 1 + np.sum(scores > exact_scores[:, None], axis=1)
        image_rr += float(np.sum(1.0 / exact_ranks))

        identity_mask = labels[None, :] == labels[start:end, None]
        best_identity_scores = np.where(identity_mask, scores, -np.inf).max(axis=1)
        identity_valid_mask = np.isfinite(best_identity_scores)
        if np.any(identity_valid_mask):
            identity_ranks = 1 + np.sum(scores[identity_valid_mask] > best_identity_scores[identity_valid_mask, None], axis=1)
            identity_rr += float(np.sum(1.0 / identity_ranks))
            identity_valid += int(identity_valid_mask.sum())
        for k in top_k:
            image_recall_counts[k] += int(np.sum(exact_ranks <= k))
            if np.any(identity_valid_mask):
                identity_recall_counts[k] += int(np.sum(identity_ranks <= k))
    result: dict[str, float] = {
        "text_to_image_mrr": image_rr / max(1, image_valid),
        "text_to_image_valid_queries": float(image_valid),
        "text_to_identity_mrr": identity_rr / max(1, identity_valid),
        "text_to_identity_valid_queries": float(identity_valid),
    }
    for k in top_k:
        result[f"text_to_image_recall@{k}"] = image_recall_counts[k] / max(1, len(text_embeddings))
        result[f"text_to_identity_recall@{k}"] = identity_recall_counts[k] / max(1, len(text_embeddings))
    return result


def image_identity_retrieval_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: Iterable[int] = (1, 5, 10),
    batch_size: int = 512,
) -> dict[str, float]:
    top_k = tuple(int(k) for k in top_k)
    embeddings = embeddings.astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    labels = labels.astype("int64")
    recall_counts = {k: 0 for k in top_k}
    reciprocal_rank = 0.0
    valid_queries = 0
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        scores = embeddings[start:end] @ embeddings.T
        query_positions = np.arange(start, end)
        scores[np.arange(end - start), query_positions] = -np.inf
        positive_mask = labels[None, :] == labels[start:end, None]
        positive_mask[np.arange(end - start), query_positions] = False
        best_positive_scores = np.where(positive_mask, scores, -np.inf).max(axis=1)
        valid_mask = np.isfinite(best_positive_scores)
        if not np.any(valid_mask):
            continue
        ranks = 1 + np.sum(scores[valid_mask] > best_positive_scores[valid_mask, None], axis=1)
        valid_queries += int(valid_mask.sum())
        reciprocal_rank += float(np.sum(1.0 / ranks))
        for k in top_k:
            recall_counts[k] += int(np.sum(ranks <= k))
    metrics = {f"recall@{k}": recall_counts[k] / max(1, valid_queries) for k in top_k}
    metrics["mrr"] = reciprocal_rank / max(1, valid_queries)
    metrics["valid_queries"] = float(valid_queries)
    return metrics


def evaluate_joint_clip_model(
    model: BaseJointClipModel,
    manifest_path: str | Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    include_character_text: bool = False,
) -> dict[str, float]:
    dataset = JointImageTextDataset(manifest_path, include_character_text=include_character_text)
    image_embeddings, text_embeddings, labels, row_indices = extract_joint_embeddings(model, dataset, batch_size, workers, device)
    image_metrics = image_identity_retrieval_metrics(image_embeddings, labels, top_k=(1, 5, 10))
    metrics: dict[str, float] = {
        f"image_to_image_{key}": value for key, value in image_metrics.items() if key != "valid_queries"
    }
    metrics["image_to_image_valid_queries"] = image_metrics["valid_queries"]
    metrics.update(text_to_image_metrics(text_embeddings, image_embeddings, labels, row_indices, top_k=(1, 5, 10)))
    return metrics


def run_joint_clip_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = create_joint_clip_model_from_values(
        backend=args.backend,
        model_name=args.model_name,
        pretrained=True,
        trust_remote_code=args.trust_remote_code,
        open_clip_pretrained=args.open_clip_pretrained,
        train_mode=args.train_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        include_character_text=args.include_character_text,
        hf_token=args.hf_token,
    ).to(device)
    train_dataset = JointImageTextDataset(args.train_manifest, include_character_text=args.include_character_text)
    val_manifest = args.val_manifest
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=model.create_collator(),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are available for the selected joint CLIP train_mode.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_joint_scheduler(optimizer, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run_config = vars(args).copy()
    run_config.pop("hf_token", None)
    run_config["checkpoint_type"] = CHECKPOINT_TYPE
    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "class_to_idx.json", train_dataset.class_to_idx)
    write_json(output_dir / "parameter_counts.json", count_parameters(model))

    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_joint_clip_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=args.amp and device.type == "cuda",
            grad_accum_steps=args.grad_accum_steps,
            max_batches=args.batches_per_epoch,
        )
        metrics = {"lr": current_lr, **{f"train/{key}": value for key, value in train_metrics.items()}}
        if val_manifest:
            val_metrics = evaluate_joint_clip_model(
                model=model,
                manifest_path=val_manifest,
                batch_size=args.eval_batch_size,
                workers=args.workers,
                device=device,
                include_character_text=args.include_character_text,
            )
            metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            score = float(val_metrics.get("text_to_identity_recall@1", 0.0))
        else:
            score = -float(train_metrics["loss"])

        history.append({"epoch": epoch, "metrics": metrics})
        save_training_artifacts(output_dir, history)
        save_joint_clip_checkpoint(output_dir / "last.pt", model, epoch, metrics, train_dataset.class_to_idx, run_config)
        if score > best_score:
            best_score = score
            save_joint_clip_checkpoint(output_dir / "best.pt", model, epoch, metrics, train_dataset.class_to_idx, run_config)
        scheduler.step()
        print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True), flush=True)

    result = {
        "output_dir": str(output_dir.resolve()),
        "elapsed_sec": time.time() - started,
        "best_score": best_score,
        "epochs": args.epochs,
        **count_parameters(model),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        result["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        result["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1024**3
    write_json(output_dir / "training_summary.json", result)
    return result


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


def parse_optional_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP/SigLIP/OpenCLIP joint image-text embedding model.")
    parser.add_argument("--backend", required=True, choices=["hf-transformers-clip", "open-clip"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--open-clip-pretrained", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-mode", default="lora_vision", choices=["frozen", "projection", "lora_vision", "full_vision"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=parse_optional_csv, default=None)
    parser.add_argument("--include-character-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=523)
    return parser.parse_args()


def main() -> None:
    run_joint_clip_training(parse_args())


if __name__ == "__main__":
    main()
