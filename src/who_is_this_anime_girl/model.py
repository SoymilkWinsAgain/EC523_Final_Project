from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import timm


@dataclass
class ModelConfig:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    embedding_dim: int = 256
    projection_hidden_dim: int = 512
    num_classes: int | None = None


class AnimeEmbeddingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=0)
        feature_dim = int(self.backbone.num_features)

        if config.projection_hidden_dim and config.projection_hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, config.projection_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projection_hidden_dim, config.embedding_dim),
            )
        else:
            self.projection = nn.Linear(feature_dim, config.embedding_dim)

        self.classifier = nn.Linear(config.embedding_dim, config.num_classes) if config.num_classes else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        embedding = F.normalize(self.projection(features), dim=1)
        outputs = {"embedding": embedding}
        if self.classifier is not None:
            outputs["logits"] = self.classifier(embedding)
        return outputs

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def create_model(
    model_name: str,
    pretrained: bool,
    embedding_dim: int,
    projection_hidden_dim: int,
    num_classes: int | None = None,
) -> AnimeEmbeddingModel:
    config = ModelConfig(
        model_name=model_name,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        projection_hidden_dim=projection_hidden_dim,
        num_classes=num_classes,
    )
    return AnimeEmbeddingModel(config)


def save_checkpoint(
    path: str | Path,
    model: AnimeEmbeddingModel,
    epoch: int,
    metrics: dict[str, float],
    class_to_idx: dict[str, int],
    run_config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "metrics": metrics,
            "class_to_idx": class_to_idx,
            "run_config": run_config,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[AnimeEmbeddingModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    model_config = ModelConfig(**checkpoint["model_config"])
    model_config.pretrained = False
    model = AnimeEmbeddingModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
