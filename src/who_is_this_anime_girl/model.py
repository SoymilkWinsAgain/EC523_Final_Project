from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None


@dataclass
class ModelConfig:
    backbone_backend: str = "timm"
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    embedding_dim: int = 256
    projection_hidden_dim: int = 512
    num_classes: int | None = None
    trust_remote_code: bool = False
    timm_kwargs: dict[str, Any] | None = None
    finetune_mode: str = "full"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scaling


def _name_matches(name: str, target_modules: Iterable[str] | None) -> bool:
    targets = [target for target in (target_modules or []) if target]
    if not targets:
        return True
    return any(name.endswith(target) or target in name for target in targets)


def apply_linear_lora(module: nn.Module, rank: int, alpha: int, dropout: float, target_modules: list[str] | None) -> int:
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and _name_matches(name, target_modules):
            parent_path, _, child_name = name.rpartition(".")
            parent = module.get_submodule(parent_path) if parent_path else module
            replacements.append((parent, child_name, child))

    for parent, child_name, child in replacements:
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
    return len(replacements)


class TimmBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        kwargs = dict(config.timm_kwargs or {})
        self.model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=0, **kwargs)
        self.feature_dim = int(self.model.num_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class TransformersBackbone(nn.Module):
    def __init__(self, config: ModelConfig, hf_token: str | None = None) -> None:
        super().__init__()
        if config.pretrained:
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=config.trust_remote_code,
                token=hf_token,
            )
        else:
            hf_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
            self.model = AutoModel.from_config(hf_config, trust_remote_code=config.trust_remote_code)
        self.feature_dim = int(
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "projection_dim", None)
            or getattr(self.model.config, "embed_dim", None)
            or 0
        )
        if self.feature_dim <= 0:
            raise ValueError(f"Could not infer feature dimension for transformers model: {config.model_name}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state[:, 0]
        if isinstance(outputs, tuple) and outputs:
            value = outputs[0]
            return value[:, 0] if value.ndim == 3 else value
        raise ValueError("Transformers model output does not contain a usable image feature tensor.")


class AnimeEmbeddingModel(nn.Module):
    def __init__(self, config: ModelConfig, hf_token: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = self._create_backbone(config, hf_token=hf_token)
        feature_dim = int(self.backbone.feature_dim)

        if config.projection_hidden_dim and config.projection_hidden_dim > 0:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, config.projection_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projection_hidden_dim, config.embedding_dim),
            )
        else:
            self.projection = nn.Linear(feature_dim, config.embedding_dim)

        self.classifier = nn.Linear(config.embedding_dim, config.num_classes) if config.num_classes else None
        self.configure_finetuning()

    def _create_backbone(self, config: ModelConfig, hf_token: str | None) -> nn.Module:
        backend = config.backbone_backend.lower()
        if backend in {"timm", "hf-timm"}:
            return TimmBackbone(config)
        if backend in {"hf-transformers", "transformers"}:
            return TransformersBackbone(config, hf_token=hf_token)
        if backend == "lvface":
            raise ValueError(
                "LVFace is distributed on Hugging Face as project-specific .pt/.onnx weights. "
                "Use scripts/download_hf.py to fetch those files, or provide a compatible timm/transformers backbone."
            )
        raise ValueError(f"Unsupported backbone backend: {config.backbone_backend}")

    def configure_finetuning(self) -> None:
        mode = self.config.finetune_mode.lower()
        if mode not in {"full", "projection", "frozen", "lora"}:
            raise ValueError("finetune_mode must be one of: full, projection, frozen, lora")

        if mode == "full":
            return

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if mode == "lora":
            if self.config.backbone_backend.lower() in {"hf-transformers", "transformers"}:
                if LoraConfig is None or get_peft_model is None:
                    raise ImportError("peft is required for transformers LoRA fine-tuning.")
                target_modules = self.config.lora_target_modules or ["query", "value", "qkv"]
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                )
                self.backbone.model = get_peft_model(self.backbone.model, lora_config)
            else:
                replaced = apply_linear_lora(
                    self.backbone,
                    rank=self.config.lora_r,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                )
                if replaced == 0:
                    raise ValueError("No Linear modules matched the requested LoRA target modules.")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        embedding = F.normalize(self.projection(features), dim=1)
        outputs = {"embedding": embedding}
        if self.classifier is not None:
            outputs["logits"] = self.classifier(embedding)
        return outputs

    def set_backbone_trainable(self, trainable: bool) -> None:
        if self.config.finetune_mode.lower() != "full":
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def trainable_parameter_count(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return trainable, total


def create_model(
    backbone_backend: str,
    model_name: str,
    pretrained: bool,
    embedding_dim: int,
    projection_hidden_dim: int,
    num_classes: int | None = None,
    trust_remote_code: bool = False,
    timm_kwargs: dict[str, Any] | None = None,
    finetune_mode: str = "full",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    hf_token: str | None = None,
) -> AnimeEmbeddingModel:
    config = ModelConfig(
        backbone_backend=backbone_backend,
        model_name=model_name,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        projection_hidden_dim=projection_hidden_dim,
        num_classes=num_classes,
        trust_remote_code=trust_remote_code,
        timm_kwargs=timm_kwargs,
        finetune_mode=finetune_mode,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    return AnimeEmbeddingModel(config, hf_token=hf_token)


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
    model_config = ModelConfig(**{**asdict(ModelConfig()), **checkpoint["model_config"]})
    model_config.pretrained = False
    model = AnimeEmbeddingModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
