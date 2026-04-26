from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PKBatchSampler, make_dataset
from .losses import supervised_contrastive_loss
from .metrics import extract_embeddings, retrieval_metrics
from .model import create_model, save_checkpoint
from .utils import load_yaml, resolve_device, set_seed, write_json


DEFAULTS: dict[str, Any] = {
    "train_dir": "data/train",
    "val_dir": None,
    "output_dir": "runs/vit_baseline",
    "backbone_backend": "timm",
    "model_name": "vit_base_patch16_224",
    "pretrained": True,
    "trust_remote_code": False,
    "timm_kwargs": {},
    "finetune_mode": "full",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": None,
    "image_size": 224,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "embedding_dim": 256,
    "projection_hidden_dim": 512,
    "epochs": 10,
    "identities_per_batch": 8,
    "samples_per_identity": 4,
    "batches_per_epoch": None,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "temperature": 0.07,
    "classification_weight": 0.1,
    "freeze_backbone_epochs": 0,
    "workers": 4,
    "device": "auto",
    "amp": True,
    "seed": 523,
}


def parse_json_mapping(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def parse_optional_csv(value: str | list[str] | None) -> list[str] | None:
    if value is None or isinstance(value, list):
        return value
    cleaned = [item.strip() for item in value.split(",") if item.strip()]
    return cleaned or None


def parse_float_triplet(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated float values.")
    return values


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional YAML config path.")
    known_args, _ = pre_parser.parse_known_args()

    defaults = dict(DEFAULTS)
    if known_args.config:
        defaults.update(load_yaml(known_args.config))

    parser = argparse.ArgumentParser(parents=[pre_parser], description="Train an anime character embedding model.")
    parser.add_argument("--train-dir", default=defaults["train_dir"], help="ImageFolder training directory.")
    parser.add_argument("--train-manifest", default=defaults.get("train_manifest"), help="JSONL training manifest.")
    parser.add_argument("--val-dir", default=defaults["val_dir"], help="Optional ImageFolder validation directory.")
    parser.add_argument("--val-manifest", default=defaults.get("val_manifest"), help="Optional JSONL validation manifest.")
    parser.add_argument("--output-dir", default=defaults["output_dir"], help="Directory for checkpoints and logs.")
    parser.add_argument(
        "--backbone-backend",
        default=defaults["backbone_backend"],
        choices=["timm", "hf-timm", "hf-transformers", "transformers", "lvface"],
        help="Backbone loader backend.",
    )
    parser.add_argument("--model-name", default=defaults["model_name"], help="timm model name or Hugging Face model ID.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=defaults["pretrained"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=defaults["trust_remote_code"])
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Prefer HF_TOKEN in the environment.")
    parser.add_argument("--timm-kwargs", type=parse_json_mapping, default=defaults["timm_kwargs"], help="JSON kwargs for timm.create_model.")
    parser.add_argument(
        "--finetune-mode",
        default=defaults["finetune_mode"],
        choices=["full", "projection", "frozen", "lora"],
        help="Which parameters to train.",
    )
    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--lora-target-modules",
        type=parse_optional_csv,
        default=defaults["lora_target_modules"],
        help="Comma-separated module name fragments for LoRA.",
    )
    parser.add_argument("--image-size", type=int, default=defaults["image_size"])
    parser.add_argument("--image-mean", type=parse_float_triplet, default=defaults["image_mean"])
    parser.add_argument("--image-std", type=parse_float_triplet, default=defaults["image_std"])
    parser.add_argument("--embedding-dim", type=int, default=defaults["embedding_dim"])
    parser.add_argument("--projection-hidden-dim", type=int, default=defaults["projection_hidden_dim"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--identities-per-batch", type=int, default=defaults["identities_per_batch"])
    parser.add_argument("--samples-per-identity", type=int, default=defaults["samples_per_identity"])
    parser.add_argument("--batches-per-epoch", type=int, default=defaults["batches_per_epoch"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--classification-weight", type=float, default=defaults["classification_weight"])
    parser.add_argument("--freeze-backbone-epochs", type=int, default=defaults["freeze_backbone_epochs"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--device", default=defaults["device"], help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults["amp"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    return parser.parse_args()


def namespace_to_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config.pop("hf_token", None)
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temperature: float,
    classification_weight: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_classifier = 0.0
    batches = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            contrastive_loss = supervised_contrastive_loss(outputs["embedding"], labels, temperature=temperature)
            classifier_loss = torch.zeros((), device=device)
            if classification_weight > 0 and "logits" in outputs:
                classifier_loss = F.cross_entropy(outputs["logits"], labels)
            loss = contrastive_loss + classification_weight * classifier_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu())
        total_contrastive += float(contrastive_loss.detach().cpu())
        total_classifier += float(classifier_loss.detach().cpu())
        batches += 1

    return {
        "loss": total_loss / max(1, batches),
        "contrastive_loss": total_contrastive / max(1, batches),
        "classifier_loss": total_classifier / max(1, batches),
    }


def evaluate(model: torch.nn.Module, dataset, workers: int, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    embeddings, labels = extract_embeddings(model, loader, device)
    return retrieval_metrics(embeddings, labels, top_k=(1, 5, 10))


def run_training(args: argparse.Namespace) -> None:
    args.timm_kwargs = parse_json_mapping(args.timm_kwargs)
    args.lora_target_modules = parse_optional_csv(args.lora_target_modules)
    args.image_mean = parse_float_triplet(args.image_mean)
    args.image_std = parse_float_triplet(args.image_std)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_dataset = make_dataset(
        args.train_dir,
        args.train_manifest,
        image_size=args.image_size,
        train=True,
        mean=args.image_mean,
        std=args.image_std,
    )
    val_dataset = None
    if args.val_manifest:
        val_dataset = make_dataset(
            None,
            args.val_manifest,
            image_size=args.image_size,
            train=False,
            mean=args.image_mean,
            std=args.image_std,
        )
    elif args.val_dir:
        val_path = Path(args.val_dir)
        if val_path.exists():
            val_dataset = make_dataset(
                val_path,
                None,
                image_size=args.image_size,
                train=False,
                mean=args.image_mean,
                std=args.image_std,
            )

    sampler = PKBatchSampler(
        train_dataset.targets,
        identities_per_batch=args.identities_per_batch,
        samples_per_identity=args.samples_per_identity,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(train_dataset.classes) if args.classification_weight > 0 else None
    model = create_model(
        backbone_backend=args.backbone_backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
        embedding_dim=args.embedding_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        num_classes=num_classes,
        trust_remote_code=args.trust_remote_code,
        timm_kwargs=args.timm_kwargs,
        finetune_mode=args.finetune_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        hf_token=args.hf_token,
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are available for the selected fine-tuning mode.")
    trainable_count, total_count = model.trainable_parameter_count()
    print(json.dumps({"trainable_parameters": trainable_count, "total_parameters": total_count}, sort_keys=True))

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    run_config = namespace_to_config(args)

    write_json(output_dir / "config.json", run_config)
    write_json(output_dir / "class_to_idx.json", train_dataset.class_to_idx)

    best_score = -float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.set_backbone_trainable(epoch > args.freeze_backbone_epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            temperature=args.temperature,
            classification_weight=args.classification_weight,
            amp_enabled=args.amp and device.type == "cuda",
        )

        metrics: dict[str, float] = {f"train/{key}": value for key, value in train_metrics.items()}
        if val_dataset is not None:
            val_metrics = evaluate(model, val_dataset, workers=args.workers, device=device)
            metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            score = metrics["val/recall@1"]
        else:
            score = -metrics["train/loss"]

        history.append({"epoch": epoch, "metrics": metrics})
        print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True))

        save_checkpoint(output_dir / "last.pt", model, epoch, metrics, train_dataset.class_to_idx, run_config)
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, epoch, metrics, train_dataset.class_to_idx, run_config)
        write_json(output_dir / "history.json", history)


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
