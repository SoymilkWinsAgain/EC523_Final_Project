from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .data import make_dataset
from .metrics import extract_embeddings, retrieval_metrics, retrieval_metrics_between
from .model import ModelConfig, TimmBackbone, TransformersBackbone, load_checkpoint
from .utils import resolve_device, write_json
from .train import parse_float_triplet


class BackboneOnlyModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, hf_token: str | None = None) -> None:
        super().__init__()
        self.config = config
        backend = config.backbone_backend.lower()
        if backend in {"timm", "hf-timm"}:
            self.backbone = TimmBackbone(config)
        elif backend in {"hf-transformers", "transformers"}:
            self.backbone = TransformersBackbone(config, hf_token=hf_token)
        else:
            raise ValueError(f"Unsupported raw evaluation backend: {config.backbone_backend}")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embedding": F.normalize(self.backbone(images), dim=1)}


def parse_csv_ints(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_spec(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        if str(path).endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["name", "type", "protocol", "recall@1", "recall@5", "recall@10", "mrr", "valid_queries"]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison_bars(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    names = [str(row["name"]) for row in rows]
    recall_keys = [key for key in ["recall@1", "recall@5", "recall@10"] if any(key in row for row in rows)]
    if recall_keys:
        x_positions = list(range(len(names)))
        width = 0.22
        plt.figure(figsize=(10, 5.6))
        for offset, key in enumerate(recall_keys):
            values = [float(row.get(key, 0.0)) for row in rows]
            positions = [x + (offset - (len(recall_keys) - 1) / 2) * width for x in x_positions]
            plt.bar(positions, values, width=width, label=key)
        plt.xticks(x_positions, names, rotation=15, ha="right")
        plt.ylim(0.0, 1.0)
        plt.ylabel("Recall")
        plt.title("Retrieval Recall Comparison")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_recall.png", dpi=180)
        plt.close()

    if any("mrr" in row for row in rows):
        plt.figure(figsize=(8, 5.2))
        values = [float(row.get("mrr", 0.0)) for row in rows]
        plt.bar(names, values, color="#0f766e")
        plt.ylim(0.0, 1.0)
        plt.ylabel("MRR")
        plt.title("Retrieval MRR Comparison")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_mrr.png", dpi=180)
        plt.close()


def default_model_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint:
        return {
            "name": Path(args.checkpoint).stem,
            "type": "checkpoint",
            "checkpoint": args.checkpoint,
            "image_size": args.image_size,
            "image_mean": parse_float_triplet(args.image_mean),
            "image_std": parse_float_triplet(args.image_std),
        }
    return {
        "name": args.model_name,
        "type": "raw",
        "backbone_backend": args.backbone_backend,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "trust_remote_code": args.trust_remote_code,
        "timm_kwargs": json.loads(args.timm_kwargs) if args.timm_kwargs else {},
        "image_size": args.image_size,
        "image_mean": parse_float_triplet(args.image_mean),
        "image_std": parse_float_triplet(args.image_std),
    }


def create_eval_model(model_spec: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_type = model_spec.get("type", "checkpoint")
    if model_type == "checkpoint":
        model, checkpoint = load_checkpoint(model_spec["checkpoint"], map_location=device)
        model.to(device)
        run_config = checkpoint.get("run_config", {})
        metadata = {
            "image_size": int(model_spec.get("image_size", run_config.get("image_size", 224))),
            "image_mean": model_spec.get("image_mean", run_config.get("image_mean", [0.485, 0.456, 0.406])),
            "image_std": model_spec.get("image_std", run_config.get("image_std", [0.229, 0.224, 0.225])),
            "model_config": checkpoint.get("model_config", {}),
        }
        return model, metadata

    if model_type == "raw":
        config = ModelConfig(
            backbone_backend=model_spec.get("backbone_backend", "timm"),
            model_name=model_spec["model_name"],
            pretrained=bool(model_spec.get("pretrained", True)),
            embedding_dim=0,
            projection_hidden_dim=0,
            trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
            timm_kwargs=model_spec.get("timm_kwargs", {}),
        )
        model = BackboneOnlyModel(config, hf_token=model_spec.get("hf_token")).to(device)
        metadata = {
            "image_size": int(model_spec.get("image_size", 224)),
            "image_mean": model_spec.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": model_spec.get("image_std", [0.229, 0.224, 0.225]),
            "model_config": asdict(config),
        }
        return model, metadata

    raise ValueError(f"Unsupported model spec type: {model_type}")


def make_eval_dataset(dataset_spec: dict[str, Any], model_metadata: dict[str, Any], prefix: str = "eval"):
    directory = dataset_spec.get(f"{prefix}_dir") or dataset_spec.get("dir")
    manifest = dataset_spec.get(f"{prefix}_manifest") or dataset_spec.get("manifest")
    return make_dataset(
        directory,
        manifest,
        image_size=int(dataset_spec.get("image_size", model_metadata["image_size"])),
        train=False,
        mean=dataset_spec.get("image_mean", model_metadata["image_mean"]),
        std=dataset_spec.get("image_std", model_metadata["image_std"]),
    )


def evaluate_model(
    model_spec: dict[str, Any],
    dataset_spec: dict[str, Any],
    top_k: list[int],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = create_eval_model(model_spec, device=device)
    name = model_spec.get("name") or model_spec.get("checkpoint") or model_spec.get("model_name")
    model_type = model_spec.get("type", "checkpoint")

    if dataset_spec.get("query_manifest") or dataset_spec.get("query_dir"):
        query_dataset = make_eval_dataset(dataset_spec, metadata, prefix="query")
        gallery_dataset = make_eval_dataset(dataset_spec, metadata, prefix="gallery")
        query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        query_embeddings, query_labels = extract_embeddings(model, query_loader, device)
        gallery_embeddings, gallery_labels = extract_embeddings(model, gallery_loader, device)
        metrics = retrieval_metrics_between(query_embeddings, query_labels, gallery_embeddings, gallery_labels, top_k=top_k)
        protocol = "query-gallery"
    else:
        dataset = make_eval_dataset(dataset_spec, metadata, prefix="eval")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
        embeddings, labels = extract_embeddings(model, loader, device)
        metrics = retrieval_metrics(embeddings, labels, top_k=top_k)
        protocol = "self-retrieval"

    return {
        "name": str(name),
        "type": model_type,
        "protocol": protocol,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and trained models on the same retrieval evaluation protocol.")
    parser.add_argument("--spec", default=None, help="YAML or JSON comparison spec.")
    parser.add_argument("--eval-dir", default=None, help="ImageFolder eval directory for simple one-model runs.")
    parser.add_argument("--eval-manifest", default=None, help="JSONL eval manifest for simple one-model runs.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for simple one-model runs.")
    parser.add_argument("--backbone-backend", default="timm", choices=["timm", "hf-timm", "hf-transformers", "transformers"])
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timm-kwargs", default=None, help="JSON kwargs for raw timm model construction.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-mean", default="0.485,0.456,0.406")
    parser.add_argument("--image-std", default="0.229,0.224,0.225")
    parser.add_argument("--top-k", default="1,5,10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)
    dataset_spec = spec.get("dataset", {})
    if args.eval_dir or args.eval_manifest:
        dataset_spec.update({"dir": args.eval_dir, "manifest": args.eval_manifest})
    if not dataset_spec:
        raise ValueError("Evaluation dataset must be provided through --spec, --eval-dir, or --eval-manifest.")

    model_specs = spec.get("models") or [default_model_from_args(args)]
    top_k = parse_csv_ints(spec.get("top_k", args.top_k))
    batch_size = int(spec.get("batch_size", args.batch_size))
    workers = int(spec.get("workers", args.workers))
    device = resolve_device(spec.get("device", args.device))

    rows = [evaluate_model(model_spec, dataset_spec, top_k, batch_size, workers, device) for model_spec in model_specs]
    print(json.dumps(rows, indent=2, sort_keys=True))

    output_dir = spec.get("output_dir") or args.output_dir
    if output_dir:
        output_path = Path(output_dir)
        write_json(output_path / "comparison_metrics.json", rows)
        write_csv(output_path / "comparison_metrics.csv", rows)
        plot_comparison_bars(output_path, rows)


if __name__ == "__main__":
    main()
