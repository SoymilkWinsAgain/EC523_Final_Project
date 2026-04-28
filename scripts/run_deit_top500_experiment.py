#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.data import PKBatchSampler, make_dataset
from who_is_this_anime_girl.evaluate import evaluate_model, plot_comparison_bars, write_csv
from who_is_this_anime_girl.losses import supervised_contrastive_loss
from who_is_this_anime_girl.model import create_model
from who_is_this_anime_girl.train import DEFAULTS, run_training
from who_is_this_anime_girl.utils import resolve_device, write_json


MODEL_NAME = "facebook/deit-small-patch16-224"
MANIFEST_DIR = ROOT / "data" / "manifests" / "danbooru2018_top500"
SUMMARY_PATH = ROOT / "data" / "danbooru2018_acr_summary.json"
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
REPORT_DIR = ROOT / "artifacts" / "deit_top500_experiment"
SCRATCH_RUN = ROOT / "runs" / "deit_small_scratch_top500"
FINETUNE_RUN = ROOT / "runs" / "deit_small_hf_finetune_top500"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def generate_manifests(seed: int = 523, top_identities: int = 500) -> dict[str, Any]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    identities = list(summary["identity_mapping"].items())
    identities.sort(key=lambda item: (-int(item[1]["available"]), str(item[0])))
    selected = identities[:top_identities]
    rng = random.Random(seed)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    identity_summary: dict[str, dict[str, Any]] = {}

    for identity, metadata in selected:
        train_paths = sorted((TRAIN_DIR / identity).glob("*.jpg"))
        val_paths = sorted((VAL_DIR / identity).glob("*.jpg"))
        if len(train_paths) < 80 or len(val_paths) < 20:
            raise ValueError(f"Identity {identity} does not have the expected 80/20 split.")

        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        train_paths = train_paths[:80]
        val_paths = val_paths[:20]
        gallery_paths = val_paths[:5]
        query_paths = val_paths[5:20]

        base = {
            "identity": identity,
            "original_identity": metadata["original_name"],
            "tag_id": int(metadata["tag_id"]),
        }
        for split, paths, target in [
            ("train", train_paths, train_rows),
            ("val", val_paths, val_rows),
            ("gallery", gallery_paths, gallery_rows),
            ("query", query_paths, query_rows),
        ]:
            for path in paths:
                target.append({**base, "path": str(path.resolve()), "split": split})

        identity_summary[identity] = {
            "available": int(metadata["available"]),
            "tag_id": int(metadata["tag_id"]),
            "original_identity": metadata["original_name"],
            "train": len(train_paths),
            "val": len(val_paths),
            "gallery": len(gallery_paths),
            "query": len(query_paths),
        }

    for name, rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("gallery", gallery_rows),
        ("query", query_rows),
    ]:
        write_jsonl(MANIFEST_DIR / f"{name}.jsonl", rows)

    manifest_summary = {
        "seed": seed,
        "top_identities": top_identities,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "gallery_rows": len(gallery_rows),
        "query_rows": len(query_rows),
        "identities": identity_summary,
    }
    write_json(MANIFEST_DIR / "summary.json", manifest_summary)
    return manifest_summary


def assert_manifest(path: Path, expected_rows: int, expected_identities: int) -> None:
    rows = 0
    identities: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            identities.add(str(row["identity"]))
            if not Path(row["path"]).exists():
                raise FileNotFoundError(row["path"])
    if rows != expected_rows:
        raise AssertionError(f"{path} has {rows} rows, expected {expected_rows}.")
    if len(identities) != expected_identities:
        raise AssertionError(f"{path} has {len(identities)} identities, expected {expected_identities}.")


def smoke_model(pretrained: bool, train_manifest: Path, seed: int) -> dict[str, Any]:
    device = resolve_device("auto")
    dataset = make_dataset(
        None,
        train_manifest,
        image_size=224,
        train=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    sampler = PKBatchSampler(dataset.targets, identities_per_batch=8, samples_per_identity=4, batches_per_epoch=20, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=device.type == "cuda")
    model = create_model(
        backbone_backend="hf-transformers",
        model_name=MODEL_NAME,
        pretrained=pretrained,
        embedding_dim=256,
        projection_hidden_dim=512,
        num_classes=len(dataset.classes),
        trust_remote_code=False,
        finetune_mode="full",
    ).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    start = time.perf_counter()
    batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = supervised_contrastive_loss(outputs["embedding"], labels, temperature=0.07)
            loss = loss + 0.1 * F.cross_entropy(outputs["logits"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    del model, optimizer, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "pretrained": pretrained,
        "batches": batches,
        "elapsed_sec": elapsed,
        "sec_per_batch": elapsed / max(1, batches),
        "peak_gb": peak_gb,
    }


def training_args(output_dir: Path, pretrained: bool, lr: float, epochs: int, batches_per_epoch: int) -> Namespace:
    values = dict(DEFAULTS)
    values.update(
        {
            "train_dir": None,
            "train_manifest": str(MANIFEST_DIR / "train.jsonl"),
            "val_dir": None,
            "val_manifest": str(MANIFEST_DIR / "val.jsonl"),
            "output_dir": str(output_dir),
            "backbone_backend": "hf-transformers",
            "model_name": MODEL_NAME,
            "pretrained": pretrained,
            "trust_remote_code": False,
            "hf_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            "timm_kwargs": {},
            "finetune_mode": "full",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": None,
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "embedding_dim": 256,
            "projection_hidden_dim": 512,
            "epochs": epochs,
            "identities_per_batch": 8,
            "samples_per_identity": 4,
            "batches_per_epoch": batches_per_epoch,
            "lr": lr,
            "weight_decay": 1e-4,
            "scheduler": "cosine-warmup",
            "min_lr": lr / 100,
            "warmup_epochs": 1,
            "step_size": 5,
            "gamma": 0.1,
            "milestones": None,
            "temperature": 0.07,
            "classification_weight": 0.1,
            "freeze_backbone_epochs": 0,
            "workers": 4,
            "device": "auto",
            "amp": True,
            "seed": 523,
        }
    )
    return Namespace(**values)


def run_training_job(name: str, output_dir: Path, pretrained: bool, lr: float, epochs: int, batches_per_epoch: int) -> dict[str, Any]:
    start = time.perf_counter()
    run_training(training_args(output_dir, pretrained=pretrained, lr=lr, epochs=epochs, batches_per_epoch=batches_per_epoch))
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "output_dir": str(output_dir),
        "pretrained": pretrained,
        "lr": lr,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
    }


def evaluate_experiment() -> list[dict[str, Any]]:
    device = resolve_device("auto")
    dataset_spec = {
        "query_manifest": str(MANIFEST_DIR / "query.jsonl"),
        "gallery_manifest": str(MANIFEST_DIR / "gallery.jsonl"),
    }
    models = [
        {
            "name": "deit_small_raw_hf",
            "type": "raw",
            "backbone_backend": "hf-transformers",
            "model_name": MODEL_NAME,
            "pretrained": True,
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        {"name": "deit_small_scratch_top500", "type": "checkpoint", "checkpoint": str(SCRATCH_RUN / "best.pt")},
        {"name": "deit_small_hf_finetune_top500", "type": "checkpoint", "checkpoint": str(FINETUNE_RUN / "best.pt")},
    ]
    rows = [evaluate_model(spec, dataset_spec, top_k=[1, 5, 10], batch_size=64, workers=4, device=device) for spec in models]
    write_json(REPORT_DIR / "comparison_metrics.json", rows)
    write_csv(REPORT_DIR / "comparison_metrics.csv", rows)
    plot_comparison_bars(REPORT_DIR, rows)
    return rows


def command_text(output_dir: Path, pretrained: bool, lr: float, epochs: int, batches_per_epoch: int) -> str:
    pretrained_flag = "--pretrained" if pretrained else "--no-pretrained"
    return (
        "conda run -n jigsaw python scripts/train.py "
        f"--train-manifest {MANIFEST_DIR / 'train.jsonl'} "
        f"--val-manifest {MANIFEST_DIR / 'val.jsonl'} "
        f"--output-dir {output_dir} --backbone-backend hf-transformers "
        f"--model-name {MODEL_NAME} {pretrained_flag} --finetune-mode full "
        "--image-mean 0.5,0.5,0.5 --image-std 0.5,0.5,0.5 "
        f"--epochs {epochs} --batches-per-epoch {batches_per_epoch} "
        "--identities-per-batch 8 --samples-per-identity 4 "
        f"--lr {lr} --scheduler cosine-warmup --warmup-epochs 1 --amp"
    )


def write_report(manifest_summary: dict[str, Any], smoke: list[dict[str, Any]], runs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    lines = [
        "# DeiT-Small Top-500 Experiment",
        "",
        "## Hardware",
        "",
        f"- GPU: {gpu.stdout.strip() or 'unknown'}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "## Data",
        "",
        f"- Identities: {manifest_summary['top_identities']}",
        f"- Train rows: {manifest_summary['train_rows']}",
        f"- Validation rows: {manifest_summary['val_rows']}",
        f"- Gallery/query rows: {manifest_summary['gallery_rows']} / {manifest_summary['query_rows']}",
        "",
        "## Smoke Test",
        "",
        "| model | batches | sec/batch | peak GB |",
        "|---|---:|---:|---:|",
    ]
    for item in smoke:
        label = "hf_pretrained" if item["pretrained"] else "scratch"
        lines.append(f"| {label} | {item['batches']} | {item['sec_per_batch']:.3f} | {item['peak_gb']:.2f} |")
    lines.extend(["", "## Training Runs", ""])
    for run in runs:
        lines.extend(
            [
                f"### {run['name']}",
                "",
                f"- Output: `{run['output_dir']}`",
                f"- Runtime: {run['elapsed_min']:.1f} min",
                f"- Epochs / batches per epoch: {run['epochs']} / {run['batches_per_epoch']}",
                f"- Command: `{command_text(Path(run['output_dir']), run['pretrained'], run['lr'], run['epochs'], run['batches_per_epoch'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Retrieval Comparison",
            "",
            "| model | recall@1 | recall@5 | recall@10 | mrr | valid queries |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['name']} | {row.get('recall@1', 0):.4f} | {row.get('recall@5', 0):.4f} | "
            f"{row.get('recall@10', 0):.4f} | {row.get('mrr', 0):.4f} | {row.get('valid_queries', 0):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Slide Figures",
            "",
            "- `comparison_recall.png`",
            "- `comparison_mrr.png`",
            "- each run: `curves/loss.png`, `curves/retrieval.png`, `curves/accuracy.png`, `curves/lr.png`",
            "",
        ]
    )
    (REPORT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_summary = generate_manifests()
    assert_manifest(MANIFEST_DIR / "train.jsonl", 40000, 500)
    assert_manifest(MANIFEST_DIR / "val.jsonl", 10000, 500)
    assert_manifest(MANIFEST_DIR / "gallery.jsonl", 2500, 500)
    assert_manifest(MANIFEST_DIR / "query.jsonl", 7500, 500)

    smoke = [
        smoke_model(pretrained=False, train_manifest=MANIFEST_DIR / "train.jsonl", seed=523),
        smoke_model(pretrained=True, train_manifest=MANIFEST_DIR / "train.jsonl", seed=524),
    ]
    write_json(REPORT_DIR / "smoke_test.json", smoke)

    peak = max(item["peak_gb"] for item in smoke)
    sec_per_batch = max(item["sec_per_batch"] for item in smoke)
    epochs = 8
    batches_per_epoch = 400
    estimated_train_min = sec_per_batch * epochs * batches_per_epoch / 60
    if peak > 6.8 or estimated_train_min > 45:
        batches_per_epoch = 300
        estimated_train_min = sec_per_batch * epochs * batches_per_epoch / 60
    if peak > 6.8 or estimated_train_min > 45:
        epochs = 6

    runs = [
        run_training_job("deit_small_scratch_top500", SCRATCH_RUN, pretrained=False, lr=3e-4, epochs=epochs, batches_per_epoch=batches_per_epoch),
        run_training_job("deit_small_hf_finetune_top500", FINETUNE_RUN, pretrained=True, lr=1e-4, epochs=epochs, batches_per_epoch=batches_per_epoch),
    ]
    write_json(REPORT_DIR / "training_runs.json", runs)
    metrics = evaluate_experiment()
    write_report(manifest_summary, smoke, runs, metrics)


if __name__ == "__main__":
    main()
