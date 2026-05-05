#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.joint_clip import (
    JointBatchCollator,
    JointImageTextDataset,
    create_joint_clip_model_from_values,
    evaluate_joint_clip_model,
    load_joint_clip_checkpoint,
    move_batch_to_device,
    run_joint_clip_training,
    sanitize_model_name,
)
from who_is_this_anime_girl.losses import symmetric_image_text_contrastive_loss
from who_is_this_anime_girl.utils import resolve_device, set_seed, write_json


MANIFEST_DIR = ROOT / "data" / "manifests" / "danbooru_clip_top500"
TRAIN_MANIFEST = MANIFEST_DIR / "train.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val.jsonl"
REPORT_DIR = ROOT / "artifacts" / "joint_clip_top500"
RUNS_DIR = ROOT / "runs"
SEED = 523
MEMORY_LIMIT_GB = 6.8
TARGET_MAX_MINUTES = 30.0
EFFECTIVE_BATCH = 128


MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "openai_clip_b32",
        "backend": "hf-transformers-clip",
        "model_name": "openai/clip-vit-base-patch32",
        "train_policy": "default",
        "trust_remote_code": False,
    },
    {
        "name": "openai_clip_b16",
        "backend": "hf-transformers-clip",
        "model_name": "openai/clip-vit-base-patch16",
        "train_policy": "default",
        "trust_remote_code": False,
    },
    {
        "name": "laion_openclip_b32",
        "backend": "open-clip",
        "model_name": "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "train_policy": "default",
    },
    {
        "name": "danbooru_clip",
        "backend": "hf-transformers-clip",
        "model_name": "OysterQAQ/DanbooruCLIP",
        "train_policy": "safe_optional",
        "smoke_candidates": [4, 8, 16],
        "smoke_batches": 5,
        "trust_remote_code": False,
    },
    {
        "name": "google_siglip_base",
        "backend": "hf-transformers-clip",
        "model_name": "google/siglip-base-patch16-224",
        "train_policy": "safe_optional",
        "trust_remote_code": False,
    },
    {
        "name": "google_siglip2_base",
        "backend": "hf-transformers-clip",
        "model_name": "google/siglip2-base-patch16-224",
        "train_policy": "safe_optional",
        "trust_remote_code": False,
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "name",
        "stage",
        "model_name",
        "text_to_identity_recall@1",
        "text_to_identity_recall@5",
        "text_to_identity_recall@10",
        "text_to_identity_mrr",
        "text_to_image_recall@1",
        "text_to_image_recall@5",
        "text_to_image_recall@10",
        "text_to_image_mrr",
        "image_to_image_recall@1",
        "image_to_image_recall@5",
        "image_to_image_recall@10",
        "image_to_image_mrr",
    ]
    ordered = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def data_checks() -> dict[str, Any]:
    result: dict[str, Any] = {}
    exact_expected_rows = {"train": 40000}
    for name in ["train", "val"]:
        path = MANIFEST_DIR / f"{name}.jsonl"
        rows = read_jsonl(path)
        identities = {row["identity"] for row in rows}
        missing = [row["path"] for row in rows if not Path(row["path"]).exists()]
        expected_rows = exact_expected_rows.get(name)
        if expected_rows is not None and len(rows) != expected_rows:
            raise AssertionError(f"{path} has {len(rows)} rows; expected {expected_rows}")
        if missing:
            raise FileNotFoundError(f"{path} has {len(missing)} missing paths; first missing path: {missing[0]}")
        result[name] = {
            "rows": len(rows),
            "identities": len(identities),
            "missing_paths": 0,
            "path": str(path.resolve()),
        }
    write_json(REPORT_DIR / "data_checks.json", result)
    return result


def environment_checks() -> dict[str, Any]:
    import open_clip
    import transformers

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    result = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "open_clip": getattr(open_clip, "__version__", "unknown"),
        "gpu": gpu.stdout.strip(),
    }
    write_json(REPORT_DIR / "environment_checks.json", result)
    return result


def spec_to_config(spec: dict[str, Any], train_mode: str) -> dict[str, Any]:
    return {
        "backend": spec["backend"],
        "model_name": spec["model_name"],
        "open_clip_pretrained": spec.get("open_clip_pretrained"),
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        "train_mode": train_mode,
        "lora_r": int(spec.get("lora_r", 8)),
        "lora_alpha": int(spec.get("lora_alpha", 16)),
        "lora_dropout": float(spec.get("lora_dropout", 0.05)),
        "lora_target_modules": spec.get("lora_target_modules"),
        "include_character_text": False,
    }


def create_model(spec: dict[str, Any], train_mode: str, device: torch.device):
    cfg = spec_to_config(spec, train_mode)
    return create_joint_clip_model_from_values(
        **cfg,
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    ).to(device)


def parameter_and_shape_check(spec: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device("auto")
    dataset = JointImageTextDataset(VAL_MANIFEST, include_character_text=False)
    rows = [dataset[index] for index in range(2)]
    model = create_model(spec, train_mode="lora_vision", device=device)
    batch = model.create_collator()(rows)
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        outputs = model(batch)
    image_norm = outputs["image_embedding"].norm(dim=1).detach().cpu().numpy()
    text_norm = outputs["text_embedding"].norm(dim=1).detach().cpu().numpy()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "name": spec["name"],
        "model_name": spec["model_name"],
        "backend": spec["backend"],
        "image_embedding_shape": list(outputs["image_embedding"].shape),
        "text_embedding_shape": list(outputs["text_embedding"].shape),
        "image_norm_min": float(image_norm.min()),
        "image_norm_max": float(image_norm.max()),
        "text_norm_min": float(text_norm.min()),
        "text_norm_max": float(text_norm.max()),
        "logit_scale": float(outputs["logit_scale"].detach().exp().cpu()),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / max(1, total)),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_row(name: str, stage: str, spec: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"name": name, "stage": stage, "model_name": spec.get("model_name", ""), "backend": spec.get("backend", "")}
    row.update({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
    return row


def frozen_eval(spec: dict[str, Any]) -> dict[str, Any]:
    output_path = REPORT_DIR / "frozen_metrics" / f"{spec['name']}.json"
    if output_path.exists():
        result = read_json(output_path)
        result["reused"] = True
        return result
    device = resolve_device("auto")
    model = create_model(spec, train_mode="frozen", device=device)
    started = time.time()
    metrics = evaluate_joint_clip_model(
        model,
        manifest_path=VAL_MANIFEST,
        batch_size=64,
        workers=0,
        device=device,
        include_character_text=False,
    )
    result = {
        "name": spec["name"],
        "stage": "frozen",
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
    }
    write_json(output_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def smoke_train(spec: dict[str, Any], batch_candidates: list[int] = [16, 32, 64]) -> dict[str, Any]:
    output_path = REPORT_DIR / "smoke" / f"{spec['name']}.json"
    if output_path.exists():
        result = read_json(output_path)
        result["reused"] = True
        return result
    batch_candidates = [int(value) for value in spec.get("smoke_candidates", batch_candidates)]
    target_batches = int(spec.get("smoke_batches", 20))
    max_candidate_sec = float(spec.get("smoke_max_candidate_sec", 180.0))
    set_seed(SEED)
    device = resolve_device("auto")
    dataset = JointImageTextDataset(TRAIN_MANIFEST, include_character_text=False)
    model = create_model(spec, train_mode="lora_vision", device=device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rows: list[dict[str, Any]] = []
    for batch_size in batch_candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
                collate_fn=model.create_collator(),
            )
            started = time.time()
            completed = 0
            stop_reason = ""
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch in loader:
                if completed >= target_batches:
                    break
                batch = move_batch_to_device(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    outputs = model(batch)
                    loss = symmetric_image_text_contrastive_loss(outputs["image_embedding"], outputs["text_embedding"], outputs["logit_scale"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                completed += 1
                if device.type == "cuda":
                    current_peak = torch.cuda.max_memory_allocated(device) / 1024**3
                    if current_peak >= MEMORY_LIMIT_GB:
                        stop_reason = "memory_gate_exceeded"
                        break
                if time.time() - started > max_candidate_sec:
                    stop_reason = "candidate_time_gate_exceeded"
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - started
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
            rows.append(
                {
                    "batch_size": batch_size,
                    "batches": completed,
                    "sec_per_batch": elapsed / max(1, completed),
                    "peak_allocated_gb": peak_allocated,
                    "peak_reserved_gb": peak_reserved,
                    "ok": completed == target_batches and peak_allocated < MEMORY_LIMIT_GB and not stop_reason,
                    "stop_reason": stop_reason,
                }
            )
            if stop_reason in {"memory_gate_exceeded", "candidate_time_gate_exceeded"}:
                break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rows.append({"batch_size": batch_size, "ok": False, "error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    safe = [row for row in rows if row.get("ok")]
    selected = max(safe, key=lambda row: int(row["batch_size"])) if safe else None
    result = {"name": spec["name"], "candidates": rows, "selected": selected}
    write_json(output_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def choose_train_plan(spec: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    selected = smoke.get("selected")
    if not selected:
        return {"train": False, "reason": "no_safe_smoke_batch"}
    batch_size = int(selected["batch_size"])
    grad_accum_steps = max(1, math.ceil(EFFECTIVE_BATCH / batch_size))
    epochs = 4
    batches_per_epoch = 250
    projected_min = float(selected["sec_per_batch"]) * epochs * batches_per_epoch / 60
    if projected_min > TARGET_MAX_MINUTES:
        batches_per_epoch = 150
        projected_min = float(selected["sec_per_batch"]) * epochs * batches_per_epoch / 60
    if projected_min > TARGET_MAX_MINUTES:
        epochs = 2
        projected_min = float(selected["sec_per_batch"]) * epochs * batches_per_epoch / 60
    optional = spec.get("train_policy") == "safe_optional"
    if optional and (batch_size < 32 or projected_min > TARGET_MAX_MINUTES):
        return {
            "train": False,
            "reason": "safe_optional_failed_batch32_or_time_gate",
            "batch_size": batch_size,
            "projected_min": projected_min,
        }
    return {
        "train": True,
        "reason": "safe",
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "projected_min": projected_min,
    }


def training_args(spec: dict[str, Any], output_dir: Path, plan: dict[str, Any], mini: bool = False) -> Namespace:
    return Namespace(
        backend=spec["backend"],
        model_name=spec["model_name"],
        open_clip_pretrained=spec.get("open_clip_pretrained"),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        train_manifest=str(TRAIN_MANIFEST),
        val_manifest=str(VAL_MANIFEST),
        output_dir=str(output_dir),
        train_mode="lora_vision",
        lora_r=int(spec.get("lora_r", 8)),
        lora_alpha=int(spec.get("lora_alpha", 16)),
        lora_dropout=float(spec.get("lora_dropout", 0.05)),
        lora_target_modules=spec.get("lora_target_modules"),
        include_character_text=False,
        batch_size=int(plan["batch_size"]),
        eval_batch_size=64,
        grad_accum_steps=int(plan["grad_accum_steps"]),
        epochs=1 if mini else int(plan["epochs"]),
        batches_per_epoch=5 if mini else int(plan["batches_per_epoch"]),
        lr=5e-5,
        weight_decay=1e-4,
        min_lr=1e-6,
        warmup_epochs=1,
        workers=0,
        device="auto",
        amp=True,
        seed=SEED,
    )


def run_training_for_spec(spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if not plan.get("train"):
        return {"name": spec["name"], "trained": False, "reason": plan.get("reason")}
    output_dir = RUNS_DIR / f"joint_clip_{spec['name']}_top500"
    if (output_dir / "best.pt").exists() and (output_dir / "training_summary.json").exists():
        result = read_json(output_dir / "training_summary.json")
        result.update({"name": spec["name"], "trained": True, "reused": True, "output_dir": str(output_dir.resolve())})
        return result
    mini_dir = REPORT_DIR / "mini_train" / spec["name"]
    if not (mini_dir / "best.pt").exists():
        run_joint_clip_training(training_args(spec, mini_dir, plan, mini=True))
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    run_joint_clip_training(training_args(spec, output_dir, plan, mini=False))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result = read_json(output_dir / "training_summary.json")
    result.update(
        {
            "name": spec["name"],
            "trained": True,
            "output_dir": str(output_dir.resolve()),
            "elapsed_wall_sec": time.time() - started,
        }
    )
    write_json(REPORT_DIR / "training_runs" / f"{spec['name']}.json", result)
    return result


def evaluate_trained_run(spec: dict[str, Any], run: dict[str, Any]) -> dict[str, Any] | None:
    if not run.get("trained"):
        return None
    output_path = REPORT_DIR / "trained_metrics" / f"{spec['name']}.json"
    if output_path.exists():
        result = read_json(output_path)
        result["reused"] = True
        return result
    device = resolve_device("auto")
    model, _ = load_joint_clip_checkpoint(Path(run["output_dir"]) / "best.pt", map_location=device)
    started = time.time()
    metrics = evaluate_joint_clip_model(
        model,
        manifest_path=VAL_MANIFEST,
        batch_size=64,
        workers=0,
        device=device,
        include_character_text=False,
    )
    result = {
        "name": spec["name"],
        "stage": "fine_tuned",
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
    }
    write_json(output_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def add_existing_baselines(rows: list[dict[str, Any]]) -> None:
    devise_metrics = REPORT_DIR.parent / "devise_transform_top500" / "text_to_image_metrics.json"
    image_space = REPORT_DIR.parent / "devise_transform_top500" / "image_space_metrics.json"
    if devise_metrics.exists():
        metrics = read_json(devise_metrics)
        row = {
            "name": "devise_transform",
            "stage": "existing",
            "model_name": "hf_finetune_v2 + Qwen3 transform",
            "backend": "devise",
            "text_to_image_recall@1": metrics.get("image_recall@1", 0.0),
            "text_to_image_recall@5": metrics.get("image_recall@5", 0.0),
            "text_to_image_recall@10": metrics.get("image_recall@10", 0.0),
            "text_to_image_mrr": metrics.get("image_mrr", 0.0),
            "text_to_identity_recall@1": metrics.get("identity_recall@1", 0.0),
            "text_to_identity_recall@5": metrics.get("identity_recall@5", 0.0),
            "text_to_identity_recall@10": metrics.get("identity_recall@10", 0.0),
            "text_to_identity_mrr": metrics.get("identity_mrr", 0.0),
        }
        if image_space.exists():
            image_metrics = read_json(image_space)
            row.update(
                {
                    "image_to_image_recall@1": image_metrics.get("recall@1", 0.0),
                    "image_to_image_recall@5": image_metrics.get("recall@5", 0.0),
                    "image_to_image_recall@10": image_metrics.get("recall@10", 0.0),
                    "image_to_image_mrr": image_metrics.get("mrr", 0.0),
                }
            )
        rows.append(row)


def plot_metric_group(rows: list[dict[str, Any]], filename: str, keys: list[str], title: str, ylabel: str) -> None:
    if not rows:
        return
    labels = [f"{row['name']}:{row['stage']}" for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(keys))
    plt.figure(figsize=(max(10, len(labels) * 0.7), 5.4))
    for offset, key in enumerate(keys):
        values = [float(row.get(key, 0.0)) for row in rows]
        plt.bar(x + (offset - (len(keys) - 1) / 2) * width, values, width=width, label=key)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=180)
    plt.close()


def plot_figures(rows: list[dict[str, Any]]) -> None:
    plot_metric_group(
        rows,
        "comparison_text_identity_recall.png",
        ["text_to_identity_recall@1", "text_to_identity_recall@5", "text_to_identity_recall@10"],
        "Text-to-Identity Retrieval",
        "Recall",
    )
    plot_metric_group(
        rows,
        "comparison_text_image_recall.png",
        ["text_to_image_recall@1", "text_to_image_recall@5", "text_to_image_recall@10"],
        "Text-to-Exact-Image Retrieval",
        "Recall",
    )
    plot_metric_group(
        rows,
        "comparison_image_recall.png",
        ["image_to_image_recall@1", "image_to_image_recall@5", "image_to_image_recall@10"],
        "Image-to-Image Retrieval",
        "Recall",
    )
    plot_metric_group(
        rows,
        "comparison_mrr.png",
        ["text_to_identity_mrr", "text_to_image_mrr", "image_to_image_mrr"],
        "MRR Comparison",
        "MRR",
    )


def write_report(
    env: dict[str, Any],
    checks: dict[str, Any],
    param_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    train_plans: list[dict[str, Any]],
    train_runs: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Joint CLIP Top-500 Experiment",
        "",
        "## Environment",
        f"- GPU: `{env.get('gpu', '')}`",
        f"- torch: `{env.get('torch')}`, transformers: `{env.get('transformers')}`, open_clip: `{env.get('open_clip')}`",
        f"- Data rows: train `{checks['train']['rows']}`, val `{checks['val']['rows']}`.",
        "",
        "## Model Parameter Checks",
        "| model | backend | total params | trainable params | trainable % | embedding dim |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in param_rows:
        if "error" in row:
            lines.append(
                f"| {row['name']} | {row['backend']} | error: `{row['error']}` | n/a | n/a | n/a |"
            )
            continue
        dim = row.get("image_embedding_shape", ["?", "?"])[-1]
        lines.append(
            f"| {row['name']} | {row['backend']} | {row['total_parameters']} | {row['trainable_parameters']} | "
            f"{100 * row['trainable_fraction']:.3f}% | {dim} |"
        )
    lines.extend(
        [
            "",
            "## Smoke Gate",
            "| model | selected batch | sec/batch | peak allocated GB | train decision |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    plan_by_name = {plan["name"]: plan for plan in train_plans}
    for smoke in smoke_rows:
        selected = smoke.get("selected") or {}
        plan = plan_by_name.get(smoke["name"], {})
        lines.append(
            f"| {smoke['name']} | {selected.get('batch_size', 'n/a')} | {float(selected.get('sec_per_batch', 0.0)):.3f} | "
            f"{float(selected.get('peak_allocated_gb', 0.0)):.2f} | {plan.get('reason', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Training Runs",
            "| model | trained | epochs | batch | grad accum | runtime min | output |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run in train_runs:
        plan = plan_by_name.get(run["name"], {})
        runtime = float(run.get("elapsed_sec", run.get("elapsed_wall_sec", 0.0))) / 60
        lines.append(
            f"| {run['name']} | {run.get('trained', False)} | {plan.get('epochs', 'n/a')} | {plan.get('batch_size', 'n/a')} | "
            f"{plan.get('grad_accum_steps', 'n/a')} | {runtime:.1f} | `{run.get('output_dir', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Comparison Metrics",
            "| model | stage | text-id R@1 | text-id R@5 | text-img R@1 | image R@1 | image MRR |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['name']} | {row['stage']} | {float(row.get('text_to_identity_recall@1', 0.0)):.4f} | "
            f"{float(row.get('text_to_identity_recall@5', 0.0)):.4f} | {float(row.get('text_to_image_recall@1', 0.0)):.4f} | "
            f"{float(row.get('image_to_image_recall@1', 0.0)):.4f} | {float(row.get('image_to_image_mrr', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Slide Figures",
            "- `comparison_text_identity_recall.png`",
            "- `comparison_text_image_recall.png`",
            "- `comparison_image_recall.png`",
            "- `comparison_mrr.png`",
        ]
    )
    (REPORT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def selected_specs(names: list[str] | None) -> list[dict[str, Any]]:
    if not names:
        return MODEL_SPECS
    wanted = set(names)
    return [spec for spec in MODEL_SPECS if spec["name"] in wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen/fine-tuned CLIP-style joint embedding experiments on Danbooru top500.")
    parser.add_argument("--models", default=None, help="Comma-separated model names from MODEL_SPECS. Default: all.")
    parser.add_argument("--skip-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-frozen-eval", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["frozen_metrics", "trained_metrics", "smoke", "training_runs", "mini_train"]:
        (REPORT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    specs = selected_specs([item.strip() for item in args.models.split(",") if item.strip()] if args.models else None)
    checks = data_checks()
    env = environment_checks()

    param_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    train_plans: list[dict[str, Any]] = []
    train_runs: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for spec in specs:
        try:
            param = parameter_and_shape_check(spec)
            param_rows.append(param)
        except Exception as exc:
            param_rows.append({"name": spec["name"], "backend": spec["backend"], "model_name": spec["model_name"], "error": str(exc)})
            continue
        if not args.skip_frozen_eval:
            frozen = frozen_eval(spec)
            comparison_rows.append(metric_row(spec["name"], "frozen", spec, frozen["metrics"]))
        else:
            cached_frozen = REPORT_DIR / "frozen_metrics" / f"{spec['name']}.json"
            if cached_frozen.exists():
                frozen = read_json(cached_frozen)
                comparison_rows.append(metric_row(spec["name"], "frozen", spec, frozen["metrics"]))
        smoke = smoke_train(spec)
        smoke_rows.append(smoke)
        plan = {"name": spec["name"], **choose_train_plan(spec, smoke)}
        train_plans.append(plan)
        if not args.skip_training:
            run = run_training_for_spec(spec, plan)
            train_runs.append(run)
            trained = evaluate_trained_run(spec, run)
            if trained is not None:
                comparison_rows.append(metric_row(spec["name"], "fine_tuned", spec, trained["metrics"]))

    add_existing_baselines(comparison_rows)
    write_json(REPORT_DIR / "model_parameter_checks.json", param_rows)
    write_json(REPORT_DIR / "smoke_summary.json", smoke_rows)
    write_json(REPORT_DIR / "training_plans.json", train_plans)
    write_json(REPORT_DIR / "training_runs.json", train_runs)
    write_json(REPORT_DIR / "comparison_metrics.json", comparison_rows)
    write_csv(REPORT_DIR / "comparison_metrics.csv", comparison_rows)
    plot_figures(comparison_rows)
    write_report(env, checks, param_rows, smoke_rows, train_plans, train_runs, comparison_rows)
    print(json.dumps({"report": str((REPORT_DIR / "experiment_report.md").resolve()), "rows": len(comparison_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
