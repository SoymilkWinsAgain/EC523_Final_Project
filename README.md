# Who Is This Anime Girl

This repository implements an anime character retrieval system. It supports image-only character embeddings, a DeVISE-style text-to-image extension, and CLIP/SigLIP/OpenCLIP-style joint image-text experiments. It trains embedding models, builds FAISS gallery indexes, and serves a small local web UI for enrolling known characters and querying unknown images.

## Technology Stack

- Python environment: dedicated conda environment `wiag`
- Modeling: PyTorch, torchvision, timm
- Hugging Face loading: transformers, huggingface_hub, peft
- Joint image-text models: Hugging Face CLIP/SigLIP wrappers plus `open_clip_torch`
- Retrieval: FAISS with cosine similarity through normalized embeddings
- Web UI: Python standard library HTTP server plus static HTML/CSS/JavaScript
- Dataset format: ImageFolder directories or JSONL manifests aligned with Danbooru-style image metadata

## Environment Setup

Use the dedicated project environment `wiag` instead of reusing an older shared environment:

```bash
conda create -y -n wiag python=3.11 pip
conda run -n wiag python -m pip install --upgrade pip
conda run -n wiag python -m pip install -r requirements.txt
conda run -n wiag python -m pip install -e .
```

`requirements.txt` covers the core image retrieval pipeline, Hugging Face fine-tuning, FAISS indexing, the DeVISE text-to-image extension, CLIP/SigLIP/OpenCLIP joint embedding experiments, and the legacy Danbooru helper scripts under `ExtraScripts/`.
The PyTorch entries are pinned to the CUDA 12.8 wheel index to match the environment used for the current experiments. On a CPU-only machine, replace the PyTorch/torchvision install with the CPU wheel instructions from PyTorch before installing the rest of the file.

Quick smoke check:

```bash
conda run -n wiag python -c "import torch, torchvision, timm, transformers, faiss, sentence_transformers; print(torch.__version__)"
conda run -n wiag python -c "import open_clip; print(open_clip.__version__)"
conda run -n wiag python scripts/train.py --help
conda run -n wiag python scripts/build_index.py --help
conda run -n wiag python scripts/query_text.py --help
conda run -n wiag python scripts/train_joint_clip.py --help
```

## Project Layout

```text
configs/                         Default training configuration
scripts/                         Command-line entry points for SCC and local runs
src/who_is_this_anime_girl/       Reusable training, indexing, inference, and web code
data/                            Local datasets and gallery images, ignored by Git
artifacts/                       Model and index outputs, ignored by Git
runs/                            Training outputs, ignored by Git
project_report/                  Project report PDF
```

## Dataset Layout

Small local experiments can use identity names as subdirectories:

```text
data/train/
  hatsune_miku/
    image_001.jpg
    image_002.jpg
  rem/
    image_001.jpg

data/val/
  hatsune_miku/
    image_101.jpg
  rem/
    image_101.jpg
```

For retrieval, `data/gallery/` uses the same layout.

Danbooru2021-scale experiments should use manifests instead of copying images into class folders. The expected manifest row format is:

```json
{"path": "/datasets/danbooru2021/512px/0001/1001.jpg", "identity": "character_name", "danbooru_id": 1001, "rating": "s"}
```

Danbooru2021 stores images in modulo buckets. For image ID `1001`, the bucket is `0001` because `1001 % 1000 == 1`, so the 512px path is `512px/0001/1001.jpg`.

For CLIP/SigLIP-style experiments, the repository uses a richer image-text JSONL format. Each row contains an image path, one identity, the Danbooru post id, structured tags, and a text field:

```json
{
  "path": "/abs/path/data/danbooru_clip_top500/images/koizumi_hanayo/1905074.jpg",
  "identity": "koizumi_hanayo",
  "post_id": 1905074,
  "tags": {
    "character": ["koizumi_hanayo"],
    "copyright": ["love_live!"],
    "artist": ["ranshin"],
    "general": ["1girl", "brown_hair", "purple_eyes"]
  },
  "text": "character: koizumi_hanayo. copyright: love_live!. artist: ranshin. tags: 1girl, brown_hair, purple_eyes."
}
```

Joint CLIP training removes character names by default and trains on source, artist, and visual tags. Name lookup is handled separately by deterministic keyword matching in the text-query system.

## Prepare Danbooru2021 Manifests

Do not download the full dataset into this repository. Point the script at an existing SCC or external dataset location:

```bash
conda run -n wiag python scripts/prepare_danbooru.py \
  --metadata /path/to/metadata.json.tar.xz \
  --image-root /path/to/danbooru2021 \
  --image-subdir 512px \
  --output-dir data/manifests/danbooru \
  --ratings s \
  --single-character-only \
  --min-images-per-identity 20 \
  --max-images-per-identity 500 \
  --max-identities 1000 \
  --require-image-exists
```

The script writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `all.jsonl`, and `summary.json`. Use `--limit-records` for fast dry runs on a small metadata slice.

## Prepare a Danbooru CLIP/SigLIP Subset

Use `scripts/prepare_danbooru_clip_subset.py` when you have a local `data/archive.zip` containing `danbooru-images/` and `danbooru-metadata/`. The script scans the zip in place and extracts only the selected images, so it does not unpack the full archive.

```bash
conda run -n wiag python scripts/prepare_danbooru_clip_subset.py \
  --archive data/archive.zip \
  --output-dir data/danbooru_clip_top500 \
  --manifest-dir data/manifests/danbooru_clip_top500 \
  --ratings s \
  --single-character-only \
  --min-general-tags 5 \
  --max-identities 500 \
  --target-total 50000 \
  --train-count 40000 \
  --val-count 10000
```

The current local CLIP subset is:

```text
data/danbooru_clip_top500/images/<identity>/*.jpg
data/manifests/danbooru_clip_top500/train.jsonl
data/manifests/danbooru_clip_top500/val.jsonl
data/manifests/danbooru_clip_top500/all.jsonl
```

Use `--dry-run` to write manifests and summaries without extracting images.

## Train an Embedding Model

Run from the repository root:

```bash
conda run -n wiag python scripts/train.py \
  --config configs/default.yaml \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir runs/vit_baseline
```

Useful SCC overrides:

```bash
conda run -n wiag python scripts/train.py \
  --train-dir /path/to/train \
  --val-dir /path/to/val \
  --output-dir /path/to/run \
  --model-name vit_base_patch16_224 \
  --epochs 20 \
  --identities-per-batch 16 \
  --samples-per-identity 4 \
  --workers 8 \
  --amp
```

The training script saves `last.pt`, `best.pt`, `config.json`, and `class_to_idx.json` in the output directory.
It also writes report-ready artifacts:

- `history.json`
- `metrics.csv`
- `curves/loss.png`
- `curves/accuracy.png`
- `curves/retrieval.png`
- `curves/lr.png`

Scheduler options include `none`, `cosine`, `cosine-warmup`, `step`, and `multistep`:

```bash
conda run -n wiag python scripts/train.py \
  --config configs/default.yaml \
  --scheduler cosine-warmup \
  --warmup-epochs 2 \
  --min-lr 0.000001
```

Manifest-based Danbooru training:

```bash
conda run -n wiag python scripts/train.py \
  --config configs/hf_vit_lora.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl
```

Full fine-tuning can be selected with `--finetune-mode full`. LoRA fine-tuning can be selected with `--finetune-mode lora --lora-r 8 --lora-alpha 16 --lora-target-modules query,value`.
From-scratch training can be run with `--no-pretrained` or `configs/scratch_vit.yaml`.

## Train CLIP/SigLIP/OpenCLIP Joint Embeddings

The joint pipeline trains image-text contrastive models with symmetric InfoNCE. It supports:

- `--backend hf-transformers-clip` for Hugging Face models with `get_image_features` and `get_text_features`, including `openai/clip-vit-base-patch32`, `openai/clip-vit-base-patch16`, `google/siglip-base-patch16-224`, and compatible Danbooru CLIP checkpoints.
- `--backend open-clip` for `open_clip_torch` models, including Hugging Face Hub OpenCLIP IDs such as `hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K`.

Default fine-tuning mode is `lora_vision`: the text tower is frozen, the vision tower gets LoRA adapters, and projection/logit-scale parameters remain trainable. This is the safest default for 8GB GPUs.

Single-model example:

```bash
conda run -n wiag python scripts/train_joint_clip.py \
  --backend hf-transformers-clip \
  --model-name openai/clip-vit-base-patch32 \
  --train-manifest data/manifests/danbooru_clip_top500/train.jsonl \
  --val-manifest data/manifests/danbooru_clip_top500/val.jsonl \
  --output-dir runs/joint_clip_openai_b32_top500 \
  --train-mode lora_vision \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --epochs 4 \
  --batches-per-epoch 250 \
  --amp
```

OpenCLIP example:

```bash
conda run -n wiag python scripts/train_joint_clip.py \
  --backend open-clip \
  --model-name hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --train-manifest data/manifests/danbooru_clip_top500/train.jsonl \
  --val-manifest data/manifests/danbooru_clip_top500/val.jsonl \
  --output-dir runs/joint_clip_laion_b32_top500 \
  --train-mode lora_vision \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --epochs 4 \
  --batches-per-epoch 150 \
  --amp
```

The joint trainer writes the same report-style artifacts as the image-only trainer:

- `best.pt`, `last.pt`, `config.json`
- `history.json`, `metrics.csv`
- `curves/loss.png`, `curves/retrieval.png`, `curves/lr.png`
- `parameter_counts.json`, `training_summary.json`

Checkpoint type is `joint_clip_v1`. The checkpoint stores trainable adapter/projection weights plus the base model configuration; it does not duplicate the full frozen base model.

## Run the Joint CLIP Top-500 Experiment

Use the automated runner to compare frozen baselines, smoke-test memory, train safe models, and generate slide-friendly tables and plots:

```bash
conda run -n wiag python scripts/run_joint_clip_top500_experiment.py \
  --models openai_clip_b32,openai_clip_b16,laion_openclip_b32,google_siglip_base
```

For a fast non-training check:

```bash
conda run -n wiag python scripts/run_joint_clip_top500_experiment.py \
  --models openai_clip_b32,laion_openclip_b32 \
  --skip-training
```

The runner writes:

```text
artifacts/joint_clip_top500/experiment_report.md
artifacts/joint_clip_top500/comparison_metrics.json
artifacts/joint_clip_top500/comparison_metrics.csv
artifacts/joint_clip_top500/comparison_text_identity_recall.png
artifacts/joint_clip_top500/comparison_text_image_recall.png
artifacts/joint_clip_top500/comparison_image_recall.png
artifacts/joint_clip_top500/comparison_mrr.png
```

Reported metrics include text-to-exact-image Recall/MRR, text-to-identity Recall/MRR, and image-to-image retrieval Recall/MRR. These CLIP/SigLIP experiments are currently for controlled model comparison; the existing production UI path remains the image-only index plus the DeVISE text-query index.

## Hugging Face Backbones

Supported loader backends:

- `--backbone-backend hf-transformers` for model IDs that work with `transformers.AutoModel`, such as `google/vit-base-patch16-224-in21k` and DINOv3 ViT checkpoints.
- `--backbone-backend hf-timm` for timm models hosted on Hugging Face, such as `hf-hub:MahmoodLab/UNI`.
- `--backbone-backend timm` for normal timm model names.

Examples:

```bash
conda run -n wiag python scripts/train.py \
  --config configs/dinov3_lora.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl

conda run -n wiag python scripts/train.py \
  --config configs/uni_full.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl
```

Some Hugging Face models are gated. Authenticate with `huggingface-cli login` or set `HF_TOKEN` before running. UNI requires accepting its access terms. DINOv3 models may also require access approval depending on the checkpoint.

LVFace is distributed on Hugging Face as project-specific `.pt` and `.onnx` artifacts rather than a standard timm or transformers model. Use the downloader to fetch selected files without downloading the whole repository:

```bash
conda run -n wiag python scripts/download_hf.py \
  --repo-id bytedance-research/LVFace \
  --filename LVFace-B_Glint360K/LVFace-B_Glint360K.pt \
  --output-dir artifacts/hf_models/lvface
```

Fine-tuning LVFace requires the upstream LVFace architecture code. The current training pipeline supports full and LoRA fine-tuning for timm and transformers backbones.

## Build a Gallery Index

```bash
conda run -n wiag python scripts/build_index.py \
  --checkpoint runs/vit_baseline/best.pt \
  --gallery-dir data/gallery \
  --output-dir artifacts/gallery_index
```

Manifest-based gallery indexing:

```bash
conda run -n wiag python scripts/build_index.py \
  --checkpoint runs/hf_vit_lora/best.pt \
  --gallery-manifest data/manifests/danbooru/all.jsonl \
  --output-dir artifacts/danbooru_index
```

## Query from the Command Line

```bash
conda run -n wiag python scripts/query.py \
  --checkpoint runs/vit_baseline/best.pt \
  --index-dir artifacts/gallery_index \
  --image /path/to/query.jpg \
  --top-k 5
```

## Compare Raw, Fine-Tuned, and Scratch Models

Use `scripts/evaluate.py` to run the same retrieval protocol across multiple model variants:

```bash
conda run -n wiag python scripts/evaluate.py \
  --spec configs/compare_example.yaml
```

The comparison spec can include raw Hugging Face backbones, fine-tuned checkpoints, and from-scratch checkpoints:

```yaml
dataset:
  query_manifest: data/manifests/danbooru/test.jsonl
  gallery_manifest: data/manifests/danbooru/all.jsonl

models:
  - name: google_vit_raw
    type: raw
    backbone_backend: hf-transformers
    model_name: google/vit-base-patch16-224-in21k
    pretrained: true
    image_mean: [0.5, 0.5, 0.5]
    image_std: [0.5, 0.5, 0.5]

  - name: fine_tuned_vit
    type: checkpoint
    checkpoint: runs/hf_vit_lora/best.pt

  - name: scratch_vit
    type: checkpoint
    checkpoint: runs/scratch_vit/best.pt
```

Outputs are written as `comparison_metrics.json` and `comparison_metrics.csv` when `output_dir` is set.

## Run the Local Web UI

```bash
conda run -n wiag python scripts/serve.py \
  --checkpoint runs/vit_baseline/best.pt \
  --gallery-dir data/gallery \
  --index-dir artifacts/gallery_index \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The UI can enroll gallery images, rebuild the FAISS index, and query a new image.

## Notes

- The default model is `vit_base_patch16_224`, but timm, Hugging Face Transformers, and Hugging Face-hosted timm backbones are supported.
- DINOv3, ViT, and UNI experiments can be represented as config presets and run directories.
- The system compares identities by nearest-neighbor retrieval rather than closed-set classification, so new gallery identities can be added by rebuilding the index.
