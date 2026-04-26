# Who Is This Anime Girl

This repository implements an image-only anime character retrieval system. It trains an embedding model, builds a FAISS gallery index, and serves a small local web UI for enrolling known characters and querying unknown images.

## Technology Stack

- Python environment: `jigsaw`
- Modeling: PyTorch, torchvision, timm
- Hugging Face loading: transformers, huggingface_hub, peft
- Retrieval: FAISS with cosine similarity through normalized embeddings
- Web UI: Python standard library HTTP server plus static HTML/CSS/JavaScript
- Dataset format: ImageFolder directories or JSONL manifests aligned with Danbooru2021 bucketed files

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

## Prepare Danbooru2021 Manifests

Do not download the full dataset into this repository. Point the script at an existing SCC or external dataset location:

```bash
conda run -n jigsaw python scripts/prepare_danbooru.py \
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

## Train an Embedding Model

Run from the repository root:

```bash
conda run -n jigsaw python scripts/train.py \
  --config configs/default.yaml \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir runs/vit_baseline
```

Useful SCC overrides:

```bash
conda run -n jigsaw python scripts/train.py \
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
conda run -n jigsaw python scripts/train.py \
  --config configs/default.yaml \
  --scheduler cosine-warmup \
  --warmup-epochs 2 \
  --min-lr 0.000001
```

Manifest-based Danbooru training:

```bash
conda run -n jigsaw python scripts/train.py \
  --config configs/hf_vit_lora.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl
```

Full fine-tuning can be selected with `--finetune-mode full`. LoRA fine-tuning can be selected with `--finetune-mode lora --lora-r 8 --lora-alpha 16 --lora-target-modules query,value`.
From-scratch training can be run with `--no-pretrained` or `configs/scratch_vit.yaml`.

## Hugging Face Backbones

Supported loader backends:

- `--backbone-backend hf-transformers` for model IDs that work with `transformers.AutoModel`, such as `google/vit-base-patch16-224-in21k` and DINOv3 ViT checkpoints.
- `--backbone-backend hf-timm` for timm models hosted on Hugging Face, such as `hf-hub:MahmoodLab/UNI`.
- `--backbone-backend timm` for normal timm model names.

Examples:

```bash
conda run -n jigsaw python scripts/train.py \
  --config configs/dinov3_lora.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl

conda run -n jigsaw python scripts/train.py \
  --config configs/uni_full.yaml \
  --train-manifest data/manifests/danbooru/train.jsonl \
  --val-manifest data/manifests/danbooru/val.jsonl
```

Some Hugging Face models are gated. Authenticate with `huggingface-cli login` or set `HF_TOKEN` before running. UNI requires accepting its access terms. DINOv3 models may also require access approval depending on the checkpoint.

LVFace is distributed on Hugging Face as project-specific `.pt` and `.onnx` artifacts rather than a standard timm or transformers model. Use the downloader to fetch selected files without downloading the whole repository:

```bash
conda run -n jigsaw python scripts/download_hf.py \
  --repo-id bytedance-research/LVFace \
  --filename LVFace-B_Glint360K/LVFace-B_Glint360K.pt \
  --output-dir artifacts/hf_models/lvface
```

Fine-tuning LVFace requires the upstream LVFace architecture code. The current training pipeline supports full and LoRA fine-tuning for timm and transformers backbones.

## Build a Gallery Index

```bash
conda run -n jigsaw python scripts/build_index.py \
  --checkpoint runs/vit_baseline/best.pt \
  --gallery-dir data/gallery \
  --output-dir artifacts/gallery_index
```

Manifest-based gallery indexing:

```bash
conda run -n jigsaw python scripts/build_index.py \
  --checkpoint runs/hf_vit_lora/best.pt \
  --gallery-manifest data/manifests/danbooru/all.jsonl \
  --output-dir artifacts/danbooru_index
```

## Query from the Command Line

```bash
conda run -n jigsaw python scripts/query.py \
  --checkpoint runs/vit_baseline/best.pt \
  --index-dir artifacts/gallery_index \
  --image /path/to/query.jpg \
  --top-k 5
```

## Compare Raw, Fine-Tuned, and Scratch Models

Use `scripts/evaluate.py` to run the same retrieval protocol across multiple model variants:

```bash
conda run -n jigsaw python scripts/evaluate.py \
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
conda run -n jigsaw python scripts/serve.py \
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
