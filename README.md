# Who Is This Anime Girl

This repository implements an image-only anime character retrieval system. It trains an embedding model, builds a FAISS gallery index, and serves a small local web UI for enrolling known characters and querying unknown images.

## Technology Stack

- Python environment: `jigsaw`
- Modeling: PyTorch, torchvision, timm
- Retrieval: FAISS with cosine similarity through normalized embeddings
- Web UI: Python standard library HTTP server plus static HTML/CSS/JavaScript
- Dataset format: folder-per-identity image directories compatible with `torchvision.datasets.ImageFolder`

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

Training and gallery directories use identity names as subdirectories:

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

## Build a Gallery Index

```bash
conda run -n jigsaw python scripts/build_index.py \
  --checkpoint runs/vit_baseline/best.pt \
  --gallery-dir data/gallery \
  --output-dir artifacts/gallery_index
```

## Query from the Command Line

```bash
conda run -n jigsaw python scripts/query.py \
  --checkpoint runs/vit_baseline/best.pt \
  --index-dir artifacts/gallery_index \
  --image /path/to/query.jpg \
  --top-k 5
```

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

- The default model is `vit_base_patch16_224`, but any timm backbone with `num_classes=0` support can be used.
- DINO, LVFace, ViT, and UNI experiments can be represented as different `--model-name` values and run directories.
- The system compares identities by nearest-neighbor retrieval rather than closed-set classification, so new gallery identities can be added by rebuilding the index.
