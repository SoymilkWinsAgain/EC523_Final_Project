# Data

This directory is for local datasets and gallery images. Git ignores generated contents in this directory.

Expected training layout:

```text
data/train/<identity_name>/*.jpg
data/val/<identity_name>/*.jpg
```

Expected gallery layout:

```text
data/gallery/<identity_name>/*.jpg
```

Danbooru2021-scale runs should use generated manifests instead of copying images into this directory:

```text
data/manifests/danbooru/train.jsonl
data/manifests/danbooru/val.jsonl
data/manifests/danbooru/test.jsonl
data/manifests/danbooru/all.jsonl
```

The generated manifests are ignored by Git.

CLIP/SigLIP/OpenCLIP joint embedding experiments use a richer image-text subset extracted from `data/archive.zip` without unpacking the full archive:

```text
data/archive.zip
data/danbooru_clip_top500/images/<identity_name>/*.jpg
data/manifests/danbooru_clip_top500/train.jsonl
data/manifests/danbooru_clip_top500/val.jsonl
data/manifests/danbooru_clip_top500/all.jsonl
data/manifests/danbooru_clip_top500/summary.json
```

Each CLIP manifest row should include `path`, `identity`, `post_id`, structured `tags`, and `text`. The joint trainer removes character names from the training text by default and uses source, artist, and visual tags for image-text contrastive learning.
