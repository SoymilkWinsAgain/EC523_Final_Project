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
