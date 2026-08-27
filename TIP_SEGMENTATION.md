# Surgical Instrument Tip Segmentation with YOLO26 and SAM 2

This repository detects retinal surgical instrument tips with a trained
Ultralytics YOLO26 model and converts high-confidence `loop` and `forceps`
detections into segmentation masks with Meta SAM 2.1.

The final high-quality workflow uses two prompts for SAM 2:

1. the YOLO tip bounding box;
2. a positive point at the center of that box.

A tip is accepted only when its YOLO confidence is above `0.80` and its best
SAM 2 predicted mask quality is above `0.70`.

## Pipeline

```text
3D-STARES image
    → visibility filtering
    → YOLO26 loop/forceps tip detection (>0.80)
    → bounding-box center point
    → SAM 2.1 box + positive-point prompting
    → best mask selection (>0.70)
    → class-aware mask and review images
```

## Requirements

- Linux
- NVIDIA GPU and current NVIDIA driver
- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit
- Approximately 40 GB for the extracted dataset
- Approximately 1 GB for the SAM 2.1 Hiera Large checkpoint

The Docker image is based on:

```text
pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
```

It contains PyTorch 2.9, CUDA 12.8, the local Ultralytics source, and Meta's
official SAM 2 source. The optional SAM 2 connected-components CUDA extension
is disabled because the runtime image does not include `nvcc`; normal image
segmentation remains available.

## Repository layout

```text
configs/
  evaluate-video1.yaml
  tip-video1-high-quality-50.yaml
scripts/
  evaluate_visibility.py
  select_for_sam2.py
  segment_high_quality_tips_sam2.py
  render_tip_mask_samples.py
Dockerfile
compose.yaml
TRAINING.md
TIP_SEGMENTATION.md
```

The dataset, checkpoints, generated masks, evaluation outputs, and dependency
source clones are intentionally excluded from Git.

## 1. Clone this repository and its dependencies

Replace the first URL with the URL of your GitHub repository:

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

mkdir -p src
git clone https://github.com/ultralytics/ultralytics.git src/ultralytics
git clone https://github.com/facebookresearch/sam2.git src/sam2
git clone https://github.com/ultralytics/yolo26.git yolo26
```

The implementation repositories are cloned locally instead of committed into
this repository. Review the Ultralytics and SAM 2 licenses for the intended use.

## 2. Prepare the 3D-STARES dataset

The expected extracted files are:

```text
data/
  images/
    video1_0.png
    ...
  expanded_detection_labels.json
```

Convert the JSON boxes to YOLO labels and create the split files:

```bash
python3 scripts/prepare_dataset.py --train-fraction 0.50 --seed 42
```

The detection annotations describe instrument **tips**, not entire shafts.
Generated YOLO classes are:

```text
0 forceps
1 light
2 loop
3 tono
```

## 3. Download the SAM 2.1 weight

The pipeline uses Meta's public SAM 2.1 Hiera Large checkpoint:

```bash
mkdir -p models
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o models/sam2.1_hiera_large.pt
```

Expected path inside Docker:

```text
/workspace/models/sam2.1_hiera_large.pt
```

## 4. Provide the trained YOLO26 weight

The default configuration expects:

```text
runs/yolo26s_3d_stares_25pct-2/weights/best.pt
```

Either copy your trained checkpoint there or update `yolo_model` in
`configs/tip-video1-high-quality-50.yaml`.

To train a new detector first, follow `TRAINING.md`:

```bash
docker compose run --rm train yolo detect train \
  cfg=/workspace/configs/train-small.yaml
```

## 5. Build and verify Docker

```bash
docker compose build train

docker compose run --rm train python -c \
  "import torch, ultralytics, sam2; print(torch.__version__, torch.cuda.is_available())"
```

The second command should report PyTorch `2.9.0+cu128` and CUDA availability as
`True`.

## 6. Calculate visibility metrics

The selector uses per-frame visibility measurements produced by the evaluator:

```bash
docker compose run --rm train python /workspace/scripts/evaluate_visibility.py \
  --config /workspace/configs/evaluate-video1.yaml
```

This produces:

```text
evaluation/video1/per_frame_metrics.csv
```

Visibility filtering currently checks brightness, contrast, sharpness, and the
fraction of dark pixels.

## 7. Select YOLO tip candidates

```bash
docker compose run --rm train python /workspace/scripts/select_for_sam2.py \
  --config /workspace/configs/tip-video1-high-quality-50.yaml
```

The default configuration builds a temporally separated pool of up to 250
visible video-1 images containing `loop` or `forceps` detections. YOLO's
threshold is `0.80`.

## 8. Generate high-quality SAM 2 tip masks

```bash
docker compose run --rm train python \
  /workspace/scripts/segment_high_quality_tips_sam2.py \
  --config /workspace/configs/tip-video1-high-quality-50.yaml
```

For every candidate, the script:

1. loads the image into SAM 2;
2. supplies the YOLO bounding box;
3. adds a positive prompt at the box center;
4. requests multiple candidate masks;
5. selects the highest-scoring mask;
6. accepts it only if its SAM 2 score is above `0.70`;
7. stops after 50 accepted images.

Output structure:

```text
selected_and_masked/video1_tip_high_quality_50/
  accepted_images/       # source images that passed both thresholds
  tip_masks/             # class-aware, single-channel PNG labels
  tip_previews/          # full-frame mask overlays
  contact_sheets/        # five overview sheets
  selection.json         # YOLO candidate metadata
  tip_annotations.json   # accepted YOLO/SAM scores and mask metadata
  rejected_tips.json     # rejected SAM candidates
```

Mask pixel values are:

| Pixel | Meaning |
|---:|---|
| 0 | Background |
| 1 | Loop tip |
| 2 | Forceps tip |

## 9. Apply masks and create enlarged review samples

```bash
docker compose run --rm train python \
  /workspace/scripts/render_tip_mask_samples.py \
  --input /workspace/selected_and_masked/video1_tip_high_quality_50
```

Additional outputs:

```text
masked_images/           # full-frame tip cutouts on black
roi_mask_samples/        # enlarged original / overlay / cutout panels
masked_contact_sheets/   # five sheets showing all 50 panels
```

The ROI panels are the quickest way to review whether the generated masks
accurately follow the tip boundaries.

## Configuration

Important settings in `configs/tip-video1-high-quality-50.yaml`:

```yaml
classes: [loop, forceps]
yolo_conf: 0.80
accepted_images: 50
sam_score_min: 0.70
min_frame_gap: 25
```

Increase `sam_score_min` for stricter masks. Increase `max_images` if the
candidate pool cannot produce the requested number of accepted images.

## Publish the code on GitHub

Large files are ignored by `.gitignore`, including `data/`, `runs/`, `*.pt`,
evaluation outputs, generated masks, and cloned dependency sources. Verify this
before committing:

```bash
git init
git status --short
git add .
git status --short
```

Confirm that no dataset images or `.pt` checkpoints appear in the staged list.
Then create an empty repository on GitHub and run:

```bash
git commit -m "Add YOLO26 and SAM 2 instrument-tip segmentation pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

### Optional: publish the trained YOLO weight

GitHub rejects ordinary files larger than 100 MB. The current YOLO checkpoint
is smaller, but model files remain ignored by default. If licensing permits
checkpoint redistribution, use Git LFS rather than ordinary Git:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add -f path/to/your/best.pt
git commit -m "Add trained model weight with Git LFS"
git push
```

Do not upload the 3D-STARES images or annotations unless their license
explicitly permits redistribution.

## Reproducibility notes

- Record the commit hashes of `src/ultralytics` and `src/sam2` in experiment
  reports.
- Video 1 contributed to detector training, so video-1 masks demonstrate the
  workflow but are not an unbiased test of generalization.
- Generated masks are pseudo-labels. Confidence filtering improves their
  reliability but does not replace expert review for clinical research.
