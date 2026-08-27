# Train YOLO26 on 3D-STARES locally

For the focused YOLO26-to-SAM 2 instrument-tip mask workflow and GitHub
publishing instructions, see [`TIP_SEGMENTATION.md`](TIP_SEGMENTATION.md).

This project trains an Ultralytics YOLO26 detector for four retinal surgical
instrument-tip classes: `forceps`, `light`, `loop`, and `tono`.

The source split ZIP has already been reconstructed and extracted into `data/`.
The requested `yolo26/` clone is retained for reference. Ultralytics identifies
`src/ultralytics/` as the canonical implementation repository, so that source is
installed in the Docker image.

## Prerequisites

- Linux with an NVIDIA GPU and a current NVIDIA driver
- Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit
- About 70 GB free for source images, the Docker image, caches, and runs

Confirm that Docker can see the GPU:

```bash
docker run --rm --gpus all pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## 1. Prepare labels and leakage-resistant splits

The source annotations already use normalized YOLO box coordinates. Convert the
expanded JSON annotations to one YOLO text file per frame:

```bash
python3 scripts/prepare_dataset.py --train-fraction 0.25 --seed 42
```

The split is by source video, not by random frames, to prevent neighboring
frames leaking between sets:

- train: videos 1–5
- validation: video 6
- test: video 7

Images without a label file are retained as background examples. Generated
labels, image lists, and `data/dataset.yaml` are ignored by Git along with the
large extracted dataset.

This creates both the full `data/dataset.yaml` and a downsampled
`data/dataset-small.yaml`. The latter retains 25% from **each** training video,
selected reproducibly with seed 42. Validation and test sets remain complete.
Change `0.25` to another value in `(0, 1]` and rerun preparation if desired.

## 2. Build the training image

```bash
docker compose build train
```

The image is based exactly on
`pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` and installs the locally cloned
Ultralytics source. Building and the first training run require internet access
for Python packages and pretrained model weights.

Verify PyTorch, CUDA, and YOLO26:

```bash
docker compose run --rm train python -c \
  "import torch, ultralytics; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), ultralytics.__version__)"
```

## 3. Train

Start with the small pretrained model. The supplied configuration uses
`batch: 0.80`, which asks Ultralytics AutoBatch to target 80% of available GPU
memory. This is appropriate for the 96 GB RTX PRO 6000 while retaining headroom
for augmentation and transient allocations:

```bash
docker compose run --rm train yolo detect train \
  cfg=/workspace/configs/train-small.yaml
```

`imgsz=960` is a useful starting point for small tool tips. If GPU memory is
tight, lower `batch` to `0.70`, use `imgsz=640`, `model=yolo26n.pt`, or a fixed
smaller batch. If utilization remains below roughly 70%, try `batch: 0.85`; avoid
going above 0.90 because validation and augmentation can cause memory spikes. Results and
checkpoints are written to `runs/yolo26s_3d_stares_25pct/`; the best checkpoint is
`weights/best.pt`.

Resume an interrupted run:

```bash
docker compose run --rm train yolo detect train \
  resume=/workspace/runs/yolo26s_3d_stares_25pct/weights/last.pt
```

## 4. Evaluate and predict

Evaluate once on the held-out video 7 test split:

```bash
docker compose run --rm train yolo detect val \
  model=/workspace/runs/yolo26s_3d_stares_25pct/weights/best.pt \
  data=/workspace/data/dataset.yaml split=test device=0
```

Run inference on an image:

```bash
docker compose run --rm train yolo detect predict \
  model=/workspace/runs/yolo26s_3d_stares_25pct/weights/best.pt \
  source=/workspace/data/images/video7_2127.png \
  device=0 save=True project=/workspace/runs name=predictions
```

## Notes

- `expanded_detection_labels.json` includes interpolated boxes and maximizes the
  labeled training set. To use only manually supplied detection labels, rerun
  preparation with `--annotations detection_labels.json`.
- Dataset images and labels may have separate licensing or usage conditions;
  check the 3D-STARES source terms before redistribution.
- Ultralytics source is offered under AGPL-3.0 or an Enterprise license. Review
  its license before production use.

## Video 1 confidence and visibility evaluation

Evaluate every video-1 frame and save threshold detection metrics, per-frame
confidence/visibility data, aggregate plots, and 50 annotated examples:

```bash
docker compose run --rm train python /workspace/scripts/evaluate_visibility.py \
  --config /workspace/configs/evaluate-video1.yaml
```

Results are written to `evaluation/video1/`. Predicted boxes use the Ultralytics
plot colors and ground-truth boxes are red. Video 1 contributed to the training
pool, so this is a training-domain diagnostic rather than a held-out estimate of
generalization; use video 7 for final test reporting.

## Select visible detections and create SAM 2 masks

The combined Docker image installs both the local YOLO26 implementation and
Meta's official SAM 2 implementation. The test configuration selects 10 temporally
separated video-1 frames with acceptable brightness, contrast, sharpness, and
dark-pixel density, then requires a `loop` or `forceps` YOLO detection with at
least 0.80 confidence:

```bash
docker compose run --rm train python /workspace/scripts/select_for_sam2.py \
  --config /workspace/configs/select-mask-video1-sam2.yaml
```

Selected originals, YOLO previews, and box/visibility metadata are saved under
`selected_and_masked/video1_sam2_test/`.

The configuration uses Meta's public SAM 2.1 Hiera Large checkpoint. Download it
once if it is not already present:

```bash
mkdir -p models
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o models/sam2.1_hiera_large.pt
```

Create masks from the selected YOLO boxes:

```bash
docker compose run --rm train python /workspace/scripts/mask_with_sam2.py \
  --config /workspace/configs/select-mask-video1-sam2.yaml
```

The output `masks/` PNGs use pixel value 0 for background, 1 for `loop`, and 2
for `forceps`; `mask_overlays/` contains visual checks and `masks.json` records
YOLO and SAM 2 scores. Because 3D-STARES boxes annotate instrument tips, these box-
prompted masks target the tip region rather than guaranteeing the full shaft.

### Fifty high-quality tip-only samples

Generate a strict 50-frame video-1 tip set. Candidates require YOLO confidence
above 0.80. SAM 2 receives both the YOLO box and its center as a positive point;
only the best mask with predicted SAM 2 quality above 0.70 is accepted:

```bash
docker compose run --rm train python /workspace/scripts/select_for_sam2.py \
  --config /workspace/configs/tip-video1-high-quality-50.yaml

docker compose run --rm train python /workspace/scripts/segment_high_quality_tips_sam2.py \
  --config /workspace/configs/tip-video1-high-quality-50.yaml
```

Outputs are under `selected_and_masked/video1_tip_high_quality_50/`. In the mask
PNGs, 0 is background, 1 is a loop tip, and 2 is a forceps tip. The final folder
contains only accepted images and masks, five contact sheets, acceptance
metadata, and a separate rejection log.

Apply the accepted masks to the source images and create enlarged
original/overlay/cutout ROI comparisons:

```bash
docker compose run --rm train python /workspace/scripts/render_tip_mask_samples.py \
  --input /workspace/selected_and_masked/video1_tip_high_quality_50
```

Black-background full-frame cutouts are written to `masked_images/`, enlarged
three-panel examples to `roi_mask_samples/`, and five overview sheets to
`masked_contact_sheets/`.
