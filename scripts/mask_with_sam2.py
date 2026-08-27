#!/usr/bin/env python3
"""Create SAM 2 masks from YOLO box prompts in selection.json."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


COLORS = {"loop": (255, 180, 0), "forceps": (0, 220, 0)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/select-mask-video1-sam2.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output"])
    metadata = json.loads((output / "selection.json").read_text())
    checkpoint = Path(cfg["sam_checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"SAM 2 checkpoint not found: {checkpoint}")

    masks_dir = output / "masks"
    overlays_dir = output / "mask_overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    device = cfg["device"] if isinstance(cfg["device"], str) else f"cuda:{cfg['device']}"
    model = build_sam2(cfg["sam_model_config"], str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    manifest = []

    for item in metadata["images"]:
        image_path = Path(item["image"])
        image = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.asarray([detection["box_xyxy"] for detection in item["detections"]], dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False
            )

        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        instances = []
        overlay = image.copy()
        for detection, mask, score in zip(item["detections"], masks, np.asarray(scores).reshape(-1)):
            binary = np.squeeze(mask) > cfg["sam_mask_threshold"]
            class_id = 1 if detection["class"] == "loop" else 2
            combined[binary] = class_id
            color = np.asarray(COLORS[detection["class"]], dtype=np.uint8)
            overlay[binary] = (0.55 * overlay[binary] + 0.45 * color).astype(np.uint8)
            x1, y1, x2, y2 = map(int, detection["box_xyxy"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(value) for value in color), 2)
            cv2.putText(
                overlay, f"{detection['class']} SAM2={float(score):.3f}",
                (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                tuple(int(value) for value in color), 2,
            )
            instances.append({**detection, "sam2_score": float(score), "mask_pixels": int(binary.sum())})

        cv2.imwrite(str(masks_dir / f"{item['frame']}.png"), combined)
        cv2.imwrite(str(overlays_dir / f"{item['frame']}.png"), overlay)
        manifest.append({"frame": item["frame"], "instances": instances})

    (output / "masks.json").write_text(json.dumps({
        "model": "SAM 2.1 Hiera Large",
        "mask_encoding": {"0": "background", "1": "loop", "2": "forceps"},
        "images": manifest,
    }, indent=2))
    print(f"Created SAM 2 masks and overlays for {len(manifest)} images in {output}")


if __name__ == "__main__":
    main()
