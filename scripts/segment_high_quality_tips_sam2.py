#!/usr/bin/env python3
"""Use a YOLO box plus its center point to create high-quality SAM 2 tip masks."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


TIP_COLOR = np.array([255, 0, 255], dtype=np.uint8)  # magenta, BGR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tip-video1-high-quality-50.yaml"))
    return parser.parse_args()


def make_contact_sheets(paths, output_dir, columns=5, rows=2, thumb_width=480):
    per_sheet = columns * rows
    for sheet_number, start in enumerate(range(0, len(paths), per_sheet), start=1):
        images = []
        for path in paths[start : start + per_sheet]:
            image = cv2.imread(str(path))
            scale = thumb_width / image.shape[1]
            images.append(cv2.resize(image, (thumb_width, round(image.shape[0] * scale))))
        height = images[0].shape[0]
        canvas = np.zeros((rows * height, columns * thumb_width, 3), dtype=np.uint8)
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            canvas[row * height : (row + 1) * height,
                   column * thumb_width : (column + 1) * thumb_width] = image
        cv2.imwrite(str(output_dir / f"contact_sheet_{sheet_number:02d}.jpg"), canvas)


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output"])
    candidates = json.loads((output / "selection.json").read_text())
    checkpoint = Path(cfg["sam_checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"SAM 2 checkpoint not found: {checkpoint}")

    accepted_dir = output / "accepted_images"
    masks_dir = output / "tip_masks"
    previews_dir = output / "tip_previews"
    sheets_dir = output / "contact_sheets"
    for directory in (accepted_dir, masks_dir, previews_dir, sheets_dir):
        directory.mkdir(exist_ok=True)

    device = cfg["device"] if isinstance(cfg["device"], str) else f"cuda:{cfg['device']}"
    predictor = SAM2ImagePredictor(build_sam2(
        cfg["sam_model_config"], str(checkpoint), device=device
    ))
    accepted, rejected = [], []
    preview_paths = []

    for item in candidates["images"]:
        image_path = Path(item["image"])
        image = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_accepted = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)
            for detection in item["detections"]:
                box = np.asarray(detection["box_xyxy"], dtype=np.float32)
                center = np.asarray([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], dtype=np.float32)
                labels = np.asarray([1], dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=center,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                score = float(scores[best])
                record = {
                    **detection,
                    "center_point_xy": center[0].tolist(),
                    "sam2_score": score,
                }
                if score > cfg["sam_score_min"]:
                    record["mask"] = np.squeeze(masks[best]).astype(bool)
                    frame_accepted.append(record)
                else:
                    rejected.append({"frame": item["frame"], **record, "reason": "sam2_score_not_above_threshold"})

        if not frame_accepted:
            continue

        label_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        overlay = image.copy()
        instances = []
        for record in frame_accepted:
            binary = record.pop("mask")
            # Class-aware tip encoding: 1=loop tip, 2=forceps tip.
            value = 1 if record["class"] == "loop" else 2
            label_mask[binary] = value
            overlay[binary] = (0.45 * overlay[binary] + 0.55 * TIP_COLOR).astype(np.uint8)
            x1, y1, x2, y2 = map(int, record["box_xyxy"])
            cx, cy = map(int, record["center_point_xy"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(map(int, TIP_COLOR)), 2)
            cv2.drawMarker(overlay, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                overlay,
                f"{record['class']} YOLO={record['confidence']:.2f} SAM2={record['sam2_score']:.2f}",
                (max(5, x1), max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2,
            )
            record["mask_pixels"] = int(binary.sum())
            instances.append(record)

        destination = accepted_dir / image_path.name
        shutil.copy2(image_path, destination)
        mask_path = masks_dir / f"{item['frame']}.png"
        preview_path = previews_dir / f"{item['frame']}.jpg"
        cv2.imwrite(str(mask_path), label_mask)
        cv2.imwrite(str(preview_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        preview_paths.append(preview_path)
        accepted.append({"frame": item["frame"], "image": str(destination), "instances": instances})
        if len(accepted) >= cfg["accepted_images"]:
            break

    make_contact_sheets(preview_paths, sheets_dir)
    result = {
        "model": "YOLO26 box + center point -> SAM 2.1 Hiera Large",
        "acceptance": {"yolo_confidence_strictly_above": cfg["yolo_conf"],
                       "sam2_score_strictly_above": cfg["sam_score_min"]},
        "mask_encoding": {"0": "background", "1": "loop tip", "2": "forceps tip"},
        "accepted_images": accepted,
        "rejected_detections_checked": len(rejected),
    }
    (output / "tip_annotations.json").write_text(json.dumps(result, indent=2))
    (output / "rejected_tips.json").write_text(json.dumps(rejected, indent=2))
    print(f"Accepted {len(accepted)} high-quality tip images; rejected {len(rejected)} detections")
    if len(accepted) < cfg["accepted_images"]:
        raise SystemExit("Candidate pool did not contain enough accepted tips; increase max_images")


if __name__ == "__main__":
    main()
