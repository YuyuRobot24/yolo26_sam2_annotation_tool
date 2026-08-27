#!/usr/bin/env python3
"""Select diverse, visible frames for YOLO-to-SAM 2 mask generation."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/select-mask-video1-sam2.yaml"))
    return parser.parse_args()


def frame_number(frame: str) -> int:
    return int(frame.rsplit("_", 1)[1])


def visible(row: dict, cfg: dict) -> bool:
    return (
        float(row["brightness"]) >= cfg["min_brightness"]
        and float(row["contrast"]) >= cfg["min_contrast"]
        and float(row["sharpness"]) >= cfg["min_sharpness"]
        and float(row["dark_density"]) <= cfg["max_dark_density"]
    )


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output"])
    original_dir = output / "selected_images"
    preview_dir = output / "yolo_previews"
    original_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    with Path(cfg["visibility_csv"]).open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["frame"].startswith(f"video{cfg['video']}_") and visible(row, cfg)]
    # Prioritize frames where the earlier evaluation already had strong detections,
    # then let the class-specific YOLO pass make the final decision.
    rows.sort(key=lambda row: (float(row["mean_confidence"]), float(row["contrast"])), reverse=True)

    model = YOLO(cfg["yolo_model"])
    wanted = set(cfg["classes"])
    selected = []
    accepted_frames = []
    batch_size = int(cfg["batch"])
    image_dir = Path(cfg["image_dir"])

    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        paths = [image_dir / f"{row['frame']}.png" for row in chunk_rows]
        results = model.predict(
            source=[str(path) for path in paths], imgsz=cfg["imgsz"], batch=batch_size,
            conf=cfg["yolo_conf"], iou=cfg["yolo_iou"], device=cfg["device"],
            stream=False, verbose=False,
        )
        for row, path, result in zip(chunk_rows, paths, results):
            number = frame_number(row["frame"])
            if any(abs(number - existing) < cfg["min_frame_gap"] for existing in accepted_frames):
                continue
            detections = []
            for box, cls_id, confidence in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                class_name = model.names[int(cls_id)]
                if class_name in wanted and confidence >= cfg["yolo_conf"]:
                    detections.append({
                        "class": class_name,
                        "class_id": int(cls_id),
                        "confidence": float(confidence),
                        "box_xyxy": [float(value) for value in box],
                    })
            if not detections:
                continue
            shutil.copy2(path, original_dir / path.name)
            cv2.imwrite(str(preview_dir / path.name), result.plot())
            selected.append({
                "frame": row["frame"],
                "image": str(original_dir / path.name),
                "visibility": {key: float(row[key]) for key in (
                    "brightness", "contrast", "light_density", "glare_density",
                    "dark_density", "saturation", "sharpness"
                )},
                "detections": detections,
            })
            accepted_frames.append(number)
            if len(selected) >= cfg["max_images"]:
                break
        if len(selected) >= cfg["max_images"]:
            break

    metadata = {
        "video": cfg["video"],
        "selection_rules": {key: cfg[key] for key in (
            "classes", "min_brightness", "min_contrast", "min_sharpness",
            "max_dark_density", "yolo_conf", "min_frame_gap"
        )},
        "images": selected,
    }
    (output / "selection.json").write_text(json.dumps(metadata, indent=2))
    print(f"Selected {len(selected)} images into {output}")
    if len(selected) < cfg["max_images"]:
        print("Warning: fewer images passed all selection rules than requested")


if __name__ == "__main__":
    main()
