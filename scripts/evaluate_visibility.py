#!/usr/bin/env python3
"""Evaluate one 3D-STARES video by detection confidence and image visibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate-video1.yaml"))
    return parser.parse_args()


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0)
    xy1 = np.maximum(box[:2], boxes[:, :2])
    xy2 = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(xy2 - xy1, 0, None).prod(axis=1)
    area1 = np.clip(box[2:] - box[:2], 0, None).prod()
    area2 = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(axis=1)
    return intersection / np.clip(area1 + area2 - intersection, 1e-9, None)


def match_detections(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, iou_threshold):
    order = np.argsort(-pred_conf)
    used = set()
    tp = np.zeros(len(pred_boxes), dtype=bool)
    for index in order:
        candidates = np.flatnonzero(gt_cls == pred_cls[index])
        candidates = np.array([candidate for candidate in candidates if candidate not in used], dtype=int)
        if len(candidates) == 0:
            continue
        ious = box_iou(pred_boxes[index], gt_boxes[candidates])
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            tp[index] = True
            used.add(int(candidates[best]))
    return int(tp.sum()), int(len(pred_boxes) - tp.sum()), int(len(gt_boxes) - tp.sum())


def visibility_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "light_density": float(np.mean(gray >= 200)),
        "glare_density": float(np.mean(gray >= 245)),
        "dark_density": float(np.mean(gray <= 50)),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def load_ground_truth(label_path: Path, width: int, height: int):
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, values)
            boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                          (xc + bw / 2) * width, (yc + bh / 2) * height])
            classes.append(int(cls))
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(classes, dtype=int)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize(rows: list[dict], names: dict, config: dict) -> dict:
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    summary = {
        "scope": f"3D-STARES video {config['video']} (training-domain evaluation)",
        "frames": len(rows),
        "ground_truth_objects": sum(row["ground_truth_count"] for row in rows),
        "predictions": sum(row["prediction_count"] for row in rows),
        "confidence_threshold": config["conf"],
        "iou_threshold": config["iou"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_at_threshold": safe_ratio(tp, tp + fp),
        "recall_at_threshold": safe_ratio(tp, tp + fn),
        "f1_at_threshold": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "mean_detection_confidence": float(np.mean([r["mean_confidence"] for r in rows if r["prediction_count"]]))
        if any(r["prediction_count"] for r in rows) else 0.0,
        "classes": names,
        "visibility_definitions": {
            "brightness": "mean grayscale intensity / 255",
            "contrast": "grayscale standard deviation / 255",
            "light_density": "fraction of pixels with grayscale intensity >= 200",
            "glare_density": "fraction of pixels with grayscale intensity >= 245",
            "dark_density": "fraction of pixels with grayscale intensity <= 50",
            "saturation": "mean HSV saturation / 255",
            "sharpness": "variance of the grayscale Laplacian",
        },
    }
    return summary


def visibility_bins(rows: list[dict], metric: str) -> list[dict]:
    values = np.asarray([row[metric] for row in rows])
    edges = np.unique(np.quantile(values, [0, 0.25, 0.5, 0.75, 1.0]))
    output = []
    for index in range(len(edges) - 1):
        include_right = index == len(edges) - 2
        selected = [row for row in rows if edges[index] <= row[metric] and
                    (row[metric] <= edges[index + 1] if include_right else row[metric] < edges[index + 1])]
        if not selected:
            continue
        tp, fp, fn = (sum(row[key] for row in selected) for key in ("tp", "fp", "fn"))
        output.append({
            "visibility_metric": metric,
            "bin": index + 1,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "frames": len(selected),
            "precision": safe_ratio(tp, tp + fp),
            "recall": safe_ratio(tp, tp + fn),
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
            "mean_confidence": float(np.mean([row["mean_confidence"] for row in selected])),
        })
    return output


def save_plots(rows: list[dict], bins: list[dict], output: Path):
    metrics = ["brightness", "light_density", "glare_density", "contrast", "sharpness"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, metric in zip(axes.flat, metrics):
        axis.scatter([row[metric] for row in rows], [row["mean_confidence"] for row in rows],
                     s=4, alpha=0.25)
        axis.set(xlabel=metric, ylabel="mean detection confidence", ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Detection confidence versus image visibility")
    fig.tight_layout()
    fig.savefig(output / "confidence_vs_visibility.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), sharey=True)
    for axis, metric in zip(axes, metrics):
        subset = [row for row in bins if row["visibility_metric"] == metric]
        axis.plot([row["bin"] for row in subset], [row["f1"] for row in subset], marker="o", label="F1")
        axis.plot([row["bin"] for row in subset], [row["mean_confidence"] for row in subset], marker="s", label="confidence")
        axis.set(title=metric, xlabel="visibility quartile", ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("score")
    axes[-1].legend()
    fig.suptitle("Performance by visibility quartile (low to high)")
    fig.tight_layout()
    fig.savefig(output / "metrics_by_visibility.png", dpi=180)
    plt.close(fig)


def predict_in_batches(model: YOLO, image_paths: list[Path], config: dict):
    """Avoid Ultralytics treating a Python source list as one in-memory batch."""
    batch_size = int(config["batch"])
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        yield from model.predict(
            source=[str(path) for path in chunk], imgsz=config["imgsz"], batch=batch_size,
            conf=config["conf"], iou=0.7, device=config["device"], stream=False, verbose=False,
        )


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    annotated_dir = output / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)
    data_dir = Path(config["data"]).parent
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    image_paths = sorted(image_dir.glob(f"video{config['video']}_*.png"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not image_paths:
        raise SystemExit(f"No images found for video {config['video']} in {image_dir}")

    model = YOLO(config["model"])
    names = {int(key): value for key, value in model.names.items()}
    selected_indices = set(np.linspace(0, len(image_paths) - 1,
                                       min(config["max_annotated_images"], len(image_paths)), dtype=int))
    rows = []
    results = predict_in_batches(model, image_paths, config)
    for index, (image_path, result) in enumerate(zip(image_paths, results)):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt_boxes, gt_cls = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        if result.boxes is None:
            pred_boxes, pred_cls, pred_conf = np.empty((0, 4)), np.empty(0, int), np.empty(0)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
        tp, fp, fn = match_detections(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, config["iou"])
        visibility = visibility_metrics(image)
        rows.append({
            "frame": image_path.stem,
            "ground_truth_count": len(gt_boxes),
            "prediction_count": len(pred_boxes),
            "mean_confidence": float(pred_conf.mean()) if len(pred_conf) else 0.0,
            "max_confidence": float(pred_conf.max()) if len(pred_conf) else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            **visibility,
        })
        if index in selected_indices:
            plotted = result.plot()
            for box, cls in zip(gt_boxes.astype(int), gt_cls):
                x1, y1, x2, y2 = box
                cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(plotted, f"GT {names.get(int(cls), cls)}", (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            text = f"light={visibility['light_density']:.3f} bright={visibility['brightness']:.3f} sharp={visibility['sharpness']:.0f}"
            cv2.putText(plotted, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.imwrite(str(annotated_dir / image_path.name), plotted)

    with (output / "per_frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    binned = []
    for metric in ("brightness", "light_density", "glare_density", "contrast", "sharpness"):
        binned.extend(visibility_bins(rows, metric))
    with (output / "visibility_bin_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=binned[0].keys())
        writer.writeheader()
        writer.writerows(binned)
    summary = summarize(rows, names, config)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    save_plots(rows, binned, output)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
