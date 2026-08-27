#!/usr/bin/env python3
"""Convert 3D-STARES JSON boxes to Ultralytics YOLO detection format."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path, PurePosixPath


DEFAULT_SPLITS = {
    "train": {1, 2, 3, 4, 5},
    "val": {6},
    "test": {7},
}
VIDEO_RE = re.compile(r"^video(\d+)_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--annotations", default="expanded_detection_labels.json",
        help="JSON file within data-dir (expanded labels are the default).",
    )
    parser.add_argument(
        "--container-data-dir", default="/workspace/data",
        help="Data path written to image lists and dataset.yaml.",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Seeded fraction of each training video to retain (default: 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction <= 1.0:
        raise SystemExit("--train-fraction must be greater than 0 and at most 1")
    data_dir = args.data_dir.resolve()
    image_dir = data_dir / "images"
    annotation_path = data_dir / args.annotations
    if not image_dir.is_dir() or not annotation_path.is_file():
        raise SystemExit(f"Missing {image_dir} or {annotation_path}")

    annotations = json.loads(annotation_path.read_text())
    classes = sorted({box["label"] for boxes in annotations.values() for box in boxes})
    class_ids = {name: index for index, name in enumerate(classes)}
    label_dir = data_dir / "labels"
    split_dir = data_dir / "splits"
    label_dir.mkdir(exist_ok=True)
    split_dir.mkdir(exist_ok=True)

    invalid = []
    box_counts = Counter()
    for frame, boxes in annotations.items():
        lines = []
        for box in boxes:
            values = [box[k] for k in ("x_center", "y_center", "width", "height")]
            if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
                invalid.append((frame, box))
                continue
            box_counts[box["label"]] += 1
            lines.append(f"{class_ids[box['label']]} " + " ".join(f"{value:.8f}" for value in values))
        (label_dir / f"{frame}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    if invalid:
        raise SystemExit(f"Found {len(invalid)} invalid boxes; first: {invalid[0]}")

    split_counts = Counter()
    all_images = sorted(image_dir.glob("*.png"))
    split_paths = {}
    for split, video_ids in DEFAULT_SPLITS.items():
        paths = []
        for image in all_images:
            match = VIDEO_RE.match(image.stem)
            if match and int(match.group(1)) in video_ids:
                paths.append(str(PurePosixPath(args.container_data_dir) / "images" / image.name))
        (split_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
        split_paths[split] = paths
        split_counts[split] = len(paths)

    yaml_lines = [
        f"path: {args.container_data_dir}",
        "train: splits/train.txt",
        "val: splits/val.txt",
        "test: splits/test.txt",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(classes)],
        "",
    ]
    (data_dir / "dataset.yaml").write_text("\n".join(yaml_lines))

    if args.train_fraction < 1.0:
        rng = random.Random(args.seed)
        sampled = []
        for video_id in sorted(DEFAULT_SPLITS["train"]):
            marker = f"/video{video_id}_"
            video_paths = [path for path in split_paths["train"] if marker in path]
            keep = max(1, round(len(video_paths) * args.train_fraction))
            sampled.extend(rng.sample(video_paths, keep))
        sampled.sort()
        (split_dir / "train_sampled.txt").write_text("\n".join(sampled) + "\n")
        small_yaml = yaml_lines.copy()
        small_yaml[1] = "train: splits/train_sampled.txt"
        (data_dir / "dataset-small.yaml").write_text("\n".join(small_yaml))
        print(f"Sampled training images: {len(sampled)} ({args.train_fraction:.1%}, seed={args.seed})")
        print(f"Wrote {data_dir / 'dataset-small.yaml'}")
    print(f"Images by split: {dict(split_counts)}")
    print(f"Boxes by class: {dict(box_counts)}")
    print(f"Classes: {class_ids}")
    print(f"Wrote {data_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
