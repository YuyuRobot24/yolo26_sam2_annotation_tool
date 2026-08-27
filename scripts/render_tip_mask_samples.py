#!/usr/bin/env python3
"""Apply accepted tip masks and render enlarged ROI review panels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("selected_and_masked/video1_tip_high_quality_50"),
    )
    parser.add_argument("--roi-scale", type=float, default=2.5)
    return parser.parse_args()


def crop_bounds(box, width, height, scale):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * scale
    return (
        max(0, round(cx - size / 2)), max(0, round(cy - size / 2)),
        min(width, round(cx + size / 2)), min(height, round(cy + size / 2)),
    )


def contact_sheets(paths, output, columns=5, rows=2, cell_width=720, cell_height=360):
    for sheet, start in enumerate(range(0, len(paths), columns * rows), start=1):
        panels = []
        for path in paths[start : start + columns * rows]:
            panel = cv2.imread(str(path))
            factor = min(cell_width / panel.shape[1], cell_height / panel.shape[0])
            resized = cv2.resize(panel, (round(panel.shape[1] * factor), round(panel.shape[0] * factor)))
            cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            y = (cell_height - resized.shape[0]) // 2
            x = (cell_width - resized.shape[1]) // 2
            cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            panels.append(cell)
        canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
        for index, panel in enumerate(panels):
            row, column = divmod(index, columns)
            canvas[row * cell_height : (row + 1) * cell_height,
                   column * cell_width : (column + 1) * cell_width] = panel
        cv2.imwrite(str(output / f"masked_contact_sheet_{sheet:02d}.jpg"), canvas)


def main():
    args = parse_args()
    root = args.input
    metadata = json.loads((root / "tip_annotations.json").read_text())
    masked_dir = root / "masked_images"
    roi_dir = root / "roi_mask_samples"
    sheets_dir = root / "masked_contact_sheets"
    for directory in (masked_dir, roi_dir, sheets_dir):
        directory.mkdir(exist_ok=True)

    panels = []
    for item in metadata["accepted_images"]:
        frame = item["frame"]
        image = cv2.imread(item["image"])
        labels = cv2.imread(str(root / "tip_masks" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        foreground = labels > 0
        cutout = np.zeros_like(image)
        cutout[foreground] = image[foreground]
        cv2.imwrite(str(masked_dir / f"{frame}.png"), cutout)

        detection = item["instances"][0]
        x1, y1, x2, y2 = crop_bounds(
            detection["box_xyxy"], image.shape[1], image.shape[0], args.roi_scale
        )
        original_roi = image[y1:y2, x1:x2]
        mask_roi = foreground[y1:y2, x1:x2]
        overlay_roi = original_roi.copy()
        magenta = np.array([255, 0, 255], dtype=np.uint8)
        overlay_roi[mask_roi] = (0.4 * overlay_roi[mask_roi] + 0.6 * magenta).astype(np.uint8)
        cutout_roi = np.zeros_like(original_roi)
        cutout_roi[mask_roi] = original_roi[mask_roi]

        target_height = 300
        tiles = []
        for tile in (original_roi, overlay_roi, cutout_roi):
            factor = target_height / tile.shape[0]
            tiles.append(cv2.resize(tile, (round(tile.shape[1] * factor), target_height)))
        panel = np.hstack(tiles)
        title = (f"{frame} {detection['class']}  YOLO={detection['confidence']:.3f}  "
                 f"SAM2={detection['sam2_score']:.3f} | ORIGINAL / MASK / CUTOUT")
        header = np.zeros((45, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, title, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 2)
        panel = np.vstack((header, panel))
        path = roi_dir / f"{frame}.jpg"
        cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
        panels.append(path)

    contact_sheets(panels, sheets_dir)
    print(f"Applied masks to {len(panels)} images and saved ROI panels under {root}")


if __name__ == "__main__":
    main()
