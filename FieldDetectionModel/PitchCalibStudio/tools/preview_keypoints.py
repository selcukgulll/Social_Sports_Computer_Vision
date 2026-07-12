from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_utils import (  # noqa: E402
    list_image_label_pairs,
    load_yaml,
    normalize_names,
    parse_pose_row,
    slugify,
)
from src.visualization import (  # noqa: E402
    draw_keypoint_prediction,
    make_contact_sheet,
    write_html_gallery,
)


def _denormalize_annotation(
    annotation: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, box_width, box_height = annotation["bbox"]
    bbox = np.asarray(
        [
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        ],
        dtype=float,
    )
    points = np.asarray(annotation["keypoints"], dtype=float)
    xy = np.column_stack((points[:, 0] * width, points[:, 1] * height))
    confidence = points[:, 2] / 2.0 if points.shape[1] == 3 else np.ones(len(points))
    return bbox, xy, confidence


def generate_previews(
    data_yaml: str | Path,
    split: str,
    count: int,
    output: str | Path,
    seed: int = 42,
) -> list[Path]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_yaml(data_yaml)
    kpt_shape = data.get("kpt_shape")
    if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
        raise ValueError("Bu dataset geçerli kpt_shape içermiyor; YOLO pose değil.")
    keypoint_count, keypoint_dims = (int(kpt_shape[0]), int(kpt_shape[1]))
    names = normalize_names(data.get("names"))
    pairs = [
        pair
        for pair in list_image_label_pairs(data_yaml, split)
        if pair.label.exists() and pair.label.stat().st_size > 0
    ]
    if not pairs:
        raise ValueError(f"{split} splitinde etiketli görsel bulunamadı.")
    random.Random(seed).shuffle(pairs)
    selected = pairs[: min(max(1, count), len(pairs))]
    saved: list[Path] = []
    for order, pair in enumerate(selected, start=1):
        image = cv2.imread(str(pair.image))
        if image is None:
            continue
        height, width = image.shape[:2]
        canvas = image.copy()
        valid_objects = 0
        for line in pair.label.read_text(encoding="utf-8", errors="replace").splitlines():
            annotation, errors = parse_pose_row(line, keypoint_count, keypoint_dims)
            if annotation is None or errors:
                continue
            bbox, xy, confidence = _denormalize_annotation(annotation, width, height)
            canvas = draw_keypoint_prediction(
                canvas,
                bbox,
                xy,
                confidence,
                schema=None,
                confidence_threshold=0.0,
                class_name=names.get(annotation["class_id"], str(annotation["class_id"])),
                large_indices=True,
            )
            valid_objects += 1
        if not valid_objects:
            continue
        target = output / f"{order:03d}_{slugify(pair.image.stem)}.jpg"
        cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(target)
    if saved:
        write_html_gallery(saved, output / "index.html")
        make_contact_sheet(saved[: min(12, len(saved))], output / "contact_sheet.jpg")
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO pose etiketlerini büyük keypoint indeksleriyle önizler."
    )
    parser.add_argument("--data", required=True, help="Dataset data.yaml yolu")
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "valid", "test"]
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    saved = generate_previews(args.data, args.split, args.count, args.output, args.seed)
    print(f"{len(saved)} önizleme kaydedildi: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
