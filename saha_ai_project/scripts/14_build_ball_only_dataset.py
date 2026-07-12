#!/usr/bin/env python
"""Birden fazla Roboflow YOLO kaynağını tek sınıflı ball datasetine dönüştürür."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BALL_NAMES = {"ball", "football", "soccer ball", "top"}
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO detection/segmentation kaynaklarını yalnız ball sınıfıyla birleştir."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Aynı diskte hardlink hızlıdır ve ek disk alanı kullanmaz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Çıktı klasörü doluysa güvenli şekilde silip yeniden oluştur.",
    )
    return parser.parse_args()


def class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def safe_slug(value: str, limit: int = 30) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (normalized or "dataset")[:limit]


def source_group(stem: str) -> str:
    """Roboflow augmentasyonlarını aynı split içinde tutacak temel adı döndür."""
    return stem.split(".rf.", 1)[0]


def deterministic_split(dataset: str, group: str, seed: int) -> str:
    value = f"{seed}|{dataset}|{group}".encode("utf-8")
    ratio = int(hashlib.sha1(value).hexdigest()[:12], 16) / float(16**12)
    if ratio < 0.80:
        return "train"
    if ratio < 0.90:
        return "val"
    return "test"


def convert_ball_rows(
    label_path: Path,
    ball_ids: set[int],
) -> tuple[list[str], int, int, int]:
    """Detection kutusunu korur; segmentasyon polygonunu bbox'a dönüştürür."""
    output: list[str] = []
    detection_rows = 0
    polygon_rows = 0
    rejected_rows = 0
    if not label_path.is_file():
        return output, detection_rows, polygon_rows, rejected_rows

    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(item) for item in parts[1:]]
        except ValueError:
            rejected_rows += 1
            continue
        if class_id not in ball_ids:
            continue

        if len(values) == 4:
            x, y, width, height = values
            detection_rows += 1
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            polygon_rows += 1
        else:
            rejected_rows += 1
            continue

        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        width = min(1.0, max(0.0, width))
        height = min(1.0, max(0.0, height))
        if width <= 0.0 or height <= 0.0:
            rejected_rows += 1
            continue
        output.append(f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
    return output, detection_rows, polygon_rows, rejected_rows


def link_or_copy(source: Path, target: Path, mode: str) -> str:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: {source_root}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Çıktı klasörü dolu: {output}\n"
            "Hazır datasetin üzerine yazılmadı; yeni bir çıktı klasörü seçin."
        )
    if output.exists() and args.overwrite:
        project_root = Path(__file__).resolve().parents[1].resolve()
        resolved_output = output.resolve()
        if project_root not in resolved_output.parents:
            raise ValueError(f"Güvenlik için proje dışı çıktı silinmedi: {resolved_output}")
        shutil.rmtree(output)

    datasets = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "data.yaml").is_file()
    )
    if not datasets:
        raise FileNotFoundError(f"data.yaml içeren dataset bulunamadı: {source_root}")

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []

    for dataset in datasets:
        data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8-sig")) or {}
        names = class_names(data)
        ball_ids = {
            index
            for index, name in enumerate(names)
            if name.strip().lower() in BALL_NAMES
        }
        if not ball_ids:
            raise ValueError(f"Ball sınıfı bulunamadı: {dataset} names={names}")

        existing_source_splits = [
            split for split in SPLIT_MAP if (dataset / split / "images").is_dir()
        ]
        only_train = existing_source_splits == ["train"]
        slug = safe_slug(dataset.name)
        report: Counter[str] = Counter()

        for source_split in existing_source_splits:
            image_dir = dataset / source_split / "images"
            label_dir = dataset / source_split / "labels"
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target_split = (
                    deterministic_split(dataset.name, source_group(image_path.stem), args.seed)
                    if only_train
                    else SPLIT_MAP[source_split]
                )
                label_path = label_dir / f"{image_path.stem}.txt"
                rows, detections, polygons, rejected = convert_ball_rows(
                    label_path, ball_ids
                )
                identity = hashlib.sha1(
                    f"{dataset.name}|{source_split}|{image_path.name}".encode("utf-8")
                ).hexdigest()[:10]
                short_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", image_path.stem)[:75]
                target_name = f"{slug}__{short_stem}__{identity}{image_path.suffix.lower()}"
                target_image = output / "images" / target_split / target_name
                target_label = output / "labels" / target_split / f"{Path(target_name).stem}.txt"
                actual_copy_mode = link_or_copy(image_path, target_image, args.copy_mode)
                target_label.write_text(
                    "\n".join(rows) + ("\n" if rows else ""),
                    encoding="utf-8",
                )

                report["images"] += 1
                report[f"{target_split}_images"] += 1
                report["ball_boxes"] += len(rows)
                report["positive_images"] += int(bool(rows))
                report["negative_images"] += int(not rows)
                report["detection_rows"] += detections
                report["polygon_rows_converted"] += polygons
                report["rejected_rows"] += rejected
                manifest_rows.append(
                    {
                        "dataset": dataset.name,
                        "source_split": source_split,
                        "target_split": target_split,
                        "source_image": str(image_path),
                        "target_image": str(target_image),
                        "ball_boxes": len(rows),
                        "polygon_boxes": polygons,
                        "negative": int(not rows),
                        "copy_mode": actual_copy_mode,
                    }
                )

        dataset_reports.append(
            {
                "dataset": dataset.name,
                "class_names": names,
                "ball_source_ids": sorted(ball_ids),
                **dict(report),
            }
        )
    data_yaml = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["ball"],
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts = Counter(row["target_split"] for row in manifest_rows)
    summary = {
        "source_root": str(source_root),
        "output": str(output),
        "seed": args.seed,
        "copy_mode_requested": args.copy_mode,
        "class_names": ["ball"],
        "split_images": dict(sorted(split_counts.items())),
        "images": len(manifest_rows),
        "ball_boxes": sum(int(row["ball_boxes"]) for row in manifest_rows),
        "positive_images": sum(1 for row in manifest_rows if int(row["ball_boxes"]) > 0),
        "negative_images": sum(1 for row in manifest_rows if int(row["ball_boxes"]) == 0),
        "polygon_boxes_converted": sum(int(row["polygon_boxes"]) for row in manifest_rows),
        "datasets": dataset_reports,
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Ball-only dataset hazır: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
