#!/usr/bin/env python
"""Create a ball training dataset whose test split is the real CVAT case set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base ball dataset train/val + gerçek-case CVAT test splitini birleştirir."
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/master_v3_augmented",
    )
    parser.add_argument(
        "--real-test",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/real_case_test_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/master_v8_realcase",
    )
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Çıktı zaten var: {output}. --overwrite kullanın.")
        allowed = (PROJECT_ROOT / "datasets/BallOnly").resolve()
        if allowed not in output.resolve().parents:
            raise ValueError(f"Güvenlik için proje dışı çıktı silinmedi: {output}")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path, mode: str) -> str:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def copy_split(
    source_dataset: Path,
    source_split: str,
    output: Path,
    target_split: str,
    copy_mode: str,
    prefix: str,
) -> dict[str, int]:
    source_images = source_dataset / "images" / source_split
    source_labels = source_dataset / "labels" / source_split
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Split eksik: {source_dataset} {source_split}")
    images = sorted(
        path for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    boxes = 0
    positives = 0
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Label bulunamadı: {label_path}")
        target_stem = f"{prefix}__{image_path.stem}"
        target_image = output / "images" / target_split / f"{target_stem}{image_path.suffix.lower()}"
        target_label = output / "labels" / target_split / f"{target_stem}.txt"
        link_or_copy(image_path, target_image, copy_mode)
        shutil.copy2(label_path, target_label)
        label_boxes = sum(1 for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.strip())
        boxes += label_boxes
        positives += int(label_boxes > 0)
    return {"images": len(images), "boxes": boxes, "positive_images": positives}


def write_data_yaml(output: Path) -> None:
    data_yaml = (
        f'path: "{output.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- ball\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_dataset = resolve(args.base_dataset)
    real_test = resolve(args.real_test)
    output = resolve(args.output)
    if not (base_dataset / "data.yaml").is_file():
        raise FileNotFoundError(f"Base dataset data.yaml bulunamadı: {base_dataset / 'data.yaml'}")
    if not (real_test / "data.yaml").is_file():
        raise FileNotFoundError(f"Real test data.yaml bulunamadı: {real_test / 'data.yaml'}")

    reset_output(output, args.overwrite)
    split_summary = {
        "train": copy_split(base_dataset, "train", output, "train", args.copy_mode, "train"),
        "val": copy_split(base_dataset, "val", output, "val", args.copy_mode, "val"),
        "test": copy_split(real_test, "test", output, "test", args.copy_mode, "realcase"),
    }
    write_data_yaml(output)
    summary = {
        "base_dataset": str(base_dataset),
        "real_test": str(real_test),
        "output": str(output),
        "copy_mode": args.copy_mode,
        "splits": split_summary,
    }
    (output / "realcase_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Ball v8 real-case dataset hazır: {output}")
    print(f"data.yaml: {output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
