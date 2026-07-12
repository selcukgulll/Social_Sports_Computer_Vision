#!/usr/bin/env python
"""Build a ball-only YOLO dataset from separate image and CVAT label folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    box_count: int

    @property
    def positive(self) -> bool:
        return self.box_count > 0


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ayri image/CVAT label kaynaklarindan ball train/val/test dataseti kurar."
    )
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_label_dir(root: Path) -> Path:
    candidates = (root / "labels" / "train", root / "labels", root)
    ranked = [(len(list(candidate.glob("*.txt"))), candidate) for candidate in candidates if candidate.is_dir()]
    ranked = [(count, candidate) for count, candidate in ranked if count > 0]
    if not ranked:
        raise FileNotFoundError(f"YOLO label .txt dosyalari bulunamadi: {root}")
    return max(ranked, key=lambda item: item[0])[1]


def validate_label(path: Path) -> int:
    count = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Gecersiz YOLO label ({path}:{line_number}): {line}")
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Sayisal olmayan label ({path}:{line_number}): {line}") from exc
        if class_id != 0:
            raise ValueError(f"Ball disinda class bulundu ({path}:{line_number}): {class_id}")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"BBox merkezi aralik disinda ({path}:{line_number}): {line}")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"BBox boyutu aralik disinda ({path}:{line_number}): {line}")
        count += 1
    return count


def collect_samples(images_dir: Path, labels_root: Path) -> tuple[list[Sample], Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image klasoru bulunamadi: {images_dir}")
    label_dir = find_label_dir(labels_root)
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"Image klasoru bos: {images_dir}")

    image_stems = {image.stem for image in images}
    label_paths = {path.stem: path for path in label_dir.glob("*.txt")}
    orphan_labels = sorted(set(label_paths) - image_stems)
    if orphan_labels:
        raise ValueError(
            f"Gorseli bulunmayan label var: {label_dir / (orphan_labels[0] + '.txt')}"
        )

    samples: list[Sample] = []
    for image in images:
        label = label_paths.get(image.stem)
        samples.append(Sample(image, label, validate_label(label) if label else 0))
    return samples, label_dir


def stratified_split(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("Validation orani 0 ile 0.5 arasinda olmalidir.")
    rng = random.Random(seed)
    groups = ([sample for sample in samples if sample.positive], [sample for sample in samples if not sample.positive])
    train: list[Sample] = []
    val: list[Sample] = []
    for group in groups:
        rng.shuffle(group)
        val_count = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    return sorted(train, key=lambda item: item.image.name), sorted(val, key=lambda item: item.image.name)


def reset_output(output: Path, overwrite: bool) -> None:
    allowed_parent = (PROJECT_ROOT / "datasets" / "BallOnly").resolve()
    if allowed_parent not in output.parents:
        raise ValueError(f"Cikti BallOnly klasoru altinda olmalidir: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Cikti zaten var; --overwrite kullanin: {output}")
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


def write_split(
    samples: list[Sample],
    output: Path,
    split: str,
    prefix: str,
    copy_mode: str,
    manifest: list[dict[str, object]],
) -> dict[str, int]:
    boxes = 0
    positives = 0
    negatives = 0
    for index, sample in enumerate(samples, start=1):
        target_stem = f"{prefix}__{sample.image.stem}"
        target_image = output / "images" / split / f"{target_stem}{sample.image.suffix.lower()}"
        target_label = output / "labels" / split / f"{target_stem}.txt"
        actual_mode = link_or_copy(sample.image, target_image, copy_mode)
        if sample.label:
            shutil.copy2(sample.label, target_label)
        else:
            target_label.write_text("", encoding="utf-8")
        boxes += sample.box_count
        positives += int(sample.positive)
        negatives += int(not sample.positive)
        manifest.append(
            {
                "split": split,
                "source_image": str(sample.image),
                "source_label": str(sample.label or ""),
                "target_image": str(target_image),
                "boxes": sample.box_count,
                "negative": not sample.positive,
                "copy_mode": actual_mode,
            }
        )
        if index % 100 == 0 or index == len(samples):
            print(f"{split}: {index}/{len(samples)}", flush=True)
    return {
        "images": len(samples),
        "boxes": boxes,
        "positive_images": positives,
        "negative_images": negatives,
    }


def write_data_yaml(output: Path) -> None:
    content = (
        f'path: "{output.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- ball\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    train_images = resolve(args.train_images)
    train_labels = resolve(args.train_labels)
    test_images = resolve(args.test_images)
    test_labels = resolve(args.test_labels)
    output = resolve(args.output)

    print("BALL CUSTOM DATASET HAZIRLAMA", flush=True)
    print(f"Train images: {train_images}", flush=True)
    print(f"Train labels: {train_labels}", flush=True)
    print(f"Test images : {test_images}", flush=True)
    print(f"Test labels : {test_labels}", flush=True)
    print(f"Output      : {output}", flush=True)

    train_samples, resolved_train_labels = collect_samples(train_images, train_labels)
    test_samples, resolved_test_labels = collect_samples(test_images, test_labels)
    train_split, val_split = stratified_split(train_samples, args.val_ratio, args.seed)
    print(
        f"Kaynaklar dogrulandi: train-source={len(train_samples)}, test={len(test_samples)}",
        flush=True,
    )
    print(f"Label klasorleri: {resolved_train_labels} | {resolved_test_labels}", flush=True)
    print(
        f"Split: train={len(train_split)}, val={len(val_split)}, test={len(test_samples)}",
        flush=True,
    )

    reset_output(output, args.overwrite)
    manifest: list[dict[str, object]] = []
    summaries = {
        "train": write_split(train_split, output, "train", "newtrain", args.copy_mode, manifest),
        "val": write_split(val_split, output, "val", "newval", args.copy_mode, manifest),
        "test": write_split(test_samples, output, "test", "realtest", args.copy_mode, manifest),
    }
    write_data_yaml(output)
    summary = {
        "train_images_source": str(train_images),
        "train_labels_source": str(train_labels),
        "test_images_source": str(test_images),
        "test_labels_source": str(test_labels),
        "output": str(output),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "missing_labels_are_negative": True,
        "augmentation_ran": False,
        "splits": summaries,
    }
    (output / "custom_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print("", flush=True)
    print("DATASET HAZIR", flush=True)
    for split in ("train", "val", "test"):
        values = summaries[split]
        print(
            f"{split:>5}: {values['images']} image, {values['boxes']} box, "
            f"{values['positive_images']} pozitif, {values['negative_images']} negatif",
            flush=True,
        )
    print(f"data.yaml: {output / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
