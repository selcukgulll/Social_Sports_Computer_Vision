#!/usr/bin/env python
"""Build BallOnly v3 dataset with hard-case augmentations.

This script does not modify the source dataset.  It creates a new YOLO dataset
where train images contain the original samples plus generated variants for:

- small/far/low-resolution ball appearance
- motion blur
- low-light + low-contrast scenes
- a combined hard case

All default augmentations are geometry-preserving, so YOLO labels are copied
without changing coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["ball"]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    split: str
    has_ball: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BallOnly master_v1 üzerine küçük/top-blur/az ışık augmentasyonlu v3 dataset üretir."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/master_v1",
        help="Kaynak YOLO dataset kökü.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/master_v3_augmented",
        help="Yeni dataset çıktı klasörü.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Orijinal görseller hardlink ile bağlanabilir; augmented görseller yeniden yazılır.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Augmented JPG kalite değeri.",
    )
    parser.add_argument(
        "--negative-augment-ratio",
        type=float,
        default=0.25,
        help="Top olmayan train karelerinin ne kadarı low-light negatif olarak çoğaltılsın.",
    )
    parser.add_argument(
        "--combo-ratio",
        type=float,
        default=0.50,
        help="Pozitif train karelerinin ne kadarına ayrıca hard_combo eklensin.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Smoke test için train image sayısını sınırlar; 0=tümü.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_remove_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Çıktı klasörü zaten var: {output}\n"
            "Yeniden oluşturmak için --overwrite kullanın."
        )
    resolved_output = output.resolve()
    allowed_parent = (PROJECT_ROOT / "datasets/BallOnly").resolve()
    if allowed_parent not in resolved_output.parents:
        raise ValueError(f"Güvenlik için bu çıktı silinmedi: {resolved_output}")
    shutil.rmtree(output)


def read_label_has_ball(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    try:
        for raw_line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            try:
                if int(float(parts[0])) == 0:
                    return True
            except ValueError:
                continue
    except OSError:
        return False
    return False


def collect_samples(dataset: Path, max_train_images: int = 0) -> dict[str, list[Sample]]:
    samples: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Dataset split eksik: {split}")
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if split == "train" and max_train_images > 0:
            image_paths = image_paths[:max_train_images]
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Label bulunamadı: {label_path}")
            samples[split].append(
                Sample(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                    has_ball=read_label_has_ball(label_path),
                )
            )
    return samples


def ensure_dirs(output: Path) -> None:
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


def copy_label(source: Path, target: Path) -> None:
    shutil.copy2(source, target)


def write_image(path: Path, image: np.ndarray, quality: int) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        quality = min(100, max(1, int(quality)))
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Görsel yazılamadı: {path}")
    path.write_bytes(encoded.tobytes())


def downscale_upscale(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate a small/far ball by removing fine pixel detail."""
    height, width = image.shape[:2]
    factor = rng.uniform(0.28, 0.55)
    small_w = max(16, int(width * factor))
    small_h = max(16, int(height * factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    sharpen_weight = rng.uniform(0.0, 0.25)
    if sharpen_weight > 0:
        blurred = cv2.GaussianBlur(restored, (0, 0), 1.0)
        restored = cv2.addWeighted(restored, 1.0 + sharpen_weight, blurred, -sharpen_weight, 0)
    return restored


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply directional motion blur like a fast ball/camera movement."""
    kernel_size = rng.choice([7, 9, 11, 13, 15, 17, 21])
    angle = rng.uniform(0.0, 180.0)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        return image
    kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)


def low_light_low_contrast(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate dark/low-contrast corners of the pitch."""
    working = image.astype(np.float32) / 255.0
    gamma = rng.uniform(1.35, 2.40)
    contrast = rng.uniform(0.45, 0.78)
    brightness = rng.uniform(-0.12, 0.02)
    working = np.power(np.clip(working, 0.0, 1.0), gamma)
    mean = working.mean(axis=(0, 1), keepdims=True)
    working = (working - mean) * contrast + mean + brightness

    if rng.random() < 0.85:
        height, width = working.shape[:2]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        cx = rng.uniform(-0.35, 0.35)
        cy = rng.uniform(-0.35, 0.35)
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        vignette = 1.0 - rng.uniform(0.18, 0.42) * np.clip(radius, 0.0, 1.2)
        working *= vignette[..., None]

    if rng.random() < 0.70:
        noise_sigma = rng.uniform(0.008, 0.030)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, working.shape)
        working += noise.astype(np.float32)

    return np.clip(working * 255.0, 0, 255).astype(np.uint8)


def hard_combo(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small + moving + dark.  This is intentionally harsh."""
    image = downscale_upscale(image, rng)
    if rng.random() < 0.85:
        image = motion_blur(image, rng)
    image = low_light_low_contrast(image, rng)
    return image


def output_name(sample: Sample, suffix: str) -> str:
    return f"{sample.image_path.stem}__{suffix}{sample.image_path.suffix.lower()}"


def add_original(sample: Sample, output: Path, copy_mode: str, manifest: list[dict[str, object]]) -> None:
    target_image = output / "images" / sample.split / sample.image_path.name
    target_label = output / "labels" / sample.split / f"{sample.image_path.stem}.txt"
    mode = link_or_copy(sample.image_path, target_image, copy_mode)
    copy_label(sample.label_path, target_label)
    manifest.append(
        {
            "split": sample.split,
            "kind": "original",
            "image": str(target_image),
            "label": str(target_label),
            "source_image": str(sample.image_path),
            "has_ball": int(sample.has_ball),
            "copy_mode": mode,
        }
    )


def add_augmented(
    sample: Sample,
    output: Path,
    suffix: str,
    transform: Callable[[np.ndarray, random.Random], np.ndarray],
    rng: random.Random,
    quality: int,
    manifest: list[dict[str, object]],
) -> None:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Görsel okunamadı: {sample.image_path}")
    augmented = transform(image, rng)
    target_image_name = output_name(sample, suffix)
    target_stem = Path(target_image_name).stem
    target_image = output / "images" / "train" / target_image_name
    target_label = output / "labels" / "train" / f"{target_stem}.txt"
    write_image(target_image, augmented, quality)
    copy_label(sample.label_path, target_label)
    manifest.append(
        {
            "split": "train",
            "kind": suffix,
            "image": str(target_image),
            "label": str(target_label),
            "source_image": str(sample.image_path),
            "has_ball": int(sample.has_ball),
            "copy_mode": "generated",
        }
    )


def write_data_yaml(output: Path) -> None:
    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- ball\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")


def write_manifest(output: Path, manifest: list[dict[str, object]], summary: dict[str, object]) -> None:
    with (output / "augmentation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    dataset = resolve(args.dataset)
    output = resolve(args.output)
    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(f"Kaynak data.yaml bulunamadı: {dataset / 'data.yaml'}")
    if not 0.0 <= args.negative_augment_ratio <= 1.0:
        raise ValueError("--negative-augment-ratio 0 ile 1 arasında olmalı.")
    if not 0.0 <= args.combo_ratio <= 1.0:
        raise ValueError("--combo-ratio 0 ile 1 arasında olmalı.")

    rng = random.Random(args.seed)
    safe_remove_output(output, args.overwrite)
    ensure_dirs(output)

    samples_by_split = collect_samples(dataset, int(args.max_train_images))
    manifest: list[dict[str, object]] = []
    counters = {
        "original_train": 0,
        "original_val": 0,
        "original_test": 0,
        "aug_small_lowres": 0,
        "aug_motion_blur": 0,
        "aug_low_light": 0,
        "aug_hard_combo": 0,
        "aug_negative_low_light": 0,
    }

    for split in ("train", "val", "test"):
        for sample in samples_by_split[split]:
            add_original(sample, output, args.copy_mode, manifest)
            counters[f"original_{split}"] += 1

    train_samples = samples_by_split["train"]
    positive_samples = [sample for sample in train_samples if sample.has_ball]
    negative_samples = [sample for sample in train_samples if not sample.has_ball]
    combo_ids = {
        sample.image_path.name
        for sample in rng.sample(
            positive_samples,
            k=int(round(len(positive_samples) * float(args.combo_ratio))),
        )
    } if positive_samples else set()
    negative_ids = {
        sample.image_path.name
        for sample in rng.sample(
            negative_samples,
            k=int(round(len(negative_samples) * float(args.negative_augment_ratio))),
        )
    } if negative_samples else set()

    for index, sample in enumerate(positive_samples, start=1):
        add_augmented(sample, output, "aug_small_lowres", downscale_upscale, rng, args.jpeg_quality, manifest)
        counters["aug_small_lowres"] += 1
        add_augmented(sample, output, "aug_motion_blur", motion_blur, rng, args.jpeg_quality, manifest)
        counters["aug_motion_blur"] += 1
        add_augmented(sample, output, "aug_low_light", low_light_low_contrast, rng, args.jpeg_quality, manifest)
        counters["aug_low_light"] += 1
        if sample.image_path.name in combo_ids:
            add_augmented(sample, output, "aug_hard_combo", hard_combo, rng, args.jpeg_quality, manifest)
            counters["aug_hard_combo"] += 1
        if index % 250 == 0:
            print(f"Pozitif augmentasyon ilerleme: {index}/{len(positive_samples)}")

    for index, sample in enumerate(negative_samples, start=1):
        if sample.image_path.name not in negative_ids:
            continue
        add_augmented(
            sample,
            output,
            "aug_negative_low_light",
            low_light_low_contrast,
            rng,
            args.jpeg_quality,
            manifest,
        )
        counters["aug_negative_low_light"] += 1
        if index % 500 == 0:
            print(f"Negatif augmentasyon ilerleme: {index}/{len(negative_samples)}")

    write_data_yaml(output)
    image_counts = {}
    label_counts = {}
    for split in ("train", "val", "test"):
        image_counts[split] = len(list((output / "images" / split).glob("*")))
        label_counts[split] = len(list((output / "labels" / split).glob("*.txt")))
    summary = {
        "source_dataset": str(dataset),
        "output": str(output),
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "image_counts": image_counts,
        "label_counts": label_counts,
        "positive_train_images": len(positive_samples),
        "negative_train_images": len(negative_samples),
        "negative_augment_ratio": args.negative_augment_ratio,
        "combo_ratio": args.combo_ratio,
        **counters,
    }
    write_manifest(output, manifest, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Ball v3 augmented dataset hazır: {output}")
    print(f"data.yaml: {output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
