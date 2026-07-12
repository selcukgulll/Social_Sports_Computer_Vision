#!/usr/bin/env python
"""Ayrı train ve test CVAT kaynaklarından ball eğitim dataseti oluşturur."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_SCRIPT = PROJECT_ROOT / "scripts/17_import_cvat_ball_test_set.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_cvat_import_module():
    spec = importlib.util.spec_from_file_location("cvat_ball_import", IMPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"CVAT import scripti yüklenemedi: {IMPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cvat_ball_import"] = module
    spec.loader.exec_module(module)
    return module


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/validation için bir CVAT kaynağı, test için ayrı bir CVAT kaynağı "
            "kullanarak ball dataseti oluşturur."
        )
    )
    parser.add_argument("--train-labels-source", type=Path, required=True)
    parser.add_argument("--train-images-source", type=Path, required=True)
    parser.add_argument("--test-labels-source", type=Path, required=True)
    parser.add_argument("--test-images-source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/manuel_dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
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


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def train_val_split(stem: str, seed: int, val_ratio: float) -> str:
    ratio = int(hashlib.sha1(f"{seed}|{stem}".encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    return "val" if ratio < val_ratio else "train"


def ordered_stems(
    cvat,
    labels_source: Path,
    images: dict[str, Path],
    labels: dict[str, Path],
    max_images: int,
) -> list[str]:
    image_list_path = cvat.default_image_list(labels_source)
    if image_list_path is not None and image_list_path.is_file():
        stems = cvat.read_image_list(image_list_path, max_images)
        return [stem for stem in stems if stem in images and stem in labels]
    stems = sorted(set(labels) & set(images))
    if max_images > 0:
        stems = stems[:max_images]
    return stems


def import_rows(cvat, label_path: Path, ball_ids: set[int]) -> tuple[list[str], int]:
    rows: list[str] = []
    invalid = 0
    for raw_line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        row = cvat.parse_row(raw_line, ball_ids)
        if row is None:
            invalid += 1
        else:
            rows.append(row)
    return rows, invalid


def write_pair(
    cvat,
    output: Path,
    split: str,
    stem: str,
    image_path: Path,
    rows: list[str],
    prefix: str,
    copy_mode: str,
) -> dict[str, object]:
    target_name = cvat.safe_name(f"{prefix}__{stem}", image_path.suffix)
    target_stem = Path(target_name).stem
    target_image = output / "images" / split / target_name
    target_label = output / "labels" / split / f"{target_stem}.txt"
    link_or_copy(image_path, target_image, copy_mode)
    target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return {
        "split": split,
        "source_image": str(image_path),
        "target_image": str(target_image),
        "target_label": str(target_label),
        "ball_boxes": len(rows),
    }


def split_summary(output: Path, split: str) -> dict[str, int]:
    images = [
        path
        for path in sorted((output / "images" / split).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    boxes = 0
    positives = 0
    for image_path in images:
        label_path = output / "labels" / split / f"{image_path.stem}.txt"
        count = sum(
            1
            for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            if line.strip()
        )
        boxes += count
        positives += int(count > 0)
    return {"images": len(images), "boxes": boxes, "positive_images": positives}


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


def import_source(
    cvat,
    *,
    labels_source: Path,
    images_source: Path,
    output: Path,
    ball_ids: set[int],
    prefix: str,
    copy_mode: str,
    max_images: int,
    split_mode: str,
    seed: int,
    val_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    images = cvat.find_images_from_roots([images_source])
    labels = cvat.find_labels(labels_source)
    stems = ordered_stems(cvat, labels_source, images, labels, max_images)
    manifest: list[dict[str, object]] = []
    stats = {
        "requested": len(stems),
        "imported": 0,
        "missing_images": 0,
        "missing_labels": 0,
        "invalid_rows": 0,
    }

    for index, stem in enumerate(stems, start=1):
        image_path = images.get(stem)
        label_path = labels.get(stem)
        if image_path is None:
            stats["missing_images"] += 1
            continue
        if label_path is None:
            stats["missing_labels"] += 1
            continue

        rows, invalid = import_rows(cvat, label_path, ball_ids)
        stats["invalid_rows"] += invalid
        if split_mode == "train_val":
            split = train_val_split(stem, seed, val_ratio)
        else:
            split = "test"

        manifest.append(
            write_pair(
                cvat,
                output,
                split,
                stem,
                image_path,
                rows,
                prefix,
                copy_mode,
            )
        )
        stats["imported"] += 1
        if index % 100 == 0 or index == len(stems):
            print(f"{prefix} kareleri hazirlaniyor: {index}/{len(stems)}", flush=True)

    return manifest, stats


def main() -> int:
    args = parse_args()
    cvat = load_cvat_import_module()
    train_labels = resolve(args.train_labels_source)
    train_images = resolve(args.train_images_source)
    test_labels = resolve(args.test_labels_source)
    test_images = resolve(args.test_images_source)
    output = resolve(args.output)

    for name, path in (
        ("train labels", train_labels),
        ("train images", train_images),
        ("test labels", test_labels),
        ("test images", test_images),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} klasoru bulunamadi: {path}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio 0 ile 1 arasinda olmali.")
    allowed_parent = (PROJECT_ROOT / "datasets/BallOnly").resolve()
    if allowed_parent not in output.parents:
        raise ValueError(f"Cikti guvenli BallOnly alani disinda: {output}")

    train_class_names = cvat.find_class_names(train_labels)
    test_class_names = cvat.find_class_names(test_labels)
    train_ball_ids = cvat.ball_source_ids(
        train_class_names,
        list(cvat.find_labels(train_labels).values()),
    )
    test_ball_ids = cvat.ball_source_ids(
        test_class_names,
        list(cvat.find_labels(test_labels).values()),
    )

    reset_output(output, args.overwrite)
    train_manifest, train_stats = import_source(
        cvat,
        labels_source=train_labels,
        images_source=train_images,
        output=output,
        ball_ids=train_ball_ids,
        prefix="traincase",
        copy_mode=args.copy_mode,
        max_images=int(args.max_train_images),
        split_mode="train_val",
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    test_manifest, test_stats = import_source(
        cvat,
        labels_source=test_labels,
        images_source=test_images,
        output=output,
        ball_ids=test_ball_ids,
        prefix="realcase",
        copy_mode=args.copy_mode,
        max_images=int(args.max_test_images),
        split_mode="test",
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if not train_manifest:
        raise RuntimeError("Train kaynagindan hic kare aktarilamadi.")
    if not test_manifest:
        raise RuntimeError("Test kaynagindan hic kare aktarilamadi.")

    write_data_yaml(output)
    summaries = {split: split_summary(output, split) for split in ("train", "val", "test")}
    summary = {
        "output": str(output),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "copy_mode": args.copy_mode,
        "train_source": {
            "labels": str(train_labels),
            "images": str(train_images),
            "class_names": train_class_names,
            "ball_source_ids": sorted(train_ball_ids),
            **train_stats,
        },
        "test_source": {
            "labels": str(test_labels),
            "images": str(test_images),
            "class_names": test_class_names,
            "ball_source_ids": sorted(test_ball_ids),
            **test_stats,
        },
        "splits": summaries,
        "manifest_counts": {
            "train_source": len(train_manifest),
            "test_source": len(test_manifest),
        },
    }
    (output / "mixed_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("", flush=True)
    print("BALL MIXED DATASET HAZIR", flush=True)
    for split in ("train", "val", "test"):
        values = summaries[split]
        print(
            f"{split:>5}: {values['images']:,} gorsel, "
            f"{values['boxes']:,} top kutusu, "
            f"{values['positive_images']:,} pozitif",
            flush=True,
        )
    print(f"data.yaml: {output / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
