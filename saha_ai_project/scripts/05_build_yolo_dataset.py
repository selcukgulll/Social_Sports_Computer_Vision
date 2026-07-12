#!/usr/bin/env python
"""Build a reproducible YOLO detection dataset from manually labeled images."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "player", 1: "ball", 2: "outside_person"}


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manuel YOLO etiketlerinden train/val/test veri kümesi oluşturur."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("labels/images"),
    )
    parser.add_argument("--labels-dir", type=Path, default=Path("labels/manual"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/dataset_v1"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Etiket dosyası olmayan görselleri boş etiketle negatif örnek olarak ekle.",
    )
    return parser.parse_args()


def validate_label(label_path: Path) -> tuple[dict[int, int], int]:
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    invalid_rows = 0
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        print(f"UYARI: Etiket okunamadı ({label_path}): {exc}")
        return counts, 1

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        problems: list[str] = []
        if len(parts) != 5:
            problems.append(f"5 değer bekleniyordu, {len(parts)} bulundu")
        else:
            try:
                class_value = float(parts[0])
                if not class_value.is_integer() or int(class_value) not in CLASS_NAMES:
                    problems.append("class id 0, 1 veya 2 olmalı")
                    class_id = None
                else:
                    class_id = int(class_value)
            except ValueError:
                problems.append("class id sayı değil")
                class_id = None

            try:
                coordinates = [float(value) for value in parts[1:]]
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    problems.append("normalize koordinatlar 0 ile 1 arasında olmalı")
            except ValueError:
                problems.append("koordinatlardan en az biri sayı değil")

            if not problems and class_id is not None:
                counts[class_id] += 1

        if problems:
            invalid_rows += 1
            print(
                f"UYARI: Geçersiz YOLO satırı: {label_path.name}:{line_number} - "
                + "; ".join(problems)
            )

    return counts, invalid_rows


def split_counts(total: int, ratios: list[float]) -> list[int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(raw_counts[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def reset_split_directories(out_dir: Path) -> dict[str, tuple[Path, Path]]:
    result: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        for directory in (image_dir, label_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        result[split] = (image_dir, label_dir)
    return result


def main() -> int:
    args = parse_args()
    ratios = [args.train, args.val, args.test]
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        print("HATA: --train, --val ve --test negatif olmamalı ve toplamları 1.0 olmalıdır.")
        return 2

    images_dir = project_path(args.images_dir)
    labels_dir = project_path(args.labels_dir)
    out_dir = project_path(args.out_dir)

    if not images_dir.exists():
        print(f"HATA: Görsel klasörü bulunamadı: {images_dir}")
        return 1
    if not labels_dir.exists() and not args.include_unlabeled:
        print(
            f"UYARI: Etiket klasörü bulunamadı: {labels_dir}. "
            "Etiketsiz görseller varsayılan olarak atlanacak."
        )

    images = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    print(f"Kaynak görsel sayısı: {len(images)}")
    print(f"Etiket klasörü: {labels_dir}")

    samples: list[dict[str, Any]] = []
    skipped_unlabeled = 0
    included_unlabeled = 0
    invalid_label_rows = 0
    bbox_count_by_class = {class_id: 0 for class_id in CLASS_NAMES}

    seen_stems: set[str] = set()
    for image_path in images:
        stem_key = image_path.stem.lower()
        if stem_key in seen_stems:
            print(f"UYARI: Yinelenen görsel stem'i atlandı: {image_path.name}")
            continue
        seen_stems.add(stem_key)

        label_path = labels_dir / f"{image_path.stem}.txt"
        has_label = label_path.exists()
        if not has_label and not args.include_unlabeled:
            skipped_unlabeled += 1
            print(f"UYARI: Etiket bulunamadı, görsel atlandı: {image_path.name}")
            continue
        if not has_label:
            included_unlabeled += 1
        else:
            label_counts, invalid_rows = validate_label(label_path)
            invalid_label_rows += invalid_rows
            for class_id, count in label_counts.items():
                bbox_count_by_class[class_id] += count

        samples.append(
            {
                "image_path": image_path,
                "label_path": label_path if has_label else None,
            }
        )

    random.Random(args.seed).shuffle(samples)
    train_count, val_count, test_count = split_counts(len(samples), ratios)
    split_samples = {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + val_count],
        "test": samples[train_count + val_count :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    split_directories = reset_split_directories(out_dir)
    for split, records in split_samples.items():
        target_images, target_labels = split_directories[split]
        print(f"{split}: {len(records)} örnek kopyalanıyor...")
        for record in records:
            image_path = record["image_path"]
            shutil.copy2(image_path, target_images / image_path.name)
            target_label = target_labels / f"{image_path.stem}.txt"
            if record["label_path"] is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(record["label_path"], target_label)

    dataset_path_text = out_dir.resolve().as_posix().replace('"', '\\"')
    data_yaml = (
        f'path: "{dataset_path_text}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: player\n"
        "  1: ball\n"
        "  2: outside_person\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "total_images": len(samples),
        "train_images": len(split_samples["train"]),
        "val_images": len(split_samples["val"]),
        "test_images": len(split_samples["test"]),
        "skipped_unlabeled": skipped_unlabeled,
        "included_unlabeled": included_unlabeled,
        "bbox_count_by_class": {
            str(class_id): bbox_count_by_class[class_id] for class_id in CLASS_NAMES
        },
        "invalid_label_rows": invalid_label_rows,
        "seed": args.seed,
    }
    summary_path = out_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"YOLO veri kümesi hazır: {out_dir}")
    print(
        f"Toplam={len(samples)}, train={train_count}, val={val_count}, test={test_count}, "
        f"etiketsiz atlanan={skipped_unlabeled}"
    )
    if invalid_label_rows:
        print(
            f"UYARI: {invalid_label_rows} geçersiz etiket satırı bulundu. "
            "Eğitimden önce yukarıdaki dosyaları düzeltin."
        )
    print(f"Yapılandırma: {out_dir / 'data.yaml'}")
    print(f"Özet: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
