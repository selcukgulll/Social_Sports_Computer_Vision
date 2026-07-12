#!/usr/bin/env python
"""CVAT YOLO etiketleri ve orijinal karelerden sıfırdan ball eğitim dataseti oluşturur."""

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
            "CVAT YOLO exportunu ve orijinal kareleri kullanarak "
            "train/validation/test içeren yeni bir ball dataseti oluşturur."
        )
    )
    parser.add_argument("--labels-source", type=Path, required=True)
    parser.add_argument(
        "--images-source",
        type=Path,
        action="append",
        required=True,
        help="CVAT ZIP'inde olmayan orijinal karelerin klasörü. Birden fazla verilebilir.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="İlk N kareyi al. 0=tüm eşleşen kareler.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/manuel_dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def deterministic_split(stem: str, seed: int) -> str:
    ratio = int(hashlib.sha1(f"{seed}|{stem}".encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    if ratio < 0.80:
        return "train"
    if ratio < 0.90:
        return "val"
    return "test"


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


def split_summary(output: Path, split: str) -> dict[str, int]:
    images = [
        path
        for path in sorted((output / "images" / split).iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def main() -> int:
    args = parse_args()
    cvat = load_cvat_import_module()
    labels_source = resolve(args.labels_source)
    image_sources = [resolve(path) for path in args.images_source]
    output = resolve(args.output)

    if not labels_source.is_dir():
        raise FileNotFoundError(f"CVAT etiket klasörü bulunamadı: {labels_source}")
    for image_source in image_sources:
        if not image_source.is_dir():
            raise FileNotFoundError(f"Görsel klasörü bulunamadı: {image_source}")
    allowed_parent = (PROJECT_ROOT / "datasets/BallOnly").resolve()
    if allowed_parent not in output.parents:
        raise ValueError(f"Çıktı güvenli BallOnly alanı dışında: {output}")

    images = cvat.find_images_from_roots(image_sources)
    labels = cvat.find_labels(labels_source)
    if not labels:
        raise RuntimeError(f"Label kaynağında YOLO txt bulunamadı: {labels_source}")
    if not images:
        raise RuntimeError(
            "Görsel bulunamadı. --images-source içinde images/ alt klasörünü kontrol edin."
        )

    label_paths = list(labels.values())
    class_names = cvat.find_class_names(labels_source)
    ball_ids = cvat.ball_source_ids(class_names, label_paths)

    image_list_path = cvat.default_image_list(labels_source)
    if image_list_path is not None and image_list_path.is_file():
        ordered_stems = cvat.read_image_list(image_list_path, int(args.max_images))
    else:
        ordered_stems = sorted(set(labels) & set(images))
        if args.max_images > 0:
            ordered_stems = ordered_stems[: int(args.max_images)]
    if not ordered_stems:
        raise RuntimeError("İçe aktarılacak görsel listesi boş.")

    reset_output(output, args.overwrite)
    manifest: list[dict[str, object]] = []
    missing_source_images = 0
    missing_label_images = 0
    invalid_or_ignored_rows = 0

    for index, stem in enumerate(ordered_stems, start=1):
        image_path = images.get(stem)
        if image_path is None:
            missing_source_images += 1
            continue
        label_path = labels.get(stem)
        if label_path is None:
            missing_label_images += 1
            continue

        rows: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if not raw_line.strip():
                continue
            row = cvat.parse_row(raw_line, ball_ids)
            if row is None:
                invalid_or_ignored_rows += 1
            else:
                rows.append(row)

        target_split = deterministic_split(stem, args.seed)
        target_name = cvat.safe_name(stem, image_path.suffix)
        target_stem = Path(target_name).stem
        target_image = output / "images" / target_split / target_name
        target_label = output / "labels" / target_split / f"{target_stem}.txt"
        link_or_copy(image_path, target_image, args.copy_mode)
        target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        manifest.append(
            {
                "index": index,
                "split": target_split,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "target_image": str(target_image),
                "target_label": str(target_label),
                "ball_boxes": len(rows),
            }
        )
        if index % 100 == 0 or index == len(ordered_stems):
            print(f"Kareler hazırlanıyor: {index}/{len(ordered_stems)}", flush=True)

    if not manifest:
        raise RuntimeError(
            "Hiç görsel eşleşmedi. Label isimleri ile --images-source görsel isimlerini kontrol edin."
        )

    write_data_yaml(output)
    summaries = {split: split_summary(output, split) for split in ("train", "val", "test")}
    summary = {
        "labels_source": str(labels_source),
        "images_source": [str(path) for path in image_sources],
        "image_list": str(image_list_path) if image_list_path else "",
        "max_images": int(args.max_images),
        "output": str(output),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "class_names": class_names,
        "ball_source_ids": sorted(ball_ids),
        "images": len(manifest),
        "ball_boxes": sum(int(item["ball_boxes"]) for item in manifest),
        "positive_images": sum(1 for item in manifest if int(item["ball_boxes"]) > 0),
        "negative_images": sum(1 for item in manifest if int(item["ball_boxes"]) == 0),
        "missing_label_images": missing_label_images,
        "missing_source_images": missing_source_images,
        "invalid_or_ignored_rows": invalid_or_ignored_rows,
        "splits": summaries,
    }
    (output / "manuel_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("", flush=True)
    print("BALL MANUEL DATASET HAZIR", flush=True)
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
