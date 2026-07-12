#!/usr/bin/env python
"""CVAT YOLO ZIP dışa aktarımını mevcut staging/manuel etiket alanına al."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVAT YOLO ZIP etiketlerini içe aktarır.")
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("frames/label_staging/images"),
    )
    parser.add_argument(
        "--labeled-images-dir",
        type=Path,
        default=Path("labels/images"),
        help="Etiketlenmiş görsellerin kalıcı arşivi.",
    )
    parser.add_argument("--labels-dir", type=Path, default=Path("labels/manual"))
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=Path("labels/cvat_exports"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Aynı adlı mevcut manuel etiketi yeni dosyayla değiştir.",
    )
    return parser.parse_args()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"ZIP içinde güvensiz yol bulundu: {member.filename}") from exc
    archive.extractall(destination)


def is_yolo_label(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    lowered_parts = {part.lower() for part in path.parts}
    if path.name.lower() in {"train.txt", "test.txt", "val.txt", "classes.txt"}:
        return False
    return "labels" in lowered_parts or "obj_train_data" in lowered_parts


def main() -> int:
    args = parse_args()
    zip_path = project_path(args.zip_path)
    images_dir = project_path(args.images_dir)
    labeled_images_dir = project_path(args.labeled_images_dir)
    labels_dir = project_path(args.labels_dir)
    exports_dir = project_path(args.exports_dir)

    if not zip_path.is_file():
        print(f"HATA: ZIP bulunamadı: {zip_path}")
        return 1

    images_dir.mkdir(parents=True, exist_ok=True)
    labeled_images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    existing_images = {
        path.stem.lower(): path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    imported_labels = 0
    imported_images = 0
    skipped_conflicts = 0
    missing_images: list[str] = []

    with tempfile.TemporaryDirectory(prefix="saha_cvat_") as temp_name:
        temp_dir = Path(temp_name)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                safe_extract(archive, temp_dir)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            print(f"HATA: ZIP açılamadı: {exc}")
            return 1

        extracted_images = {
            path.stem.lower(): path
            for path in temp_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        for stem, source_image in extracted_images.items():
            if stem in existing_images:
                continue
            destination = images_dir / source_image.name
            shutil.copy2(source_image, destination)
            existing_images[stem] = destination
            imported_images += 1

        label_files = sorted(
            path for path in temp_dir.rglob("*.txt") if path.is_file() and is_yolo_label(path)
        )
        if not label_files:
            print("HATA: ZIP içinde YOLO label .txt dosyası bulunamadı.")
            return 1

        for source_label in label_files:
            stem = source_label.stem
            if stem.lower() not in existing_images:
                missing_images.append(stem)
                continue
            destination = labels_dir / f"{stem}.txt"
            if destination.exists() and not args.overwrite:
                if destination.read_bytes() != source_label.read_bytes():
                    print(f"UYARI: Mevcut farklı etiket korundu: {destination.name}")
                    skipped_conflicts += 1
            else:
                shutil.copy2(source_label, destination)
                imported_labels += 1

            source_image = existing_images[stem.lower()]
            archived_image = labeled_images_dir / source_image.name
            if not archived_image.exists() or args.overwrite:
                shutil.copy2(source_image, archived_image)

    archive_destination = exports_dir / zip_path.name
    if archive_destination.resolve() != zip_path.resolve():
        if archive_destination.exists():
            archive_destination = exports_dir / (
                f"{zip_path.stem}_{int(zip_path.stat().st_mtime)}{zip_path.suffix}"
            )
        shutil.copy2(zip_path, archive_destination)

    report = {
        "source_zip": str(zip_path),
        "imported_labels": imported_labels,
        "imported_images": imported_images,
        "labeled_images_archive": str(labeled_images_dir),
        "skipped_conflicts": skipped_conflicts,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
    }
    report_path = exports_dir / f"{zip_path.stem}_import_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"İçe aktarılan etiket: {imported_labels}")
    print(f"ZIP'ten eklenen görsel: {imported_images}")
    print(f"Korunan çakışmalı etiket: {skipped_conflicts}")
    print(f"Karşılık gelen görseli bulunamayan etiket: {len(missing_images)}")
    print(f"Rapor: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
