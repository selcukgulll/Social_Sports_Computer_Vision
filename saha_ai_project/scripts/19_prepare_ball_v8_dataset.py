#!/usr/bin/env python
"""Mevcut Ball v8 train/val verisini koruyup CVAT test splitini yeniler."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mevcut Ball v8 train/validation verisine dokunmadan CVAT gercek-case "
            "test splitini yeniler."
        )
    )
    parser.add_argument("--labels-source", type=Path, required=True)
    parser.add_argument(
        "--images-source",
        type=Path,
        action="append",
        required=True,
        help="CVAT ZIP'inde olmayan orijinal karelerin klasoru. Birden fazla verilebilir.",
    )
    parser.add_argument("--max-images", type=int, default=210)
    parser.add_argument(
        "--real-test-output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/real_case_test_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/master_v8_realcase",
        help="Train/val bolumu daha once hazirlanmis final Ball v8 dataseti.",
    )
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_summary(
    dataset: Path,
    split: str,
    *,
    verify_pairs: bool = True,
    count_boxes: bool = True,
) -> dict[str, int]:
    images = image_paths(dataset / "images" / split)
    labels_dir = dataset / "labels" / split
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label split klasoru bulunamadi: {labels_dir}")
    label_files = list(labels_dir.glob("*.txt"))
    if verify_pairs:
        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in label_files}
        missing = image_stems - label_stems
        extra = label_stems - image_stems
        if missing:
            raise FileNotFoundError(f"Gorsele ait label bulunamadi: {next(iter(missing))}")
        if extra:
            raise ValueError(
                f"{split} splitinde gorselsiz label bulundu: {next(iter(extra))}"
            )

    boxes = 0
    positive_images = 0
    if count_boxes:
        for label_path in label_files:
            count = sum(
                1
                for line in label_path.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines()
                if line.strip()
            )
            boxes += count
            positive_images += int(count > 0)
    return {
        "images": len(images),
        "boxes": boxes,
        "positive_images": positive_images,
    }


def validate_existing_training_dataset(output: Path) -> dict[str, dict[str, int]]:
    if not (output / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Hazir Ball v8 data.yaml bulunamadi: {output / 'data.yaml'}"
        )
    cached: dict[str, object] = {}
    summary_path = output / "realcase_dataset_summary.json"
    try:
        cached = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        pass
    cached_splits = cached.get("splits", {}) if isinstance(cached, dict) else {}
    summaries: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        quick = split_summary(output, split, count_boxes=False)
        previous = cached_splits.get(split, {}) if isinstance(cached_splits, dict) else {}
        if isinstance(previous, dict) and int(previous.get("images", -1)) == quick["images"]:
            quick["boxes"] = int(previous.get("boxes", 0))
            quick["positive_images"] = int(previous.get("positive_images", 0))
        else:
            quick = split_summary(output, split, count_boxes=True)
        summaries[split] = quick
    for split, values in summaries.items():
        if values["images"] == 0:
            raise ValueError(f"Hazir Ball v8 {split} split'i bos: {output}")
    return summaries


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def replace_test_split(real_test: Path, output: Path, copy_mode: str) -> None:
    source_images = image_paths(real_test / "images" / "test")
    source_labels = real_test / "labels" / "test"
    if not source_images:
        raise ValueError(f"CVAT gercek test splitinde gorsel yok: {real_test}")
    if not source_labels.is_dir():
        raise FileNotFoundError(f"CVAT gercek test label klasoru yok: {source_labels}")

    token = str(os.getpid())
    stage_root = output / f".test_refresh_{token}"
    stage_images = stage_root / "images"
    stage_labels = stage_root / "labels"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_images.mkdir(parents=True)
    stage_labels.mkdir(parents=True)

    try:
        for index, image_path in enumerate(source_images, start=1):
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"CVAT test label bulunamadi: {label_path}")
            target_stem = f"realcase__{image_path.stem}"
            link_or_copy(
                image_path,
                stage_images / f"{target_stem}{image_path.suffix.lower()}",
                copy_mode,
            )
            shutil.copy2(label_path, stage_labels / f"{target_stem}.txt")
            if index % 100 == 0 or index == len(source_images):
                print(
                    f"Test kareleri hazirlaniyor: {index}/{len(source_images)}",
                    flush=True,
                )

        target_images = output / "images" / "test"
        target_labels = output / "labels" / "test"
        backup_images = output / "images" / f".test_backup_{token}"
        backup_labels = output / "labels" / f".test_backup_{token}"
        target_images.parent.mkdir(parents=True, exist_ok=True)
        target_labels.parent.mkdir(parents=True, exist_ok=True)

        moved_old_images = False
        moved_old_labels = False
        installed_images = False
        installed_labels = False
        try:
            if target_images.exists():
                os.replace(target_images, backup_images)
                moved_old_images = True
            if target_labels.exists():
                os.replace(target_labels, backup_labels)
                moved_old_labels = True
            os.replace(stage_images, target_images)
            installed_images = True
            os.replace(stage_labels, target_labels)
            installed_labels = True
        except OSError:
            if installed_images and target_images.exists():
                shutil.rmtree(target_images)
            if installed_labels and target_labels.exists():
                shutil.rmtree(target_labels)
            if moved_old_images and backup_images.exists():
                os.replace(backup_images, target_images)
            if moved_old_labels and backup_labels.exists():
                os.replace(backup_labels, target_labels)
            raise
        else:
            if backup_images.exists():
                shutil.rmtree(backup_images)
            if backup_labels.exists():
                shutil.rmtree(backup_labels)
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


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


def run_import(command: list[str]) -> None:
    print("", flush=True)
    print("=" * 72, flush=True)
    print("[1/2] CVAT etiketlerinden gercek-case test setini olustur", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    print("=" * 72, flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"CVAT test importu basarisiz oldu (hata kodu {completed.returncode})."
        )
    print("[1/2] TAMAMLANDI: CVAT test seti olusturuldu", flush=True)


def main() -> int:
    args = parse_args()
    labels_source = resolve(args.labels_source)
    image_sources = [resolve(path) for path in args.images_source]
    real_test_output = resolve(args.real_test_output)
    output = resolve(args.output)

    if not labels_source.is_dir():
        raise FileNotFoundError(f"CVAT etiket klasoru bulunamadi: {labels_source}")
    for image_source in image_sources:
        if not image_source.is_dir():
            raise FileNotFoundError(f"Gorsel klasoru bulunamadi: {image_source}")
    if args.max_images < 0:
        raise ValueError("Test kare sayisi negatif olamaz. 0 tum kareler anlamina gelir.")
    if real_test_output == output:
        raise ValueError("Gecici CVAT test cikti klasoru final dataset ile ayni olamaz.")
    allowed_parent = (PROJECT_ROOT / "datasets/BallOnly").resolve()
    for name, path in (("gercek test", real_test_output), ("final dataset", output)):
        if allowed_parent not in path.parents:
            raise ValueError(f"{name} guvenli BallOnly alani disinda: {path}")

    has_existing_train_val = (
        (output / "data.yaml").is_file()
        and (output / "images" / "train").is_dir()
        and (output / "images" / "val").is_dir()
    )
    if not has_existing_train_val:
        build_script = PROJECT_ROOT / "scripts/20_build_ball_training_dataset_from_cvat.py"
        if not build_script.is_file():
            raise FileNotFoundError(
                f"Hazir Ball v8 data.yaml bulunamadi: {output / 'data.yaml'}\n"
                "Yeni dataset olusturmak icin 20_build_ball_training_dataset_from_cvat.py gerekli."
            )
        command = [
            sys.executable,
            "-u",
            str(build_script),
            "--labels-source",
            str(labels_source),
            "--max-images",
            str(args.max_images),
            "--output",
            str(output),
            "--overwrite",
        ]
        for image_source in image_sources:
            command.extend(["--images-source", str(image_source)])
        print(
            "Hazir train/validation bulunamadi; CVAT etiketlerinden yeni dataset olusturuluyor...",
            flush=True,
        )
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        return int(completed.returncode)

    print("BALL V8 CVAT TEST YENILEME", flush=True)
    print("Hazir train/validation dosyalari kontrol ediliyor...", flush=True)
    preserved = validate_existing_training_dataset(output)
    print(
        f"TRAIN KORUNUYOR: {preserved['train']['images']:,} gorsel, "
        f"{preserved['train']['boxes']:,} top kutusu",
        flush=True,
    )
    print(
        f"VALIDATION KORUNUYOR: {preserved['val']['images']:,} gorsel, "
        f"{preserved['val']['boxes']:,} top kutusu",
        flush=True,
    )
    print("Augmentation calistirilmayacak.", flush=True)

    import_script = PROJECT_ROOT / "scripts/17_import_cvat_ball_test_set.py"
    command = [
        sys.executable,
        "-u",
        str(import_script),
        "--labels-source",
        str(labels_source),
    ]
    for image_source in image_sources:
        command.extend(["--images-source", str(image_source)])
    command.extend(
        [
            "--max-images",
            str(args.max_images),
            "--output",
            str(real_test_output),
            "--overwrite",
        ]
    )
    run_import(command)

    print("", flush=True)
    print("=" * 72, flush=True)
    print("[2/2] Yalniz final datasetin test splitini yenile", flush=True)
    print("Train ve validation klasorlerine dokunulmayacak.", flush=True)
    print("=" * 72, flush=True)
    replace_test_split(real_test_output, output, args.copy_mode)
    write_data_yaml(output)

    summaries = {
        "train": preserved["train"],
        "val": preserved["val"],
        "test": split_summary(output, "test"),
    }
    summary = {
        "output": str(output),
        "real_test": str(real_test_output),
        "train_val_preserved": True,
        "augmentation_ran": False,
        "splits": summaries,
    }
    (output / "realcase_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[2/2] TAMAMLANDI: test split yenilendi", flush=True)
    print("", flush=True)
    print("BALL V8 EGITIME HAZIR", flush=True)
    for split in ("train", "val", "test"):
        values = summaries[split]
        print(
            f"{split:>5}: {values['images']:,} gorsel, "
            f"{values['boxes']:,} top kutusu, "
            f"{values['positive_images']:,} pozitif",
            flush=True,
        )
    print(f"data.yaml: {output / 'data.yaml'}", flush=True)
    print("Simdi egitimi baslatabilirsiniz.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
