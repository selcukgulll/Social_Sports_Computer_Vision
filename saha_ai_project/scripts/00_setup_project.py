#!/usr/bin/env python
"""Create the directory skeleton and default files for the training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = [
    "app",
    "videos/raw",
    "videos/test",
    "frames/candidate_frames",
    "frames/hard_frames",
    "frames/label_staging",
    "labels/images",
    "labels/manual",
    "datasets/dataset_v1",
    "datasets/dataset_v2",
    "models/base",
    "models/trained",
    "experiments/configs",
    "experiments/outputs",
    "experiments/reports",
    "scripts",
]

CLASSES = ["player", "ball", "outside_person"]

DEFAULT_CONFIG = {
    "expected_players": 14,
    "sample_sec": 0.1,
    "model": "yolo11x.pt",
    "conf": 0.12,
    "field_mapper": "poly2",
    "use_sahi": True,
    "frame_extract_every_sec": 2.0,
    "hard_frame_top_k": 200,
    "train_split": 0.70,
    "val_split": 0.20,
    "test_split": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saha AI eğitim/deney proje klasörlerini ve varsayılan dosyaları hazırlar."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Proje kökü (varsayılan: bu scriptin üst klasörü).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mevcut classes.txt ve default_config.json dosyalarını varsayılanlarla yenile.",
    )
    return parser.parse_args()


def write_if_missing(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        print(f"Korundu (zaten var): {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Yazıldı: {path}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    print(f"Proje hazırlanıyor: {project_root}")
    for relative_dir in REQUIRED_DIRECTORIES:
        directory = project_root / relative_dir
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Hazır: {directory}")

    classes_path = project_root / "classes.txt"
    write_if_missing(classes_path, "\n".join(CLASSES) + "\n", args.force)

    config_path = project_root / "experiments/configs/default_config.json"
    config_text = json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n"
    write_if_missing(config_path, config_text, args.force)

    v8_path = project_root / "app/process_video_to_csv_v8.py"
    if v8_path.exists():
        print(f"V8 ana script bulundu: {v8_path}")
    else:
        print(
            "UYARI: app/process_video_to_csv_v8.py bulunamadı. "
            "Baseline deneylerinden önce V8 scriptini bu konuma kopyalayın."
        )

    print("Kurulum tamamlandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
