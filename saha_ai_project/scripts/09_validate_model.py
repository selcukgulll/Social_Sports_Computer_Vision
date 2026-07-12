#!/usr/bin/env python
"""Eğitilmiş bir YOLO modelini seçilen veri kümesinde değerlendir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO model doğrulaması")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="model_validation")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    model_path = resolve(args.model)
    dataset_path = resolve(args.dataset)
    data_yaml = dataset_path / "data.yaml" if dataset_path.is_dir() else dataset_path
    if not model_path.is_file():
        print(f"HATA: Model bulunamadı: {model_path}")
        return 1
    if not data_yaml.is_file():
        print(f"HATA: data.yaml bulunamadı: {data_yaml}")
        return 1

    output_root = PROJECT_ROOT / "experiments/outputs"
    print(f"Model: {model_path}")
    print(f"Veri kümesi: {data_yaml}")
    metrics = YOLO(str(model_path)).val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        project=str(output_root),
        name=args.name,
        split=args.split,
        plots=True,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    summary = {
        "model": str(model_path),
        "data": str(data_yaml),
        "split": args.split,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    result_dir = Path(metrics.save_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Sonuç: "
        f"P={summary['precision']:.4f}, R={summary['recall']:.4f}, "
        f"F1={summary['f1_score']:.4f}, "
        f"mAP50={summary['map50']:.4f}, mAP50-95={summary['map50_95']:.4f}"
    )
    print(f"Rapor klasörü: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
