#!/usr/bin/env python
"""Tek sınıflı ball detection modelini güvenli varsayılanlarla eğitir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ball-only YOLO detection eğitimi")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--name", default="ball_only_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="PyTorch CPU thread siniri. 0 ise sistem varsayilani kullanilir.",
    )
    parser.add_argument("--mosaic", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = resolve(args.dataset)
    data_yaml = dataset / "data.yaml" if dataset.is_dir() else dataset
    base_model = resolve(args.base_model)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Ball data.yaml bulunamadı: {data_yaml}")
    if not base_model.is_file():
        raise FileNotFoundError(f"Başlangıç modeli bulunamadı: {base_model}")
    if not args.name.strip():
        raise ValueError("Eğitim adı boş olamaz.")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise ValueError("Epoch/imgsz pozitif, batch sıfırdan farklı olmalı.")
    if args.workers < 0:
        raise ValueError("Worker sayısı negatif olamaz.")
    if args.cpu_threads < 0:
        raise ValueError("CPU thread sayısı negatif olamaz.")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    names = data.get("names")
    if data.get("nc") != 1 or names not in (["ball"], {0: "ball"}, {"0": "ball"}):
        raise ValueError(
            f"Bu eğitim yalnız ball datasetini kabul eder: nc={data.get('nc')} names={names}"
        )

    if args.cpu_threads:
        thread_count = str(args.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count

    # Thread ortam değişkenleri torch yüklenmeden önce ayarlanmış olsun.
    from ultralytics import YOLO

    if args.cpu_threads:
        import torch

        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, args.cpu_threads // 2)))
        except RuntimeError:
            pass

    output_root = PROJECT_ROOT / "models" / "trained"
    model = YOLO(str(base_model))
    print(f"Dataset: {data_yaml}")
    print(f"Başlangıç modeli: {base_model}")
    print(
        f"Ayarlar: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={args.device}, workers={args.workers}, "
        f"cpu_threads={args.cpu_threads or 'otomatik'}"
    )
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=str(output_root),
        name=args.name,
        task="detect",
        single_cls=False,
        mosaic=args.mosaic,
        close_mosaic=10,
        mixup=0.0,
        scale=0.25,
        fliplr=0.5,
        flipud=0.0,
        amp=True,
        cache=False,
        plots=True,
        seed=42,
        exist_ok=False,
    )
    save_dir = Path(result.save_dir).resolve()
    summary = {
        "dataset": str(data_yaml),
        "base_model": str(base_model),
        "save_dir": str(save_dir),
        "best_model": str(save_dir / "weights" / "best.pt"),
        "last_model": str(save_dir / "weights" / "last.pt"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cpu_threads": args.cpu_threads,
    }
    (save_dir / "ball_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
