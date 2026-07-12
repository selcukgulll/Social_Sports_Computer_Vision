#!/usr/bin/env python
"""Generate, save, and optionally execute an Ultralytics YOLO training command."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def portable_project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO eğitim komutunu üretir; --run verilirse komutu çalıştırır."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/dataset_v1"))
    parser.add_argument("--base-model", default="models/base/yolo26m.pt")
    parser.add_argument("--name", default="player_ball_v1")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=8)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0 or args.epochs <= 0 or args.batch == 0 or args.nbs <= 0:
        print(
            "HATA: --imgsz, --epochs ve --nbs pozitif; "
            "--batch sıfırdan farklı olmalıdır."
        )
        return 2
    if not 0.0 <= args.mosaic <= 1.0:
        print("HATA: --mosaic 0.0 ile 1.0 arasında olmalıdır.")
        return 2
    if not args.name.strip():
        print("HATA: --name boş olamaz.")
        return 2

    dataset_dir = project_path(args.dataset)
    data_yaml = dataset_dir / "data.yaml"
    reports_dir = PROJECT_ROOT / "experiments/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_argument = portable_project_path(data_yaml)
    command = [
        "yolo",
        "detect",
        "train",
        f"model={args.base_model}",
        f"data={data_argument}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"nbs={args.nbs}",
        f"mosaic={args.mosaic}",
        "mixup=0.0",
        "scale=0.2",
        f"device={args.device}",
        f"project={(PROJECT_ROOT / 'models/trained').resolve()}",
        f"name={args.name}",
    ]
    command_text = subprocess.list2cmdline(command)
    report_path = reports_dir / f"train_command_{args.name}.txt"
    report_path.write_text(command_text + "\n", encoding="utf-8")

    if not data_yaml.exists():
        print(f"UYARI: data.yaml henüz bulunamadı: {data_yaml}")
    print("YOLO eğitim komutu:")
    print(command_text)
    print(f"Komut dosyası yazıldı: {report_path}")

    best_path = PROJECT_ROOT / "models/trained" / args.name / "weights/best.pt"
    print(f"Eğitim tamamlanınca best.pt beklenen konum: {best_path}")

    if not args.run:
        print("Komut çalıştırılmadı. Çalıştırmak için aynı komuta --run ekleyin.")
        return 0

    yolo_executable = shutil.which("yolo")
    if yolo_executable is None:
        print(
            "HATA: 'yolo' komutu PATH üzerinde bulunamadı. "
            "Ultralytics kurulumunu/aktif sanal ortamı kontrol edin."
        )
        return 1

    command[0] = yolo_executable
    print("Eğitim başlatılıyor...")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        print(f"HATA: YOLO eğitimi {completed.returncode} çıkış koduyla sona erdi.")
        return completed.returncode
    print(f"Eğitim komutu tamamlandı. best.pt konumu: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
