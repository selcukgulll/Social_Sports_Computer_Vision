#!/usr/bin/env python
"""Run process_video_to_csv_v8.py over every video in a directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V8 takip scriptini klasördeki her video için çalıştırır."
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("videos/raw"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments/outputs/baseline_v8"),
    )
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--sample-sec", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--field-mapper", choices=["homography", "poly2", "poly3"], default="poly2")
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument(
        "--no-force-calibration",
        action="store_true",
        help="V8 komutuna --force-calibration ekleme.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("app/process_video_to_csv_v8.py"),
        help="V8 ana script yolu.",
    )
    return parser.parse_args()


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.exists():
        return []
    return sorted(
        (
            path
            for path in videos_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def main() -> int:
    args = parse_args()
    if args.sample_sec <= 0:
        print("HATA: --sample-sec sıfırdan büyük olmalıdır.")
        return 2
    if not 0 <= args.conf <= 1:
        print("HATA: --conf 0 ile 1 arasında olmalıdır.")
        return 2

    videos_dir = project_path(args.videos_dir)
    out_root = project_path(args.out_root)
    script_path = project_path(args.script)

    if not script_path.exists():
        print(f"HATA: V8 scripti bulunamadı: {script_path}")
        return 1

    videos = list_videos(videos_dir)
    print(f"Video klasörü: {videos_dir}")
    print(f"İşlenecek video sayısı: {len(videos)}")
    if not videos:
        print("Bilgi: Desteklenen video bulunamadı; deney çalıştırılmadı.")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, int]] = []

    for number, video_path in enumerate(videos, start=1):
        video_out = out_root / video_path.stem
        video_out.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(script_path),
            "--video",
            str(video_path),
            "--out",
            str(video_out),
            "--sample-sec",
            str(args.sample_sec),
            "--model",
            args.model,
            "--field-mapper",
            args.field_mapper,
            "--conf",
            str(args.conf),
        ]
        if not args.no_force_calibration:
            command.append("--force-calibration")
        if args.sahi:
            command.append("--sahi")

        print()
        print(f"[{number}/{len(videos)}] Now running calibration/processing for video: {video_path}")
        print("Komut:", subprocess.list2cmdline(command))
        print("Kalibrasyon penceresi bu işlem tarafından gizlenmez; ekrandaki adımları tamamlayın.")

        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            failures.append((video_path, completed.returncode))
            print(
                f"UYARI: İşlem {completed.returncode} koduyla bitti; "
                "sıradaki videoya geçiliyor."
            )
        else:
            print(f"Video başarıyla tamamlandı. Çıktı: {video_out}")

    print()
    print(f"Baseline çalışması bitti. Başarılı: {len(videos) - len(failures)}, hatalı: {len(failures)}")
    for video_path, return_code in failures:
        print(f"  Hatalı: {video_path} (çıkış kodu {return_code})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
