#!/usr/bin/env python
"""Extract regularly sampled candidate frames from match videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import cv2
except ImportError:  # Allows --help and a clearer runtime error.
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
INDEX_COLUMNS = ["image_path", "video_path", "video_stem", "time_sec", "frame_idx"]


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="videos/raw içindeki videolardan düzenli aralıklarla aday eğitim kareleri çıkarır."
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("videos/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("frames/candidate_frames"))
    parser.add_argument("--every-sec", type=float, default=2.0)
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="0 ise video başına sınır uygulanmaz.",
    )
    return parser.parse_args()


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.exists():
        return []
    return sorted(
        (
            path
            for path in videos_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(videos_dir)).lower(),
    )


def extract_video(
    video_path: Path,
    out_dir: Path,
    every_sec: float,
    max_frames: int,
) -> list[dict[str, object]]:
    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"UYARI: Video açılamadı, atlanıyor: {video_path}")
        return []

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    duration_text = f"{duration_sec:.2f} sn" if duration_sec > 0 else "bilinmiyor"
    print(
        f"Video işleniyor: {video_path.name} | FPS={fps:.3f} | "
        f"süre={duration_text}"
    )

    records: list[dict[str, object]] = []
    sample_number = 1
    consecutive_read_failures = 0

    while True:
        if max_frames > 0 and len(records) >= max_frames:
            break

        time_sec = sample_number * every_sec
        if duration_sec > 0 and time_sec >= duration_sec - 1e-9:
            break

        capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            consecutive_read_failures += 1
            if duration_sec > 0 or consecutive_read_failures >= 2:
                break
            sample_number += 1
            continue

        consecutive_read_failures = 0
        reported_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES) - 1))
        frame_idx = (
            reported_frame
            if reported_frame >= 0
            else int(round(time_sec * fps)) if fps > 0 else sample_number
        )
        image_name = f"{video_path.stem}_t{time_sec:09.2f}.jpg"
        image_path = video_out / image_name

        if not cv2.imwrite(str(image_path), frame):
            print(f"UYARI: Kare yazılamadı: {image_path}")
            sample_number += 1
            continue

        records.append(
            {
                "image_path": str(image_path.resolve()),
                "video_path": str(video_path.resolve()),
                "video_stem": video_path.stem,
                "time_sec": round(time_sec, 6),
                "frame_idx": frame_idx,
            }
        )
        if len(records) % 25 == 0:
            print(f"  {len(records)} kare çıkarıldı...")
        sample_number += 1

    capture.release()
    print(f"Tamamlandı: {video_path.name} -> {len(records)} kare")
    return records


def main() -> int:
    args = parse_args()
    if cv2 is None:
        print("HATA: opencv-python kurulu değil. Kurulum: pip install opencv-python")
        return 1
    if args.every_sec <= 0:
        print("HATA: --every-sec sıfırdan büyük olmalıdır.")
        return 2
    if args.max_frames_per_video < 0:
        print("HATA: --max-frames-per-video negatif olamaz.")
        return 2

    videos_dir = project_path(args.videos_dir)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir)
    print(f"Video klasörü: {videos_dir}")
    print(f"Bulunan desteklenen video sayısı: {len(videos)}")

    all_records: list[dict[str, object]] = []
    for video_path in videos:
        all_records.extend(
            extract_video(
                video_path,
                out_dir,
                args.every_sec,
                args.max_frames_per_video,
            )
        )

    index_path = out_dir / "frame_index.csv"
    pd.DataFrame(all_records, columns=INDEX_COLUMNS).to_csv(
        index_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Frame indeksi yazıldı: {index_path}")
    print(f"Toplam çıkarılan kare: {len(all_records)}")
    if not videos:
        print("Bilgi: Video bulunamadı; başlıkları içeren boş frame_index.csv oluşturuldu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
