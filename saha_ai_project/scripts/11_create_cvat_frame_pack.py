#!/usr/bin/env python
"""Create a CVAT-ready ZIP by sampling frames from selected videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


IMAGE_QUALITY = 95


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    index: int
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class SelectedFrame:
    video: VideoInfo
    frame_idx: int
    time_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seçili videolardan CVAT'a direkt yüklenebilecek frame ZIP'i oluşturur."
    )
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        required=True,
        help="Kaynak MP4/MOV/AVI video dosyaları.",
    )
    parser.add_argument("--count", type=int, required=True, help="Toplam çıkarılacak frame sayısı.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Paketlerin yazılacağı klasör.")
    parser.add_argument(
        "--strategy",
        choices=("uniform", "random"),
        default="uniform",
        help="uniform=eșit aralıklı, random=video içinde rastgele.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=IMAGE_QUALITY)
    return parser.parse_args()


def safe_slug(value: str, limit: int = 70) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._-")
    return (slug or "video")[:limit]


def open_video_info(path: Path, index: int) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Video açılamadı: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    finally:
        capture.release()
    if fps <= 0:
        fps = 25.0
    if frame_count <= 0:
        raise ValueError(f"Video frame sayısı okunamadı: {path}")
    return VideoInfo(
        path=path.resolve(),
        index=index,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def allocate_counts(total: int, videos: list[VideoInfo]) -> dict[int, int]:
    """Distribute the total count as evenly as possible across videos."""
    if total <= 0:
        raise ValueError("--count pozitif olmalıdır.")
    if not videos:
        raise ValueError("En az bir video gerekli.")

    remaining = min(total, sum(video.frame_count for video in videos))
    allocation = {video.index: 0 for video in videos}
    active = {video.index for video in videos}
    by_index = {video.index: video for video in videos}

    while remaining > 0 and active:
        share = max(1, math.ceil(remaining / len(active)))
        progressed = False
        for video_index in list(sorted(active)):
            if remaining <= 0:
                break
            video = by_index[video_index]
            capacity = video.frame_count - allocation[video_index]
            if capacity <= 0:
                active.remove(video_index)
                continue
            take = min(share, capacity, remaining)
            allocation[video_index] += take
            remaining -= take
            progressed = progressed or take > 0
            if allocation[video_index] >= video.frame_count:
                active.remove(video_index)
        if not progressed:
            break
    return allocation


def uniform_indices(frame_count: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= frame_count:
        return list(range(frame_count))
    values: list[int] = []
    seen: set[int] = set()
    for index in range(count):
        value = int(round(((index + 0.5) * frame_count / count) - 0.5))
        value = min(frame_count - 1, max(0, value))
        while value in seen and value + 1 < frame_count:
            value += 1
        while value in seen and value > 0:
            value -= 1
        seen.add(value)
        values.append(value)
    return sorted(values)


def random_indices(frame_count: int, count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    if count >= frame_count:
        return list(range(frame_count))
    return sorted(rng.sample(range(frame_count), count))


def select_frames(
    videos: list[VideoInfo],
    count: int,
    strategy: str,
    seed: int,
) -> list[SelectedFrame]:
    rng = random.Random(seed)
    allocation = allocate_counts(count, videos)
    selected: list[SelectedFrame] = []
    for video in videos:
        wanted = allocation.get(video.index, 0)
        if strategy == "random":
            indices = random_indices(video.frame_count, wanted, rng)
        else:
            indices = uniform_indices(video.frame_count, wanted)
        selected.extend(
            SelectedFrame(video=video, frame_idx=frame_idx, time_sec=frame_idx / video.fps)
            for frame_idx in indices
        )
    return selected


def write_jpeg(path: Path, frame, quality: int) -> None:
    quality = min(100, max(1, int(quality)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"JPG encode başarısız: {path}")
    path.write_bytes(encoded.tobytes())


def extract_frames(
    selected: list[SelectedFrame],
    images_dir: Path,
    quality: int,
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    by_video: dict[int, list[SelectedFrame]] = {}
    for item in selected:
        by_video.setdefault(item.video.index, []).append(item)

    global_index = 0
    for video_index in sorted(by_video):
        items = sorted(by_video[video_index], key=lambda item: item.frame_idx)
        if not items:
            continue
        video = items[0].video
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            raise OSError(f"Video açılamadı: {video.path}")
        try:
            video_slug = safe_slug(video.path.stem)
            for item in items:
                capture.set(cv2.CAP_PROP_POS_FRAMES, item.frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(
                        f"UYARI: frame okunamadı, atlandı: {video.path.name} #{item.frame_idx}",
                        file=sys.stderr,
                    )
                    continue
                image_name = (
                    f"{global_index:06d}__v{video.index:02d}__{video_slug}"
                    f"__f{item.frame_idx:08d}__t{item.time_sec:010.3f}.jpg"
                )
                image_path = images_dir / image_name
                write_jpeg(image_path, frame, quality)
                manifest.append(
                    {
                        "image": image_name,
                        "video_index": video.index,
                        "video": str(video.path),
                        "video_name": video.path.name,
                        "frame_idx": item.frame_idx,
                        "time_sec": round(item.time_sec, 6),
                        "fps": round(video.fps, 6),
                        "video_frame_count": video.frame_count,
                        "width": video.width,
                        "height": video.height,
                    }
                )
                global_index += 1
                if global_index % 100 == 0:
                    print(f"{global_index} frame yazıldı...")
        finally:
            capture.release()
    return manifest


def write_manifest(run_dir: Path, manifest: list[dict[str, object]], summary: dict[str, object]) -> None:
    manifest_path = run_dir / "manifest.csv"
    if manifest:
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    else:
        manifest_path.write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_zip(images_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for image_path in sorted(images_dir.glob("*.jpg")):
            archive.write(image_path, arcname=image_path.name)


def main() -> int:
    args = parse_args()
    videos = [
        open_video_info(path.expanduser().resolve(), index)
        for index, path in enumerate(args.videos, start=1)
    ]
    requested_count = int(args.count)
    selected = select_frames(videos, requested_count, args.strategy, int(args.seed))

    output_root = args.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"cvat_frames_{timestamp}"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    print("Kaynak videolar:")
    for video in videos:
        duration = video.frame_count / video.fps
        print(
            f"- v{video.index:02d}: {video.path} | "
            f"{video.frame_count} frame | {video.fps:.3f} fps | {duration:.1f} sn"
        )
    print(f"İstenen frame: {requested_count}")
    print(f"Seçilen frame: {len(selected)}")
    print(f"Strateji: {args.strategy}")
    print(f"Çıktı: {run_dir}")

    manifest = extract_frames(selected, images_dir, int(args.jpeg_quality))
    zip_path = run_dir / "cvat_upload.zip"
    create_zip(images_dir, zip_path)

    summary = {
        "requested_count": requested_count,
        "selected_count": len(selected),
        "written_count": len(manifest),
        "strategy": args.strategy,
        "seed": int(args.seed),
        "jpeg_quality": int(args.jpeg_quality),
        "run_dir": str(run_dir),
        "images_dir": str(images_dir),
        "zip_path": str(zip_path),
        "videos": [
            {
                "index": video.index,
                "path": str(video.path),
                "fps": video.fps,
                "frame_count": video.frame_count,
                "width": video.width,
                "height": video.height,
            }
            for video in videos
        ],
    }
    write_manifest(run_dir, manifest, summary)
    print(f"CVAT ZIP hazır: {zip_path}")
    print(f"Manifest: {run_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
