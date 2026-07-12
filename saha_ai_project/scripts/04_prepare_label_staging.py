#!/usr/bin/env python
"""Combine hard and regular candidate frames into a clean labeling workspace."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["player", "ball", "outside_person"]
INDEX_COLUMNS = [
    "staged_image_path",
    "empty_label_path",
    "source_image_path",
    "source_type",
    "video_path",
    "video_stem",
    "time_sec",
    "hard_score",
    "reasons",
]


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aday ve zor karelerden manuel etiketleme için temiz bir staging klasörü hazırlar."
    )
    parser.add_argument(
        "--candidate-index",
        type=Path,
        default=Path("frames/candidate_frames/frame_index.csv"),
    )
    parser.add_argument(
        "--hard-index",
        type=Path,
        default=Path("frames/hard_frames/baseline_v8/hard_frames_index.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("frames/label_staging"))
    parser.add_argument("--max-candidate", type=int, default=500)
    parser.add_argument("--max-hard", type=int, default=300)
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=0.75,
        help="Aynı videonun çok yakın zamanlı karelerini tekrar eklemeyi engeller.",
    )
    return parser.parse_args()


def resolve_indexed_path(text: Any) -> Path:
    path = Path(str(text)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_index(path: Path, required: bool) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            print(
                f"HATA: Aday kare indeksi bulunamadı: {path}\n"
                "Önce python scripts/01_extract_candidate_frames.py komutunu çalıştırın."
            )
        else:
            print(f"Bilgi: Zor kare indeksi henüz yok; yalnız aday kareler kullanılacak: {path}")
        return None
    try:
        dataframe = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        print(f"HATA: İndeks okunamadı ({path}): {exc}")
        return None
    if "image_path" not in dataframe.columns:
        print(f"HATA: İndekste image_path kolonu yok: {path}")
        return None
    return dataframe


def candidate_round_robin(dataframe: pd.DataFrame) -> Iterable[dict[str, Any]]:
    """Spread ordinary candidates across videos instead of exhausting one video first."""
    if dataframe.empty:
        return []
    group_column = "video_path" if "video_path" in dataframe.columns else "video_stem"
    if group_column not in dataframe.columns:
        return dataframe.to_dict("records")

    groups: deque[deque[dict[str, Any]]] = deque()
    for _, group in dataframe.groupby(group_column, sort=True, dropna=False):
        if "time_sec" in group.columns:
            group = group.assign(
                _sort_time=pd.to_numeric(group["time_sec"], errors="coerce")
            ).sort_values("_sort_time", na_position="last")
        groups.append(deque(group.to_dict("records")))

    ordered: list[dict[str, Any]] = []
    while groups:
        group_records = groups.popleft()
        if group_records:
            ordered.append(group_records.popleft())
        if group_records:
            groups.append(group_records)
    return ordered


def record_video_key(record: dict[str, Any]) -> str:
    for column in ("video_path", "video_stem"):
        value = record.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().lower()
    return str(record.get("image_path", "")).lower()


def record_time(record: dict[str, Any]) -> float | None:
    try:
        value = float(record.get("time_sec"))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def clear_generated_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def main() -> int:
    args = parse_args()
    if args.max_candidate < 0 or args.max_hard < 0 or args.min_gap_sec < 0:
        print("HATA: Limitler ve --min-gap-sec negatif olamaz.")
        return 2

    candidate_index = project_path(args.candidate_index)
    hard_index = project_path(args.hard_index)
    out_dir = project_path(args.out_dir)

    candidates = read_index(candidate_index, required=True)
    if candidates is None:
        return 1
    hard_frames = read_index(hard_index, required=False)

    images_dir = out_dir / "images"
    empty_labels_dir = out_dir / "labels_empty"
    clear_generated_files(images_dir)
    clear_generated_files(empty_labels_dir)
    print(f"Temiz staging klasörleri hazırlandı: {out_dir}")

    hard_records: list[dict[str, Any]] = []
    if hard_frames is not None and not hard_frames.empty:
        if "hard_score" in hard_frames.columns:
            hard_frames = hard_frames.assign(
                _score=pd.to_numeric(hard_frames["hard_score"], errors="coerce").fillna(0.0)
            ).sort_values("_score", ascending=False)
        hard_records = hard_frames.to_dict("records")

    ordered_sources = [
        ("hard", hard_records, args.max_hard),
        ("candidate", list(candidate_round_robin(candidates)), args.max_candidate),
    ]

    seen_source_paths: set[str] = set()
    selected_times: dict[str, list[float]] = {}
    used_destination_names: dict[str, str] = {}
    staged_records: list[dict[str, Any]] = []
    counts = {"hard": 0, "candidate": 0}

    for source_type, records, limit in ordered_sources:
        if limit == 0:
            continue
        for record in records:
            if counts[source_type] >= limit:
                break
            source_path = resolve_indexed_path(record["image_path"])
            source_key = str(source_path).lower()
            if source_key in seen_source_paths:
                continue
            if not source_path.exists():
                print(f"UYARI: Kaynak görsel bulunamadı, atlandı: {source_path}")
                continue

            video_key = record_video_key(record)
            time_sec = record_time(record)
            prior_times = selected_times.setdefault(video_key, [])
            if time_sec is not None and any(
                abs(time_sec - prior) < args.min_gap_sec - 1e-9 for prior in prior_times
            ):
                continue

            destination_name = source_path.name
            prior_source = used_destination_names.get(destination_name.lower())
            if prior_source is not None and prior_source != source_key:
                digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
                destination_name = f"{source_path.stem}_{digest}{source_path.suffix.lower()}"

            destination_image = images_dir / destination_name
            destination_label = empty_labels_dir / f"{Path(destination_name).stem}.txt"
            shutil.copy2(source_path, destination_image)
            destination_label.write_text("", encoding="utf-8")

            seen_source_paths.add(source_key)
            used_destination_names[destination_name.lower()] = source_key
            if time_sec is not None:
                prior_times.append(time_sec)
            counts[source_type] += 1
            staged_records.append(
                {
                    "staged_image_path": str(destination_image.resolve()),
                    "empty_label_path": str(destination_label.resolve()),
                    "source_image_path": str(source_path),
                    "source_type": source_type,
                    "video_path": record.get("video_path", ""),
                    "video_stem": record.get("video_stem", ""),
                    "time_sec": time_sec if time_sec is not None else "",
                    "hard_score": record.get("hard_score", ""),
                    "reasons": record.get("reasons", ""),
                }
            )

    classes_path = out_dir / "classes.txt"
    classes_path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    label_index_path = out_dir / "label_index.csv"
    pd.DataFrame(staged_records, columns=INDEX_COLUMNS).to_csv(
        label_index_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Staging tamamlandı: {counts['hard']} zor kare + "
        f"{counts['candidate']} aday kare = {len(staged_records)} görsel"
    )
    print(f"Etiket indeksi: {label_index_path}")
    print("Manual labeling step:")
    print("Open frames/label_staging/images in your labeling tool.")
    print("Classes: 0 player, 1 ball, 2 outside_person.")
    print("Save YOLO .txt labels into labels/manual or frames/label_staging/labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
