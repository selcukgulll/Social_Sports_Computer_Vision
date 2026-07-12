#!/usr/bin/env python
"""Büyük tracking_all.csv dosyasından düşük bellekli viewer CSV üret."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


VIEWER_COLUMNS = [
    "viewer_id",
    "stable_id",
    "time_sec",
    "field_x",
    "field_y",
    "team",
    "team_hint",
    "entity",
    "variant",
    "position_conf",
    "det_conf",
    "source",
    "is_interpolated",
]
REQUIRED_COLUMNS = {"stable_id", "time_sec", "field_x", "field_y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="tracking_all.csv -> tracking_viewer.csv")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name("tracking_viewer.csv")
    )
    if not input_path.is_file():
        print(f"HATA: Kaynak CSV bulunamadı: {input_path}")
        return 1
    if output_path == input_path:
        print("HATA: Kaynak ve çıktı dosyası aynı olamaz.")
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    row_count = 0
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            available = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - available
            if missing:
                print(f"HATA: Gerekli kolonlar eksik: {', '.join(sorted(missing))}")
                return 2
            columns = [name for name in VIEWER_COLUMNS if name in available]
            if "viewer_id" not in columns:
                columns.insert(0, "viewer_id")

            with temp_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    if "viewer_id" not in available:
                        entity = row.get("entity") or (
                            "ball" if str(row.get("team", "")).lower() == "ball" else "player"
                        )
                        row["viewer_id"] = f"{entity}_{row.get('stable_id', 'unknown')}"
                    writer.writerow(row)
                    row_count += 1
                    if row_count % 25000 == 0:
                        print(f"{row_count} satır dönüştürüldü...", flush=True)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    source_mb = input_path.stat().st_size / 1024**2
    output_mb = output_path.stat().st_size / 1024**2
    print(f"Viewer CSV hazır: {output_path}")
    print(f"Satır: {row_count}")
    print(f"Boyut: {source_mb:.1f} MB -> {output_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
