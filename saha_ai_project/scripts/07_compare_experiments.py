#!/usr/bin/env python
"""Compare V8 baseline/custom tracking runs using repeatable heuristic metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = [
    "run",
    "tracking_files",
    "avg_players_per_frame",
    "median_players_per_frame",
    "frames_below_expected_players",
    "ball_found_frame_ratio",
    "avg_player_conf",
    "avg_position_conf",
    "interpolated_ratio",
    "unknown_team_ratio",
    "slice_detection_ratio",
    "total_rows",
    "total_unique_player_ids",
]


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Birden çok deney klasöründeki tracking_all.csv çıktılarını karşılaştırır."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        default=[
            Path("experiments/outputs/baseline_v8"),
            Path("experiments/outputs/custom_v1"),
        ],
    )
    parser.add_argument("--expected-players", type=int, default=14)
    return parser.parse_args()


def read_summary(csv_path: Path) -> dict[str, Any]:
    summary_path = csv_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"UYARI: summary.json okunamadı ({summary_path}): {exc}")
        return {}


def canonical_variant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "variant" not in df.columns:
        return df.copy()
    entities = (
        df["entity"].fillna("unknown").astype(str).str.lower()
        if "entity" in df.columns
        else pd.Series("all", index=df.index)
    )
    variants = df["variant"].fillna("").astype(str).str.lower()
    parts: list[pd.DataFrame] = []
    for entity in entities.unique():
        entity_mask = entities.eq(entity)
        if variants[entity_mask].eq("interpolated").any():
            parts.append(df[entity_mask & variants.eq("interpolated")])
        elif variants[entity_mask].eq("raw").any():
            parts.append(df[entity_mask & variants.eq("raw")])
        else:
            parts.append(df[entity_mask])
    return pd.concat(parts, ignore_index=True, sort=False) if parts else df.iloc[0:0].copy()


def interpolation_mask(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=df.index)
    if "is_interpolated" in df.columns:
        result |= (
            df["is_interpolated"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    for column in ("source", "match_type"):
        if column in df.columns:
            result |= (
                df[column]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("interpol", regex=False)
            )
    return result


def infer_sample_sec(data: pd.DataFrame, summary: dict[str, Any]) -> float:
    try:
        value = float(summary.get("sample_sec", 0.0))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    if {"sample_idx", "time_sec"}.issubset(data.columns):
        pairs = pd.DataFrame(
            {
                "sample_idx": pd.to_numeric(data["sample_idx"], errors="coerce"),
                "time_sec": pd.to_numeric(data["time_sec"], errors="coerce"),
            }
        ).dropna()
        pairs = pairs.drop_duplicates("sample_idx").sort_values("sample_idx")
        estimates = (pairs["time_sec"].diff() / pairs["sample_idx"].diff()).replace(
            [np.inf, -np.inf], np.nan
        )
        estimates = estimates[estimates > 0].dropna()
        if not estimates.empty:
            return float(estimates.median())
    return 0.1


def analyze_tracking_file(
    csv_path: Path,
    expected_players: int,
) -> dict[str, Any]:
    original = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    summary = read_summary(csv_path)
    data = canonical_variant(original)

    if "entity" not in data.columns:
        if "cls_name" in data.columns:
            names = data["cls_name"].fillna("").astype(str).str.lower()
            data["entity"] = np.where(names.str.contains("ball"), "ball", "player")
        else:
            data["entity"] = "player"
    entities = data["entity"].fillna("").astype(str).str.lower()
    players = data[entities.eq("player")].copy()
    balls = data[entities.eq("ball")].copy()

    use_sample_idx = "sample_idx" in data.columns
    if use_sample_idx:
        data["_group"] = pd.to_numeric(data["sample_idx"], errors="coerce").round().astype("Int64")
    elif "time_sec" in data.columns:
        data["_group"] = pd.to_numeric(data["time_sec"], errors="coerce").round(6)
    else:
        data["_group"] = np.arange(len(data))

    player_counts: dict[Any, int] = {}
    ball_groups: set[Any] = set()
    for group_key, group in data[data["_group"].notna()].groupby("_group"):
        group_entities = group["entity"].fillna("").astype(str).str.lower()
        group_players = group[group_entities.eq("player")]
        group_balls = group[group_entities.eq("ball")]
        if "stable_id" in group_players.columns and group_players["stable_id"].notna().any():
            count = int(group_players["stable_id"].nunique())
        else:
            count = int(len(group_players))
        normalized_key = int(group_key) if use_sample_idx else group_key
        player_counts[normalized_key] = count
        if not group_balls.empty:
            ball_groups.add(normalized_key)

    if use_sample_idx:
        observed_keys = player_counts.keys() | ball_groups
        maximum = max(observed_keys, default=0)
        sample_sec = infer_sample_sec(data, summary)
        try:
            duration = float(summary.get("duration_sec_used", summary.get("duration_sec", 0.0)))
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            maximum = max(maximum, int(math.floor(duration / sample_sec + 1e-6)))
        if not observed_keys and duration <= 0:
            frame_keys = []
        elif maximum <= 1_000_000:
            frame_keys = list(range(maximum + 1))
        else:
            frame_keys = sorted(observed_keys)
    else:
        frame_keys = sorted(player_counts.keys() | ball_groups)

    frame_player_counts = [player_counts.get(key, 0) for key in frame_keys]
    ball_found_count = sum(1 for key in frame_keys if key in ball_groups)

    confidence_column = "det_conf" if "det_conf" in players.columns else "conf"
    player_conf = (
        pd.to_numeric(players[confidence_column], errors="coerce")
        if confidence_column in players.columns
        else pd.Series(dtype=float)
    )
    position_conf = (
        pd.to_numeric(players["position_conf"], errors="coerce")
        if "position_conf" in players.columns
        else pd.Series(dtype=float)
    )
    interpolated = interpolation_mask(players)
    if "team" in players.columns and not players.empty:
        team_text = players["team"].fillna("unknown").astype(str).str.strip().str.lower()
        unknown_count = int(team_text.isin({"", "unknown", "nan", "none"}).sum())
    else:
        unknown_count = 0
    if "det_source" in players.columns and not players.empty:
        slice_count = int(
            players["det_source"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains("slice", regex=False)
            .sum()
        )
    else:
        slice_count = 0

    id_column = "stable_id" if "stable_id" in players.columns else "raw_track_id"
    unique_ids = (
        int(players[id_column].dropna().nunique()) if id_column in players.columns else 0
    )
    return {
        "total_rows": int(len(original)),
        "frame_player_counts": frame_player_counts,
        "ball_found_count": ball_found_count,
        "player_conf_sum": float(player_conf.sum()),
        "player_conf_count": int(player_conf.notna().sum()),
        "position_conf_sum": float(position_conf.sum()),
        "position_conf_count": int(position_conf.notna().sum()),
        "interpolated_count": int(interpolated.sum()),
        "player_row_count": int(len(players)),
        "unknown_team_count": unknown_count,
        "slice_detection_count": slice_count,
        "unique_player_ids": unique_ids,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def analyze_run(run_path: Path, expected_players: int) -> dict[str, Any]:
    csv_paths = sorted(run_path.rglob("tracking_all.csv")) if run_path.exists() else []
    print(f"Deney inceleniyor: {run_path} | tracking dosyası: {len(csv_paths)}")

    aggregate = {
        "total_rows": 0,
        "frame_player_counts": [],
        "ball_found_count": 0,
        "player_conf_sum": 0.0,
        "player_conf_count": 0,
        "position_conf_sum": 0.0,
        "position_conf_count": 0,
        "interpolated_count": 0,
        "player_row_count": 0,
        "unknown_team_count": 0,
        "slice_detection_count": 0,
        "unique_player_ids": 0,
    }
    successful_files = 0
    for csv_path in csv_paths:
        try:
            values = analyze_tracking_file(csv_path, expected_players)
        except Exception as exc:
            print(f"UYARI: Takip çıktısı okunamadı, atlandı ({csv_path}): {exc}")
            continue
        successful_files += 1
        aggregate["frame_player_counts"].extend(values["frame_player_counts"])
        for key in aggregate:
            if key != "frame_player_counts":
                aggregate[key] += values[key]

    counts = aggregate["frame_player_counts"]
    frame_count = len(counts)
    run_name = run_path.name or str(run_path)
    return {
        "run": run_name,
        "tracking_files": successful_files,
        "avg_players_per_frame": float(np.mean(counts)) if counts else math.nan,
        "median_players_per_frame": float(np.median(counts)) if counts else math.nan,
        "frames_below_expected_players": (
            int(sum(count < expected_players for count in counts)) if counts else 0
        ),
        "ball_found_frame_ratio": safe_ratio(aggregate["ball_found_count"], frame_count),
        "avg_player_conf": safe_ratio(
            aggregate["player_conf_sum"], aggregate["player_conf_count"]
        ),
        "avg_position_conf": safe_ratio(
            aggregate["position_conf_sum"], aggregate["position_conf_count"]
        ),
        "interpolated_ratio": safe_ratio(
            aggregate["interpolated_count"], aggregate["player_row_count"]
        ),
        "unknown_team_ratio": safe_ratio(
            aggregate["unknown_team_count"], aggregate["player_row_count"]
        ),
        "slice_detection_ratio": safe_ratio(
            aggregate["slice_detection_count"], aggregate["player_row_count"]
        ),
        "total_rows": aggregate["total_rows"],
        "total_unique_player_ids": aggregate["unique_player_ids"],
    }


def markdown_value(column: str, value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if column in {
        "tracking_files",
        "frames_below_expected_players",
        "total_rows",
        "total_unique_player_ids",
    }:
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def build_markdown(results: list[dict[str, Any]], expected_players: int) -> str:
    headers = METRIC_COLUMNS
    lines = [
        "# Deney Karşılaştırması",
        "",
        f"Beklenen oyuncu sayısı: **{expected_players}**",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(markdown_value(column, result.get(column)) for column in headers)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metrikler ne anlatır?",
            "",
            "- `avg_players_per_frame` / `median_players_per_frame`: Bir karede görülen tipik oyuncu sayısı.",
            "- `frames_below_expected_players`: Oyuncu sayısı beklenen değerin altında kalan kare sayısı.",
            "- `ball_found_frame_ratio`: En az bir top kaydı bulunan karelerin tüm karelere oranı.",
            "- `avg_player_conf`: Oyuncu tespit güvenlerinin ortalaması.",
            "- `avg_position_conf`: Oyuncu saha konumu güvenlerinin ortalaması.",
            "- `interpolated_ratio`: Son oyuncu görünümündeki interpolasyon satırlarının oranı.",
            "- `unknown_team_ratio`: Takımı `unknown` kalan oyuncu satırlarının oranı.",
            "- `slice_detection_ratio`: `det_source` alanında `slice` geçen oyuncu satırlarının oranı.",
            "- `total_rows`: tracking_all.csv dosyalarındaki ham toplam satır sayısı.",
            "- `total_unique_player_ids`: Video başına benzersiz oyuncu kimliklerinin toplamı.",
            "",
            "> Not: Bunlar ground-truth etiketleriyle ölçülmüş doğruluk metrikleri değil, "
            "deneyleri hızlı kıyaslamak için kullanılan sezgisel kalite göstergeleridir. "
            "V8 çıktısındaki `raw` ve `interpolated` tekrarlarını çift saymamak için "
            "kare metriklerinde son `interpolated` görünüm kullanılır.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.expected_players <= 0:
        print("HATA: --expected-players sıfırdan büyük olmalıdır.")
        return 2

    run_paths = [project_path(path) for path in args.runs]
    results = [analyze_run(path, args.expected_players) for path in run_paths]

    reports_dir = PROJECT_ROOT / "experiments/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "experiment_comparison.csv"
    markdown_path = reports_dir / "experiment_comparison.md"

    pd.DataFrame(results, columns=METRIC_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    markdown_path.write_text(
        build_markdown(results, args.expected_players),
        encoding="utf-8",
    )
    print(f"Karşılaştırma CSV raporu: {csv_path}")
    print(f"Karşılaştırma Markdown raporu: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
