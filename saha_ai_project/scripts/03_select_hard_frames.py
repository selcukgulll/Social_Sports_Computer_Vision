#!/usr/bin/env python
"""Score tracking output frames and extract the hardest examples for labeling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # Allows --help and a clearer runtime error.
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
INDEX_COLUMNS = [
    "image_path",
    "video_path",
    "output_dir",
    "video_stem",
    "time_sec",
    "frame_idx",
    "hard_score",
    "reasons",
]


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="tracking_all.csv dosyalarından zor/hatalı kareleri seçip videodan çıkarır."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("experiments/outputs/baseline_v8"),
    )
    parser.add_argument("--videos-dir", type=Path, default=Path("videos/raw"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("frames/hard_frames/baseline_v8"),
    )
    parser.add_argument("--expected-players", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=1.0,
        help="Aynı videodan seçilen iki kare arasındaki en düşük zaman farkı.",
    )
    return parser.parse_args()


def read_summary(csv_path: Path) -> dict[str, Any]:
    summary_path = csv_path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"UYARI: summary.json okunamadı ({summary_path}): {exc}")
        return {}


def canonical_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Use V8's final interpolated view once, avoiding raw/interpolated duplicates."""
    if df.empty or "variant" not in df.columns:
        return df.copy()

    entity_series = (
        df["entity"].fillna("unknown").astype(str).str.lower()
        if "entity" in df.columns
        else pd.Series("all", index=df.index)
    )
    variant_series = df["variant"].fillna("").astype(str).str.lower()
    parts: list[pd.DataFrame] = []

    for entity in entity_series.unique():
        entity_mask = entity_series == entity
        entity_variants = variant_series[entity_mask]
        if (entity_variants == "interpolated").any():
            parts.append(df[entity_mask & (variant_series == "interpolated")])
        elif (entity_variants == "raw").any():
            parts.append(df[entity_mask & (variant_series == "raw")])
        else:
            parts.append(df[entity_mask])

    return pd.concat(parts, ignore_index=True, sort=False) if parts else df.iloc[0:0].copy()


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def interpolation_mask(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=df.index)
    if "is_interpolated" in df.columns:
        text = df["is_interpolated"].fillna("").astype(str).str.lower()
        result |= text.isin({"true", "1", "yes"})
    for column in ("source", "match_type"):
        if column in df.columns:
            result |= df[column].fillna("").astype(str).str.lower().str.contains("interpol", regex=False)
    return result


def infer_sample_sec(df: pd.DataFrame, summary: dict[str, Any]) -> float:
    try:
        sample_sec = float(summary.get("sample_sec", 0.0))
        if sample_sec > 0:
            return sample_sec
    except (TypeError, ValueError):
        pass

    if {"sample_idx", "time_sec"}.issubset(df.columns):
        pairs = pd.DataFrame(
            {
                "sample_idx": pd.to_numeric(df["sample_idx"], errors="coerce"),
                "time_sec": pd.to_numeric(df["time_sec"], errors="coerce"),
            }
        ).dropna()
        pairs = pairs.drop_duplicates("sample_idx").sort_values("sample_idx")
        index_delta = pairs["sample_idx"].diff()
        time_delta = pairs["time_sec"].diff()
        estimates = (time_delta / index_delta).replace([np.inf, -np.inf], np.nan)
        estimates = estimates[estimates > 0].dropna()
        if not estimates.empty:
            return float(estimates.median())
    return 0.1


def frame_rows(
    df: pd.DataFrame,
    summary: dict[str, Any],
    expected_players: int,
) -> pd.DataFrame:
    data = canonical_variant(df)
    if "entity" not in data.columns:
        if "cls_name" in data.columns:
            names = data["cls_name"].fillna("").astype(str).str.lower()
            data["entity"] = np.where(names.str.contains("ball"), "ball", "player")
        else:
            data["entity"] = "player"

    data["_entity"] = data["entity"].fillna("").astype(str).str.lower()
    if "sample_idx" in data.columns:
        data["_sample_idx"] = pd.to_numeric(data["sample_idx"], errors="coerce")
    else:
        data["_sample_idx"] = np.nan
    data["_time_sec"] = numeric_series(data, "time_sec")
    data["_frame_idx"] = numeric_series(data, "frame_idx")
    data["_det_conf"] = numeric_series(data, "det_conf")
    if data["_det_conf"].isna().all() and "conf" in data.columns:
        data["_det_conf"] = numeric_series(data, "conf")
    data["_position_conf"] = numeric_series(data, "position_conf")
    data["_field_x"] = numeric_series(data, "field_x")
    data["_field_y"] = numeric_series(data, "field_y")
    data["_interpolated"] = interpolation_mask(data)

    # Even a completely empty V8 CSV still declares sample_idx in its header.
    # In that case summary.json lets us reconstruct missed frames.
    use_sample_idx = "sample_idx" in data.columns
    if use_sample_idx:
        data["_group"] = data["_sample_idx"].round().astype("Int64")
    else:
        data["_group"] = data["_time_sec"].round(6)

    sample_sec = infer_sample_sec(data, summary)
    try:
        fps = float(summary.get("fps", 0.0))
    except (TypeError, ValueError):
        fps = 0.0
    try:
        pitch_length = float(summary.get("pitch_length", 100.0))
        pitch_width = float(summary.get("pitch_width", 60.0))
    except (TypeError, ValueError):
        pitch_length, pitch_width = 100.0, 60.0

    grouped_records: dict[float | int, dict[str, Any]] = {}
    valid_data = data[data["_group"].notna()]
    for group_key, group in valid_data.groupby("_group", sort=True):
        players = group[group["_entity"].eq("player")]
        balls = group[group["_entity"].eq("ball")]

        if "stable_id" in players.columns and players["stable_id"].notna().any():
            player_count = int(players["stable_id"].nunique())
        else:
            player_count = int(len(players))

        if not players.empty:
            outside = (
                players["_field_x"].isna()
                | players["_field_y"].isna()
                | players["_field_x"].lt(0.0)
                | players["_field_x"].gt(pitch_length)
                | players["_field_y"].lt(0.0)
                | players["_field_y"].gt(pitch_width)
            )
            outside_ratio = float(outside.mean())
        else:
            outside_ratio = 0.0

        if "team" in players.columns and not players.empty:
            teams = players["team"].fillna("unknown").astype(str).str.strip().str.lower()
            unknown_team_ratio = float(teams.isin({"", "unknown", "nan", "none"}).mean())
        else:
            unknown_team_ratio = 0.0

        if "det_source" in players.columns and not players.empty:
            slice_ratio = float(
                players["det_source"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("slice", regex=False)
                .mean()
            )
        else:
            slice_ratio = 0.0

        time_values = group["_time_sec"].dropna()
        frame_values = group["_frame_idx"].dropna()
        time_sec = (
            float(time_values.median())
            if not time_values.empty
            else float(group_key) * sample_sec if use_sample_idx else float(group_key)
        )
        frame_idx = (
            int(round(float(frame_values.median())))
            if not frame_values.empty
            else int(round(time_sec * fps)) if fps > 0 else 0
        )
        ball_x = float(balls["_field_x"].mean()) if not balls.empty else math.nan
        ball_y = float(balls["_field_y"].mean()) if not balls.empty else math.nan

        grouped_records[int(group_key) if use_sample_idx else float(group_key)] = {
            "sample_idx": int(group_key) if use_sample_idx else math.nan,
            "time_sec": time_sec,
            "frame_idx": frame_idx,
            "player_count": player_count,
            "ball_found": not balls.empty,
            "avg_player_conf": float(players["_det_conf"].mean()) if not players.empty else math.nan,
            "avg_position_conf": (
                float(players["_position_conf"].mean()) if not players.empty else math.nan
            ),
            "interpolated_ratio": (
                float(players["_interpolated"].mean()) if not players.empty else 0.0
            ),
            "outside_pitch_ratio": outside_ratio,
            "unknown_team_ratio": unknown_team_ratio,
            "slice_detection_ratio": slice_ratio,
            "ball_x": ball_x,
            "ball_y": ball_y,
        }

    if use_sample_idx:
        observed_max = max(grouped_records.keys(), default=0)
        try:
            duration = float(summary.get("duration_sec_used", summary.get("duration_sec", 0.0)))
        except (TypeError, ValueError):
            duration = 0.0
        duration_max = int(math.floor(duration / sample_sec + 1e-6)) if duration > 0 else 0
        maximum = max(int(observed_max), duration_max)
        if maximum > 1_000_000:
            print("UYARI: Olağandışı frame aralığı görüldü; yalnız gözlenen sample_idx değerleri kullanılacak.")
        else:
            for sample_idx in range(maximum + 1):
                if sample_idx not in grouped_records:
                    time_sec = sample_idx * sample_sec
                    grouped_records[sample_idx] = {
                        "sample_idx": sample_idx,
                        "time_sec": time_sec,
                        "frame_idx": int(round(time_sec * fps)) if fps > 0 else sample_idx,
                        "player_count": 0,
                        "ball_found": False,
                        "avg_player_conf": math.nan,
                        "avg_position_conf": math.nan,
                        "interpolated_ratio": 0.0,
                        "outside_pitch_ratio": 0.0,
                        "unknown_team_ratio": 0.0,
                        "slice_detection_ratio": 0.0,
                        "ball_x": math.nan,
                        "ball_y": math.nan,
                    }

    frames = pd.DataFrame(grouped_records.values())
    if frames.empty:
        return frames
    frames = frames.sort_values("time_sec").reset_index(drop=True)

    previous_count = frames["player_count"].shift(1)
    next_count = frames["player_count"].shift(-1)
    neighbor_count = pd.concat([previous_count, next_count], axis=1).mean(axis=1)
    frames["player_count_change"] = (frames["player_count"] - neighbor_count).abs().fillna(0.0)

    found_time = frames["time_sec"].where(frames["ball_found"])
    previous_ball_time = found_time.ffill().shift(1)
    previous_ball_x = frames["ball_x"].where(frames["ball_found"]).ffill().shift(1)
    previous_ball_y = frames["ball_y"].where(frames["ball_found"]).ffill().shift(1)
    ball_dt = frames["time_sec"] - previous_ball_time
    ball_distance = np.hypot(
        frames["ball_x"] - previous_ball_x,
        frames["ball_y"] - previous_ball_y,
    )
    frames["ball_speed"] = np.where(
        frames["ball_found"] & ball_dt.gt(0),
        ball_distance / ball_dt,
        np.nan,
    )

    scores: list[float] = []
    reasons_list: list[str] = []
    for row in frames.to_dict("records"):
        score = 0.0
        reasons: list[str] = []
        player_count = int(row["player_count"])

        if player_count < expected_players:
            deficit = expected_players - player_count
            score += 4.0 * deficit / max(expected_players, 1)
            reasons.append(f"eksik_oyuncu={player_count}/{expected_players}")

        if not bool(row["ball_found"]):
            score += 2.0
            reasons.append("top_yok")

        det_conf = row["avg_player_conf"]
        if pd.notna(det_conf):
            score += 1.5 * max(0.0, 0.55 - float(det_conf)) / 0.55
            if float(det_conf) < 0.50:
                reasons.append(f"dusuk_det_conf={float(det_conf):.3f}")

        position_conf = row["avg_position_conf"]
        if pd.notna(position_conf):
            score += 1.25 * max(0.0, 0.55 - float(position_conf)) / 0.55
            if float(position_conf) < 0.50:
                reasons.append(f"dusuk_position_conf={float(position_conf):.3f}")

        interpolation_ratio = float(row["interpolated_ratio"])
        score += 1.5 * interpolation_ratio
        if interpolation_ratio > 0:
            reasons.append(f"interpolasyon={interpolation_ratio:.2f}")

        outside_ratio = float(row["outside_pitch_ratio"])
        score += 2.5 * outside_ratio
        if outside_ratio > 0:
            reasons.append(f"saha_disi_konum={outside_ratio:.2f}")

        unknown_ratio = float(row["unknown_team_ratio"])
        score += unknown_ratio
        if unknown_ratio >= 0.25:
            reasons.append(f"bilinmeyen_takim={unknown_ratio:.2f}")

        slice_ratio = float(row["slice_detection_ratio"])
        score += 0.4 * slice_ratio
        if slice_ratio > 0:
            reasons.append(f"slice_tespiti={slice_ratio:.2f}")

        count_change = float(row["player_count_change"])
        score += min(2.0, 2.0 * count_change / max(expected_players, 1))
        if count_change >= max(2.0, expected_players * 0.20):
            reasons.append(f"ani_oyuncu_sayisi_deg_={count_change:.1f}")

        ball_speed = row["ball_speed"]
        if pd.notna(ball_speed) and float(ball_speed) > 25.0:
            score += min(2.0, (float(ball_speed) - 25.0) / 25.0)
            reasons.append(f"gercekdisi_top_hizi={float(ball_speed):.1f}")

        scores.append(round(score, 6))
        reasons_list.append("; ".join(reasons) if reasons else "dusuk_oncelikli_kalite_kontrolu")

    frames["hard_score"] = scores
    frames["reasons"] = reasons_list
    return frames


def video_index(videos_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not videos_dir.exists():
        return result
    for path in videos_dir.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            result.setdefault(path.stem.lower(), []).append(path.resolve())
    for paths in result.values():
        paths.sort(key=lambda item: item.name.lower())
    return result


def main() -> int:
    args = parse_args()
    if cv2 is None:
        print("HATA: opencv-python kurulu değil. Kurulum: pip install opencv-python")
        return 1
    if args.expected_players <= 0:
        print("HATA: --expected-players sıfırdan büyük olmalıdır.")
        return 2
    if args.top_k < 0 or args.min_gap_sec < 0:
        print("HATA: --top-k ve --min-gap-sec negatif olamaz.")
        return 2

    outputs_root = project_path(args.outputs_root)
    videos_dir = project_path(args.videos_dir)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(outputs_root.rglob("tracking_all.csv")) if outputs_root.exists() else []
    videos_by_stem = video_index(videos_dir)
    print(f"Takip çıktısı kökü: {outputs_root}")
    print(f"Bulunan tracking_all.csv sayısı: {len(csv_paths)}")

    all_candidates: list[dict[str, Any]] = []
    for csv_number, csv_path in enumerate(csv_paths, start=1):
        video_stem = csv_path.parent.name
        matches = videos_by_stem.get(video_stem.lower(), [])
        if not matches:
            print(
                f"UYARI: Kaynak video bulunamadı; çıktı atlanıyor: "
                f"{csv_path} (aranan stem: {video_stem})"
            )
            continue
        if len(matches) > 1:
            print(
                f"UYARI: {video_stem} için birden çok video bulundu; "
                f"ilk eşleşme kullanılacak: {matches[0]}"
            )
        video_path = matches[0]

        print(f"[{csv_number}/{len(csv_paths)}] Puanlanıyor: {csv_path}")
        try:
            dataframe = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
            summary = read_summary(csv_path)
            frames = frame_rows(dataframe, summary, args.expected_players)
        except Exception as exc:
            print(f"UYARI: CSV puanlanamadı, atlanıyor ({csv_path}): {exc}")
            continue

        for record in frames.to_dict("records"):
            record.update(
                {
                    "video_path": str(video_path),
                    "video_stem": video_stem,
                    "output_dir": str(csv_path.parent.resolve()),
                }
            )
            all_candidates.append(record)
        print(f"  Puanlanan zaman noktası: {len(frames)}")

    ranked = sorted(
        all_candidates,
        key=lambda row: (-float(row["hard_score"]), str(row["video_path"]), float(row["time_sec"])),
    )

    selected_times: dict[str, list[float]] = {}
    captures: dict[str, Any] = {}
    index_records: list[dict[str, Any]] = []
    try:
        for candidate in ranked:
            if len(index_records) >= args.top_k:
                break
            video_path_text = str(candidate["video_path"])
            time_sec = float(candidate["time_sec"])
            prior_times = selected_times.setdefault(video_path_text.lower(), [])
            if any(abs(time_sec - prior) < args.min_gap_sec - 1e-9 for prior in prior_times):
                continue

            capture = captures.get(video_path_text)
            if capture is None:
                capture = cv2.VideoCapture(video_path_text)
                if not capture.isOpened():
                    print(f"UYARI: Video açılamadı, seçili kareler atlanacak: {video_path_text}")
                    captures[video_path_text] = False
                    continue
                captures[video_path_text] = capture
            if capture is False:
                continue

            capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                print(f"UYARI: Kare okunamadı: {video_path_text} @ {time_sec:.3f} sn")
                continue

            video_stem = str(candidate["video_stem"])
            image_folder = out_dir / video_stem
            image_folder.mkdir(parents=True, exist_ok=True)
            image_path = image_folder / f"{video_stem}_t{time_sec:09.2f}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                print(f"UYARI: Zor kare yazılamadı: {image_path}")
                continue

            prior_times.append(time_sec)
            index_records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "video_path": video_path_text,
                    "output_dir": str(candidate["output_dir"]),
                    "video_stem": video_stem,
                    "time_sec": round(time_sec, 6),
                    "frame_idx": int(round(float(candidate["frame_idx"]))),
                    "hard_score": round(float(candidate["hard_score"]), 6),
                    "reasons": str(candidate["reasons"]),
                }
            )
            if len(index_records) % 25 == 0:
                print(f"  {len(index_records)} zor kare çıkarıldı...")
    finally:
        for capture in captures.values():
            if capture is not False:
                capture.release()

    index_path = out_dir / "hard_frames_index.csv"
    pd.DataFrame(index_records, columns=INDEX_COLUMNS).to_csv(
        index_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Zor kare indeksi yazıldı: {index_path}")
    print(f"Seçilen ve çıkarılan zor kare: {len(index_records)} / hedef {args.top_k}")
    if not csv_paths:
        print("Bilgi: Henüz baseline tracking_all.csv çıktısı yok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
