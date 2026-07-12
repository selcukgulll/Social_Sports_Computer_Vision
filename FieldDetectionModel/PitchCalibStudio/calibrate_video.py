from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_image import analyze_image_array  # noqa: E402
from src.dataset_utils import load_yaml, save_json, slugify, timestamp  # noqa: E402
from src.homography_utils import field_to_image  # noqa: E402
from src.keypoint_utils import load_pitch_schema  # noqa: E402
from src.yolo_utils import load_pose_model, require_pitch_class  # noqa: E402


def _stability_scores(candidates: list[dict[str, Any]], diagonal: float) -> None:
    valid = [item for item in candidates if item.get("homography")]
    if not valid:
        return
    corners = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    projected = np.asarray(
        [
            field_to_image(corners, item["homography"]["H_field_to_image"])
            for item in valid
        ]
    )
    median = np.median(projected, axis=0)
    for item, points in zip(valid, projected):
        jitter = float(np.mean(np.linalg.norm(points - median, axis=1)) / max(diagonal, 1))
        stability = float(math.exp(-jitter * 8.0))
        item["stability_score"] = stability
        item["selection_score"] = 0.7 * item["confidence_score"] + 0.3 * stability


def calibrate_video(
    video_path: str | Path,
    model_path: str | Path,
    schema_path: str | Path,
    seconds: float,
    sample_fps: float,
    confidence_threshold: float,
    output: str | Path,
    config_path: str | Path | None = None,
    min_keypoints: int | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    schema_path = Path(schema_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    if seconds <= 0 or sample_fps <= 0:
        raise ValueError("seconds ve sample-fps sıfırdan büyük olmalı.")
    config = load_yaml(config_path) if config_path and Path(config_path).exists() else {}
    calibration_config = config.get("calibration", {})
    schema = load_pitch_schema(schema_path)
    model = load_pose_model(model_path)
    require_pitch_class(model)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested_frames = max(1, int(round(seconds * fps)))
    max_frame = min(total_frames, requested_frames) if total_frames > 0 else requested_frames
    step = max(1, int(round(fps / sample_fps)))
    target_frames = list(range(0, max_frame, step))
    candidates: list[dict[str, Any]] = []
    analysis_kwargs = {
        "min_keypoints": int(
            min_keypoints
            if min_keypoints is not None
            else calibration_config.get("min_keypoints", 4)
        ),
        "ransac_threshold_px": float(
            calibration_config.get("ransac_reprojection_threshold_px", 8.0)
        ),
        "max_error_px": float(
            calibration_config.get("max_mean_reprojection_error_px", 15.0)
        ),
        "min_confidence_score": float(
            calibration_config.get("min_confidence_score", 0.35)
        ),
        "min_area_ratio": float(
            calibration_config.get("min_projected_pitch_area_ratio", 0.005)
        ),
        "max_area_ratio": float(
            calibration_config.get("max_projected_pitch_area_ratio", 12.0)
        ),
    }
    try:
        for frame_index in target_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / fps,
                        "error": "frame okunamadı",
                    }
                )
                continue
            try:
                result = analyze_image_array(
                    frame,
                    model,
                    schema,
                    confidence_threshold,
                    **analysis_kwargs,
                )
                compact_result = {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "overlay",
                        "top_view",
                        "keypoint_rows",
                        "bbox",
                        "keypoints_xy",
                        "confidences",
                    }
                }
                candidates.append(
                    {
                        **compact_result,
                        "frame_index": frame_index,
                        "time_seconds": frame_index / fps,
                    }
                )
            except Exception as exc:
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / fps,
                        "error": str(exc),
                    }
                )
    finally:
        capture.release()

    diagonal = math.hypot(width, height)
    _stability_scores(candidates, diagonal)
    valid = [item for item in candidates if item.get("homography")]
    if not valid:
        raise RuntimeError("Örneklenen karelerin hiçbirinde homography kurulamadı.")
    selected = max(valid, key=lambda item: item.get("selection_score", 0.0))
    selected_frame = int(selected["frame_index"])

    visual_capture = cv2.VideoCapture(str(video_path))
    try:
        visual_capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ok, selected_image = visual_capture.read()
        if not ok:
            raise RuntimeError("Seçilen en iyi kare yeniden okunamadı.")
    finally:
        visual_capture.release()
    selected_visual = analyze_image_array(
        selected_image,
        model,
        schema,
        confidence_threshold,
        **analysis_kwargs,
    )

    run_dir = output / f"{timestamp()}_{slugify(video_path.stem)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = run_dir / "best_frame_overlay.jpg"
    top_view_path = run_dir / "top_view.jpg"
    cv2.imwrite(str(overlay_path), selected_visual["overlay"])
    cv2.imwrite(str(top_view_path), selected_visual["top_view"])

    report_rows: list[dict[str, Any]] = []
    for item in candidates:
        homography = item.get("homography")
        report_rows.append(
            {
                "frame_index": item["frame_index"],
                "time_seconds": item["time_seconds"],
                "keypoints_used": len(item.get("used_indices", [])),
                "mean_reprojection_error_px": (
                    homography["mean_reprojection_error_px"] if homography else None
                ),
                "inlier_count": homography["inlier_count"] if homography else None,
                "confidence_score": item.get("confidence_score"),
                "stability_score": item.get("stability_score"),
                "selection_score": item.get("selection_score"),
                "accepted_auto": item.get("accepted_auto", False),
                "error": item.get("error"),
                "selected": item["frame_index"] == selected_frame,
            }
        )
    csv_path = run_dir / "sampled_frame_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    homography = selected["homography"]
    calibration = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "image_width": width,
        "image_height": height,
        "fps": fps,
        "sampled_frames": len(target_frames),
        "selected_frame_index": selected_frame,
        "selected_time_seconds": selected["time_seconds"],
        "schema_path": str(schema_path),
        "keypoints_used": selected["used_indices"],
        "image_points": selected["image_points"].tolist(),
        "field_points": selected["field_points"].tolist(),
        "H_image_to_field": homography["H_image_to_field"].tolist(),
        "H_field_to_image": homography["H_field_to_image"].tolist(),
        "mean_reprojection_error_px": homography["mean_reprojection_error_px"],
        "confidence_score": selected["confidence_score"],
        "stability_score": selected.get("stability_score", 0.0),
        "selection_score": selected.get("selection_score", 0.0),
        "accepted_auto": selected["accepted_auto"],
        "sanity": selected["sanity"],
        "overlay_path": str(overlay_path),
        "top_view_path": str(top_view_path),
        "sampled_frame_report": str(csv_path),
    }
    calibration_path = save_json(run_dir / "calibration.json", calibration)
    calibration["calibration_path"] = str(calibration_path)
    calibration["run_dir"] = str(run_dir)
    return calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Videonun ilk saniyelerinden sabit kamera saha kalibrasyonu çıkarır."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--schema", default=str(PROJECT_ROOT / "configs" / "pitch_schema.yaml")
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-keypoints", type=int)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "outputs" / "calibrations")
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "app_config.yaml")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = calibrate_video(
        args.video,
        args.model,
        args.schema,
        args.seconds,
        args.sample_fps,
        args.conf,
        args.output,
        args.config,
        args.min_keypoints,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
