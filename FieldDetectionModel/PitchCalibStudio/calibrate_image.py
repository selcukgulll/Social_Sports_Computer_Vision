from __future__ import annotations

import argparse
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

from src.dataset_utils import load_yaml, save_json, slugify, timestamp  # noqa: E402
from src.homography_utils import (  # noqa: E402
    estimate_homography_from_keypoints,
    sanity_check_homography,
)
from src.keypoint_utils import (  # noqa: E402
    load_pitch_schema,
    prediction_to_numpy,
    schema_correspondences,
)
from src.visualization import draw_calibration_debug, draw_top_view_pitch  # noqa: E402
from src.yolo_utils import load_pose_model, require_pitch_class  # noqa: E402


def _matrix_list(value: np.ndarray | None) -> list | None:
    return value.tolist() if value is not None else None


def analyze_image_array(
    image: np.ndarray,
    model: Any,
    schema: dict[str, Any],
    confidence_threshold: float,
    min_keypoints: int = 4,
    ransac_threshold_px: float = 8.0,
    max_error_px: float = 15.0,
    min_confidence_score: float = 0.35,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 12.0,
) -> dict[str, Any]:
    results = model.predict(source=image, conf=confidence_threshold, verbose=False)
    if not results:
        raise RuntimeError("Model sonuç döndürmedi.")
    bbox, keypoints_xy, confidences = prediction_to_numpy(results[0])
    if bbox is None or keypoints_xy is None or confidences is None:
        raise RuntimeError("Pitch pose tespiti bulunamadı.")
    image_points, field_points, used_indices = schema_correspondences(
        keypoints_xy,
        confidences,
        schema,
        confidence_threshold,
    )
    homography = estimate_homography_from_keypoints(
        image_points,
        field_points,
        min_points=min_keypoints,
        ransac_threshold_px=ransac_threshold_px,
    )
    sane = False
    sanity: dict[str, Any] = {}
    confidence_score = 0.0
    if homography:
        sane, sanity = sanity_check_homography(
            homography["H_image_to_field"],
            image.shape,
            min_area_ratio,
            max_area_ratio,
        )
        mean_confidence = float(np.mean(confidences[used_indices])) if used_indices else 0.0
        inlier_ratio = homography["inlier_count"] / max(1, len(used_indices))
        error_factor = math.exp(
            -homography["mean_reprojection_error_px"] / max(1.0, max_error_px)
        )
        confidence_score = float(mean_confidence * inlier_ratio * error_factor)
    accepted = bool(
        homography
        and len(used_indices) >= min_keypoints
        and homography["mean_reprojection_error_px"] <= max_error_px
        and sane
        and confidence_score >= min_confidence_score
    )
    overlay = draw_calibration_debug(
        image,
        bbox,
        keypoints_xy,
        confidences,
        schema,
        homography["H_field_to_image"] if homography else None,
        confidence_threshold,
    )
    top_labels = [
        f"{index} {schema['keypoints'][index]['name']}" for index in used_indices
    ]
    top_view = draw_top_view_pitch(field_points, top_labels)
    keypoint_rows = []
    for index, point in enumerate(keypoints_xy):
        item = schema["keypoints"].get(index, {})
        keypoint_rows.append(
            {
                "index": index,
                "name": item.get("name"),
                "x": float(point[0]),
                "y": float(point[1]),
                "confidence": float(confidences[index]),
                "used_for_homography": index in used_indices,
                "field_xy": item.get("field_xy"),
            }
        )
    return {
        "bbox": bbox,
        "keypoints_xy": keypoints_xy,
        "confidences": confidences,
        "image_points": image_points,
        "field_points": field_points,
        "used_indices": used_indices,
        "homography": homography,
        "sanity": sanity,
        "confidence_score": confidence_score,
        "accepted_auto": accepted,
        "overlay": overlay,
        "top_view": top_view,
        "keypoint_rows": keypoint_rows,
    }


def save_image_calibration(
    image_path: str | Path,
    model_path: str | Path,
    schema_path: str | Path,
    output: str | Path,
    confidence_threshold: float = 0.25,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    schema_path = Path(schema_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Görsel okunamadı: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    schema = load_pitch_schema(schema_path)
    config = (
        load_yaml(config_path).get("calibration", {})
        if config_path and Path(config_path).exists()
        else {}
    )
    model = load_pose_model(model_path)
    require_pitch_class(model)
    result = analyze_image_array(
        image,
        model,
        schema,
        confidence_threshold,
        min_keypoints=int(config.get("min_keypoints", 4)),
        ransac_threshold_px=float(config.get("ransac_reprojection_threshold_px", 8.0)),
        max_error_px=float(config.get("max_mean_reprojection_error_px", 15.0)),
        min_confidence_score=float(config.get("min_confidence_score", 0.35)),
        min_area_ratio=float(config.get("min_projected_pitch_area_ratio", 0.005)),
        max_area_ratio=float(config.get("max_projected_pitch_area_ratio", 12.0)),
    )
    run_dir = output / f"{timestamp()}_{slugify(image_path.stem)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    overlay_path = run_dir / "overlay.jpg"
    top_view_path = run_dir / "top_view.jpg"
    cv2.imwrite(str(overlay_path), result["overlay"])
    cv2.imwrite(str(top_view_path), result["top_view"])
    prediction = {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "confidence_threshold": confidence_threshold,
        "bbox_xyxy": _matrix_list(result["bbox"]),
        "keypoints": result["keypoint_rows"],
        "confidence_score": result["confidence_score"],
        "accepted_auto": result["accepted_auto"],
        "overlay_path": str(overlay_path),
        "top_view_path": str(top_view_path),
    }
    save_json(run_dir / "prediction.json", prediction)
    calibration_path: Path | None = None
    if result["homography"]:
        homography = result["homography"]
        calibration = {
            **prediction,
            "keypoints_used": result["used_indices"],
            "image_points": result["image_points"].tolist(),
            "field_points": result["field_points"].tolist(),
            "H_image_to_field": _matrix_list(homography["H_image_to_field"]),
            "H_field_to_image": _matrix_list(homography["H_field_to_image"]),
            "mean_reprojection_error_px": homography["mean_reprojection_error_px"],
            "point_errors_px": homography["point_errors_px"],
            "inliers": homography["inliers"].astype(bool).tolist(),
            "sanity": result["sanity"],
        }
        calibration_path = save_json(run_dir / "calibration.json", calibration)
        prediction["calibration"] = calibration
    prediction["run_dir"] = str(run_dir)
    prediction["calibration_path"] = str(calibration_path) if calibration_path else None
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tek karede pitch keypoint tahmini ve homography kalibrasyonu yapar."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--schema", default=str(PROJECT_ROOT / "configs" / "pitch_schema.yaml")
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "outputs" / "predictions")
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "app_config.yaml")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = save_image_calibration(
        args.image,
        args.model,
        args.schema,
        args.output,
        args.conf,
        args.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
