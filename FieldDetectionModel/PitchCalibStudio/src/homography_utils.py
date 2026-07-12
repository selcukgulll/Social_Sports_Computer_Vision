from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def image_to_field(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def field_to_image(points: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(H_inv, dtype=np.float64)).reshape(-1, 2)


def compute_reprojection_error(
    H: np.ndarray,
    image_points: np.ndarray,
    field_points: np.ndarray,
) -> tuple[float, list[float]]:
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    projected = field_to_image(field_points, H_inv)
    errors = np.linalg.norm(projected - np.asarray(image_points, dtype=float), axis=1)
    return float(np.mean(errors)), [float(value) for value in errors]


def estimate_homography_from_keypoints(
    image_keypoints: np.ndarray,
    schema_keypoints: np.ndarray,
    min_points: int = 4,
    ransac_threshold_px: float = 8.0,
) -> dict[str, Any] | None:
    image_points = np.asarray(image_keypoints, dtype=np.float32).reshape(-1, 2)
    field_points = np.asarray(schema_keypoints, dtype=np.float32).reshape(-1, 2)
    if len(image_points) < min_points or len(field_points) != len(image_points):
        return None
    # RANSAC threshold is expressed in destination units. Estimating
    # field->image first lets us use a meaningful pixel threshold.
    H_field_to_image, mask = cv2.findHomography(
        field_points,
        image_points,
        cv2.RANSAC,
        float(ransac_threshold_px),
    )
    if H_field_to_image is None or not np.all(np.isfinite(H_field_to_image)):
        return None
    try:
        H_image_to_field = np.linalg.inv(H_field_to_image)
        mean_error, point_errors = compute_reprojection_error(
            H_image_to_field, image_points, field_points
        )
    except np.linalg.LinAlgError:
        return None
    inliers = (
        mask.reshape(-1).astype(bool)
        if mask is not None
        else np.ones(len(image_points), dtype=bool)
    )
    return {
        "H_image_to_field": H_image_to_field,
        "H_field_to_image": H_field_to_image,
        "mean_reprojection_error_px": mean_error,
        "point_errors_px": point_errors,
        "inliers": inliers,
        "inlier_count": int(inliers.sum()),
    }


def sanity_check_homography(
    H: np.ndarray,
    image_shape: tuple[int, ...],
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 12.0,
) -> tuple[bool, dict[str, Any]]:
    height, width = image_shape[:2]
    diagnostics: dict[str, Any] = {}
    try:
        H = np.asarray(H, dtype=np.float64)
        H_inv = np.linalg.inv(H)
        condition_number = float(np.linalg.cond(H))
        corners = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        projected = field_to_image(corners, H_inv)
        area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        image_area = float(width * height)
        area_ratio = area / image_area if image_area else 0.0
        convex = bool(cv2.isContourConvex(projected.astype(np.float32)))
        finite = bool(np.all(np.isfinite(projected)))
        diagnostics.update(
            {
                "condition_number": condition_number,
                "projected_pitch_area_ratio": area_ratio,
                "projected_pitch_convex": convex,
                "projected_corners": projected.tolist(),
            }
        )
        accepted = (
            finite
            and convex
            and condition_number < 1e12
            and min_area_ratio <= area_ratio <= max_area_ratio
        )
        return accepted, diagnostics
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        diagnostics["error"] = str(exc)
        return False, diagnostics


def bbox_bottom_centers_to_field(bboxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    bottom_centers = np.column_stack(
        ((bboxes[:, 0] + bboxes[:, 2]) / 2.0, bboxes[:, 3])
    )
    return image_to_field(bottom_centers, H)
