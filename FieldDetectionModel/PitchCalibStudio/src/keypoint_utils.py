from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_utils import load_yaml


def load_pitch_schema(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    schema = load_yaml(path)
    raw_keypoints = schema.get("keypoints", {})
    if not isinstance(raw_keypoints, dict) or not raw_keypoints:
        raise ValueError(f"Şemada keypoints bulunamadı: {path}")
    keypoints: dict[int, dict[str, Any]] = {}
    for raw_index, value in raw_keypoints.items():
        index = int(raw_index)
        if not isinstance(value, dict) or "field_xy" not in value:
            raise ValueError(f"Şema keypoint {index} için field_xy eksik.")
        xy = value["field_xy"]
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError(f"Şema keypoint {index} field_xy iki değer olmalı.")
        keypoints[index] = {
            **value,
            "name": str(value.get("name", f"keypoint_{index}")),
            "field_xy": [float(xy[0]), float(xy[1])],
        }
    expected = list(range(max(keypoints) + 1))
    if sorted(keypoints) != expected:
        raise ValueError("Şema keypoint indeksleri 0'dan başlayıp kesintisiz ilerlemeli.")
    schema["keypoints"] = keypoints
    schema["path"] = str(path)
    return schema


def schema_keypoint_count(schema: dict[str, Any]) -> int:
    return len(schema["keypoints"])


def remap_keypoints(
    source: list[list[float]],
    source_to_target: dict[int | str, int | str | None],
    target_count: int,
    target_dims: int = 3,
) -> list[list[float]]:
    missing = [0.0] * target_dims
    target = [missing.copy() for _ in range(target_count)]
    used_targets: set[int] = set()
    for raw_source, raw_target in source_to_target.items():
        if raw_target is None:
            continue
        source_index, target_index = int(raw_source), int(raw_target)
        if source_index < 0 or source_index >= len(source):
            raise ValueError(f"Kaynak keypoint indeksi aralık dışında: {source_index}")
        if target_index < 0 or target_index >= target_count:
            raise ValueError(f"Hedef keypoint indeksi aralık dışında: {target_index}")
        if target_index in used_targets:
            raise ValueError(f"Hedef keypoint iki kez eşlendi: {target_index}")
        used_targets.add(target_index)
        point = list(source[source_index])
        if target_dims == 3 and len(point) == 2:
            point.append(2.0)
        target[target_index] = point[:target_dims]
    return target


def prediction_to_numpy(result: Any) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return None, None, None
    box_conf = boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(box_conf))
    bbox = boxes.xyxy[best].detach().cpu().numpy().astype(float)
    xy = keypoints.xy[best].detach().cpu().numpy().astype(float)
    conf_tensor = getattr(keypoints, "conf", None)
    if conf_tensor is None:
        conf = np.ones((xy.shape[0],), dtype=float)
    else:
        conf = conf_tensor[best].detach().cpu().numpy().astype(float)
    return bbox, xy, conf


def schema_correspondences(
    image_xy: np.ndarray,
    confidences: np.ndarray,
    schema: dict[str, Any],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    image_points: list[list[float]] = []
    field_points: list[list[float]] = []
    indices: list[int] = []
    for index, item in schema["keypoints"].items():
        if index >= len(image_xy) or index >= len(confidences):
            continue
        point = image_xy[index]
        confidence = float(confidences[index])
        if confidence < confidence_threshold or not np.all(np.isfinite(point)):
            continue
        if float(point[0]) == 0.0 and float(point[1]) == 0.0:
            continue
        image_points.append([float(point[0]), float(point[1])])
        field_points.append([float(v) for v in item["field_xy"]])
        indices.append(index)
    return (
        np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
        np.asarray(field_points, dtype=np.float32).reshape(-1, 2),
        indices,
    )


def field_to_meters(points: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    points[:, 0] *= float(schema.get("pitch_length_m", 1.0))
    points[:, 1] *= float(schema.get("pitch_width_m", 1.0))
    return points
