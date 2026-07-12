#!/usr/bin/env python
"""Ball-only YOLO modeliyle videoyu analiz eder ve Frame İncele CSV'si üretir."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


ROW_COLUMNS = [
    "sample_idx", "frame_idx", "time_sec", "stable_id", "raw_id", "entity",
    "team", "det_conf", "position_conf", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "bbox_w", "bbox_h", "image_x", "image_y", "field_x", "field_y",
    "source", "match_type", "is_interpolated", "missing_gap_sec", "det_source",
]


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def merge_detections(detections: list[Detection]) -> list[Detection]:
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        cx, cy = detection.center
        duplicate = False
        for kept in merged:
            kx, ky = kept.center
            kw = max(1.0, kept.bbox[2] - kept.bbox[0])
            kh = max(1.0, kept.bbox[3] - kept.bbox[1])
            if bbox_iou(detection.bbox, kept.bbox) >= 0.35 or math.hypot(cx - kx, cy - ky) <= max(5.0, 0.5 * max(kw, kh)):
                duplicate = True
                break
        if not duplicate:
            merged.append(detection)
    return merged


def ball_class_ids(model: YOLO) -> list[int]:
    names = getattr(model, "names", {}) or {}
    items = list(names.items()) if isinstance(names, dict) else list(enumerate(names))
    result = [int(index) for index, name in items if "ball" in str(name).strip().lower() or str(name).strip().lower() == "top"]
    if not result and len(items) == 1:
        index, name = items[0]
        print(f"UYARI: Model tek sınıflı ama sınıf adı 'ball' değil ({name!r}). Bu sınıf top kabul edilecek.")
        return [int(index)]
    if not result:
        raise ValueError(f"Modelde ball sınıfı bulunamadı. Sınıflar: {names}")
    return result


def result_detections(result: Any, *, offset_x: int = 0, offset_y: int = 0, source: str) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    output: list[Detection] = []
    for box, confidence in zip(xyxy, confidences):
        x1, y1, x2, y2 = [float(value) for value in box]
        output.append(Detection((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), float(confidence), source))
    return output


def tile_boxes(width: int, height: int, size: int, overlap: float, maximum: int) -> list[tuple[int, int, int, int]]:
    size = max(160, size)
    step = max(80, int(round(size * (1.0 - overlap))))
    xs = list(range(0, max(1, width - size + 1), step))
    ys = list(range(0, max(1, height - size + 1), step))
    if not xs or xs[-1] + size < width:
        xs.append(max(0, width - size))
    if not ys or ys[-1] + size < height:
        ys.append(max(0, height - size))
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(width, x + size), min(height, y + size)))
            if len(boxes) >= maximum:
                return boxes
    return boxes


class SingleBallTracker:
    def __init__(self, max_lost_sec: float = 2.0) -> None:
        self.max_lost_sec = max_lost_sec
        self.last: Detection | None = None
        self.last_time: float | None = None
        self.velocity = np.zeros(2, dtype=float)

    def choose(self, detections: list[Detection], time_sec: float) -> tuple[Detection | None, str]:
        if not detections:
            return None, "missing"
        if self.last is None or self.last_time is None:
            selected = max(detections, key=lambda item: item.confidence)
            self._update(selected, time_sec)
            return selected, "new"
        gap = max(0.001, time_sec - self.last_time)
        if gap > self.max_lost_sec:
            selected = max(detections, key=lambda item: item.confidence)
            self._update(selected, time_sec, reset_velocity=True)
            return selected, "reacquired"
        predicted = np.asarray(self.last.center) + self.velocity * gap
        gate = 90.0 + 300.0 * min(gap, 1.0)
        scored: list[tuple[float, Detection]] = []
        for detection in detections:
            center = np.asarray(detection.center)
            distance = float(np.linalg.norm(center - predicted))
            if distance > gate and detection.confidence < 0.55:
                continue
            continuity = max(0.0, 1.0 - distance / max(gate, 1.0))
            score = detection.confidence + 0.35 * continuity + 0.15 * bbox_iou(detection.bbox, self.last.bbox)
            scored.append((score, detection))
        selected = max(scored, key=lambda item: item[0])[1] if scored else max(detections, key=lambda item: item.confidence)
        match_type = "continued" if scored else "reacquired"
        self._update(selected, time_sec, reset_velocity=not scored)
        return selected, match_type

    def _update(self, detection: Detection, time_sec: float, reset_velocity: bool = False) -> None:
        if self.last is not None and self.last_time is not None and not reset_velocity:
            elapsed = max(0.001, time_sec - self.last_time)
            measured = (np.asarray(detection.center) - np.asarray(self.last.center)) / elapsed
            self.velocity = 0.65 * self.velocity + 0.35 * measured
        elif reset_velocity:
            self.velocity[:] = 0.0
        self.last = detection
        self.last_time = time_sec


def detection_row(detection: Detection, sample_index: int, frame_index: int, time_sec: float, match_type: str) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox
    center_x, center_y = detection.center
    return {
        "sample_idx": sample_index, "frame_idx": frame_index, "time_sec": round(time_sec, 6),
        "stable_id": "B1", "raw_id": "", "entity": "ball", "team": "ball",
        "det_conf": detection.confidence, "position_conf": detection.confidence,
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "bbox_w": x2 - x1, "bbox_h": y2 - y1, "image_x": center_x, "image_y": center_y,
        "field_x": "", "field_y": "", "source": "detected", "match_type": match_type,
        "is_interpolated": False, "missing_gap_sec": 0.0, "det_source": detection.source,
    }


def interpolate_rows(rows: list[dict[str, Any]], sample_sec: float, max_gap_sec: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    output: list[dict[str, Any]] = [dict(rows[0])]
    numeric = ("time_sec", "frame_idx", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_w", "bbox_h", "image_x", "image_y", "det_conf", "position_conf")
    for before, after in zip(rows, rows[1:]):
        missing = int(after["sample_idx"]) - int(before["sample_idx"]) - 1
        gap = float(after["time_sec"]) - float(before["time_sec"])
        if missing > 0 and gap <= max_gap_sec + sample_sec + 1e-9:
            for step in range(1, missing + 1):
                alpha = step / float(missing + 1)
                row = dict(before)
                row["sample_idx"] = int(before["sample_idx"]) + step
                for key in numeric:
                    row[key] = float(before[key]) + alpha * (float(after[key]) - float(before[key]))
                row["frame_idx"] = int(round(row["frame_idx"]))
                row["source"] = "interpolated"
                row["match_type"] = "interpolated"
                row["is_interpolated"] = True
                row["missing_gap_sec"] = gap
                row["det_source"] = "interpolation"
                output.append(row)
        output.append(dict(after))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ball-only video detection ve tracking")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-sec", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--interpolate-gap", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {video}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    if args.sample_sec <= 0 or args.imgsz <= 0:
        raise ValueError("sample-sec ve imgsz pozitif olmalı.")
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = frame_count / fps if frame_count and fps else 0.0
    start = max(0.0, args.start)
    end = video_duration if args.duration <= 0 else min(video_duration, start + args.duration)
    model = YOLO(str(model_path))
    class_ids = ball_class_ids(model)
    tracker = SingleBallTracker()
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    sample_times = np.arange(start, end + 1e-9, args.sample_sec)
    for sample_index, time_sec in tqdm(enumerate(sample_times), total=len(sample_times), desc="Top aranıyor"):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        predictions = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, classes=class_ids, device=args.device, verbose=False)
        detections = result_detections(predictions[0], source="full") if predictions else []
        if args.tile:
            height, width = frame.shape[:2]
            for tile_index, (x1, y1, x2, y2) in enumerate(tile_boxes(width, height, args.tile_size, args.tile_overlap, args.max_tiles)):
                tiled = model.predict(frame[y1:y2, x1:x2], conf=args.conf, iou=args.iou, imgsz=args.tile_size, classes=class_ids, device=args.device, verbose=False)
                if tiled:
                    detections.extend(result_detections(tiled[0], offset_x=x1, offset_y=y1, source=f"tile_{tile_index}"))
        detections = merge_detections(detections)
        candidate_count += len(detections)
        selected, match_type = tracker.choose(detections, float(time_sec))
        if selected is not None:
            rows.append(detection_row(selected, sample_index, int(round(float(time_sec) * fps)), float(time_sec), match_type))
    capture.release()
    raw = pd.DataFrame(rows, columns=ROW_COLUMNS)
    interpolated_rows = interpolate_rows(rows, args.sample_sec, args.interpolate_gap)
    interpolated = pd.DataFrame(interpolated_rows, columns=ROW_COLUMNS)
    raw["variant"] = "raw"
    interpolated["variant"] = "interpolated"
    all_rows = pd.concat([raw, interpolated], ignore_index=True)
    if not all_rows.empty:
        all_rows = all_rows.sort_values(["frame_idx", "variant"]).reset_index(drop=True)
    all_rows.to_csv(output / "tracking_all.csv", index=False, encoding="utf-8-sig")
    viewer = all_rows.copy()
    if "stable_id" in viewer:
        viewer.insert(0, "viewer_id", "ball_" + viewer["stable_id"].astype(str))
    viewer.to_csv(output / "tracking_viewer.csv", index=False, encoding="utf-8-sig")
    summary = {
        "mode": "ball_only", "video": str(video), "model": str(model_path), "out_dir": str(output),
        "fps": fps, "frame_count": frame_count, "duration_sec_used": max(0.0, end - start),
        "sample_sec": args.sample_sec, "confidence": args.conf, "imgsz": args.imgsz,
        "tile": args.tile, "candidate_detections": candidate_count, "ball_raw_rows": len(raw),
        "ball_interpolated_rows": len(interpolated), "max_players": 0, "created_at_unix": time.time(),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Ball-only sonuç hazır: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
