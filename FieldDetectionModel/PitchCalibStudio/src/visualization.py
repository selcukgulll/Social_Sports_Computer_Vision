from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from src.homography_utils import field_to_image


GREEN = (40, 220, 80)
ORANGE = (20, 165, 255)
WHITE = (245, 245, 245)
RED = (40, 40, 235)


def _put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.55,
) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )
    cv2.rectangle(
        image,
        (x - 2, y - height - 4),
        (x + width + 3, y + baseline + 2),
        (15, 15, 15),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_keypoint_prediction(
    image: np.ndarray,
    bbox: Iterable[float] | None,
    keypoints_xy: np.ndarray,
    confidences: np.ndarray | None = None,
    schema: dict[str, Any] | None = None,
    confidence_threshold: float = 0.0,
    class_name: str = "pitch",
    large_indices: bool = False,
) -> np.ndarray:
    canvas = image.copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ORANGE, 2)
        _put_label(canvas, class_name, (x1, max(22, y1)), ORANGE)
    keypoints_xy = np.asarray(keypoints_xy, dtype=float).reshape(-1, 2)
    if confidences is None:
        confidences = np.ones(len(keypoints_xy), dtype=float)
    visible_legend: list[tuple[int, float]] = []
    for index, ((x, y), confidence) in enumerate(zip(keypoints_xy, confidences)):
        if float(confidence) < confidence_threshold or (x == 0 and y == 0):
            continue
        point = (int(round(x)), int(round(y)))
        point_color = GREEN if float(confidence) > 0 else (115, 115, 115)
        thickness = -1 if float(confidence) > 0 else 2
        cv2.circle(
            canvas,
            point,
            6 if large_indices else 4,
            point_color,
            thickness,
            cv2.LINE_AA,
        )
        if large_indices:
            text = str(index)
            origin = (point[0] + 8, max(24, point[1] - 8))
            cv2.putText(
                canvas,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                WHITE if float(confidence) > 0 else (175, 175, 175),
                2,
                cv2.LINE_AA,
            )
            visible_legend.append((index, float(confidence)))
            continue
        name = ""
        if schema and index in schema.get("keypoints", {}):
            name = schema["keypoints"][index]["name"]
        label = f"{index}"
        if name:
            label += f" {name}"
        label += f" {float(confidence):.2f}"
        _put_label(
            canvas,
            label,
            (point[0] + 7, max(20, point[1] - 7)),
            WHITE,
            0.45,
        )
    if large_indices:
        panel_width = 190
        panel = np.full((canvas.shape[0], panel_width, 3), (18, 24, 31), dtype=np.uint8)
        cv2.putText(
            panel,
            "INDEX / VIS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ORANGE,
            2,
            cv2.LINE_AA,
        )
        for row, (index, confidence) in enumerate(visible_legend):
            y = 58 + row * 27
            if y >= panel.shape[0] - 5:
                break
            legend_color = GREEN if confidence > 0 else (115, 115, 115)
            cv2.circle(panel, (18, y - 5), 5, legend_color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"{index:>2}   {confidence:.2f}",
                (32, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                WHITE,
                1,
                cv2.LINE_AA,
            )
        canvas = np.hstack((canvas, panel))
    return canvas


def draw_pitch_projection_on_image(
    image: np.ndarray,
    H_field_to_image: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    segments = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, 0.0]),
        ([0.5, 0.0], [0.5, 1.0]),
        ([0.0, 0.25], [0.2, 0.25]),
        ([0.2, 0.25], [0.2, 0.75]),
        ([0.2, 0.75], [0.0, 0.75]),
        ([1.0, 0.25], [0.8, 0.25]),
        ([0.8, 0.25], [0.8, 0.75]),
        ([0.8, 0.75], [1.0, 0.75]),
    ]
    for start, end in segments:
        projected = field_to_image(
            np.asarray([start, end], dtype=np.float32), H_field_to_image
        )
        if np.all(np.isfinite(projected)):
            cv2.line(
                canvas,
                tuple(np.round(projected[0]).astype(int)),
                tuple(np.round(projected[1]).astype(int)),
                ORANGE,
                3,
                cv2.LINE_AA,
            )
    circle = np.asarray(
        [
            [0.5 + 0.075 * np.cos(angle), 0.5 + 0.15 * np.sin(angle)]
            for angle in np.linspace(0, 2 * np.pi, 64)
        ],
        dtype=np.float32,
    )
    projected_circle = field_to_image(circle, H_field_to_image)
    if np.all(np.isfinite(projected_circle)):
        cv2.polylines(
            canvas,
            [np.round(projected_circle).astype(np.int32)],
            True,
            ORANGE,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_top_view_pitch(
    field_points: np.ndarray | None = None,
    labels: Iterable[str] | None = None,
    width: int = 1000,
    height: int = 600,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (44, 118, 55), dtype=np.uint8)
    margin = 55

    def pixel(point: Iterable[float]) -> tuple[int, int]:
        x, y = point
        return (
            int(round(margin + float(x) * (width - 2 * margin))),
            int(round(margin + float(y) * (height - 2 * margin))),
        )

    cv2.rectangle(canvas, pixel((0, 0)), pixel((1, 1)), WHITE, 3)
    cv2.line(canvas, pixel((0.5, 0)), pixel((0.5, 1)), WHITE, 3)
    center = pixel((0.5, 0.5))
    circle_points = np.asarray(
        [
            pixel((0.5 + 0.075 * np.cos(angle), 0.5 + 0.15 * np.sin(angle)))
            for angle in np.linspace(0, 2 * np.pi, 64)
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [circle_points], True, WHITE, 3, cv2.LINE_AA)
    cv2.circle(canvas, center, 5, WHITE, -1)
    cv2.polylines(
        canvas,
        [
            np.asarray(
                [pixel((0.0, 0.25)), pixel((0.2, 0.25)), pixel((0.2, 0.75)), pixel((0.0, 0.75))],
                dtype=np.int32,
            )
        ],
        False,
        WHITE,
        3,
        cv2.LINE_AA,
    )
    cv2.polylines(
        canvas,
        [
            np.asarray(
                [pixel((1.0, 0.25)), pixel((0.8, 0.25)), pixel((0.8, 0.75)), pixel((1.0, 0.75))],
                dtype=np.int32,
            )
        ],
        False,
        WHITE,
        3,
        cv2.LINE_AA,
    )
    cv2.line(canvas, pixel((0, 0.425)), pixel((0, 0.575)), ORANGE, 7)
    cv2.line(canvas, pixel((1, 0.425)), pixel((1, 0.575)), ORANGE, 7)

    if field_points is not None:
        labels = list(labels or [])
        for index, point in enumerate(np.asarray(field_points).reshape(-1, 2)):
            x, y = pixel(point)
            inside = 0 <= point[0] <= 1 and 0 <= point[1] <= 1
            cv2.circle(canvas, (x, y), 7, GREEN if inside else RED, -1)
            text = labels[index] if index < len(labels) else str(index)
            _put_label(canvas, text, (x + 9, y - 7), WHITE, 0.45)
    return canvas


def draw_calibration_debug(
    image: np.ndarray,
    bbox: np.ndarray | None,
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
    schema: dict[str, Any],
    H_field_to_image: np.ndarray | None,
    confidence_threshold: float,
) -> np.ndarray:
    canvas = draw_keypoint_prediction(
        image,
        bbox,
        keypoints_xy,
        confidences,
        schema,
        confidence_threshold,
    )
    if H_field_to_image is not None:
        canvas = draw_pitch_projection_on_image(canvas, H_field_to_image)
    return canvas


def make_contact_sheet(
    image_paths: Iterable[str | Path],
    output_path: str | Path,
    columns: int = 3,
    thumb_width: int = 520,
) -> Path:
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("Contact sheet için görsel yok.")
    thumbnails: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbnails.append(image.resize((thumb_width, int(image.height * ratio))))
    row_heights = [
        max(image.height for image in thumbnails[index : index + columns])
        for index in range(0, len(thumbnails), columns)
    ]
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, sum(row_heights)),
        (20, 20, 20),
    )
    y = 0
    for row_index, row_height in enumerate(row_heights):
        row = thumbnails[row_index * columns : (row_index + 1) * columns]
        for column, image in enumerate(row):
            sheet.paste(image, (column * thumb_width, y))
        y += row_height
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def write_html_gallery(image_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for path in image_paths:
        path = Path(path)
        relative = path.relative_to(output_path.parent).as_posix()
        cards.append(
            f'<figure><img src="{html.escape(relative)}" loading="lazy">'
            f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        )
    document = f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Keypoint Previews</title>
<style>
body{{background:#10151f;color:#edf2f7;font-family:Segoe UI,sans-serif;margin:24px}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
figure{{margin:0;background:#192231;border-radius:12px;padding:10px}}
img{{width:100%;height:auto;border-radius:8px}}figcaption{{padding:8px 2px}}
</style></head><body><h1>PitchCalibStudio Keypoint Previews</h1>
<main>{''.join(cards)}</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
