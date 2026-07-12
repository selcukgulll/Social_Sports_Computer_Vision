from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GOOD_DATASET_HINTS = ("cancha_f5", "cancha-f5", "futsal-pitch-detection", "keypoints")
IGNORED_DATASET_HINTS = ("halisaha", "hali_saha", "halı", "low-quality")


@dataclass(frozen=True)
class ImageLabelPair:
    image: Path
    label: Path
    split: str


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "dataset"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML sözlük olmalı: {path}")
    return data


def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
    return path


def save_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["message"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def discover_data_yamls(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset kökü bulunamadı: {root}")
    return sorted(
        path for path in root.rglob("data.yaml")
        if not any(hint in str(path).lower() for hint in IGNORED_DATASET_HINTS)
    )


def is_allowed_dataset(path: str | Path) -> bool:
    lowered = str(path).lower().replace(" ", "_")
    if any(hint in lowered for hint in IGNORED_DATASET_HINTS):
        return False
    return any(hint in lowered for hint in GOOD_DATASET_HINTS)


def _split_aliases(split: str) -> list[str]:
    split = split.lower()
    if split in {"val", "valid", "validation"}:
        return ["val", "valid", "validation"]
    if split == "test":
        return ["test"]
    return ["train"]


def resolve_split_images(data_yaml: str | Path, split: str) -> Path | None:
    data_yaml = Path(data_yaml).expanduser().resolve()
    data = load_yaml(data_yaml)
    base = data_yaml.parent
    aliases = _split_aliases(split)
    yaml_keys = aliases if split != "valid" else ["val", "valid", "validation"]
    candidates: list[Path] = []
    for key in yaml_keys:
        value = data.get(key)
        if isinstance(value, str):
            candidates.append((base / value).resolve())
            candidates.append((base / value.replace("../", "", 1)).resolve())
    for alias in aliases:
        candidates.extend(
            [
                (base / alias / "images").resolve(),
                (base / "images" / alias).resolve(),
            ]
        )
    for candidate in candidates:
        if candidate.is_dir() and any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in candidate.iterdir()
        ):
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def label_for_image(image: Path, images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        labels_dir = Path(*parts)
    else:
        labels_dir = images_dir.parent / "labels"
    relative = image.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def list_image_label_pairs(data_yaml: str | Path, split: str) -> list[ImageLabelPair]:
    images_dir = resolve_split_images(data_yaml, split)
    if images_dir is None:
        return []
    images = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [
        ImageLabelPair(image=image, label=label_for_image(image, images_dir), split=split)
        for image in images
    ]


def parse_pose_row(
    line: str,
    keypoint_count: int,
    keypoint_dims: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    parts = line.strip().split()
    expected = 5 + keypoint_count * keypoint_dims
    if len(parts) != expected:
        return None, [f"sütun sayısı {len(parts)}, beklenen {expected}"]
    try:
        values = [float(value) for value in parts]
    except ValueError:
        return None, ["sayısal olmayan değer"]
    class_id = int(values[0])
    if values[0] != class_id or class_id < 0:
        errors.append("geçersiz class id")
    bbox = values[1:5]
    epsilon = 1e-6
    if any(value < -epsilon or value > 1.0 + epsilon for value in bbox):
        errors.append("bbox normalize değil")
    if bbox[2] <= 0 or bbox[3] <= 0:
        errors.append("bbox genişlik/yükseklik sıfır")
    flat = values[5:]
    keypoints = [
        flat[index : index + keypoint_dims]
        for index in range(0, len(flat), keypoint_dims)
    ]
    for index, point in enumerate(keypoints):
        if (
            point[0] < -epsilon
            or point[0] > 1.0 + epsilon
            or point[1] < -epsilon
            or point[1] > 1.0 + epsilon
        ):
            errors.append(f"keypoint {index} x/y normalize değil")
        if keypoint_dims == 3 and (point[2] < 0.0 or point[2] > 2.0):
            errors.append(f"keypoint {index} görünürlük değeri 0..2 dışında")
    return {"class_id": class_id, "bbox": bbox, "keypoints": keypoints}, errors


def inspect_dataset(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml = Path(data_yaml).expanduser().resolve()
    data = load_yaml(data_yaml)
    kpt_shape = data.get("kpt_shape")
    names = normalize_names(data.get("names"))
    report: dict[str, Any] = {
        "dataset_name": data_yaml.parent.name,
        "data_yaml": str(data_yaml),
        "task": data.get("task"),
        "names": names,
        "nc": data.get("nc", len(names)),
        "kpt_shape": kpt_shape,
        "is_pose": False,
        "valid_pose_rows": 0,
        "broken_rows": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "image_count": 0,
        "label_count": 0,
        "splits": {},
        "warnings": [],
        "errors": [],
        "broken_examples": [],
    }
    if (
        not isinstance(kpt_shape, list)
        or len(kpt_shape) != 2
        or not all(isinstance(value, int) for value in kpt_shape)
        or kpt_shape[0] < 1
        or kpt_shape[1] not in {2, 3}
    ):
        report["errors"].append("Geçerli kpt_shape [nokta_sayısı, 2|3] bulunamadı.")
        return report

    keypoint_count, keypoint_dims = kpt_shape
    for canonical, requested in (("train", "train"), ("val", "val"), ("test", "test")):
        pairs = list_image_label_pairs(data_yaml, requested)
        split_report = {
            "images_path": str(resolve_split_images(data_yaml, requested) or ""),
            "images": len(pairs),
            "labels": 0,
            "missing_labels": 0,
            "empty_labels": 0,
            "valid_rows": 0,
            "broken_rows": 0,
        }
        for pair in pairs:
            report["image_count"] += 1
            if not pair.label.exists():
                report["missing_labels"] += 1
                split_report["missing_labels"] += 1
                continue
            report["label_count"] += 1
            split_report["labels"] += 1
            text = pair.label.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                report["empty_labels"] += 1
                split_report["empty_labels"] += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                parsed, row_errors = parse_pose_row(line, keypoint_count, keypoint_dims)
                if parsed is not None and parsed["class_id"] not in names:
                    row_errors.append(
                        f"class id {parsed['class_id']} names/nc içinde değil"
                    )
                if parsed is None or row_errors:
                    report["broken_rows"] += 1
                    split_report["broken_rows"] += 1
                    if len(report["broken_examples"]) < 20:
                        report["broken_examples"].append(
                            {
                                "label": str(pair.label),
                                "line": line_number,
                                "errors": row_errors,
                            }
                        )
                else:
                    report["valid_pose_rows"] += 1
                    split_report["valid_rows"] += 1
        report["splits"][canonical] = split_report
    report["is_pose"] = report["valid_pose_rows"] > 0 and report["broken_rows"] == 0
    if report["missing_labels"]:
        report["warnings"].append(f"{report['missing_labels']} görselin etiketi eksik.")
    if report["empty_labels"]:
        report["warnings"].append(f"{report['empty_labels']} etiket dosyası boş.")
    if report["broken_rows"]:
        report["errors"].append(f"{report['broken_rows']} bozuk pose etiketi satırı bulundu.")
    if not report["valid_pose_rows"]:
        report["errors"].append("Geçerli YOLO pose etiketi bulunamadı.")
    return report


def compact_audit_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": report["dataset_name"],
        "is_pose": report["is_pose"],
        "kpt_shape": json.dumps(report["kpt_shape"]),
        "names": json.dumps(report["names"], ensure_ascii=False),
        "nc": report["nc"],
        "images": report["image_count"],
        "labels": report["label_count"],
        "missing_labels": report["missing_labels"],
        "empty_labels": report["empty_labels"],
        "valid_pose_rows": report["valid_pose_rows"],
        "broken_rows": report["broken_rows"],
        "warnings": " | ".join(report["warnings"]),
        "errors": " | ".join(report["errors"]),
        "data_yaml": report["data_yaml"],
    }
