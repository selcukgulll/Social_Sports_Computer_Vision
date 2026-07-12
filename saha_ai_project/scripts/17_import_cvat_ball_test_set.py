#!/usr/bin/env python
"""Import a CVAT YOLO export as a ball-only real-case test set."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BALL_NAMES = {"ball", "top", "football", "soccer ball", "sports ball"}


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CVAT YOLO ZIP exportunu yalnız ball içeren test datasetine çevirir."
    )
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument(
        "--labels-source",
        type=Path,
        help="CVAT ZIP yerine kullanılacak YOLO label dataset klasörü.",
    )
    parser.add_argument(
        "--images-source",
        type=Path,
        action="append",
        default=[],
        help="ZIP içinde image yoksa görsellerin aranacağı klasör. Birden fazla verilebilir.",
    )
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Hangi görsellerin hangi sırayla alınacağını belirleyen train.txt/list file.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Image list kullanılıyorsa ilk N görseli al. 0=tümü.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/BallOnly/real_case_test_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"ZIP içinde güvensiz yol bulundu: {member.filename}") from exc
    archive.extractall(destination)


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig", errors="replace")) or {}


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        def key(value: Any) -> tuple[int, str]:
            try:
                return (0, f"{int(value):08d}")
            except (TypeError, ValueError):
                return (1, str(value))

        return [str(names[item]) for item in sorted(names, key=key)]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def find_class_names(root: Path) -> list[str]:
    for data_yaml in sorted(root.rglob("data.yaml")):
        names = normalize_names(read_yaml(data_yaml).get("names"))
        if names:
            return names
    for name_file in sorted(root.rglob("obj.names")) + sorted(root.rglob("classes.txt")):
        lines = [
            line.strip()
            for line in name_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            if line.strip()
        ]
        if lines:
            return lines
    return []


def ball_source_ids(class_names: list[str], label_paths: list[Path]) -> set[int]:
    if class_names:
        ids = {
            index
            for index, name in enumerate(class_names)
            if name.strip().lower() in BALL_NAMES or "ball" in name.strip().lower()
        }
        if ids:
            return ids
        if len(class_names) == 1:
            return {0}
        raise ValueError(f"Ball sınıfı bulunamadı. Sınıflar: {class_names}")

    seen: set[int] = set()
    for label_path in label_paths[:500]:
        for raw_line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            try:
                seen.add(int(float(parts[0])))
            except ValueError:
                continue
    if not seen or seen == {0}:
        return {0}
    raise ValueError(f"Sınıf isimleri yok ve birden fazla class id görüldü: {sorted(seen)}")


def find_images(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def find_images_from_roots(roots: list[Path]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for stem, path in find_images(root).items():
            images.setdefault(stem, path)
    return images


def is_label_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    lowered = path.name.lower()
    if lowered in {"classes.txt", "obj.names", "train.txt", "val.txt", "valid.txt", "test.txt"}:
        return False
    return True


def find_labels(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(root.rglob("*.txt"))
        if path.is_file() and is_label_file(path)
    }


def read_image_list(path: Path, max_images: int = 0) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        stem = Path(line).stem
        if stem and stem not in seen:
            stems.append(stem)
            seen.add(stem)
        if max_images > 0 and len(stems) >= max_images:
            break
    return stems


def default_image_list(label_root: Path) -> Path | None:
    for name in ("train.txt", "val.txt", "valid.txt", "test.txt"):
        path = label_root / name
        if path.is_file():
            return path
    return None


def parse_row(raw_line: str, ball_ids: set[int]) -> str | None:
    parts = raw_line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(item) for item in parts[1:]]
    except ValueError:
        return None
    if class_id not in ball_ids:
        return None

    if len(values) == 4:
        x, y, w, h = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
    else:
        return None

    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0, max(0.0, w))
    h = min(1.0, max(0.0, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}"


def safe_name(stem: str, suffix: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("._-") or "frame"
    return f"{clean[:120]}{suffix.lower()}"


def reset_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Çıktı zaten var: {output}. --overwrite kullanın.")
        allowed = (PROJECT_ROOT / "datasets/BallOnly").resolve()
        if allowed not in output.resolve().parents:
            raise ValueError(f"Güvenlik için proje dışı çıktı silinmedi: {output}")
        shutil.rmtree(output)
    (output / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "test").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    zip_path = resolve(args.zip_path) if args.zip_path else None
    labels_source = resolve(args.labels_source) if args.labels_source else None
    output = resolve(args.output)
    image_sources = [resolve(path) for path in args.images_source]
    if zip_path is None and labels_source is None:
        raise ValueError("--zip veya --labels-source verilmelidir.")
    if zip_path is not None and not zip_path.is_file():
        raise FileNotFoundError(f"CVAT ZIP bulunamadı: {zip_path}")
    if labels_source is not None and not labels_source.exists():
        raise FileNotFoundError(f"Label kaynağı bulunamadı: {labels_source}")

    temp_context = tempfile.TemporaryDirectory(prefix="saha_ball_cvat_") if zip_path else None
    try:
        extracted_root: Path | None = None
        if zip_path is not None:
            assert temp_context is not None
            extracted_root = Path(temp_context.name)
            with zipfile.ZipFile(zip_path) as archive:
                safe_extract(archive, extracted_root)

        label_root = labels_source or extracted_root
        if label_root is None:
            raise RuntimeError("Label kaynağı belirlenemedi.")

        image_roots: list[Path] = []
        if extracted_root is not None:
            image_roots.append(extracted_root)
        image_roots.extend(image_sources)

        images = find_images_from_roots(image_roots)
        labels = find_labels(label_root)
        if not labels:
            raise RuntimeError(f"Label kaynağında YOLO txt bulunamadı: {label_root}")
        if not images:
            raise RuntimeError(
                "Görsel bulunamadı. Label-only CVAT export için --images-source verilmeli."
            )
        label_paths = list(labels.values())
        class_names = find_class_names(label_root)
        ball_ids = ball_source_ids(class_names, label_paths)

        image_list_path = resolve(args.image_list) if args.image_list else default_image_list(label_root)
        if image_list_path is not None and image_list_path.is_file():
            ordered_stems = read_image_list(image_list_path, int(args.max_images))
            image_list_used = str(image_list_path)
        else:
            ordered_stems = sorted(images)
            if args.max_images > 0:
                ordered_stems = ordered_stems[: int(args.max_images)]
            image_list_used = ""
        if not ordered_stems:
            raise RuntimeError("İçe aktarılacak görsel listesi boş.")

        reset_output(output, args.overwrite)
        total_boxes = 0
        positive_images = 0
        negative_images = 0
        missing_label_images = 0
        missing_source_images = 0
        invalid_or_ignored_rows = 0
        manifest: list[dict[str, object]] = []

        for index, stem in enumerate(ordered_stems, start=1):
            image_path = images.get(stem)
            if image_path is None:
                missing_source_images += 1
                continue
            target_name = safe_name(stem, image_path.suffix)
            target_stem = Path(target_name).stem
            target_image = output / "images" / "test" / target_name
            target_label = output / "labels" / "test" / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)

            rows: list[str] = []
            label_path = labels.get(stem)
            if label_path is None:
                missing_label_images += 1
            else:
                for raw_line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    if not raw_line.strip():
                        continue
                    row = parse_row(raw_line, ball_ids)
                    if row is None:
                        invalid_or_ignored_rows += 1
                    else:
                        rows.append(row)
            target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            total_boxes += len(rows)
            positive_images += int(bool(rows))
            negative_images += int(not rows)
            manifest.append(
                {
                    "index": index,
                    "source_image": str(image_path),
                    "source_label": str(label_path) if label_path else "",
                    "target_image": str(target_image),
                    "target_label": str(target_label),
                    "ball_boxes": len(rows),
                }
            )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    if not manifest:
        raise RuntimeError(
            "Hiç görsel eşleşmedi. Label isimleri ile --images-source görsel isimlerini kontrol edin."
        )
    data_yaml = (
        f'path: "{output.as_posix()}"\n'
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- ball\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary = {
        "source_zip": str(zip_path) if zip_path else "",
        "labels_source": str(labels_source) if labels_source else "",
        "images_source": [str(path) for path in image_sources],
        "image_list": image_list_used,
        "max_images": int(args.max_images),
        "output": str(output),
        "class_names": class_names,
        "ball_source_ids": sorted(ball_ids),
        "images": len(manifest),
        "ball_boxes": total_boxes,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "missing_label_images": missing_label_images,
        "missing_source_images": missing_source_images,
        "invalid_or_ignored_rows": invalid_or_ignored_rows,
    }
    (output / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Gerçek-case ball test set hazır: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
