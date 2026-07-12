#!/usr/bin/env python
"""Build a balanced player/ball YOLO dataset from ``datasets/Categories``.

The source folders are Roboflow-style YOLO exports.  Their class names and
split layouts are not consistent, so this script applies an explicit JSON plan:

- maps source classes into the local schema: player, ball, outside_person
- ignores non-useful classes such as goal posts
- keeps target-like TestData out of training
- caps very large sources so a single generic dataset cannot dominate training
- converts YOLO polygon rows into bounding boxes when needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is installed with Ultralytics
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLITS = ("train", "valid", "val", "test")
TARGET_NAMES = {0: "player", 1: "ball", 2: "outside_person"}
TARGET_NAME_TO_ID = {name: idx for idx, name in TARGET_NAMES.items()}


@dataclass(frozen=True)
class Sample:
    source_name: str
    source_dataset: Path
    source_split: str
    output_split: str
    image_path: Path
    label_path: Path | None
    mapping: dict[str, str]


def project_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or PROJECT_ROOT) / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categories kaynaklarından dengeli player/ball master dataset oluşturur."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "datasets/Categories/dataset_plan_player_v8.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/CuratedFootball/builds/player_ball_v8_master",
    )
    parser.add_argument(
        "--storage",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Görselleri hardlink ile bağla veya kopyala.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML bulunamadı. Lütfen `pip install pyyaml` çalıştırın.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except UnicodeDecodeError:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Geçersiz YAML: {path}")
    return data


def read_source_names(dataset: Path) -> list[str]:
    data_yaml = dataset / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")
    raw_names = load_yaml(data_yaml).get("names", [])
    if isinstance(raw_names, dict):
        def sort_key(value: Any) -> tuple[int, str]:
            try:
                return (0, f"{int(value):08d}")
            except (TypeError, ValueError):
                return (1, str(value))

        return [str(raw_names[key]) for key in sorted(raw_names, key=sort_key)]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    raise ValueError(f"names alanı okunamadı: {data_yaml}")


def iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        (
            Path(entry.path)
            for entry in os.scandir(image_dir)
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def stable_score(seed: int, *parts: str) -> str:
    text = "|".join([str(seed), *parts])
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def safe_slug(value: str) -> str:
    keep: list[str] = []
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
        elif char in ("-", "_", ".", " "):
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "dataset"


def normalize_mapping(source: dict[str, Any], source_names: list[str]) -> dict[str, str]:
    raw = source.get("class_map") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"class_map dict olmalı: {source.get('name')}")
    mapping: dict[str, str] = {}
    for index, name in enumerate(source_names):
        value = raw.get(str(index), raw.get(name, raw.get(name.lower(), "ignore")))
        value_text = str(value).strip().lower()
        if value_text in {"x", "-", "none", "null", "ignore", "yok say"}:
            mapping[str(index)] = "ignore"
        elif value_text in TARGET_NAME_TO_ID:
            mapping[str(index)] = value_text
        else:
            raise ValueError(
                f"Geçersiz hedef sınıf: {source.get('name')} class {index}:{name} -> {value}"
            )
    return mapping


def collect_samples(plan: dict[str, Any], plan_path: Path) -> tuple[list[Sample], dict[str, Any]]:
    seed = int(plan.get("seed", 42))
    plan_base = plan_path.parent
    all_samples: list[Sample] = []
    source_reports: list[dict[str, Any]] = []

    for source in plan.get("sources", []):
        if not bool(source.get("enabled", True)):
            continue
        name = str(source.get("name") or source.get("path") or "dataset")
        dataset = project_path(source["path"], base=plan_base)
        source_names = read_source_names(dataset)
        mapping = normalize_mapping(source, source_names)
        split_map = source.get("split_map") or {}
        if not isinstance(split_map, dict):
            raise ValueError(f"split_map dict olmalı: {name}")
        max_images = source.get("max_images") or {}
        if not isinstance(max_images, dict):
            raise ValueError(f"max_images dict olmalı: {name}")

        by_output: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
        discovered: dict[str, int] = {}
        for source_split in SOURCE_SPLITS:
            output_split = split_map.get(source_split)
            if output_split is None and source_split == "valid":
                output_split = split_map.get("val")
            if output_split is None:
                continue
            output_split = str(output_split).strip().lower()
            if output_split in {"ignore", "x", "-"}:
                continue
            if output_split not in by_output:
                raise ValueError(f"Geçersiz hedef split: {name} {source_split}->{output_split}")

            image_dir = dataset / source_split / "images"
            label_dir = dataset / source_split / "labels"
            images = iter_images(image_dir)
            discovered[source_split] = len(images)
            for image_path in images:
                label_path = label_dir / f"{image_path.stem}.txt"
                by_output[output_split].append(
                    Sample(
                        source_name=name,
                        source_dataset=dataset,
                        source_split=source_split,
                        output_split=output_split,
                        image_path=image_path,
                        label_path=label_path if label_path.is_file() else None,
                        mapping=mapping,
                    )
                )

        selected_for_source = 0
        for output_split, samples in by_output.items():
            cap_value = max_images.get(output_split)
            if cap_value is not None:
                cap = int(cap_value)
                samples = sorted(
                    samples,
                    key=lambda sample: stable_score(
                        seed,
                        name,
                        sample.output_split,
                        sample.image_path.relative_to(sample.source_dataset).as_posix(),
                    ),
                )[:cap]
            selected_for_source += len(samples)
            all_samples.extend(samples)

        source_reports.append(
            {
                "name": name,
                "path": str(dataset),
                "source_names": source_names,
                "mapping": mapping,
                "discovered_images_by_source_split": discovered,
                "selected_images": selected_for_source,
                "note": source.get("note", ""),
            }
        )

    all_samples.sort(
        key=lambda sample: (
            sample.output_split,
            stable_score(
                seed,
                sample.source_name,
                sample.source_split,
                sample.image_path.name,
            ),
        )
    )
    return all_samples, {"seed": seed, "sources": source_reports}


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def parse_label_row(raw_line: str) -> tuple[int, float, float, float, float, bool] | None:
    parts = raw_line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(values) == 4:
        x, y, w, h = values
        return class_id, clip01(x), clip01(y), clip01(w), clip01(h), False

    if len(values) >= 6 and len(values) % 2 == 0:
        xs = [clip01(values[index]) for index in range(0, len(values), 2)]
        ys = [clip01(values[index]) for index in range(1, len(values), 2)]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return class_id, clip01((x1 + x2) / 2), clip01((y1 + y2) / 2), clip01(w), clip01(h), True

    return None


def convert_label(sample: Sample) -> tuple[list[str], dict[str, int]]:
    stats = {
        "missing_label_files": 0,
        "empty_label_files": 0,
        "invalid_rows": 0,
        "ignored_rows": 0,
        "polygon_rows_converted": 0,
    }
    if sample.label_path is None:
        stats["missing_label_files"] += 1
        return [], stats
    try:
        raw_lines = sample.label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        stats["missing_label_files"] += 1
        return [], stats
    if not any(line.strip() for line in raw_lines):
        stats["empty_label_files"] += 1
        return [], stats

    output_rows: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        parsed = parse_label_row(line)
        if parsed is None:
            stats["invalid_rows"] += 1
            continue
        class_id, x, y, w, h, converted_polygon = parsed
        target_name = sample.mapping.get(str(class_id), "ignore")
        if target_name == "ignore":
            stats["ignored_rows"] += 1
            continue
        if w <= 0.0 or h <= 0.0:
            stats["invalid_rows"] += 1
            continue
        if converted_polygon:
            stats["polygon_rows_converted"] += 1
        target_id = TARGET_NAME_TO_ID[target_name]
        output_rows.append(f"{target_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return output_rows, stats


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            if (output_dir / "data.yaml").is_file():
                raise FileExistsError(
                    f"Çıktı zaten var: {output_dir}. Yeniden oluşturmak için --overwrite kullanın."
                )
            raise FileExistsError(
                f"Çıktı klasörü var ama data.yaml yok: {output_dir}. --overwrite kullanın."
            )
        resolved = output_dir.resolve()
        if PROJECT_ROOT.resolve() not in resolved.parents:
            raise ValueError(f"Güvenlik için proje dışı çıktı silinmedi: {resolved}")
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy_image(source: Path, target: Path, storage: str) -> str:
    if storage == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    shutil.copy2(source, target)
    return "copy"


def write_dataset(samples: list[Sample], output_dir: Path, storage: str) -> dict[str, Any]:
    images_by_split = {"train": 0, "val": 0, "test": 0}
    boxes_by_class = {str(class_id): 0 for class_id in TARGET_NAMES}
    images_by_source: dict[str, dict[str, int]] = {}
    stat_totals = {
        "missing_label_files": 0,
        "empty_label_files": 0,
        "invalid_rows": 0,
        "ignored_rows": 0,
        "polygon_rows_converted": 0,
        "copy_fallbacks": 0,
    }
    used_names: set[str] = set()

    for sample in samples:
        source_slug = safe_slug(sample.source_name)
        digest = stable_score(
            0,
            sample.source_name,
            sample.source_split,
            sample.image_path.relative_to(sample.source_dataset).as_posix(),
        )[:10]
        target_stem = f"{source_slug}_{sample.source_split}_{sample.image_path.stem}_{digest}"
        while target_stem.lower() in used_names:
            target_stem = f"{target_stem}_{len(used_names)}"
        used_names.add(target_stem.lower())

        target_image = output_dir / "images" / sample.output_split / f"{target_stem}{sample.image_path.suffix.lower()}"
        target_label = output_dir / "labels" / sample.output_split / f"{target_stem}.txt"

        storage_mode = link_or_copy_image(sample.image_path, target_image, storage)
        if storage_mode == "copy_fallback":
            stat_totals["copy_fallbacks"] += 1

        rows, row_stats = convert_label(sample)
        for key, value in row_stats.items():
            stat_totals[key] += value
        for row in rows:
            class_id = row.split(maxsplit=1)[0]
            boxes_by_class[class_id] = boxes_by_class.get(class_id, 0) + 1
        target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        images_by_split[sample.output_split] += 1
        source_bucket = images_by_source.setdefault(sample.source_name, {"train": 0, "val": 0, "test": 0})
        source_bucket[sample.output_split] += 1

    data_yaml = (
        f'path: "{output_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        "names:\n"
        "  0: player\n"
        "  1: ball\n"
        "  2: outside_person\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {
        "images_by_split": images_by_split,
        "boxes_by_class": boxes_by_class,
        "images_by_source": images_by_source,
        **stat_totals,
    }


def main() -> int:
    args = parse_args()
    plan_path = project_path(args.plan)
    output_dir = project_path(args.output_dir)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan bulunamadı: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if plan.get("target_names") and list(plan["target_names"]) != list(TARGET_NAMES.values()):
        raise ValueError(
            "Plan target_names alanı bu scriptin hedef şemasıyla aynı olmalı: "
            f"{list(TARGET_NAMES.values())}"
        )

    samples, collect_report = collect_samples(plan, plan_path)
    if not samples:
        raise RuntimeError("Plan hiçbir örnek seçmedi.")

    prepare_output_dirs(output_dir, args.overwrite)
    write_report = write_dataset(samples, output_dir, args.storage)
    summary = {
        "plan": str(plan_path),
        "output_dir": str(output_dir),
        "storage": args.storage,
        **collect_report,
        **write_report,
    }
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Player/ball v8 master dataset hazır: {output_dir}")
    print(
        "Splitler: "
        + ", ".join(f"{split}={count}" for split, count in write_report["images_by_split"].items())
    )
    print(
        "Kutular: "
        + ", ".join(
            f"{TARGET_NAMES[int(class_id)]}={count}"
            for class_id, count in sorted(write_report["boxes_by_class"].items())
        )
    )
    if write_report["invalid_rows"]:
        print(f"UYARI: {write_report['invalid_rows']} geçersiz label satırı atlandı.")
    if write_report["polygon_rows_converted"]:
        print(f"Polygon->bbox dönüştürülen satır: {write_report['polygon_rows_converted']}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")
    print(f"Özet: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
