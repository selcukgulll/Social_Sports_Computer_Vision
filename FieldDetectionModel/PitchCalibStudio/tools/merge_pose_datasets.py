from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_utils import (  # noqa: E402
    discover_data_yamls,
    inspect_dataset,
    is_allowed_dataset,
    list_image_label_pairs,
    load_yaml,
    normalize_names,
    parse_pose_row,
    save_csv,
    save_json,
    slugify,
)
from src.keypoint_utils import (  # noqa: E402
    load_pitch_schema,
    remap_keypoints,
    schema_keypoint_count,
)


def _load_remap(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return load_yaml(path)


def _dataset_remap(remap: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    entries = remap.get("datasets", {})
    if dataset_name in entries:
        return entries[dataset_name]
    lowered = dataset_name.lower()
    for name, entry in entries.items():
        if str(name).lower() in lowered or lowered in str(name).lower():
            return entry
    return None


def _validate_merge_contract(
    yamls: list[Path],
    schema: dict[str, Any],
    remap: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    target_count = schema_keypoint_count(schema)
    contracts: list[dict[str, Any]] = []
    for data_yaml in yamls:
        audit = inspect_dataset(data_yaml)
        if not audit["is_pose"]:
            raise ValueError(f"YOLO pose olarak geçersiz: {data_yaml}\n{audit['errors']}")
        dataset_name = data_yaml.parent.name
        names = normalize_names(load_yaml(data_yaml).get("names"))
        entry = _dataset_remap(remap, dataset_name) if remap else None
        source_count, source_dims = audit["kpt_shape"]

        if entry is None:
            if source_count != target_count:
                raise ValueError(
                    f"{dataset_name}: {source_count} nokta var, şema {target_count}. "
                    "Doğrulanmış remap dosyası zorunlu."
                )
            if set(names.values()) != {"pitch"}:
                raise ValueError(
                    f"{dataset_name}: sınıf {list(names.values())}; hedef 'pitch'. "
                    "Açık class_map/remap olmadan birleştirilmeyecek."
                )
            raise ValueError(
                f"{dataset_name}: indeks sırası için confirmed_order içeren remap "
                "gerekli. Yapısal eşitlik anlamsal eşitlik değildir."
            )

        if not entry.get("confirmed_order", False):
            raise ValueError(
                f"{dataset_name}: remap içindeki confirmed_order=false. "
                "Önizlemeleri kontrol edip eşlemeyi tamamlamadan birleştirme durduruldu."
            )
        declared_shape = entry.get("source_kpt_shape")
        if declared_shape and list(declared_shape) != [source_count, source_dims]:
            raise ValueError(
                f"{dataset_name}: remap source_kpt_shape {declared_shape}, "
                f"gerçek {[source_count, source_dims]}."
            )
        mapping = entry.get("source_to_target")
        if not isinstance(mapping, dict) or not any(
            target is not None for target in mapping.values()
        ):
            raise ValueError(f"{dataset_name}: source_to_target eşlemesi boş.")
        class_map = remap.get("class_map", {})
        unmapped = [name for name in names.values() if class_map.get(name) != "pitch"]
        if unmapped:
            raise ValueError(f"{dataset_name}: pitch'e eşlenmeyen sınıflar: {unmapped}")
        # A dummy remap catches duplicate and out-of-range target indexes early.
        try:
            remap_keypoints(
                [[0.0] * source_dims for _ in range(source_count)],
                mapping,
                target_count,
                source_dims,
            )
        except ValueError as exc:
            raise ValueError(f"{dataset_name}: {exc}") from exc
        contracts.append(
            {
                "data_yaml": data_yaml,
                "dataset_name": dataset_name,
                "audit": audit,
                "mapping": mapping,
                "source_dims": source_dims,
            }
        )
    return contracts


def _format_pose_row(
    annotation: dict[str, Any],
    mapping: dict[int | str, int | str | None],
    target_count: int,
    target_dims: int,
) -> str:
    remapped = remap_keypoints(
        annotation["keypoints"], mapping, target_count, target_dims
    )
    values: list[float | int] = [0, *annotation["bbox"]]
    for point in remapped:
        values.extend(point)
    return " ".join(
        str(value) if isinstance(value, int) else f"{float(value):.8g}"
        for value in values
    )


def merge_datasets(
    datasets_root: str | Path,
    output: str | Path,
    schema_path: str | Path,
    remap_path: str | Path | None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    include_dataset_names: set[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split oranları train>0, val>=0 ve train+val<1 olmalı.")
    datasets_root = Path(datasets_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Çıktı klasörü boş değil; üzerine yazılmadı: {output}"
        )
    schema = load_pitch_schema(schema_path)
    remap = _load_remap(remap_path)
    yamls = [
        path
        for path in discover_data_yamls(datasets_root)
        if is_allowed_dataset(path)
        and (
            include_dataset_names is None
            or path.parent.name in include_dataset_names
        )
    ]
    if not yamls:
        raise FileNotFoundError("İzin verilen üç dataset altında data.yaml bulunamadı.")
    contracts = _validate_merge_contract(yamls, schema, remap)
    target_count = schema_keypoint_count(schema)
    target_dims = 3

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for contract in contracts:
        for source_split in ("train", "val", "test"):
            for pair in list_image_label_pairs(contract["data_yaml"], source_split):
                if not pair.label.exists() or pair.label.stat().st_size == 0:
                    rejected.append(
                        {
                            "dataset": contract["dataset_name"],
                            "image": str(pair.image),
                            "reason": "etiket eksik veya boş",
                        }
                    )
                    continue
                output_rows: list[str] = []
                for line in pair.label.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    annotation, errors = parse_pose_row(
                        line,
                        contract["audit"]["kpt_shape"][0],
                        contract["audit"]["kpt_shape"][1],
                    )
                    if annotation is None or errors:
                        rejected.append(
                            {
                                "dataset": contract["dataset_name"],
                                "image": str(pair.image),
                                "reason": "; ".join(errors),
                            }
                        )
                        output_rows = []
                        break
                    output_rows.append(
                        _format_pose_row(
                            annotation,
                            contract["mapping"],
                            target_count,
                            target_dims,
                        )
                    )
                if output_rows:
                    records.append(
                        {
                            "dataset": contract["dataset_name"],
                            "image": pair.image,
                            "rows": output_rows,
                            "source_split": source_split,
                        }
                    )
    if not records:
        raise ValueError("Birleştirilecek geçerli görsel/etiket çifti kalmadı.")

    random.Random(seed).shuffle(records)
    train_end = int(len(records) * train_ratio)
    val_end = train_end + int(len(records) * val_ratio)
    if train_end == 0:
        train_end = 1
    assignments = (
        [("train", record) for record in records[:train_end]]
        + [("val", record) for record in records[train_end:val_end]]
        + [("test", record) for record in records[val_end:]]
    )
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    for index, (split, record) in enumerate(assignments):
        prefix = slugify(record["dataset"])
        stem = f"{prefix}_{index:06d}_{slugify(record['image'].stem)}"
        image_target = output / "images" / split / f"{stem}{record['image'].suffix.lower()}"
        label_target = output / "labels" / split / f"{stem}.txt"
        shutil.copy2(record["image"], image_target)
        label_target.write_text("\n".join(record["rows"]) + "\n", encoding="utf-8")
        report_rows.append(
            {
                "dataset": record["dataset"],
                "source_split": record["source_split"],
                "target_split": split,
                "source_image": str(record["image"]),
                "target_image": str(image_target),
            }
        )

    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [target_count, target_dims],
        "flip_idx": list(range(target_count)),
        "nc": 1,
        "names": ["pitch"],
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, allow_unicode=True, sort_keys=False)
    counts = {
        split: sum(1 for target_split, _ in assignments if target_split == split)
        for split in ("train", "val", "test")
    }
    report = {
        "datasets_root": str(datasets_root),
        "output": str(output),
        "schema": str(Path(schema_path).resolve()),
        "remap": str(Path(remap_path).resolve()) if remap_path else None,
        "seed": seed,
        "target_kpt_shape": [target_count, target_dims],
        "counts": counts,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "datasets": [contract["dataset_name"] for contract in contracts],
    }
    save_json(output / "merge_report.json", report)
    save_csv(output / "merge_report.csv", report_rows)
    print(f"Birleştirme tamamlandı: {output}")
    print(f"Split sayıları: {counts}; reddedilen: {len(rejected)}")
    return output, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Doğrulanmış remap ile seçili YOLO pose datasetlerini güvenli biçimde birleştirir."
        )
    )
    parser.add_argument("--datasets_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--remap", help="Doğrulanmış dataset remap YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    merge_datasets(
        args.datasets_root,
        args.output,
        args.schema,
        args.remap,
        args.seed,
        args.train_ratio,
        args.val_ratio,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
