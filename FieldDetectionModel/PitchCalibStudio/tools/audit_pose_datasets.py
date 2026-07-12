from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_utils import (  # noqa: E402
    compact_audit_row,
    discover_data_yamls,
    inspect_dataset,
    save_csv,
    save_json,
    slugify,
    timestamp,
)
from tools.preview_keypoints import generate_previews  # noqa: E402


def run_audit(
    datasets_root: str | Path,
    output: str | Path,
    preview_count: int = 20,
    seed: int = 42,
) -> tuple[Path, list[dict]]:
    datasets_root = Path(datasets_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    run_dir = output / f"audit_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    previews_root = output.parent / "previews" / run_dir.name
    reports: list[dict] = []

    yamls = discover_data_yamls(datasets_root)
    if not yamls:
        raise FileNotFoundError(f"data.yaml bulunamadı: {datasets_root}")
    for data_yaml in yamls:
        report = inspect_dataset(data_yaml)
        reports.append(report)
        print(
            f"\n[{report['dataset_name']}]\n"
            f"  task: {report['task']}\n"
            f"  names: {report['names']}\n"
            f"  nc: {report['nc']}\n"
            f"  kpt_shape: {report['kpt_shape']}\n"
            f"  train: {report['splits'].get('train', {}).get('images_path', '')}\n"
            f"  val: {report['splits'].get('val', {}).get('images_path', '')}\n"
            f"  test: {report['splits'].get('test', {}).get('images_path', '')}\n"
            f"  images/labels: {report['image_count']}/{report['label_count']}\n"
            f"  valid/broken rows: {report['valid_pose_rows']}/{report['broken_rows']}"
        )
        if report["is_pose"] and preview_count > 0:
            preview_dir = previews_root / slugify(report["dataset_name"])
            try:
                report["preview_paths"] = [
                    str(path)
                    for path in generate_previews(
                        data_yaml, "train", preview_count, preview_dir, seed
                    )
                ]
            except Exception as exc:
                report["warnings"].append(f"Önizleme üretilemedi: {exc}")

    shapes = {
        tuple(report["kpt_shape"])
        for report in reports
        if isinstance(report.get("kpt_shape"), list)
    }
    classes = {
        tuple(report["names"].values())
        for report in reports
        if report.get("names")
    }
    merge_safe = (
        all(report["is_pose"] for report in reports)
        and len(shapes) == 1
        and len(classes) == 1
    )
    summary = {
        "datasets_root": str(datasets_root),
        "created_at": timestamp(),
        "dataset_count": len(reports),
        "merge_safe_structurally": merge_safe,
        "manual_keypoint_order_confirmation_required": True,
        "reports": reports,
    }
    save_json(run_dir / "audit_report.json", summary)
    save_csv(run_dir / "audit_report.csv", [compact_audit_row(report) for report in reports])
    print(f"\nAudit tamamlandı: {run_dir}")
    print(
        "Birleştirme durumu: "
        + (
            "Yapısal olarak uyumlu; yine de indeks sırası görsel doğrulanmalı."
            if merge_safe
            else "GÜVENLİ DEĞİL; kpt_shape/sınıf farkı için doğrulanmış remap gerekli."
        )
    )
    return run_dir, reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yerel YOLO pose datasetlerini denetler ve indeksli önizleme üretir."
    )
    parser.add_argument("--datasets_root", required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "audits"))
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_audit(args.datasets_root, args.output, args.preview_count, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
