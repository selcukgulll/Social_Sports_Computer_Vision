from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.merge_pose_datasets import merge_datasets  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tek bir YOLO pose datasetini doğrulanmış eşleme ve saha şemasıyla remap eder."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset klasörü veya onu içeren kök klasör",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--remap", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    root = dataset.parent if (dataset / "data.yaml").exists() else dataset
    merge_datasets(
        root,
        args.output,
        args.schema,
        args.remap,
        seed=args.seed,
        include_dataset_names={dataset.name} if (dataset / "data.yaml").exists() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
