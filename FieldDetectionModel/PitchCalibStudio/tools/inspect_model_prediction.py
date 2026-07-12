from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_image import save_image_calibration  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bir pose modelinin tek görsel tahminini ayrıntılı kaydeder."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--schema", default=str(PROJECT_ROOT / "configs" / "pitch_schema.yaml")
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "outputs" / "predictions")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = save_image_calibration(
        args.image,
        args.model,
        args.schema,
        args.output,
        args.conf,
        PROJECT_ROOT / "configs" / "app_config.yaml",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
