from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_utils import save_json, timestamp  # noqa: E402
from src.yolo_utils import (  # noqa: E402
    default_device,
    load_pose_model,
    model_search_dirs,
    resolve_pose_model,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO ile pitch landmark pose modeli eğitir."
    )
    parser.add_argument("--data", required=True, help="Birleştirilmiş data.yaml")
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.0,
        help="Semantik flip_idx doğrulanmadıysa 0 bırakın",
    )
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Yerel model yoksa Ultralytics'in ağırlık indirmesine izin ver",
    )
    return parser


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_path}")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not data.get("kpt_shape"):
        raise ValueError("data.yaml kpt_shape içermiyor; detection datası eğitilemez.")
    model_path, model_note = resolve_pose_model(
        args.model,
        model_search_dirs(PROJECT_ROOT),
        allow_ultralytics_download=args.allow_download,
    )
    print(model_note)
    print(f"Kullanılan model: {model_path}")
    model = load_pose_model(model_path)

    run_stamp = timestamp()
    run_name = f"pitch_pose_{run_stamp}"
    runs_root = PROJECT_ROOT / "outputs" / "training_runs"
    command = [sys.executable, str(Path(__file__).resolve())]
    for key, value in vars(args).items():
        if key == "allow_download":
            if value:
                command.append("--allow-download")
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    command_text = subprocess.list2cmdline(command)
    print(f"Komut: {command_text}")

    train_result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        fliplr=args.fliplr,
        flipud=args.flipud,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
        task="pose",
    )
    save_dir = Path(getattr(train_result, "save_dir", runs_root / run_name)).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "training_command.txt").write_text(command_text + "\n", encoding="utf-8")
    shutil.copy2(data_path, save_dir / "data_used.yaml")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    copied_models: dict[str, str] = {}
    for kind in ("best", "last"):
        source = save_dir / "weights" / f"{kind}.pt"
        if source.exists():
            target = models_dir / f"pitch_pose_{kind}_{run_stamp}.pt"
            shutil.copy2(source, target)
            copied_models[kind] = str(target)

    metrics: dict[str, Any] = {}
    best_path = copied_models.get("best")
    if best_path:
        best_model = load_pose_model(best_path)
        validation = best_model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(save_dir),
            name="validation",
        )
        metrics = _jsonable(getattr(validation, "results_dict", {}))
    result = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(save_dir),
        "data": str(data_path),
        "model_requested": args.model,
        "model_used": model_path,
        "model_note": model_note,
        "copied_models": copied_models,
        "metrics": metrics,
        "command": command_text,
    }
    save_json(save_dir / "validation_metrics.json", metrics)
    save_json(save_dir / "training_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    args = build_parser().parse_args()
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
