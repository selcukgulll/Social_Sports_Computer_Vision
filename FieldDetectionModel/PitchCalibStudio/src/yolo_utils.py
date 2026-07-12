from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


POSE_MODEL_PREFERENCE = (
    "yolo26m-pose.pt",
    "yolo26s-pose.pt",
    "yolo26n-pose.pt",
    "yolo11m-pose.pt",
    "yolo11s-pose.pt",
    "yolo11n-pose.pt",
    "yolov8m-pose.pt",
    "yolov8n-pose.pt",
)


def default_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def model_search_dirs(project_root: str | Path) -> list[Path]:
    project_root = Path(project_root).resolve()
    return [
        project_root / "models",
        project_root,
        project_root.parent,
        project_root.parent.parent / "saha_ai_project" / "models" / "base",
        Path.cwd(),
    ]


def resolve_pose_model(
    requested: str | Path | None,
    search_dirs: Iterable[str | Path],
    allow_ultralytics_download: bool = False,
) -> tuple[str, str]:
    dirs = [Path(directory).expanduser().resolve() for directory in search_dirs]
    if requested:
        request_path = Path(str(requested)).expanduser()
        if request_path.is_file():
            return str(request_path.resolve()), "Seçilen yerel pose model"
        for directory in dirs:
            candidate = directory / request_path.name
            if candidate.is_file():
                return str(candidate.resolve()), f"Yerel model bulundu: {candidate.name}"
        if allow_ultralytics_download and request_path.name in POSE_MODEL_PREFERENCE:
            return request_path.name, "Ultralytics model adı; eksikse indirme denenebilir"
    for name in POSE_MODEL_PREFERENCE:
        for directory in dirs:
            candidate = directory / name
            if candidate.is_file():
                note = (
                    "Tercih edilen yolo26m-pose.pt bulundu"
                    if name == "yolo26m-pose.pt"
                    else f"yolo26m-pose.pt yok; en yakın yerel pose modeline geçildi: {name}"
                )
                return str(candidate), note
    searched = "\n".join(f"- {directory}" for directory in dirs)
    raise FileNotFoundError(
        "Yerel bir YOLO pose modeli bulunamadı. Detection modeli (ör. yolo26m.pt) "
        "bu iş için kullanılamaz.\nAranan klasörler:\n" + searched
    )


def load_pose_model(model_path: str | Path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    task = getattr(model, "task", None)
    if task and task != "pose":
        raise ValueError(
            f"Seçilen modelin görevi '{task}'. PitchCalibStudio için 'pose' model gerekir."
        )
    return model


def require_pitch_class(model) -> None:
    names = getattr(model, "names", None)
    if names is None and getattr(model, "model", None) is not None:
        names = getattr(model.model, "names", None)
    values = list(names.values()) if isinstance(names, dict) else list(names or [])
    if "pitch" not in {str(value).strip().lower() for value in values}:
        raise ValueError(
            "Bu pose modelinin sınıfları arasında 'pitch' yok. "
            "Başlangıç COCO pose ağırlığını değil, PitchCalibStudio ile eğitilmiş best.pt seçin. "
            f"Model sınıfları: {values}"
        )
