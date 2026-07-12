from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibrate_image import save_image_calibration  # noqa: E402
from calibrate_video import calibrate_video  # noqa: E402
from src.dataset_utils import (  # noqa: E402
    compact_audit_row,
    discover_data_yamls,
    timestamp,
)
from src.homography_utils import bbox_bottom_centers_to_field  # noqa: E402
from src.visualization import draw_top_view_pitch  # noqa: E402
from src.yolo_utils import default_device  # noqa: E402
from tools.audit_pose_datasets import run_audit  # noqa: E402
from tools.merge_pose_datasets import merge_datasets  # noqa: E402
from tools.preview_keypoints import generate_previews  # noqa: E402


DEFAULT_DATASETS = (PROJECT_ROOT.parent / "datasets").resolve()
DEFAULT_SCHEMA = (PROJECT_ROOT / "configs" / "pitch_schema.yaml").resolve()
USER_REMAP = (PROJECT_ROOT / "configs" / "dataset_remap_user.yaml").resolve()
DEFAULT_REMAP = (
    USER_REMAP
    if USER_REMAP.is_file()
    else (PROJECT_ROOT / "configs" / "dataset_remap_template.yaml").resolve()
)
DEFAULT_CONFIG = (PROJECT_ROOT / "configs" / "app_config.yaml").resolve()


st.set_page_config(
    page_title="PitchCalibStudio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
:root { --pcs-accent:#37d67a; --pcs-blue:#63a9ff; }
.stApp { background:linear-gradient(135deg,#0b1018 0%,#101a25 55%,#0c151b 100%); }
[data-testid="stSidebar"] { background:#0b121b; border-right:1px solid #203044; }
.pcs-hero { padding:1.2rem 1.4rem; border:1px solid #26445a; border-radius:18px;
 background:linear-gradient(120deg,rgba(55,214,122,.12),rgba(99,169,255,.10)); margin-bottom:1rem; }
.pcs-hero h1 { margin:0; color:#f4f8fb; font-size:2.15rem; }
.pcs-hero p { color:#b7c9d8; margin:.35rem 0 0; }
.pcs-note { padding:.8rem 1rem; border-left:4px solid var(--pcs-accent);
 background:rgba(21,40,48,.7); border-radius:8px; margin:.5rem 0 1rem; }
.stButton>button { border-radius:10px; border:1px solid #2a7e59; }
code { color:#b9f7d5 !important; }
</style>
<div class="pcs-hero">
  <h1>⚽ PitchCalibStudio</h1>
  <p>Saha landmark modeli • güvenli dataset denetimi • pose eğitimi • homography kalibrasyonu</p>
</div>
""",
    unsafe_allow_html=True,
)


def _path(value: str) -> Path:
    return Path(value.strip().strip('"')).expanduser().resolve()


def _cmd(parts: Iterable[object]) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def _show_command(parts: Iterable[object]) -> None:
    st.caption("Eşdeğer CLI komutu")
    st.code(_cmd(parts), language="powershell")


def _latest(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda item: item.stat().st_mtime)
    return matches[-1] if matches else None


def _find_pose_models() -> list[Path]:
    roots = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT.parent.parent / "saha_ai_project" / "models",
    ]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            for model in root.rglob("*pose*.pt"):
                if model.is_file() and model not in found:
                    found.append(model.resolve())
    return sorted(found, key=lambda item: item.stat().st_mtime, reverse=True)


def _save_upload(uploaded, category: str) -> Path:
    folder = PROJECT_ROOT / "data" / "raw" / category
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    target = folder / f"{timestamp()}_{Path(uploaded.name).stem}{suffix}"
    target.write_bytes(uploaded.getbuffer())
    return target


def _gallery(paths: Iterable[str | Path], limit: int = 24) -> None:
    paths = [Path(path) for path in paths if Path(path).is_file()][:limit]
    if not paths:
        st.info("Gösterilecek önizleme bulunamadı.")
        return
    columns = st.columns(3)
    for index, path in enumerate(paths):
        columns[index % 3].image(str(path), caption=path.name, width="stretch")


def _run_process(command: list[str]) -> int:
    box = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-250:]), language="text")
    return process.wait()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


pose_models = _find_pose_models()
with st.sidebar:
    st.subheader("Yerel çalışma durumu")
    st.success("Offline çalışma • Roboflow API yok")
    st.write(f"Proje: `{PROJECT_ROOT}`")
    st.write(f"CUDA cihaz varsayılanı: `{default_device()}`")
    if pose_models:
        st.write(f"Yerel pose modeli: `{pose_models[0].name}`")
    else:
        st.warning(
            "Yerel `*-pose.pt` bulunamadı. `yolo26m.pt` detection modelidir ve burada kullanılamaz."
        )
    st.divider()
    st.caption(
        "Audit ve önizleme eğitim başlatmaz. Eğitim yalnızca Train sekmesindeki düğmeyle başlar."
    )


tabs = st.tabs(
    [
        "1 · Dataset Audit",
        "2 · Keypoint Preview",
        "3 · Merge Datasets",
        "4 · Train",
        "5 · Image Viewer",
        "6 · Video Calibration",
        "7 · Test Transform",
    ]
)


with tabs[0]:
    st.subheader("Datasetleri denetle")
    st.markdown(
        '<div class="pcs-note">Birleştirmeden önce formatı ve indeks sırasını burada doğrula. '
        "Farklı <code>kpt_shape</code> veya sınıf adları otomatik olarak güvenli sayılmaz.</div>",
        unsafe_allow_html=True,
    )
    audit_root = st.text_input("Dataset kök klasörü", str(DEFAULT_DATASETS), key="audit_root")
    audit_count = st.number_input(
        "Dataset başına önizleme", min_value=1, max_value=100, value=20
    )
    audit_output = PROJECT_ROOT / "outputs" / "audits"
    _show_command(
        [
            sys.executable,
            "tools/audit_pose_datasets.py",
            "--datasets_root",
            audit_root,
            "--output",
            audit_output,
            "--preview-count",
            int(audit_count),
        ]
    )
    if st.button("Audit çalıştır", type="primary", key="audit_button"):
        try:
            with st.spinner("Etiketler okunuyor ve indeksli önizlemeler hazırlanıyor…"):
                run_dir, reports = run_audit(
                    _path(audit_root), audit_output, int(audit_count)
                )
            st.session_state["audit_run"] = str(run_dir)
            st.session_state["audit_reports"] = reports
            st.success(f"Audit tamamlandı: {run_dir}")
        except Exception as exc:
            st.exception(exc)

    run_value = st.session_state.get("audit_run")
    if not run_value:
        latest = _latest("outputs/audits/audit_*/audit_report.json")
        run_value = str(latest.parent) if latest else None
    if run_value and (Path(run_value) / "audit_report.json").exists():
        summary = _read_json(Path(run_value) / "audit_report.json")
        reports = summary["reports"]
        st.dataframe(
            pd.DataFrame([compact_audit_row(report) for report in reports]),
            width="stretch",
            hide_index=True,
        )
        shapes = [report.get("kpt_shape") for report in reports]
        if summary.get("merge_safe_structurally"):
            st.warning("Yapısal eşitlik var; indeks sırası yine görsel onay bekliyor.")
        else:
            remap_ready = DEFAULT_REMAP.is_file()
            merged_report = PROJECT_ROOT / "data" / "merged" / "pitch_pose" / "merge_report.json"
            if remap_ready and merged_report.is_file():
                st.info(
                    f"Kaynak şekilleri farklı: {shapes}. Bu beklenen durumdur; "
                    "doğrulanmış 14→22, 16→22 ve 22→22 remap hazır. "
                    "Merge sekmesindeki “Güvenli merge” akışını kullan."
                )
            else:
                st.error(
                    f"Doğrudan merge güvenli değil. Bulunan kpt_shape değerleri: {shapes}. "
                    "Devam etmek için doğrulanmış remap gereklidir."
                )
        for report in reports:
            with st.expander(
                f"{report['dataset_name']} · kpt_shape={report['kpt_shape']}"
            ):
                if report["warnings"]:
                    st.warning("\n".join(report["warnings"]))
                if report["errors"]:
                    st.error("\n".join(report["errors"]))
                _gallery(report.get("preview_paths", []), limit=9)


with tabs[1]:
    st.subheader("Keypoint indekslerini görsel doğrula")
    try:
        yaml_options = discover_data_yamls(DEFAULT_DATASETS)
    except Exception:
        yaml_options = []
    default_yaml = str(yaml_options[0]) if yaml_options else ""
    preview_yaml = st.text_input("data.yaml", default_yaml, key="preview_yaml")
    col1, col2, col3 = st.columns(3)
    preview_split = col1.selectbox("Split", ["train", "val", "test"])
    preview_count = col2.number_input("Örnek sayısı", 1, 200, 50)
    preview_seed = col3.number_input("Seed", 0, 999999, 42)
    preview_base = PROJECT_ROOT / "outputs" / "previews"
    _show_command(
        [
            sys.executable,
            "tools/preview_keypoints.py",
            "--data",
            preview_yaml,
            "--split",
            preview_split,
            "--count",
            int(preview_count),
            "--output",
            preview_base / "<tarihli_klasör>",
            "--seed",
            int(preview_seed),
        ]
    )
    if st.button("İndeksli önizleme üret", type="primary", key="preview_button"):
        try:
            data_path = _path(preview_yaml)
            output = preview_base / f"manual_{timestamp()}_{data_path.parent.name}"
            with st.spinner("Örnekler çiziliyor…"):
                saved = generate_previews(
                    data_path,
                    preview_split,
                    int(preview_count),
                    output,
                    int(preview_seed),
                )
            st.session_state["preview_paths"] = [str(path) for path in saved]
            st.success(f"{len(saved)} görsel: {output}")
        except Exception as exc:
            st.exception(exc)
    _gallery(st.session_state.get("preview_paths", []))


with tabs[2]:
    st.subheader("Doğrulanmış şemayla birleştir")
    merged_default_yaml = PROJECT_ROOT / "data" / "merged" / "pitch_pose" / "data.yaml"
    if merged_default_yaml.exists():
        st.success(
            "22 noktalı Rocktail hedef şemasıyla birleşmiş dataset hazır. "
            "İstersen doğrudan Train sekmesine geçebilirsin."
        )
    else:
        st.warning(
            "Mevcut üç dataset 14 / 16 / 22 noktalı. Ortak hedef Rocktail'in "
            "0–21 düzenidir; doğrulanmış remap olmadan merge durdurulur."
        )
    merge_root = st.text_input("Dataset kökü", str(DEFAULT_DATASETS), key="merge_root")
    merge_output = st.text_input(
        "Çıktı klasörü",
        str(PROJECT_ROOT / "data" / "merged" / "pitch_pose"),
        key="merge_output",
    )
    merge_schema = st.text_input("Pitch schema", str(DEFAULT_SCHEMA), key="merge_schema")
    merge_remap = st.text_input("Remap YAML", str(DEFAULT_REMAP), key="merge_remap")
    col1, col2, col3 = st.columns(3)
    merge_seed = col1.number_input("Seed", 0, 999999, 42, key="merge_seed")
    train_ratio = col2.number_input(
        "Train oranı", 0.1, 0.95, 0.8, step=0.05, key="train_ratio"
    )
    val_ratio = col3.number_input(
        "Validation oranı", 0.0, 0.5, 0.1, step=0.05, key="val_ratio"
    )
    with st.expander("Remap YAML'ı görüntüle / kullanıcı kopyası oluştur"):
        remap_text = ""
        try:
            remap_text = _path(merge_remap).read_text(encoding="utf-8")
        except Exception:
            pass
        edited_remap = st.text_area(
            "Eşleme dosyası",
            remap_text,
            height=420,
            help="İndeksleri yalnızca önizlemelerden anlamlarını doğruladıktan sonra doldur.",
        )
        if st.button("Kullanıcı remap dosyası olarak kaydet"):
            try:
                yaml.safe_load(edited_remap)
                target = PROJECT_ROOT / "configs" / "dataset_remap_user.yaml"
                target.write_text(edited_remap, encoding="utf-8")
                st.success(f"Kaydedildi: {target}")
            except Exception as exc:
                st.error(f"YAML geçersiz: {exc}")
    merge_command = [
        sys.executable,
        "tools/merge_pose_datasets.py",
        "--datasets_root",
        merge_root,
        "--output",
        merge_output,
        "--schema",
        merge_schema,
        "--remap",
        merge_remap,
        "--seed",
        int(merge_seed),
        "--train-ratio",
        float(train_ratio),
        "--val-ratio",
        float(val_ratio),
    ]
    _show_command(merge_command)
    if st.button("Güvenli merge başlat", type="primary", key="merge_button"):
        try:
            with st.spinner("Datasetler doğrulanıyor ve kopyalanıyor…"):
                output, report = merge_datasets(
                    _path(merge_root),
                    _path(merge_output),
                    _path(merge_schema),
                    _path(merge_remap),
                    int(merge_seed),
                    float(train_ratio),
                    float(val_ratio),
                )
            st.success(f"Birleştirildi: {output}")
            st.json(report)
            st.code((output / "data.yaml").read_text(encoding="utf-8"), language="yaml")
        except Exception as exc:
            st.exception(exc)
    report_path = _path(merge_output) / "merge_report.json"
    if report_path.exists():
        st.caption("Mevcut merge raporu")
        st.json(_read_json(report_path))


with tabs[3]:
    st.subheader("Pose modelini eğit")
    st.info("Bu sekmeyi açmak eğitim başlatmaz. Yalnızca aşağıdaki düğme başlatır.")
    train_data = st.text_input(
        "Birleştirilmiş data.yaml",
        str(PROJECT_ROOT / "data" / "merged" / "pitch_pose" / "data.yaml"),
        key="train_data",
    )
    model_default = str(pose_models[0]) if pose_models else str(PROJECT_ROOT / "models" / "yolo26m-pose.pt")
    train_model = st.text_input("Başlangıç pose modeli", model_default)
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", 1, 2000, 150)
    imgsz = c2.selectbox("Image size", [640, 768, 960, 1280], index=2)
    batch = c3.number_input("Batch", 1, 128, 8)
    device = c4.text_input("Device", default_device())
    c5, c6 = st.columns(2)
    patience = c5.number_input("Patience", 0, 500, 30)
    workers = c6.number_input("Workers", 0, 32, 4)
    train_command = [
        sys.executable,
        "train_pitch_pose.py",
        "--data",
        train_data,
        "--model",
        train_model,
        "--epochs",
        int(epochs),
        "--imgsz",
        int(imgsz),
        "--batch",
        int(batch),
        "--device",
        device,
        "--patience",
        int(patience),
        "--workers",
        int(workers),
        "--fliplr",
        0.0,
        "--flipud",
        0.0,
    ]
    _show_command(train_command)
    confirm_training = st.checkbox(
        "Dataset audit/remap kontrolünü yaptım; eğitimi başlatmak istiyorum."
    )
    if st.button(
        "Eğitimi başlat",
        type="primary",
        disabled=not confirm_training,
        key="train_button",
    ):
        with st.status("Eğitim çalışıyor…", expanded=True) as status:
            return_code = _run_process([str(value) for value in train_command])
            if return_code == 0:
                status.update(label="Eğitim tamamlandı", state="complete")
                st.success("Model kopyaları models/ klasörüne alındı.")
            else:
                status.update(label=f"Eğitim hata kodu: {return_code}", state="error")
    latest_summary = _latest("outputs/training_runs/*/training_summary.json")
    if latest_summary:
        with st.expander("Son eğitim özeti"):
            st.json(_read_json(latest_summary))


with tabs[4]:
    st.subheader("Tek karede model ve kalibrasyon testi")
    image_upload = st.file_uploader(
        "Saha karesi yükle", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    image_local = st.text_input("veya yerel görsel yolu", "", key="image_local")
    image_model = st.text_input("Eğitilmiş pose modeli", model_default, key="image_model")
    image_schema = st.text_input("Şema", str(DEFAULT_SCHEMA), key="image_schema")
    image_conf = st.slider("Confidence", 0.01, 0.95, 0.25, 0.01, key="image_conf")
    source_display = image_local or "<yüklenen_görsel>"
    _show_command(
        [
            sys.executable,
            "calibrate_image.py",
            "--image",
            source_display,
            "--model",
            image_model,
            "--schema",
            image_schema,
            "--conf",
            image_conf,
        ]
    )
    if st.button("Görseli analiz et", type="primary", key="image_button"):
        try:
            image_path = (
                _save_upload(image_upload, "images")
                if image_upload is not None
                else _path(image_local)
            )
            with st.spinner("Pose tahmini ve homography hesaplanıyor…"):
                result = save_image_calibration(
                    image_path,
                    _path(image_model),
                    _path(image_schema),
                    PROJECT_ROOT / "outputs" / "predictions",
                    image_conf,
                    DEFAULT_CONFIG,
                )
            st.session_state["image_result"] = result
        except Exception as exc:
            st.exception(exc)
    image_result = st.session_state.get("image_result")
    if image_result:
        left, right = st.columns(2)
        left.image(image_result["overlay_path"], caption="Keypoint + saha izdüşümü", width="stretch")
        right.image(image_result["top_view_path"], caption="Top view", width="stretch")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Kalibrasyon", "Kabul" if image_result["accepted_auto"] else "İncele")
        metric2.metric("Güven skoru", f"{image_result['confidence_score']:.3f}")
        calibration = image_result.get("calibration")
        metric3.metric(
            "Reprojection error",
            f"{calibration['mean_reprojection_error_px']:.2f} px"
            if calibration
            else "—",
        )
        st.dataframe(pd.DataFrame(image_result["keypoints"]), width="stretch")
        st.caption(f"Çıktılar: {image_result['run_dir']}")


with tabs[5]:
    st.subheader("Videonun ilk karelerinden sabit kamera kalibrasyonu")
    video_upload = st.file_uploader(
        "Kısa video yükle (büyük videolarda yerel yolu kullan)", type=["mp4", "mov", "avi", "mkv"]
    )
    video_local = st.text_input("veya yerel video yolu", "", key="video_local")
    video_model = st.text_input("Eğitilmiş pose modeli", model_default, key="video_model")
    video_schema = st.text_input("Şema", str(DEFAULT_SCHEMA), key="video_schema")
    c1, c2, c3, c4 = st.columns(4)
    seconds = c1.number_input("İlk kaç saniye", 0.5, 60.0, 5.0, 0.5)
    sample_fps = c2.number_input("Sample FPS", 0.5, 30.0, 5.0, 0.5)
    video_conf = c3.number_input("Confidence", 0.01, 0.95, 0.25, 0.01)
    min_kpts = c4.number_input("Min keypoint", 4, 100, 4)
    video_display = video_local or "<yüklenen_video>"
    _show_command(
        [
            sys.executable,
            "calibrate_video.py",
            "--video",
            video_display,
            "--model",
            video_model,
            "--schema",
            video_schema,
            "--seconds",
            seconds,
            "--sample-fps",
            sample_fps,
            "--conf",
            video_conf,
            "--min-keypoints",
            int(min_kpts),
        ]
    )
    if st.button("Videoyu kalibre et", type="primary", key="video_button"):
        try:
            video_path = (
                _save_upload(video_upload, "videos")
                if video_upload is not None
                else _path(video_local)
            )
            with st.spinner("Örnek kareler değerlendiriliyor…"):
                result = calibrate_video(
                    video_path,
                    _path(video_model),
                    _path(video_schema),
                    float(seconds),
                    float(sample_fps),
                    float(video_conf),
                    PROJECT_ROOT / "outputs" / "calibrations",
                    DEFAULT_CONFIG,
                    int(min_kpts),
                )
            st.session_state["video_result"] = result
        except Exception as exc:
            st.exception(exc)
    video_result = st.session_state.get("video_result")
    if video_result:
        left, right = st.columns(2)
        left.image(video_result["overlay_path"], caption="Seçilen en iyi kare", width="stretch")
        right.image(video_result["top_view_path"], caption="Top view", width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Seçilen frame", video_result["selected_frame_index"])
        m2.metric("Reprojection error", f"{video_result['mean_reprojection_error_px']:.2f} px")
        m3.metric("Auto kabul", "Evet" if video_result["accepted_auto"] else "Hayır")
        report_csv = Path(video_result["sampled_frame_report"])
        if report_csv.exists():
            st.dataframe(pd.read_csv(report_csv), width="stretch", hide_index=True)
        st.caption(f"Calibration JSON: {video_result['calibration_path']}")


with tabs[6]:
    st.subheader("Oyuncu kutularını saha koordinatına dönüştür")
    st.write(
        "Kalibrasyon JSON içindeki homography ile bbox alt-orta noktalarını normalize saha "
        "koordinatlarına çevirir. Sonradan SahaAIStudio entegrasyonunda kullanılacak köprü budur."
    )
    calibration_upload = st.file_uploader("Calibration JSON yükle", type=["json"])
    calibration_local = st.text_input("veya calibration.json yolu", "", key="cal_local")
    bbox_text = st.text_area(
        "BBox'lar — her satır: x1,y1,x2,y2",
        "100,100,180,300\n400,120,470,310",
        height=130,
    )
    bbox_csv = st.file_uploader("İsteğe bağlı bbox CSV", type=["csv"])
    if st.button("Koordinatları dönüştür", type="primary", key="transform_button"):
        try:
            if calibration_upload is not None:
                calibration = json.loads(calibration_upload.getvalue().decode("utf-8"))
            else:
                calibration = _read_json(_path(calibration_local))
            if bbox_csv is not None:
                frame = pd.read_csv(bbox_csv)
                required = ["x1", "y1", "x2", "y2"]
                if not all(column in frame.columns for column in required):
                    raise ValueError(f"CSV sütunları gerekli: {required}")
                bboxes = frame[required].to_numpy(dtype=float)
            else:
                rows = [
                    [float(value.strip()) for value in line.split(",")]
                    for line in bbox_text.splitlines()
                    if line.strip()
                ]
                if any(len(row) != 4 for row in rows):
                    raise ValueError("Her bbox satırında dört sayı olmalı.")
                bboxes = np.asarray(rows, dtype=float)
                frame = pd.DataFrame(bboxes, columns=["x1", "y1", "x2", "y2"])
            H = np.asarray(calibration["H_image_to_field"], dtype=float)
            field_xy = bbox_bottom_centers_to_field(bboxes, H)
            frame["field_x"] = field_xy[:, 0]
            frame["field_y"] = field_xy[:, 1]
            frame["inside_pitch"] = (
                (frame["field_x"] >= 0)
                & (frame["field_x"] <= 1)
                & (frame["field_y"] >= 0)
                & (frame["field_y"] <= 1)
            )
            st.dataframe(frame, width="stretch", hide_index=True)
            top_view = draw_top_view_pitch(
                field_xy, [f"bbox {index}" for index in range(len(field_xy))]
            )
            st.image(
                cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB),
                caption="BBox alt-orta noktaları",
                width="stretch",
            )
            st.download_button(
                "Sonucu CSV indir",
                frame.to_csv(index=False).encode("utf-8-sig"),
                "field_coordinates.csv",
                "text/csv",
            )
        except Exception as exc:
            st.exception(exc)
