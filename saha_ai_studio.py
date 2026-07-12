#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saha AI veri hazırlama, eğitim ve tracking iş akışları için masaüstü arayüzü."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable

from saha_tracking_player import TrackingVideoPlayer


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
VIDEO_TYPES = [
    ("Video dosyaları", "*.mp4 *.mov *.avi *.mkv *.webm"),
    ("Tüm dosyalar", "*.*"),
]
MODEL_TYPES = [("PyTorch modeli", "*.pt"), ("Tüm dosyalar", "*.*")]
ZIP_TYPES = [("ZIP arşivi", "*.zip"), ("Tüm dosyalar", "*.*")]
REMAP_TARGETS = ("player (0)", "ball (1)", "outside_person (2)", "Yok say")
MAX_REMAP_CLASSES = 8
BALL_LOAD_PROFILES = {
    "Düşük": {"batch": "1", "workers": "2", "cpu_threads": "4"},
    "Dengeli": {"batch": "2", "workers": "4", "cpu_threads": "8"},
    "Yüksek": {"batch": "3", "workers": "6", "cpu_threads": "12"},
}
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

COLORS = {
    "bg": "#F3F6FB",
    "surface": "#FFFFFF",
    "navy": "#17233C",
    "navy_2": "#223354",
    "blue": "#3478F6",
    "blue_hover": "#2468E5",
    "green": "#19A974",
    "green_hover": "#12845B",
    "red": "#DC4C64",
    "text": "#1C2942",
    "muted": "#69758A",
    "border": "#DDE4EF",
    "log": "#111827",
}


def discover_workspace() -> Path:
    starts = [Path.cwd()]
    if getattr(sys, "frozen", False):
        starts.insert(0, Path(sys.executable).resolve().parent)
    else:
        starts.insert(0, Path(__file__).resolve().parent)

    checked: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)
            if (
                (candidate / "saha_ai_project/scripts/00_setup_project.py").is_file()
                and (candidate / "random_video_clipper.py").is_file()
            ):
                return candidate
    raise FileNotFoundError(
        "Proje dosyaları bulunamadı. Uygulamayı sosyaltactics klasöründe tutun."
    )


def find_python() -> Path:
    if not getattr(sys, "frozen", False):
        current = Path(sys.executable).resolve()
        if current.name.lower() in {"python.exe", "pythonw.exe", "python"}:
            return current

    for name in ("python.exe", "python"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        python_root = Path(local_app_data) / "Programs/Python"
        candidates = sorted(
            python_root.glob("Python*/python.exe"),
            key=lambda item: item.parent.name,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
    raise FileNotFoundError("Python bulunamadı. Python 3.12 kurulumunu kontrol edin.")


def repair_mojibake(text: str) -> str:
    """Eski script çıktılarındaki UTF-8/Windows kodlama karışıklığını düzelt."""
    if not any(marker in text for marker in ("Ã", "Ä", "Å", "â€", "âœ")):
        return text
    for encoding in ("cp1252", "latin1"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
            if fixed.count("�") <= text.count("�"):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def read_yolo_class_names(dataset_dir: Path) -> list[str]:
    """Basit liste/dict biçimindeki data.yaml names alanını oku."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return []
    try:
        text = data_yaml.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    inline = re.search(r"(?m)^names:\s*(\[[^\r\n]*\]|\{[^\r\n]*\})\s*$", text)
    if inline:
        try:
            value = ast.literal_eval(inline.group(1))
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, dict):
                return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
        except (ValueError, SyntaxError, TypeError):
            pass
    block = re.search(r"(?ms)^names:\s*\r?\n((?:\s+\d+\s*:[^\r\n]+\r?\n?)+)", text)
    if block:
        pairs = re.findall(r"(?m)^\s+(\d+)\s*:\s*['\"]?([^'\"\r\n]+)", block.group(1))
        if pairs:
            values = {int(key): value.strip() for key, value in pairs}
            return [values[index] for index in sorted(values)]
    return []


def default_remap_target(class_name: str) -> str:
    normalized = class_name.strip().lower()
    if "ball" in normalized or normalized in {"top", "football"}:
        return "ball (1)"
    if any(token in normalized for token in ("outside", "spectator", "seyirci")):
        return "outside_person (2)"
    if any(token in normalized for token in ("player", "goalkeeper", "referee", "person")):
        return "player (0)"
    return "Yok say"


class SahaAIStudio:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Saha AI Studio")
        self.root.geometry("1180x860")
        self.root.minsize(1000, 720)
        self.root.configure(bg=COLORS["bg"])

        try:
            self.workspace = discover_workspace()
            self.project = self.workspace / "saha_ai_project"
            self.python = find_python()
        except (FileNotFoundError, OSError) as exc:
            messagebox.showerror("Başlatılamadı", str(exc))
            self.root.after(10, self.root.destroy)
            return

        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.active_task = ""
        self.last_output: Path | None = None
        self._success_callback: Callable[[], None] | None = None
        self._success_message = ""
        self.status_var = tk.StringVar(value="Hazır")

        self._configure_style()
        self._create_variables()
        self._build_ui()
        self.refresh_dashboard()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", font=("Segoe UI", 10), foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Surface.TFrame", background=COLORS["surface"])
        style.configure("TLabel", background=COLORS["bg"])
        style.configure(
            "Muted.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["muted"],
        )
        style.configure(
            "Card.TLabelframe",
            background=COLORS["surface"],
            bordercolor=COLORS["border"],
            relief="solid",
            borderwidth=1,
            padding=16,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Card.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["text"],
        )
        style.configure(
            "CardMuted.TLabel",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["blue"],
            foreground="white",
            borderwidth=0,
            padding=(15, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("disabled", "#A9BDE5"),
                ("active", COLORS["blue_hover"]),
            ],
            foreground=[("disabled", "#F2F5FA")],
        )
        style.configure(
            "Success.TButton",
            background=COLORS["green"],
            foreground="white",
            borderwidth=0,
            padding=(15, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Success.TButton",
            background=[("active", COLORS["green_hover"])],
        )
        style.configure("TButton", padding=(11, 7))
        style.configure("TEntry", padding=7, fieldbackground="white")
        style.configure("TCombobox", padding=6, fieldbackground="white")
        style.configure("TCheckbutton", background=COLORS["surface"])
        style.configure("TRadiobutton", background=COLORS["surface"])
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(15, 10),
            background="#E5EAF3",
            foreground=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["surface"])],
            foreground=[("selected", COLORS["blue"])],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["blue"],
            troughcolor="#DCE4F2",
            borderwidth=0,
        )

    def _create_variables(self) -> None:
        p = self.project
        w = self.workspace
        best = self._newest_best_model() or (
            p / "models/trained/player_ball_v1/weights/best.pt"
        )

        self.download_url = tk.StringVar()
        self.download_output = tk.StringVar(value=str(w / "downloads"))
        self.download_angle = tk.StringVar(value="1")
        self.download_wait = tk.StringVar(value="25")
        self.download_visible = tk.BooleanVar(value=True)

        self.split_source = tk.StringVar()
        self.split_output = tk.StringVar(value=str(p / "videos/raw"))
        self.split_count = tk.StringVar(value="100")
        self.split_minutes = tk.StringVar(value="2")
        self.split_seed = tk.StringVar(value="42")
        self.split_fast = tk.BooleanVar(value=True)

        self.frames_source = tk.StringVar(value=str(p / "videos/raw"))
        self.frames_output = tk.StringVar(value=str(p / "frames/candidate_frames"))
        self.frames_every = tk.StringVar(value="2")
        self.frames_max = tk.StringVar(value="0")
        self.staging_output = tk.StringVar(value=str(p / "frames/label_staging"))
        self.staging_max = tk.StringVar(value="500")
        self.staging_gap = tk.StringVar(value="0.75")
        self.cvat_pack_videos: list[Path] = []
        self.cvat_pack_output = tk.StringVar(value=str(p / "frames/cvat_packs"))
        self.cvat_pack_count = tk.StringVar(value="200")
        self.cvat_pack_strategy = tk.StringVar(value="Dengeli ve eşit aralıklı")
        self.cvat_pack_seed = tk.StringVar(value="42")
        self.cvat_pack_result = tk.StringVar(value="Henüz paket oluşturulmadı.")

        default_zip = p / "saha_labels_v1.zip"
        self.cvat_zip = tk.StringVar(value=str(default_zip) if default_zip.is_file() else "")
        self.cvat_overwrite = tk.BooleanVar(value=False)
        self.labeled_images = tk.StringVar(value=str(p / "labels/images"))
        self.manual_labels = tk.StringVar(value=str(p / "labels/manual"))
        self.dataset_output = tk.StringVar(value=str(p / "datasets/dataset_v1"))
        self.dataset_train = tk.StringVar(value="0.70")
        self.dataset_val = tk.StringVar(value="0.20")
        self.dataset_test = tk.StringVar(value="0.10")
        self.dataset_seed = tk.StringVar(value="42")
        self.dataset_negatives = tk.BooleanVar(value=False)
        remap_source_path = (
            p / "datasets/football-players-detection.v20-rf-detr-m.yolo26"
        )
        self.remap_source = tk.StringVar(
            value=str(remap_source_path) if remap_source_path.is_dir() else ""
        )
        self.remap_merge = tk.StringVar(value=str(p / "datasets/dataset_v1"))
        self.remap_output = tk.StringVar(
            value=str(p / "datasets/football_players_v20_merged")
        )
        self.remap_include_existing = tk.BooleanVar(value=True)
        self.remap_overwrite = tk.BooleanVar(value=False)
        initial_names = read_yolo_class_names(remap_source_path)
        self.remap_source_name_vars = [
            tk.StringVar(value=f"{index}: {initial_names[index]}" if index < len(initial_names) else "")
            for index in range(MAX_REMAP_CLASSES)
        ]
        self.remap_target_vars = [
            tk.StringVar(
                value=default_remap_target(initial_names[index])
                if index < len(initial_names)
                else "Yok say"
            )
            for index in range(MAX_REMAP_CLASSES)
        ]
        self.remap_result = tk.StringVar(value="Henüz dönüştürme yapılmadı.")
        self.category_plan = tk.StringVar(
            value=str(p / "datasets/Categories/dataset_plan_player_v8.json")
        )
        self.category_output = tk.StringVar(
            value=str(p / "datasets/CuratedFootball/builds/player_ball_v8_master")
        )
        self.category_overwrite = tk.BooleanVar(value=False)
        category_ready = (
            p / "datasets/CuratedFootball/builds/player_ball_v8_master/data.yaml"
        ).is_file()
        self.category_result = tk.StringVar(
            value="Master dataset hazır." if category_ready else "Henüz master dataset oluşturulmadı."
        )

        master_dataset = p / "datasets/CuratedFootball/builds/player_ball_v8_master"
        player_v2_best = p / "models/trained/player_ball_v2-7/weights/best.pt"
        self.train_dataset = tk.StringVar(
            value=str(master_dataset if (master_dataset / "data.yaml").is_file() else p / "datasets/dataset_v1")
        )
        player_v8_base = (
            player_v2_best if player_v2_best.is_file() else p / "models/base/yolo26m.pt"
        )
        self.train_base = tk.StringVar(
            value=str(
                player_v8_base
                if (master_dataset / "data.yaml").is_file()
                else best if best.is_file()
                else p / "models/base/yolo26m.pt"
            )
        )
        self.train_name = tk.StringVar(value="player_ball_v8")
        self.train_epochs = tk.StringVar(value="80")
        self.train_imgsz = tk.StringVar(value="1024")
        self.train_batch = tk.StringVar(value="2")
        self.train_device = tk.StringVar(value="0")
        self.val_model = tk.StringVar(value=str(best))
        self.val_name = tk.StringVar(value="model_validation")
        self.val_split = tk.StringVar(value="val")

        self.track_video = tk.StringVar()
        self.track_model = tk.StringVar(value=str(best))
        self.track_output = tk.StringVar()
        self.track_mode = tk.StringVar(value="30 saniyelik hızlı test")
        self.track_duration = tk.StringVar(value="30")
        self.track_sample = tk.StringVar(value="0.2")
        self.track_conf = tk.StringVar(value="0.12")
        self.track_players = tk.StringVar(value="14")
        self.track_calibration_mode = tk.StringVar(value="four_points")
        self.track_calibration_file = tk.StringVar()
        self.track_force_calibration = tk.BooleanVar(value=False)
        self.track_sahi = tk.BooleanVar(value=False)
        self.track_manual_players = tk.BooleanVar(value=False)

        ball_source = Path(r"D:\Dev\sosyaltactics datasets\ball_datasets")
        ball_v8_dataset = p / "datasets/BallOnly/master_v8_realcase"
        ball_v3_dataset = p / "datasets/BallOnly/master_v3_augmented"
        ball_v1_dataset = p / "datasets/BallOnly/master_v1"
        if (ball_v8_dataset / "data.yaml").is_file():
            ball_dataset = ball_v8_dataset
            ball_dataset_status = "Hazır: Ball v8 real-case test setli dataset."
        elif (ball_v3_dataset / "data.yaml").is_file():
            ball_dataset = ball_v3_dataset
            ball_dataset_status = "Hazır: Ball v3 augmented dataset."
        else:
            ball_dataset = ball_v1_dataset
            ball_dataset_status = (
                "Hazır: 8.031 kare, 7.172 top kutusu."
                if (ball_dataset / "data.yaml").is_file()
                else "Ball-only dataset henüz hazırlanmadı."
            )
        ball_model = self._newest_ball_model() or (
            p / "models/trained/ball_only_v1/weights/best.pt"
        )
        self.ball_source_root = tk.StringVar(
            value=str(ball_source) if ball_source.is_dir() else ""
        )
        self.ball_train_labels_source = tk.StringVar(
            value=r"D:\Dev\sosyaltactics datasets\ball_datasets\BallDetectNew"
        )
        self.ball_train_images_source = tk.StringVar(
            value=(
                r"D:\Dev\Github\sosyaltactics\saha_ai_project\frames\cvat_packs"
                r"\cvat_frames_20260711_203715\images"
            )
        )
        self.ball_test_labels_source = tk.StringVar(
            value=r"D:\Dev\sosyaltactics datasets\AaalBall"
        )
        self.ball_test_images_source = tk.StringVar(
            value=r"D:\Dev\sosyaltactics datasets\cvat_frames_20260709_173500\images"
        )
        self.ball_real_labels_source = self.ball_test_labels_source
        self.ball_real_images_source = self.ball_test_images_source
        self.ball_real_max_images = tk.StringVar(value="0")
        self.ball_dataset = tk.StringVar(
            value=str(p / "datasets/BallOnly/ball_detect_new_v1")
        )
        custom_ball_dataset = p / "datasets/BallOnly/ball_detect_new_v1"
        if (custom_ball_dataset / "data.yaml").is_file():
            ball_dataset_status = "Hazır: BallDetectNew train ve AaalBall test dataseti."
        elif (ball_v8_dataset / "data.yaml").is_file():
            ball_dataset_status = "Hazır: Ball v8 real-case test setli dataset."
        elif (ball_v3_dataset / "data.yaml").is_file():
            ball_dataset_status = "Hazır: Ball v3 augmented dataset."
        else:
            ball_dataset_status = (
                "Hazır: 8.031 kare, 7.172 top kutusu."
                if (ball_v1_dataset / "data.yaml").is_file()
                else "Ball-only dataset henüz hazırlanmadı."
            )
        self.ball_dataset_status = tk.StringVar(value=ball_dataset_status)
        ball_v2_model = p / "models/trained/ball_only_v2/weights/best.pt"
        self.ball_base_model = tk.StringVar(
            value=str(ball_v2_model if ball_v2_model.is_file() else p / "models/base/yolo26m.pt")
        )
        self.ball_train_name = tk.StringVar(value="ball_only_newdata_v1")
        self.ball_epochs = tk.StringVar(value="80")
        self.ball_imgsz = tk.StringVar(value="1024")
        self.ball_batch = tk.StringVar(value="2")
        self.ball_workers = tk.StringVar(value="4")
        self.ball_cpu_threads = tk.StringVar(value="8")
        self.ball_load_profile = tk.StringVar(value="Dengeli")
        self.ball_load_status = tk.StringVar(
            value="Dengeli: batch 2, 4 veri worker'ı, 8 CPU thread."
        )
        self.ball_device = tk.StringVar(value="0")
        self.ball_patience = tk.StringVar(value="20")
        self.ball_model = tk.StringVar(value=str(ball_model))
        self.ball_video = tk.StringVar()
        self.ball_output = tk.StringVar(
            value=str(p / "experiments/outputs/ball_only_test")
        )
        self.ball_test_mode = tk.StringVar(value="30 saniyelik hızlı test")
        self.ball_duration = tk.StringVar(value="30")
        self.ball_sample = tk.StringVar(value="0.1")
        self.ball_conf = tk.StringVar(value="0.05")
        self.ball_test_imgsz = tk.StringVar(value="1280")
        self.ball_tiled = tk.BooleanVar(value=False)
        self.ball_tile_size = tk.StringVar(value="640")

        self.stat_vars = {
            "raw": tk.StringVar(value="0"),
            "test": tk.StringVar(value="0"),
            "frames": tk.StringVar(value="0"),
            "images": tk.StringVar(value="0"),
            "labels": tk.StringVar(value="0"),
            "models": tk.StringVar(value="0"),
        }

    def _build_ui(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=COLORS["navy"], height=76)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Saha AI Studio",
            bg=COLORS["navy"],
            fg="white",
            font=("Segoe UI Semibold", 20),
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(12, 0))
        tk.Label(
            header,
            text="Videodan veri setine, eğitimden saha takibine tek çalışma alanı",
            bg=COLORS["navy"],
            fg="#B9C5DA",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="w", padx=23, pady=(0, 10))
        ttk.Button(
            header,
            text="Proje klasörünü aç",
            command=lambda: self._open_path(self.project),
        ).grid(row=0, column=1, rowspan=2, padx=(8, 22), pady=18)

        body = ttk.Frame(self.root, padding=(16, 12, 16, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(body)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.tabs: list[ttk.Frame] = []
        tab_specs = [
            ("Genel Bakış", self._build_dashboard_tab),
            ("1. Video İndir", self._build_download_tab),
            ("2. Video Parçala", self._build_split_tab),
            ("3. Frame & Etiket", self._build_frames_tab),
            ("4. Veri Seti", self._build_dataset_tab),
            ("5. Eğitim", self._build_training_tab),
            ("6. Test & Tracking", self._build_tracking_tab),
            ("7. Top Modeli", self._build_ball_tab),
            ("Araçlar", self._build_tools_tab),
        ]
        for title, builder in tab_specs:
            tab = ttk.Frame(self.notebook, padding=18)
            self.tabs.append(tab)
            self.notebook.add(tab, text=title)
            builder(tab)

        log_outer = ttk.Frame(self.root, padding=(16, 10, 16, 0))
        log_outer.grid(row=2, column=0, sticky="ew")
        log_outer.columnconfigure(0, weight=1)
        log_header = ttk.Frame(log_outer)
        log_header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        log_header.columnconfigure(0, weight=1)
        ttk.Label(
            log_header,
            text="İşlem günlüğü",
            font=("Segoe UI Semibold", 10),
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(log_header, text="Temizle", command=self._clear_log).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(log_header, text="Son çıktıyı aç", command=self._open_last_output).grid(
            row=0, column=2
        )
        self.log = scrolledtext.ScrolledText(
            log_outer,
            height=8,
            bg=COLORS["log"],
            fg="#D9E3F0",
            insertbackground="white",
            font=("Cascadia Mono", 9),
            relief="flat",
            padx=10,
            pady=8,
            wrap="word",
            state="disabled",
        )
        self.log.grid(row=1, column=0, sticky="ew")

        status = tk.Frame(self.root, bg=COLORS["navy_2"], height=36)
        status.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        status.grid_propagate(False)
        status.columnconfigure(0, weight=1)
        tk.Label(
            status,
            textvariable=self.status_var,
            bg=COLORS["navy_2"],
            fg="#D9E3F0",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=8)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=210)
        self.progress.grid(row=0, column=1, padx=8, pady=8)
        self.cancel_button = ttk.Button(
            status,
            text="İşlemi durdur",
            command=self._cancel_task,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=2, padx=(0, 16), pady=4)

    def _card(self, parent: ttk.Frame, title: str, column: int = 0) -> ttk.LabelFrame:
        card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe")
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 10) if column == 0 else (10, 0))
        card.columnconfigure(1, weight=1)
        return card

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        mode: str,
        filetypes=None,
    ) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        if mode == "dir":
            command = lambda: self._choose_dir(variable)
        else:
            command = lambda: self._choose_file(variable, filetypes or [])
        ttk.Button(parent, text="Seç", command=command).grid(
            row=row, column=2, padx=(8, 0), pady=5
        )

    def _value_row(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        width: int = 14,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=5)
        return entry

    def _note(self, parent, text: str, row: int, columnspan: int = 3) -> None:
        ttk.Label(
            parent,
            text=text,
            style="CardMuted.TLabel",
            wraplength=490,
            justify="left",
        ).grid(row=row, column=0, columnspan=columnspan, sticky="w", pady=(8, 4))

    def _build_dashboard_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure((0, 1, 2), weight=1)
        stats = [
            ("Ham video", "raw"),
            ("Test videosu", "test"),
            ("Aday frame", "frames"),
            ("Arşiv görseli", "images"),
            ("Manuel etiket", "labels"),
            ("Eğitilmiş model", "models"),
        ]
        for index, (label, key) in enumerate(stats):
            frame = tk.Frame(
                tab,
                bg=COLORS["surface"],
                highlightbackground=COLORS["border"],
                highlightthickness=1,
                padx=14,
                pady=10,
            )
            frame.grid(
                row=index // 3,
                column=index % 3,
                sticky="ew",
                padx=6,
                pady=6,
            )
            tk.Label(
                frame,
                textvariable=self.stat_vars[key],
                bg=COLORS["surface"],
                fg=COLORS["blue"],
                font=("Segoe UI Semibold", 20),
            ).pack(anchor="w")
            tk.Label(
                frame,
                text=label,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Segoe UI", 9),
            ).pack(anchor="w")

        pipeline = ttk.LabelFrame(tab, text="Önerilen sıra", style="Card.TLabelframe")
        pipeline.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=(18, 6))
        pipeline.columnconfigure(0, weight=1)
        steps = (
            "1. Maçı indir veya elindeki videoyu seç.\n"
            "2. Uzun videoyu rastgele kısa kliplere ayır.\n"
            "3. Frame çıkar, etiketleme alanını hazırla ve CVAT'ta kutuları çiz.\n"
            "4. CVAT ZIP'ini içe aktar; kalıcı görsel/etiket arşivi oluşsun.\n"
            "5. Veri setini kur, modeli eğit ve doğrula.\n"
            "6. Önce 30 saniyelik tracking testi, sonra tam video."
        )
        ttk.Label(
            pipeline,
            text=steps,
            style="Card.TLabel",
            justify="left",
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="nw")

        health = ttk.LabelFrame(tab, text="Durum", style="Card.TLabelframe")
        health.grid(row=2, column=2, sticky="nsew", padx=6, pady=(18, 6))
        ttk.Label(
            health,
            text="GMC düzeltmesi etkin",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 11),
            foreground=COLORS["green"],
        ).pack(anchor="w")
        ttk.Label(
            health,
            text=(
                "Varsayılan tracking akışı YOLO predict + özel stable-ID "
                "tracker kullanır. BoT-SORT optical-flow uyarıları oluşmaz."
            ),
            style="CardMuted.TLabel",
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(5, 12))
        ttk.Button(
            health,
            text="Sistem kontrolünü çalıştır",
            style="Accent.TButton",
            command=self.start_health_check,
        ).pack(anchor="w")

        ttk.Button(
            tab,
            text="Sayıları yenile",
            command=self.refresh_dashboard,
        ).grid(row=3, column=0, sticky="w", padx=6, pady=12)

    def _build_download_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        card = ttk.LabelFrame(tab, text="Sosyal Halı Saha videosu", style="Card.TLabelframe")
        card.grid(row=0, column=0, sticky="new")
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="Maç sayfası linki", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5
        )
        ttk.Entry(card, textvariable=self.download_url).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=5
        )
        self._path_row(card, 1, "Kayıt klasörü", self.download_output, mode="dir")
        ttk.Label(card, text="Kamera açısı", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            card,
            textvariable=self.download_angle,
            values=("1", "2", "3", "4"),
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(card, text="Bekleme süresi (sn)", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Entry(card, textvariable=self.download_wait, width=14).grid(
            row=3, column=1, sticky="w", pady=5
        )
        ttk.Checkbutton(
            card,
            text="Tarayıcıyı görünür aç (giriş/reklam/kamera seçimi için önerilir)",
            variable=self.download_visible,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 4))
        self._note(
            card,
            "2. kamera seçilirse uygulama açı düğmesini otomatik arar. "
            "Bulamazsa açık tarayıcıdan elle seçip videoyu başlatabilirsiniz.",
            5,
        )
        actions = ttk.Frame(card, style="Surface.TFrame")
        actions.grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(
            actions,
            text="Videoyu indir",
            style="Accent.TButton",
            command=self.start_download,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="İndirilenleri aç",
            command=lambda: self._open_path(Path(self.download_output.get())),
        ).pack(side="left", padx=8)

    def _build_split_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        card = ttk.LabelFrame(tab, text="Rastgele video klipleri", style="Card.TLabelframe")
        card.grid(row=0, column=0, sticky="new")
        card.columnconfigure(1, weight=1)
        self._path_row(card, 0, "Kaynak video", self.split_source, mode="file", filetypes=VIDEO_TYPES)
        self._path_row(card, 1, "Çıktı ana klasörü", self.split_output, mode="dir")
        self._value_row(card, 2, "Klip adedi", self.split_count)
        self._value_row(card, 3, "Her klip (dakika)", self.split_minutes)
        self._value_row(card, 4, "Random seed", self.split_seed)
        ttk.Checkbutton(
            card,
            text="Hızlı kopyalama kullan (önerilir)",
            variable=self.split_fast,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 4))
        self._note(
            card,
            "Klipler çakışabilir. Aynı seed ve ayarlar aynı başlangıç "
            "noktalarını üretir. Hassas mod daha yavaş fakat kesim noktaları daha kesindir.",
            6,
        )
        actions = ttk.Frame(card, style="Surface.TFrame")
        actions.grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(
            actions,
            text="Klipleri oluştur",
            style="Accent.TButton",
            command=self.start_split,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Çıktıyı aç",
            command=lambda: self._open_path(Path(self.split_output.get())),
        ).pack(side="left", padx=8)

    def _build_frames_tab(self, tab: ttk.Frame) -> None:
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        canvas = tk.Canvas(
            tab,
            bg=COLORS["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas, padding=(0, 0, 8, 18))
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(content_window, width=event.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )

        content.columnconfigure((0, 1), weight=1)
        extract = self._card(content, "A. Videolardan frame çıkar", 0)
        self._path_row(extract, 0, "Video klasörü", self.frames_source, mode="dir")
        self._path_row(extract, 1, "Frame klasörü", self.frames_output, mode="dir")
        self._value_row(extract, 2, "Kaç saniyede bir", self.frames_every)
        self._value_row(extract, 3, "Video başına üst sınır", self.frames_max)
        self._note(extract, "0 üst sınır olmadığı anlamına gelir.", 4)
        ttk.Button(
            extract,
            text="Frame çıkarmayı başlat",
            style="Accent.TButton",
            command=self.start_extract_frames,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))

        staging = self._card(content, "B. Etiketleme alanını hazırla", 1)
        self._path_row(staging, 0, "Staging klasörü", self.staging_output, mode="dir")
        self._value_row(staging, 1, "En fazla aday frame", self.staging_max)
        self._value_row(staging, 2, "Minimum zaman aralığı", self.staging_gap)
        self._note(
            staging,
            "Bu işlem staging/images alanını yeniden kurar. CVAT ZIP'i daha "
            "önce içe aktarıldıysa eski etiketli görseller labels/images içinde korunur.",
            3,
        )
        buttons = ttk.Frame(staging, style="Surface.TFrame")
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(
            buttons,
            text="Etiketleme alanını hazırla",
            style="Accent.TButton",
            command=self.start_prepare_staging,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="Görselleri aç",
            command=lambda: self._open_path(Path(self.staging_output.get()) / "images"),
        ).pack(side="left", padx=8)
        ttk.Button(
            buttons,
            text="CVAT'ı aç",
            command=lambda: webbrowser.open("https://app.cvat.ai/tasks"),
        ).pack(side="left")

        pack = ttk.LabelFrame(
            content,
            text="C. Seçtiğim videolardan doğrudan CVAT paketi",
            style="Card.TLabelframe",
        )
        pack.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        pack.columnconfigure(1, weight=1)
        ttk.Label(pack, text="Seçili videolar", style="Card.TLabel").grid(
            row=0, column=0, sticky="nw", padx=(0, 10), pady=5
        )
        self.cvat_pack_list = tk.Listbox(
            pack,
            height=4,
            selectmode="extended",
            bg="white",
            fg=COLORS["text"],
            selectbackground=COLORS["blue"],
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Segoe UI", 9),
        )
        self.cvat_pack_list.grid(
            row=0,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=5,
        )
        video_buttons = ttk.Frame(pack, style="Surface.TFrame")
        video_buttons.grid(row=1, column=1, columnspan=2, sticky="w", pady=(2, 6))
        ttk.Button(
            video_buttons,
            text="Videoları seç",
            command=self._choose_cvat_pack_videos,
        ).pack(side="left")
        ttk.Button(
            video_buttons,
            text="Seçilileri kaldır",
            command=self._remove_cvat_pack_videos,
        ).pack(side="left", padx=6)
        ttk.Button(
            video_buttons,
            text="Listeyi temizle",
            command=self._clear_cvat_pack_videos,
        ).pack(side="left")
        self._path_row(pack, 2, "Paket klasörü", self.cvat_pack_output, mode="dir")

        options = ttk.Frame(pack, style="Surface.TFrame")
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Label(options, text="Toplam frame", style="Card.TLabel").pack(side="left")
        ttk.Entry(options, textvariable=self.cvat_pack_count, width=10).pack(
            side="left", padx=(7, 18)
        )
        ttk.Label(options, text="Dağılım", style="Card.TLabel").pack(side="left")
        ttk.Combobox(
            options,
            textvariable=self.cvat_pack_strategy,
            values=("Dengeli ve eşit aralıklı", "Dengeli ve rastgele"),
            state="readonly",
            width=25,
        ).pack(side="left", padx=(7, 18))
        ttk.Label(options, text="Seed", style="Card.TLabel").pack(side="left")
        ttk.Entry(options, textvariable=self.cvat_pack_seed, width=10).pack(
            side="left", padx=7
        )
        ttk.Label(
            pack,
            text=(
                "Girilen sayı toplam frame adedidir; videolar arasında dengeli "
                "paylaştırılır. Sonuçtaki cvat_upload.zip doğrudan CVAT Data alanına yüklenir."
            ),
            style="CardMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(9, 4))
        pack_actions = ttk.Frame(pack, style="Surface.TFrame")
        pack_actions.grid(row=5, column=0, columnspan=3, sticky="w", pady=(9, 0))
        ttk.Button(
            pack_actions,
            text="CVAT ZIP'ini oluştur",
            style="Success.TButton",
            command=self.start_create_cvat_pack,
        ).pack(side="left")
        ttk.Button(
            pack_actions,
            text="Paketleri aç",
            command=lambda: self._open_path(Path(self.cvat_pack_output.get())),
        ).pack(side="left", padx=8)
        ttk.Button(
            pack_actions,
            text="CVAT'ı aç",
            command=lambda: webbrowser.open("https://app.cvat.ai/tasks/create"),
        ).pack(side="left")
        ttk.Label(
            pack,
            textvariable=self.cvat_pack_result,
            style="CardMuted.TLabel",
            wraplength=900,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(9, 0))

    def _build_dataset_tab(self, tab: ttk.Frame) -> None:
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        canvas = tk.Canvas(
            tab,
            bg=COLORS["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas, padding=(0, 0, 8, 18))
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(content_window, width=event.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        content.columnconfigure((0, 1), weight=1)

        importer = self._card(content, "A. CVAT ZIP'ini içe aktar", 0)
        self._path_row(importer, 0, "CVAT YOLO ZIP", self.cvat_zip, mode="file", filetypes=ZIP_TYPES)
        self._path_row(importer, 1, "Kalıcı görsel arşivi", self.labeled_images, mode="dir")
        self._path_row(importer, 2, "Manuel etiket klasörü", self.manual_labels, mode="dir")
        ttk.Checkbutton(
            importer,
            text="Aynı adlı mevcut etiketleri değiştir",
            variable=self.cvat_overwrite,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 4))
        self._note(
            importer,
            "CVAT'tan Ultralytics YOLO formatında indirdiğiniz ZIP'i seçin. "
            "Etiketli görseller kalıcı arşive kopyalanır; sonraki maçlarla birleşir.",
            4,
        )
        ttk.Button(
            importer,
            text="ZIP'i içe aktar",
            style="Accent.TButton",
            command=self.start_import_cvat,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(12, 0))

        dataset = self._card(content, "B. Train / val / test veri seti", 1)
        self._path_row(dataset, 0, "Veri seti çıkışı", self.dataset_output, mode="dir")
        self._value_row(dataset, 1, "Train oranı", self.dataset_train)
        self._value_row(dataset, 2, "Validation oranı", self.dataset_val)
        self._value_row(dataset, 3, "Test oranı", self.dataset_test)
        self._value_row(dataset, 4, "Random seed", self.dataset_seed)
        ttk.Checkbutton(
            dataset,
            text="Etiketsiz görselleri negatif örnek olarak ekle",
            variable=self.dataset_negatives,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 4))
        self._note(
            dataset,
            "Sınıflar: 0 player, 1 ball, 2 outside_person. Oranların toplamı 1 olmalıdır.",
            6,
        )
        buttons = ttk.Frame(dataset, style="Surface.TFrame")
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(
            buttons,
            text="Veri setini oluştur",
            style="Success.TButton",
            command=self.start_build_dataset,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="Veri setini aç",
            command=lambda: self._open_path(Path(self.dataset_output.get())),
        ).pack(side="left", padx=8)

        remap = ttk.LabelFrame(
            content,
            text="C. Hazır YOLO dataset'ini dönüştür ve birleştir",
            style="Card.TLabelframe",
        )
        remap.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        remap.columnconfigure(1, weight=1)
        ttk.Label(remap, text="Kaynak dataset", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=5
        )
        ttk.Entry(remap, textvariable=self.remap_source).grid(
            row=0, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            remap,
            text="Seç",
            command=self._choose_remap_source,
        ).grid(row=0, column=2, padx=(8, 0), pady=5)
        self._path_row(
            remap,
            1,
            "Mevcut dataset",
            self.remap_merge,
            mode="dir",
        )
        self._path_row(
            remap,
            2,
            "Yeni dataset çıkışı",
            self.remap_output,
            mode="dir",
        )
        ttk.Checkbutton(
            remap,
            text="Mevcut dataset_v1 ile birleştir (önerilir)",
            variable=self.remap_include_existing,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(7, 2))
        ttk.Checkbutton(
            remap,
            text="Çıktı klasörü varsa üzerine yaz",
            variable=self.remap_overwrite,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(
            remap,
            text="Kaynak sınıf",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 9),
        ).grid(row=5, column=0, sticky="w", pady=(10, 3))
        ttk.Label(
            remap,
            text="Bizim hedef sınıfımız",
            style="Card.TLabel",
            font=("Segoe UI Semibold", 9),
        ).grid(row=5, column=1, sticky="w", pady=(10, 3))
        self.remap_class_rows: list[tuple[ttk.Label, ttk.Combobox]] = []
        for index in range(MAX_REMAP_CLASSES):
            label = ttk.Label(
                remap,
                textvariable=self.remap_source_name_vars[index],
                style="Card.TLabel",
            )
            label.grid(row=6 + index, column=0, sticky="w", pady=3)
            combo = ttk.Combobox(
                remap,
                textvariable=self.remap_target_vars[index],
                values=REMAP_TARGETS,
                state="readonly",
                width=25,
            )
            combo.grid(row=6 + index, column=1, sticky="w", pady=3)
            self.remap_class_rows.append((label, combo))
        self._refresh_remap_class_rows()
        note_row = 6 + MAX_REMAP_CLASSES
        ttk.Label(
            remap,
            text=(
                "Orijinal dataset korunur. Yeni klasörde sınıflar 0 player, "
                "1 ball, 2 outside_person olacak; split'ler ve lisans README'leri korunur."
            ),
            style="CardMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=note_row, column=0, columnspan=3, sticky="w", pady=(9, 4))
        remap_actions = ttk.Frame(remap, style="Surface.TFrame")
        remap_actions.grid(
            row=note_row + 1,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(9, 0),
        )
        ttk.Button(
            remap_actions,
            text="Dönüştür ve birleştir",
            style="Success.TButton",
            command=self.start_remap_merge_dataset,
        ).pack(side="left")
        ttk.Button(
            remap_actions,
            text="Çıktıyı aç",
            command=lambda: self._open_path(Path(self.remap_output.get())),
        ).pack(side="left", padx=8)
        ttk.Label(
            remap,
            textvariable=self.remap_result,
            style="CardMuted.TLabel",
            wraplength=900,
        ).grid(
            row=note_row + 2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(9, 0),
        )

        master = ttk.LabelFrame(
            content,
            text="D. Categories kaynaklarından master dataset oluştur",
            style="Card.TLabelframe",
        )
        master.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        master.columnconfigure(1, weight=1)
        self._path_row(
            master,
            0,
            "Dataset planı",
            self.category_plan,
            mode="file",
            filetypes=[("JSON planı", "*.json"), ("Tüm dosyalar", "*.*")],
        )
        ttk.Label(master, text="Master çıktı", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=5
        )
        ttk.Entry(master, textvariable=self.category_output).grid(
            row=1, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            master,
            text="Seç",
            command=lambda: self._choose_dir(self.category_output),
        ).grid(row=1, column=2, padx=(8, 0), pady=5)
        ttk.Checkbutton(
            master,
            text="Mevcut master_v1 varsa yeniden oluştur",
            variable=self.category_overwrite,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(7, 2))
        ttk.Label(
            master,
            text=(
                "Plan: dış GeneralFootball + Halisaha → train; benim CVAT → val; "
                "SosyalHalisaha → test. Görseller NTFS hardlink ile düzenlenir; "
                "kaynaklara dokunulmaz ve ek disk alanı minimumdur."
            ),
            style="CardMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(9, 4))
        master_actions = ttk.Frame(master, style="Surface.TFrame")
        master_actions.grid(row=4, column=0, columnspan=3, sticky="w", pady=(9, 0))
        ttk.Button(
            master_actions,
            text="Master dataset oluştur",
            style="Success.TButton",
            command=self.start_build_category_master,
        ).pack(side="left")
        ttk.Button(
            master_actions,
            text="Master klasörünü aç",
            command=lambda: self._open_path(Path(self.category_output.get())),
        ).pack(side="left", padx=8)
        ttk.Button(
            master_actions,
            text="Planı aç",
            command=self._open_category_plan,
        ).pack(side="left")
        ttk.Label(
            master,
            textvariable=self.category_result,
            style="CardMuted.TLabel",
            wraplength=900,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(9, 0))

    def _build_training_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure((0, 1), weight=1)
        train = self._card(tab, "A. Modeli eğit", 0)
        self._path_row(train, 0, "Veri seti klasörü", self.train_dataset, mode="dir")
        self._path_row(train, 1, "Başlangıç modeli", self.train_base, mode="file", filetypes=MODEL_TYPES)
        self._value_row(train, 2, "Eğitim adı", self.train_name, width=22)
        self._value_row(train, 3, "Epoch", self.train_epochs)
        self._value_row(train, 4, "Görüntü boyutu", self.train_imgsz)
        self._value_row(train, 5, "Batch", self.train_batch)
        ttk.Label(train, text="Cihaz", style="Card.TLabel").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            train,
            textvariable=self.train_device,
            values=("0", "cpu"),
            width=12,
        ).grid(row=6, column=1, sticky="w", pady=5)
        self._note(
            train,
            "GPU için cihaz 0 kullanılır. Bu bilgisayarda önerilen başlangıç: "
            "1024 px, batch 2, 60 epoch.",
            7,
        )
        ttk.Button(
            train,
            text="Eğitimi başlat",
            style="Success.TButton",
            command=self.start_training,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))

        validate = self._card(tab, "B. Eğitilmiş modeli değerlendir", 1)
        self._path_row(validate, 0, "Model", self.val_model, mode="file", filetypes=MODEL_TYPES)
        self._value_row(validate, 1, "Rapor adı", self.val_name, width=22)
        ttk.Label(validate, text="Değerlendirme split'i", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            validate,
            textvariable=self.val_split,
            values=("val", "test"),
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", pady=5)
        self._note(
            validate,
            "val: yalnız senin CVAT verin. test: final-benzeri SosyalHalisaha. "
            "Precision, recall, mAP50 ve mAP50-95 hesaplanır.",
            3,
        )
        buttons = ttk.Frame(validate, style="Surface.TFrame")
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Button(
            buttons,
            text="Modeli değerlendir",
            style="Accent.TButton",
            command=self.start_validation,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="En yeni best.pt'yi seç",
            command=self.select_newest_model,
        ).pack(side="left", padx=8)
        ttk.Button(
            buttons,
            text="Model klasörünü aç",
            command=lambda: self._open_path(self.project / "models/trained"),
        ).pack(side="left")

    def _build_tracking_tab(self, tab: ttk.Frame) -> None:
        outer_tab = tab
        outer_tab.rowconfigure(0, weight=1)
        outer_tab.columnconfigure(0, weight=1)
        self.tracking_notebook = ttk.Notebook(outer_tab)
        self.tracking_notebook.grid(row=0, column=0, sticky="nsew")
        run_tab = ttk.Frame(self.tracking_notebook, padding=12)
        self.tracking_review_tab = ttk.Frame(self.tracking_notebook, padding=10)
        self.tracking_notebook.add(run_tab, text="Tracking Çalıştır")
        self.tracking_notebook.add(self.tracking_review_tab, text="Frame İncele")

        tab = run_tab
        tab.columnconfigure((0, 1), weight=1)
        source = self._card(tab, "Video ve model", 0)
        self._path_row(source, 0, "Test videosu", self.track_video, mode="file", filetypes=VIDEO_TYPES)
        self._path_row(source, 1, "Eğitilmiş model", self.track_model, mode="file", filetypes=MODEL_TYPES)
        self._path_row(source, 2, "Çıktı klasörü", self.track_output, mode="dir")
        ttk.Label(source, text="Çalışma modu", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=5
        )
        mode_combo = ttk.Combobox(
            source,
            textvariable=self.track_mode,
            values=("30 saniyelik hızlı test", "Özel süre", "Tüm video"),
            state="readonly",
            width=25,
        )
        mode_combo.grid(row=3, column=1, sticky="w", pady=5)
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_tracking_mode())
        self._value_row(source, 4, "Özel süre (sn)", self.track_duration)
        self._note(
            source,
            "Önce 30 saniyelik test önerilir. Aynı çıktı klasörüyle tam "
            "videoya geçerseniz calibration.json yeniden kullanılır.",
            5,
        )

        settings = self._card(tab, "Tracking ayarları", 1)
        self._value_row(settings, 0, "Örnekleme (sn)", self.track_sample)
        self._value_row(settings, 1, "Confidence", self.track_conf)
        self._value_row(settings, 2, "Maksimum oyuncu", self.track_players)
        ttk.Label(settings, text="Kalibrasyon", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.track_calibration_mode,
            values=("four_points", "line"),
            state="readonly",
            width=20,
        ).grid(row=3, column=1, sticky="w", pady=5)
        self._path_row(
            settings,
            4,
            "Mevcut calibration.json",
            self.track_calibration_file,
            mode="file",
            filetypes=[("JSON", "*.json"), ("Tüm dosyalar", "*.*")],
        )
        ttk.Checkbutton(
            settings,
            text="Kalibrasyonu yeniden yap",
            variable=self.track_force_calibration,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=3)
        ttk.Checkbutton(
            settings,
            text="Top için SAHI kullan (daha yavaş)",
            variable=self.track_sahi,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=3)
        ttk.Checkbutton(
            settings,
            text="İlk karede oyuncuları elle numaralandır",
            variable=self.track_manual_players,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=3)

        actions = ttk.LabelFrame(tab, text="Çalıştır", style="Card.TLabelframe")
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        ttk.Button(
            actions,
            text="Tracking'i başlat",
            style="Success.TButton",
            command=self.start_tracking,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Viewer'ı aç",
            style="Accent.TButton",
            command=self.open_viewer,
        ).pack(side="left", padx=8)
        ttk.Button(
            actions,
            text="Çıktı klasörünü aç",
            command=lambda: self._open_path(Path(self.track_output.get())),
        ).pack(side="left")
        ttk.Label(
            actions,
            text="four_points sırası: sol üst → sağ üst → sağ alt → sol alt → Enter",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=20)

        self.tracking_review_tab.rowconfigure(0, weight=1)
        self.tracking_review_tab.columnconfigure(0, weight=1)
        self.tracking_player = TrackingVideoPlayer(
            self.tracking_review_tab,
            output_getter=self.track_output.get,
            video_getter=self.track_video.get,
            on_output_changed=self.track_output.set,
            colors=COLORS,
        )
        self.tracking_player.grid(row=0, column=0, sticky="nsew")

    def _build_ball_tab(self, outer_tab: ttk.Frame) -> None:
        outer_tab.rowconfigure(0, weight=1)
        outer_tab.columnconfigure(0, weight=1)
        self.ball_notebook = ttk.Notebook(outer_tab)
        self.ball_notebook.grid(row=0, column=0, sticky="nsew")

        dataset_tab = ttk.Frame(self.ball_notebook, padding=12)
        training_tab = ttk.Frame(self.ball_notebook, padding=12)
        self.ball_training_tab = training_tab
        test_tab = ttk.Frame(self.ball_notebook, padding=12)
        self.ball_test_tab = test_tab
        self.ball_review_tab = ttk.Frame(self.ball_notebook, padding=10)
        self.ball_notebook.add(dataset_tab, text="1. Dataseti Hazırla")
        self.ball_notebook.add(training_tab, text="2. Eğit")
        self.ball_notebook.add(test_tab, text="Yalnız Top Testi")
        self.ball_notebook.add(self.ball_review_tab, text="Frame İncele")

        dataset_tab.columnconfigure((0, 1), weight=1)
        train_source = self._card(dataset_tab, "Train yapmak istediğim dataset", 0)
        self._path_row(
            train_source,
            0,
            "Imageları",
            self.ball_train_images_source,
            mode="dir",
        )
        self._path_row(
            train_source,
            1,
            "Labelları",
            self.ball_train_labels_source,
            mode="dir",
        )
        test_source = self._card(dataset_tab, "Test olarak kullanmak istediğim dataset", 1)
        self._path_row(
            test_source,
            0,
            "Imageları",
            self.ball_test_images_source,
            mode="dir",
        )
        self._path_row(
            test_source,
            1,
            "Labelları",
            self.ball_test_labels_source,
            mode="dir",
        )
        dataset = ttk.LabelFrame(
            dataset_tab,
            text="YOLO datasetini oluştur",
            style="Card.TLabelframe",
        )
        dataset.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="nsew",
            pady=(14, 0),
        )
        dataset.columnconfigure(1, weight=1)
        self._path_row(
            dataset,
            0,
            "Sonuç dataset klasörü",
            self.ball_dataset,
            mode="dir",
        )
        ttk.Label(
            dataset,
            text=(
                "Etiketsiz kareler negatif kalır; augmentation yapılmaz; "
                "bağımsız test kareleri train'e karıştırılmaz."
            ),
            style="CardMuted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 2))
        dataset_buttons = ttk.Frame(dataset, style="Surface.TFrame")
        dataset_buttons.grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(
            dataset_buttons,
            text="1. Dataseti hazırla",
            style="Accent.TButton",
            command=self.prepare_ball_dataset,
        ).pack(side="left")
        ttk.Button(
            dataset_buttons,
            text="Dataset klasörünü aç",
            command=lambda: self._open_path(Path(self.ball_dataset.get())),
        ).pack(side="left", padx=8)
        ttk.Label(
            dataset,
            textvariable=self.ball_dataset_status,
            style="CardMuted.TLabel",
            wraplength=950,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        training_tab.columnconfigure(0, weight=1)
        training = self._card(training_tab, "Yalnız top modelini eğit", 0)
        training.grid_configure(padx=0)
        self._path_row(
            training,
            0,
            "Ball-only dataset",
            self.ball_dataset,
            mode="dir",
        )
        self._path_row(
            training,
            1,
            "Başlangıç modeli",
            self.ball_base_model,
            mode="file",
            filetypes=MODEL_TYPES,
        )

        def compact_value(
            row: int,
            left_label: str,
            left_var: tk.StringVar,
            right_label: str,
            right_var: tk.StringVar,
        ) -> None:
            ttk.Label(training, text=left_label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=5, padx=(0, 10)
            )
            ttk.Entry(training, textvariable=left_var, width=18).grid(
                row=row, column=1, sticky="w", pady=5
            )
            ttk.Label(training, text=right_label, style="Card.TLabel").grid(
                row=row, column=2, sticky="w", pady=5, padx=(28, 10)
            )
            ttk.Entry(training, textvariable=right_var, width=14).grid(
                row=row, column=3, sticky="w", pady=5
            )

        compact_value(2, "Eğitim adı", self.ball_train_name, "Epoch", self.ball_epochs)
        compact_value(
            3,
            "Görüntü boyutu",
            self.ball_imgsz,
            "Patience",
            self.ball_patience,
        )
        ttk.Label(training, text="Cihaz (0 = GPU)", style="Card.TLabel").grid(
            row=4, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            training,
            textvariable=self.ball_device,
            values=("0", "cpu"),
            width=12,
            state="readonly",
        ).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(training, text="Donanım yükü", style="Card.TLabel").grid(
            row=4, column=2, sticky="w", pady=5, padx=(28, 10)
        )
        load_combo = ttk.Combobox(
            training,
            textvariable=self.ball_load_profile,
            values=tuple(BALL_LOAD_PROFILES),
            width=18,
            state="readonly",
        )
        load_combo.grid(row=4, column=3, sticky="w", pady=5)
        load_combo.bind("<<ComboboxSelected>>", self._sync_ball_load_profile)
        ttk.Label(
            training,
            textvariable=self.ball_load_status,
            style="CardMuted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 4))
        train_buttons = ttk.Frame(training, style="Surface.TFrame")
        train_buttons.grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(
            train_buttons,
            text="2. Eğitimi başlat",
            style="Success.TButton",
            command=self.start_ball_training,
        ).pack(side="left")
        ttk.Button(
            train_buttons,
            text="En yeni top modelini seç",
            command=self.select_newest_ball_model,
        ).pack(side="left", padx=8)
        ttk.Button(
            train_buttons,
            text="3. F1 / test raporu al",
            command=self.start_ball_validation,
        ).pack(side="left")

        test_tab.columnconfigure((0, 1), weight=1)
        source = self._card(test_tab, "Video ve top modeli", 0)
        self._path_row(
            source, 0, "Test videosu", self.ball_video, mode="file", filetypes=VIDEO_TYPES
        )
        self._path_row(
            source, 1, "Top modeli", self.ball_model, mode="file", filetypes=MODEL_TYPES
        )
        self._path_row(source, 2, "Çıktı klasörü", self.ball_output, mode="dir")
        ttk.Label(source, text="Çalışma modu", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=5
        )
        mode_combo = ttk.Combobox(
            source,
            textvariable=self.ball_test_mode,
            values=("30 saniyelik hızlı test", "Özel süre", "Tüm video"),
            state="readonly",
            width=25,
        )
        mode_combo.grid(row=3, column=1, sticky="w", pady=5)
        mode_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._sync_ball_test_mode()
        )
        self._value_row(source, 4, "Özel süre (sn)", self.ball_duration)
        self._note(
            source,
            "Bu test oyuncu modeli veya saha kalibrasyonu çalıştırmaz; yalnız topu "
            "bulur ve sonucu doğrudan Frame İncele'ye yükler.",
            5,
        )

        settings = self._card(test_tab, "Top tespit ayarları", 1)
        self._value_row(settings, 0, "Örnekleme (sn)", self.ball_sample)
        self._value_row(settings, 1, "Confidence", self.ball_conf)
        self._value_row(settings, 2, "Inference boyutu", self.ball_test_imgsz)
        self._value_row(settings, 3, "Tile boyutu", self.ball_tile_size)
        ttk.Checkbutton(
            settings,
            text="Çok küçük top için tiled inference (daha yavaş)",
            variable=self.ball_tiled,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=5)
        self._note(
            settings,
            "Önce tiled kapalı 30 saniyelik test yap. Uzak top kaçıyorsa tiled "
            "inference'ı açıp aynı bölümde karşılaştır.",
            5,
        )
        test_buttons = ttk.Frame(settings, style="Surface.TFrame")
        test_buttons.grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Button(
            test_buttons,
            text="Yalnız top testini başlat",
            style="Success.TButton",
            command=self.start_ball_test,
        ).pack(side="left")
        ttk.Button(
            test_buttons,
            text="En yeni top modelini seç",
            command=self.select_newest_ball_model,
        ).pack(side="left", padx=8)
        ttk.Button(
            test_buttons,
            text="Çıktı klasörünü aç",
            command=lambda: self._open_path(Path(self.ball_output.get())),
        ).pack(side="left")

        self.ball_review_tab.rowconfigure(0, weight=1)
        self.ball_review_tab.columnconfigure(0, weight=1)
        self.ball_player = TrackingVideoPlayer(
            self.ball_review_tab,
            output_getter=self.ball_output.get,
            video_getter=self.ball_video.get,
            on_output_changed=self.ball_output.set,
            colors=COLORS,
        )
        self.ball_player.grid(row=0, column=0, sticky="nsew")

    def _build_tools_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure((0, 1), weight=1)
        system = self._card(tab, "Sistem ve kurulum", 0)
        buttons = [
            ("Bağımlılıkları kontrol et", self.start_health_check),
            ("Eksik Python paketlerini kur", self.start_install_requirements),
            ("Playwright Chromium'u kur", self.start_install_browser),
            ("Proje klasörlerini hazırla", self.start_setup_project),
        ]
        for row, (text, command) in enumerate(buttons):
            ttk.Button(
                system,
                text=text,
                style="Accent.TButton" if row == 0 else "TButton",
                command=command,
            ).grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        self._note(
            system,
            "Paket kurulumu yalnız eksik olduğunda gereklidir. Chromium, "
            "Sosyal Halı Saha indirme sekmesi için kullanılır.",
            len(buttons),
        )

        paths = self._card(tab, "Hızlı erişim", 1)
        locations = [
            ("Ham videolar", self.project / "videos/raw"),
            ("Test videoları", self.project / "videos/test"),
            ("Etiketli görseller", self.project / "labels/images"),
            ("Manuel etiketler", self.project / "labels/manual"),
            ("Eğitilmiş modeller", self.project / "models/trained"),
            ("Raporlar", self.project / "experiments/reports"),
        ]
        for row, (text, path) in enumerate(locations):
            ttk.Button(
                paths,
                text=text,
                command=lambda target=path: self._open_path(target),
            ).grid(row=row, column=0, sticky="ew", pady=4)
        ttk.Label(
            paths,
            text=f"Python: {self.python}\nProje: {self.project}",
            style="CardMuted.TLabel",
            wraplength=450,
            justify="left",
        ).grid(row=len(locations), column=0, sticky="w", pady=(12, 0))

    def _choose_file(self, variable: tk.StringVar, filetypes) -> None:
        selected = filedialog.askopenfilename(filetypes=filetypes or [("Tüm dosyalar", "*.*")])
        if selected:
            variable.set(str(Path(selected).resolve()))
            if variable is self.track_video:
                video = Path(selected)
                self.track_output.set(
                    str(self.project / "experiments/outputs" / f"{video.stem}_tracked")
                )
            elif variable is self.split_source and not self.split_output.get().strip():
                self.split_output.set(str(self.project / "videos/raw"))

    def _choose_cvat_pack_videos(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Frame çıkarılacak videoları seçin",
            filetypes=VIDEO_TYPES,
        )
        existing = {path.resolve() for path in self.cvat_pack_videos}
        for value in selected:
            path = Path(value).resolve()
            if path not in existing:
                self.cvat_pack_videos.append(path)
                existing.add(path)
        self._refresh_cvat_pack_list()

    def _choose_remap_source(self) -> None:
        selected = filedialog.askdirectory(
            title="data.yaml içeren kaynak YOLO dataset klasörünü seçin",
            initialdir=self.remap_source.get().strip() or str(self.project / "datasets"),
        )
        if not selected:
            return
        dataset = Path(selected).resolve()
        if not (dataset / "data.yaml").is_file():
            messagebox.showerror(
                "Dataset bulunamadı",
                "Seçilen klasörün kökünde data.yaml bulunmalıdır.",
            )
            return
        self.remap_source.set(str(dataset))
        names = read_yolo_class_names(dataset)
        for index in range(MAX_REMAP_CLASSES):
            if index < len(names):
                self.remap_source_name_vars[index].set(f"{index}: {names[index]}")
                self.remap_target_vars[index].set(default_remap_target(names[index]))
            else:
                self.remap_source_name_vars[index].set("")
                self.remap_target_vars[index].set("Yok say")
        self._refresh_remap_class_rows()

    def _refresh_remap_class_rows(self) -> None:
        if not hasattr(self, "remap_class_rows"):
            return
        for index, (label, combo) in enumerate(self.remap_class_rows):
            if self.remap_source_name_vars[index].get().strip():
                label.grid()
                combo.grid()
            else:
                label.grid_remove()
                combo.grid_remove()

    def _refresh_cvat_pack_list(self) -> None:
        self.cvat_pack_list.delete(0, "end")
        for index, path in enumerate(self.cvat_pack_videos, start=1):
            self.cvat_pack_list.insert("end", f"{index:02d}  {path.name}  —  {path.parent}")

    def _remove_cvat_pack_videos(self) -> None:
        selected = set(self.cvat_pack_list.curselection())
        if not selected:
            return
        self.cvat_pack_videos = [
            path
            for index, path in enumerate(self.cvat_pack_videos)
            if index not in selected
        ]
        self._refresh_cvat_pack_list()

    def _clear_cvat_pack_videos(self) -> None:
        self.cvat_pack_videos.clear()
        self._refresh_cvat_pack_list()

    def _choose_dir(self, variable: tk.StringVar) -> None:
        initial = variable.get().strip()
        selected = filedialog.askdirectory(initialdir=initial if Path(initial).is_dir() else None)
        if selected:
            variable.set(str(Path(selected).resolve()))

    def _open_path(self, path: Path) -> None:
        try:
            path = path.expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Açılamadı", str(exc))

    def _open_last_output(self) -> None:
        if self.last_output is None:
            messagebox.showinfo("Çıktı yok", "Bu oturumda tamamlanan bir çıktı henüz yok.")
            return
        self._open_path(self.last_output)

    def _open_category_plan(self) -> None:
        plan = Path(self.category_plan.get().strip()).expanduser()
        if not plan.is_file():
            messagebox.showerror("Plan bulunamadı", str(plan))
            return
        os.startfile(str(plan.resolve()))

    def _newest_best_model(self) -> Path | None:
        trained = self.project / "models/trained" if hasattr(self, "project") else None
        if trained is None or not trained.is_dir():
            return None
        models = [
            model
            for model in trained.glob("*/weights/best.pt")
            if not model.parent.parent.name.lower().startswith("ball")
        ]
        return max(models, key=lambda item: item.stat().st_mtime) if models else None

    def _newest_ball_model(self) -> Path | None:
        trained = self.project / "models/trained" if hasattr(self, "project") else None
        if trained is None or not trained.is_dir():
            return None
        models = list(trained.glob("ball*/weights/best.pt"))
        return max(models, key=lambda item: item.stat().st_mtime) if models else None

    def select_newest_model(self) -> None:
        model = self._newest_best_model()
        if model is None:
            messagebox.showwarning("Model yok", "models/trained altında best.pt bulunamadı.")
            return
        self.val_model.set(str(model))
        self.track_model.set(str(model))
        self.status_var.set(f"En yeni model seçildi: {model.name}")

    def select_newest_ball_model(self) -> None:
        model = self._newest_ball_model()
        if model is None:
            messagebox.showwarning(
                "Top modeli yok",
                "models/trained/ball* altında henüz best.pt bulunamadı.",
            )
            return
        self.ball_model.set(str(model))
        self.status_var.set(f"En yeni top modeli seçildi: {model.parent.parent.name}")

    def refresh_dashboard(self) -> None:
        video_ext = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        def video_count(path: Path) -> int:
            return sum(
                1
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in video_ext
            ) if path.is_dir() else 0

        paths = {
            "raw": self.project / "videos/raw",
            "test": self.project / "videos/test",
            "frames": self.project / "frames/candidate_frames",
            "images": self.project / "labels/images",
            "labels": self.project / "labels/manual",
            "models": self.project / "models/trained",
        }
        self.stat_vars["raw"].set(str(video_count(paths["raw"])))
        self.stat_vars["test"].set(str(video_count(paths["test"])))
        self.stat_vars["frames"].set(
            str(sum(1 for item in paths["frames"].rglob("*.jpg"))) if paths["frames"].is_dir() else "0"
        )
        self.stat_vars["images"].set(
            str(sum(1 for item in paths["images"].glob("*") if item.is_file()))
            if paths["images"].is_dir() else "0"
        )
        self.stat_vars["labels"].set(
            str(sum(1 for item in paths["labels"].glob("*.txt")))
            if paths["labels"].is_dir() else "0"
        )
        self.stat_vars["models"].set(
            str(sum(1 for item in paths["models"].glob("*/weights/best.pt")))
            if paths["models"].is_dir() else "0"
        )

    @staticmethod
    def _number(value: str, label: str, kind=float, minimum=None):
        try:
            result = kind(value)
        except ValueError as exc:
            raise ValueError(f"{label} geçerli bir sayı olmalıdır.") from exc
        if minimum is not None and result < minimum:
            raise ValueError(f"{label} en az {minimum} olmalıdır.")
        return result

    @staticmethod
    def _require_file(value: str, label: str) -> Path:
        path = Path(value.strip()).expanduser()
        if not path.is_file():
            raise ValueError(f"{label} bulunamadı: {path}")
        return path.resolve()

    @staticmethod
    def _require_dir_text(value: str, label: str, create: bool = False) -> Path:
        if not value.strip():
            raise ValueError(f"{label} boş olamaz.")
        path = Path(value.strip()).expanduser().resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise ValueError(f"{label} bulunamadı: {path}")
        return path

    def start_download(self) -> None:
        try:
            url = self.download_url.get().strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("Geçerli bir http/https maç linki girin.")
            output = self._require_dir_text(self.download_output.get(), "Kayıt klasörü", True)
            angle = self._number(self.download_angle.get(), "Kamera açısı", int, 1)
            wait = self._number(self.download_wait.get(), "Bekleme süresi", int, 1)
        except (ValueError, OSError) as exc:
            messagebox.showerror("İndirme başlatılamadı", str(exc))
            return
        command = [
            str(self.python), "-u", str(self.workspace / "download_sosyalhalisaha.py"),
            url, "--output", str(output), "--angle", str(angle), "--wait", str(wait),
        ]
        if self.download_visible.get():
            command.append("--headed")
        self._run_task("Video indiriliyor", command, output=output)

    def start_split(self) -> None:
        try:
            source = self._require_file(self.split_source.get(), "Kaynak video")
            output = self._require_dir_text(self.split_output.get(), "Çıktı klasörü", True)
            count = self._number(self.split_count.get(), "Klip adedi", int, 1)
            minutes = self._number(self.split_minutes.get(), "Klip süresi", float, 0.01)
            seed = self._number(self.split_seed.get(), "Random seed", int)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Parçalama başlatılamadı", str(exc))
            return
        command = [
            str(self.python), "-u", str(self.workspace / "random_video_clipper.py"),
            "--source", str(source), "--output", str(output), "--count", str(count),
            "--duration-sec", str(minutes * 60), "--seed", str(seed),
        ]
        if not self.split_fast.get():
            command.append("--precise")
        self._run_task("Video klipleri oluşturuluyor", command, output=output)

    def start_extract_frames(self) -> None:
        try:
            source = self._require_dir_text(self.frames_source.get(), "Video klasörü")
            output = self._require_dir_text(self.frames_output.get(), "Frame klasörü", True)
            every = self._number(self.frames_every.get(), "Frame aralığı", float, 0.01)
            maximum = self._number(self.frames_max.get(), "Frame üst sınırı", int, 0)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Frame çıkarılamadı", str(exc))
            return
        script = self.project / "scripts/01_extract_candidate_frames.py"
        command = [
            str(self.python), "-u", str(script),
            "--videos-dir", str(source), "--out-dir", str(output),
            "--every-sec", str(every), "--max-frames-per-video", str(maximum),
        ]
        self._run_task(
            "Videolardan frame çıkarılıyor",
            command,
            cwd=self.project,
            output=output,
            on_success=self.refresh_dashboard,
        )

    def start_create_cvat_pack(self) -> None:
        try:
            if not self.cvat_pack_videos:
                raise ValueError("En az bir video seçin.")
            missing = [path for path in self.cvat_pack_videos if not path.is_file()]
            if missing:
                raise ValueError(f"Video bulunamadı: {missing[0]}")
            output = self._require_dir_text(
                self.cvat_pack_output.get(),
                "Paket klasörü",
                True,
            )
            count = self._number(
                self.cvat_pack_count.get(),
                "Toplam frame",
                int,
                1,
            )
            seed = self._number(self.cvat_pack_seed.get(), "Seed", int)
        except (ValueError, OSError) as exc:
            messagebox.showerror("CVAT paketi oluşturulamadı", str(exc))
            return

        strategy = (
            "random"
            if self.cvat_pack_strategy.get() == "Dengeli ve rastgele"
            else "uniform"
        )
        script = self.project / "scripts/11_create_cvat_frame_pack.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--videos",
            *[str(path) for path in self.cvat_pack_videos],
            "--count",
            str(count),
            "--output-dir",
            str(output),
            "--strategy",
            strategy,
            "--seed",
            str(seed),
        ]

        def completed() -> None:
            runs = [
                path
                for path in output.glob("cvat_frames_*")
                if path.is_dir() and (path / "cvat_upload.zip").is_file()
            ]
            if not runs:
                self.cvat_pack_result.set("Paket tamamlandı fakat ZIP bulunamadı.")
                return
            latest = max(runs, key=lambda path: path.stat().st_mtime)
            zip_path = latest / "cvat_upload.zip"
            self.last_output = latest
            self.cvat_pack_result.set(f"Hazır: {zip_path}")

        self._run_task(
            f"CVAT için {count} frame hazırlanıyor",
            command,
            cwd=self.project,
            output=output,
            on_success=completed,
            success_message=(
                "CVAT paketi hazır. Paketleri aç düğmesine basıp en yeni "
                "klasördeki cvat_upload.zip dosyasını CVAT Data alanına yükleyin."
            ),
        )

    def start_prepare_staging(self) -> None:
        try:
            candidate = Path(self.frames_output.get()).expanduser().resolve() / "frame_index.csv"
            if not candidate.is_file():
                raise ValueError("frame_index.csv bulunamadı. Önce frame çıkarın.")
            output = self._require_dir_text(self.staging_output.get(), "Staging klasörü", True)
            maximum = self._number(self.staging_max.get(), "Aday frame sınırı", int, 1)
            gap = self._number(self.staging_gap.get(), "Minimum aralık", float, 0)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Staging hazırlanamadı", str(exc))
            return
        script = self.project / "scripts/04_prepare_label_staging.py"
        command = [
            str(self.python), "-u", str(script),
            "--candidate-index", str(candidate), "--out-dir", str(output),
            "--max-candidate", str(maximum), "--max-hard", "0",
            "--min-gap-sec", str(gap),
        ]
        self._run_task("Etiketleme alanı hazırlanıyor", command, cwd=self.project, output=output)

    def start_import_cvat(self) -> None:
        try:
            zip_path = self._require_file(self.cvat_zip.get(), "CVAT ZIP")
            staging_images = Path(self.staging_output.get()).expanduser().resolve() / "images"
            if not staging_images.is_dir():
                raise ValueError("Staging görselleri bulunamadı.")
            labeled = self._require_dir_text(self.labeled_images.get(), "Görsel arşivi", True)
            labels = self._require_dir_text(self.manual_labels.get(), "Etiket klasörü", True)
        except (ValueError, OSError) as exc:
            messagebox.showerror("ZIP içe aktarılamadı", str(exc))
            return
        script = self.project / "scripts/08_import_cvat_zip.py"
        command = [
            str(self.python), "-u", str(script),
            "--zip", str(zip_path),
            "--images-dir", str(staging_images),
            "--labeled-images-dir", str(labeled),
            "--labels-dir", str(labels),
            "--exports-dir", str(self.project / "labels/cvat_exports"),
        ]
        if self.cvat_overwrite.get():
            command.append("--overwrite")
        self._run_task(
            "CVAT etiketleri içe aktarılıyor",
            command,
            cwd=self.project,
            output=labeled,
            on_success=self.refresh_dashboard,
        )

    def start_build_dataset(self) -> None:
        try:
            images = self._require_dir_text(self.labeled_images.get(), "Etiketli görseller")
            labels = self._require_dir_text(self.manual_labels.get(), "Manuel etiketler")
            output = self._require_dir_text(self.dataset_output.get(), "Veri seti çıkışı", True)
            train = self._number(self.dataset_train.get(), "Train oranı", float, 0)
            val = self._number(self.dataset_val.get(), "Validation oranı", float, 0)
            test = self._number(self.dataset_test.get(), "Test oranı", float, 0)
            seed = self._number(self.dataset_seed.get(), "Random seed", int)
            if abs(train + val + test - 1.0) > 1e-8:
                raise ValueError("Train, validation ve test oranlarının toplamı 1 olmalıdır.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Veri seti oluşturulamadı", str(exc))
            return
        script = self.project / "scripts/05_build_yolo_dataset.py"
        command = [
            str(self.python), "-u", str(script),
            "--images-dir", str(images), "--labels-dir", str(labels),
            "--out-dir", str(output), "--train", str(train),
            "--val", str(val), "--test", str(test), "--seed", str(seed),
        ]
        if self.dataset_negatives.get():
            command.append("--include-unlabeled")
        self._run_task("YOLO veri seti oluşturuluyor", command, cwd=self.project, output=output)

    def start_remap_merge_dataset(self) -> None:
        try:
            source = self._require_dir_text(self.remap_source.get(), "Kaynak dataset")
            if not (source / "data.yaml").is_file():
                raise ValueError("Kaynak dataset kökünde data.yaml bulunamadı.")
            names = read_yolo_class_names(source)
            if not names:
                raise ValueError("data.yaml içindeki names sınıf listesi okunamadı.")
            if len(names) > MAX_REMAP_CLASSES:
                raise ValueError(
                    f"Dataset {len(names)} sınıf içeriyor; arayüz en fazla "
                    f"{MAX_REMAP_CLASSES} sınıf destekliyor."
                )
            displayed = [
                variable.get().split(":", 1)[1].strip()
                for variable in self.remap_source_name_vars
                if ":" in variable.get()
            ]
            if displayed != names:
                for index in range(MAX_REMAP_CLASSES):
                    if index < len(names):
                        self.remap_source_name_vars[index].set(f"{index}: {names[index]}")
                        self.remap_target_vars[index].set(default_remap_target(names[index]))
                    else:
                        self.remap_source_name_vars[index].set("")
                        self.remap_target_vars[index].set("Yok say")
                self._refresh_remap_class_rows()
                messagebox.showinfo(
                    "Sınıflar yenilendi",
                    "Kaynak dataset sınıfları yüklendi. Eşlemeleri kontrol edip "
                    "Dönüştür ve birleştir düğmesine tekrar basın.",
                )
                return

            output_text = self.remap_output.get().strip()
            if not output_text:
                raise ValueError("Yeni dataset çıkışı boş olamaz.")
            output = Path(output_text).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and not self.remap_overwrite.get():
                raise ValueError(
                    "Çıktı klasörü zaten var. Yeni bir ad seçin veya üzerine yaz seçeneğini açın."
                )
            merge: Path | None = None
            if self.remap_include_existing.get():
                merge = self._require_dir_text(
                    self.remap_merge.get(),
                    "Birleştirilecek mevcut dataset",
                )
                if not (merge / "data.yaml").is_file():
                    raise ValueError("Birleştirilecek dataset içinde data.yaml bulunamadı.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Dataset dönüştürülemedi", str(exc))
            return

        target_ids = {
            "player (0)": "0",
            "ball (1)": "1",
            "outside_person (2)": "2",
            "Yok say": "x",
        }
        mapping = ",".join(
            f"{index}:{target_ids[self.remap_target_vars[index].get()]}"
            for index in range(len(names))
        )
        if ":0" not in mapping or ":1" not in mapping:
            messagebox.showerror(
                "Eksik eşleme",
                "En az bir kaynak sınıf player ve en az bir kaynak sınıf ball olmalıdır.",
            )
            return

        script = self.project / "scripts/12_remap_merge_yolo_dataset.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--source-dataset",
            str(source),
            "--output-dir",
            str(output),
            "--mapping",
            mapping,
        ]
        if merge is not None:
            command.extend(["--merge-dataset", str(merge)])
        if self.remap_overwrite.get():
            command.append("--overwrite")

        def completed() -> None:
            summary_path = output / "conversion_summary.json"
            message = f"Hazır: {output}"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    image_total = sum(summary.get("image_count_by_split", {}).values())
                    boxes = summary.get("bbox_count_by_class", {})
                    message = (
                        f"Hazır: {image_total} görsel | "
                        f"player={boxes.get('0', 0)}, ball={boxes.get('1', 0)}, "
                        f"outside={boxes.get('2', 0)}"
                    )
                except (OSError, ValueError):
                    pass
            self.remap_result.set(message)
            self.train_dataset.set(str(output))
            self.last_output = output

        self._run_task(
            "YOLO dataset sınıfları dönüştürülüyor",
            command,
            cwd=self.project,
            output=output,
            on_success=completed,
            success_message=(
                "Dataset dönüştürüldü ve Eğitim sekmesindeki veri seti alanına "
                "otomatik seçildi."
            ),
        )

    def start_build_category_master(self) -> None:
        try:
            plan = self._require_file(self.category_plan.get(), "Dataset planı")
            output_text = self.category_output.get().strip()
            if not output_text:
                raise ValueError("Master çıktı klasörü boş olamaz.")
            output = Path(output_text).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if (output / "data.yaml").is_file() and not self.category_overwrite.get():
                self.train_dataset.set(str(output))
                player_v2_best = self.project / "models/trained/player_ball_v2-7/weights/best.pt"
                self.train_base.set(
                    str(player_v2_best if player_v2_best.is_file() else self.project / "models/base/yolo26m.pt")
                )
                self.train_name.set("player_ball_v8")
                self.category_result.set(f"Hazır ve Eğitim sekmesine seçildi: {output}")
                messagebox.showinfo(
                    "Master dataset hazır",
                    "Mevcut master dataset Eğitim sekmesine seçildi. Kaynak planı "
                    "değiştirdiyseniz yeniden oluştur seçeneğini açın.",
                )
                return
            if output.exists() and not self.category_overwrite.get():
                raise ValueError(
                    "Çıktı klasörü var fakat geçerli data.yaml yok. "
                    "Yeniden oluştur seçeneğini açın veya başka klasör seçin."
                )
        except (ValueError, OSError) as exc:
            messagebox.showerror("Master dataset oluşturulamadı", str(exc))
            return

        script = self.project / "scripts/13_build_category_dataset.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--plan",
            str(plan),
            "--output-dir",
            str(output),
            "--storage",
            "hardlink",
        ]
        if self.category_overwrite.get():
            command.append("--overwrite")

        def completed() -> None:
            summary_path = output / "build_summary.json"
            result = f"Hazır: {output}"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    splits = summary.get("images_by_split", {})
                    result = (
                        f"Hazır: train={splits.get('train', 0)}, "
                        f"val={splits.get('val', 0)}, test={splits.get('test', 0)}"
                    )
                except (OSError, ValueError):
                    pass
            self.category_result.set(result)
            self.train_dataset.set(str(output))
            player_v2_best = self.project / "models/trained/player_ball_v2-7/weights/best.pt"
            self.train_base.set(
                str(player_v2_best if player_v2_best.is_file() else self.project / "models/base/yolo26m.pt")
            )
            self.train_name.set("player_ball_v8")
            self.last_output = output

        self._run_task(
            "Categories kaynaklarından master dataset oluşturuluyor",
            command,
            cwd=self.project,
            output=output,
            on_success=completed,
            success_message=(
                "Master dataset hazırlandı ve Eğitim sekmesindeki veri seti "
                "alanına otomatik seçildi."
            ),
        )

    def start_training(self) -> None:
        try:
            dataset = self._require_dir_text(self.train_dataset.get(), "Veri seti")
            if not (dataset / "data.yaml").is_file():
                raise ValueError("Veri setinde data.yaml bulunamadı.")
            base = self._require_file(self.train_base.get(), "Başlangıç modeli")
            name = self.train_name.get().strip()
            if not name:
                raise ValueError("Eğitim adı boş olamaz.")
            epochs = self._number(self.train_epochs.get(), "Epoch", int, 1)
            imgsz = self._number(self.train_imgsz.get(), "Görüntü boyutu", int, 1)
            batch = self._number(self.train_batch.get(), "Batch", int)
            if batch == 0:
                raise ValueError("Batch sıfır olamaz.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Eğitim başlatılamadı", str(exc))
            return
        script = self.project / "scripts/06_train_commands.py"
        command = [
            str(self.python), "-u", str(script),
            "--dataset", str(dataset), "--base-model", str(base), "--name", name,
            "--imgsz", str(imgsz), "--epochs", str(epochs), "--batch", str(batch),
            "--device", self.train_device.get().strip() or "0", "--run",
        ]
        expected = self.project / "models/trained" / name / "weights/best.pt"

        def completed() -> None:
            self.refresh_dashboard()
            if expected.is_file():
                self.val_model.set(str(expected))
                self.track_model.set(str(expected))

        self._run_task(
            f"YOLO eğitimi: {name}",
            command,
            cwd=self.project,
            output=expected.parent.parent,
            on_success=completed,
        )

    def start_validation(self) -> None:
        try:
            model = self._require_file(self.val_model.get(), "Model")
            dataset = self._require_dir_text(self.train_dataset.get(), "Veri seti")
            if not (dataset / "data.yaml").is_file():
                raise ValueError("Veri setinde data.yaml bulunamadı.")
            imgsz = self._number(self.train_imgsz.get(), "Görüntü boyutu", int, 1)
            name = self.val_name.get().strip() or "model_validation"
        except (ValueError, OSError) as exc:
            messagebox.showerror("Değerlendirme başlatılamadı", str(exc))
            return
        script = self.project / "scripts/09_validate_model.py"
        command = [
            str(self.python), "-u", str(script),
            "--model", str(model), "--dataset", str(dataset),
            "--imgsz", str(imgsz), "--device", self.train_device.get().strip() or "0",
            "--name", name, "--split", self.val_split.get(),
        ]
        output = self.project / "experiments/outputs" / name
        self._run_task("Model değerlendiriliyor", command, cwd=self.project, output=output)

    def _sync_ball_load_profile(self, _event=None) -> None:
        profile = self.ball_load_profile.get()
        settings = BALL_LOAD_PROFILES.get(profile)
        if settings:
            self.ball_batch.set(settings["batch"])
            self.ball_workers.set(settings["workers"])
            self.ball_cpu_threads.set(settings["cpu_threads"])
        self.ball_load_status.set(
            f"{profile}: batch {self.ball_batch.get()}, "
            f"{self.ball_workers.get()} veri worker'ı, "
            f"{self.ball_cpu_threads.get()} CPU thread. "
            "Bu kesin bir yüzde sınırı değildir; anlık kaynak yoğunluğunu belirler."
        )

    def prepare_ball_dataset(self) -> None:
        try:
            train_labels = self._require_dir_text(
                self.ball_train_labels_source.get(), "Train etiket klasörü"
            )
            train_images = self._require_dir_text(
                self.ball_train_images_source.get(), "Train kare klasörü"
            )
            test_labels = self._require_dir_text(
                self.ball_test_labels_source.get(), "Test etiket klasörü"
            )
            test_images = self._require_dir_text(
                self.ball_test_images_source.get(), "Test kare klasörü"
            )
            output_text = self.ball_dataset.get().strip()
            if not output_text:
                raise ValueError("Ball dataset çıkışı boş olamaz.")
            output = Path(output_text).expanduser().resolve()
            allowed_parent = (self.project / "datasets/BallOnly").resolve()
            if allowed_parent not in output.parents:
                raise ValueError(
                    "Güvenli yeniden oluşturma için çıktı, proje içindeki "
                    "datasets/BallOnly klasöründe olmalıdır."
                )
        except (ValueError, OSError) as exc:
            messagebox.showerror("Ball dataset hazırlanamadı", str(exc))
            return

        script = self.project / "scripts/20_build_custom_ball_dataset.py"
        if not script.is_file():
            messagebox.showerror("Hazırlama scripti bulunamadı", str(script))
            return
        command = [
            str(self.python),
            "-u",
            str(script),
            "--train-images",
            str(train_images),
            "--train-labels",
            str(train_labels),
            "--test-images",
            str(test_images),
            "--test-labels",
            str(test_labels),
            "--output",
            str(output),
            "--val-ratio",
            "0.10",
            "--seed",
            "42",
            "--copy-mode",
            "hardlink",
            "--overwrite",
        ]
        self.ball_dataset_status.set(
            "502 train kaynağı ve 210 ayrı test karesinden dataset oluşturuluyor."
        )

        def completed() -> None:
            self.ball_dataset.set(str(output))
            try:
                summary = json.loads(
                    (output / "custom_dataset_summary.json").read_text(
                        encoding="utf-8-sig"
                    )
                )
                splits = summary.get("splits", {})
                train_images_count = int(splits.get("train", {}).get("images", 0))
                val_images = int(splits.get("val", {}).get("images", 0))
                test_images_count = int(splits.get("test", {}).get("images", 0))
                self.ball_dataset_status.set(
                    f"Hazır: {train_images_count:,} train, {val_images:,} validation, "
                    f"{test_images_count:,} test karesi."
                    .replace(",", ".")
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.ball_dataset_status.set(f"Hazır: {output}")
            self.ball_notebook.select(self.ball_training_tab)

        self._run_task(
            "BallDetectNew train + AaalBall test dataseti hazırlanıyor",
            command,
            cwd=self.project,
            output=output,
            on_success=completed,
            success_message=(
                "Train/validation ve test split'leri hazırlandı. "
                "Şimdi aynı ekrandaki "
                "'2. Eğitimi başlat' düğmesini kullanabilirsiniz."
            ),
        )

    def start_ball_training(self) -> None:
        try:
            dataset = self._require_dir_text(self.ball_dataset.get(), "Ball-only dataset")
            if not (dataset / "data.yaml").is_file():
                raise ValueError("Ball-only dataset içinde data.yaml bulunamadı.")
            base_model = self._require_file(
                self.ball_base_model.get(), "Başlangıç modeli"
            )
            name = self.ball_train_name.get().strip()
            if not name:
                raise ValueError("Top eğitim adı boş olamaz.")
            if not name.lower().startswith("ball"):
                raise ValueError("Top eğitim adı 'ball' ile başlamalıdır.")
            epochs = self._number(self.ball_epochs.get(), "Epoch", int, 1)
            imgsz = self._number(self.ball_imgsz.get(), "Görüntü boyutu", int, 1)
            batch = self._number(self.ball_batch.get(), "Batch", int)
            workers = self._number(self.ball_workers.get(), "Veri worker sayısı", int, 0)
            cpu_threads = self._number(
                self.ball_cpu_threads.get(), "CPU thread sınırı", int, 1
            )
            patience = self._number(self.ball_patience.get(), "Patience", int, 0)
            if batch == 0:
                raise ValueError("Batch sıfır olamaz.")
            expected_run = self.project / "models/trained" / name
            if expected_run.exists():
                raise ValueError(
                    f"Bu eğitim adı zaten kullanılmış: {expected_run}\n"
                    "Eski sonucu korumak için eğitim adını değiştirin "
                    "(örnek: ball_only_v8_2)."
                )
        except (ValueError, OSError) as exc:
            messagebox.showerror("Top eğitimi başlatılamadı", str(exc))
            return

        configured_profile = BALL_LOAD_PROFILES.get(self.ball_load_profile.get())
        if configured_profile and any(
            (
                str(batch) != configured_profile["batch"],
                str(workers) != configured_profile["workers"],
                str(cpu_threads) != configured_profile["cpu_threads"],
            )
        ):
            self.ball_load_profile.set("Özel")
        self._sync_ball_load_profile()

        script = self.project / "scripts/15_train_ball_model.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--dataset",
            str(dataset),
            "--base-model",
            str(base_model),
            "--name",
            name,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--device",
            self.ball_device.get().strip() or "0",
            "--patience",
            str(patience),
            "--workers",
            str(workers),
            "--cpu-threads",
            str(cpu_threads),
        ]

        def completed() -> None:
            newest = self._newest_ball_model()
            if newest:
                self.ball_model.set(str(newest))
            self.ball_notebook.select(self.ball_test_tab)

        self._run_task(
            f"Ball-only YOLO eğitimi: {name}",
            command,
            cwd=self.project,
            output=self.project / "models/trained",
            on_success=completed,
            success_message=(
                "Top modeli tamamlandı. En yeni best.pt, Yalnız Top Testi "
                "alanına otomatik seçildi."
            ),
        )

    def start_ball_validation(self) -> None:
        try:
            model = self._require_file(self.ball_model.get(), "Top modeli")
            dataset = self._require_dir_text(self.ball_dataset.get(), "Ball-only dataset")
            imgsz = self._number(self.ball_imgsz.get(), "Görüntü boyutu", int, 1)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Top modeli değerlendirilemedi", str(exc))
            return
        name = f"ball_validation_{model.parent.parent.name}"
        script = self.project / "scripts/09_validate_model.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--imgsz",
            str(imgsz),
            "--device",
            self.ball_device.get().strip() or "0",
            "--name",
            name,
            "--split",
            "test",
        ]
        self._run_task(
            "Top modeli test splitinde değerlendiriliyor",
            command,
            cwd=self.project,
            output=self.project / "experiments/outputs" / name,
        )

    def _sync_ball_test_mode(self) -> None:
        mode = self.ball_test_mode.get()
        if mode == "30 saniyelik hızlı test":
            self.ball_duration.set("30")
            self.ball_sample.set("0.1")
        elif mode == "Tüm video":
            self.ball_duration.set("0")
            self.ball_sample.set("0.1")

    def start_ball_test(self) -> None:
        try:
            video = self._require_file(self.ball_video.get(), "Test videosu")
            model = self._require_file(self.ball_model.get(), "Top modeli")
            output = self._require_dir_text(
                self.ball_output.get(), "Top testi çıktı klasörü", True
            )
            duration = self._number(self.ball_duration.get(), "Süre", float, 0)
            sample = self._number(self.ball_sample.get(), "Örnekleme", float, 0.001)
            confidence = self._number(self.ball_conf.get(), "Confidence", float, 0.001)
            imgsz = self._number(
                self.ball_test_imgsz.get(), "Inference boyutu", int, 1
            )
            tile_size = self._number(self.ball_tile_size.get(), "Tile boyutu", int, 160)
            if confidence > 1:
                raise ValueError("Confidence 1'den büyük olamaz.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Yalnız top testi başlatılamadı", str(exc))
            return

        mode = self.ball_test_mode.get()
        if mode == "30 saniyelik hızlı test":
            duration = 30
        elif mode == "Tüm video":
            duration = 0
        script = self.project / "app/process_ball_video.py"
        command = [
            str(self.python),
            "-u",
            str(script),
            "--video",
            str(video),
            "--model",
            str(model),
            "--out",
            str(output),
            "--duration",
            str(duration),
            "--sample-sec",
            str(sample),
            "--conf",
            str(confidence),
            "--imgsz",
            str(imgsz),
            "--device",
            self.ball_device.get().strip() or "0",
            "--tile-size",
            str(tile_size),
        ]
        if self.ball_tiled.get():
            command.append("--tile")
        self._run_task(
            "Videoda yalnız top aranıyor",
            command,
            cwd=self.project,
            output=output,
            on_success=lambda: self._open_ball_review(output, video),
            success_message=(
                "Top testi tamamlandı. Sonuç Top Modeli > Frame İncele "
                "bölümüne yüklendi."
            ),
        )

    def _open_ball_review(self, output: Path, video: Path) -> None:
        self.ball_player.load_session(output, video)
        self.ball_notebook.select(self.ball_review_tab)

    def _sync_tracking_mode(self) -> None:
        mode = self.track_mode.get()
        if mode == "30 saniyelik hızlı test":
            self.track_duration.set("30")
            self.track_sample.set("0.2")
        elif mode == "Tüm video":
            self.track_duration.set("0")
            self.track_sample.set("0.1")

    def start_tracking(self) -> None:
        try:
            video = self._require_file(self.track_video.get(), "Test videosu")
            model = self._require_file(self.track_model.get(), "Model")
            output = self._require_dir_text(self.track_output.get(), "Çıktı klasörü", True)
            sample = self._number(self.track_sample.get(), "Örnekleme", float, 0.001)
            conf = self._number(self.track_conf.get(), "Confidence", float, 0.001)
            if conf > 1:
                raise ValueError("Confidence 1'den büyük olamaz.")
            players = self._number(self.track_players.get(), "Maksimum oyuncu", int, 1)
            duration = self._number(self.track_duration.get(), "Süre", float, 0)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Tracking başlatılamadı", str(exc))
            return

        mode = self.track_mode.get()
        if mode == "30 saniyelik hızlı test":
            duration = 30
        elif mode == "Tüm video":
            duration = 0
        calibration_mode = self.track_calibration_mode.get()
        mapper = "homography" if calibration_mode == "four_points" else "poly2"
        script = self.project / "app/process_video_to_csv_v8.py"
        command = [
            str(self.python), "-u", str(script),
            "--video", str(video), "--out", str(output), "--model", str(model),
            "--sample-sec", str(sample), "--conf", str(conf),
            "--yolo-classes", "0,1", "--max-players", str(players),
            "--duration", str(duration), "--calibration-mode", calibration_mode,
            "--field-mapper", mapper,
        ]
        calibration_file = self.track_calibration_file.get().strip()
        if calibration_file:
            try:
                calibration = self._require_file(calibration_file, "Calibration dosyası")
            except ValueError as exc:
                messagebox.showerror("Tracking başlatılamadı", str(exc))
                return
            command.extend(["--calibration", str(calibration)])
        if self.track_force_calibration.get():
            command.append("--force-calibration")
        if self.track_sahi.get():
            command.extend(["--sahi", "--sahi-regions", "ball", "--sahi-ball-always"])
        if self.track_manual_players.get():
            command.append("--manual-initial-players")
        self._run_task(
            "Video tracking yapılıyor",
            command,
            cwd=self.project,
            output=output,
            on_success=lambda: self._open_tracking_review(output, video),
            success_message=(
                "Tracking tamamlandı. Sonuç Frame İncele bölümüne yüklendi."
            ),
        )

    def _open_tracking_review(self, output: Path, video: Path) -> None:
        self.tracking_player.load_session(output, video)
        self.tracking_notebook.select(self.tracking_review_tab)

    def open_viewer(self) -> None:
        viewer = self.project / "app/saha_csv_viewer.html"
        if not viewer.is_file():
            messagebox.showerror("Viewer bulunamadı", str(viewer))
            return
        output_text = self.track_output.get().strip()
        if output_text:
            output_dir = Path(output_text).expanduser().resolve()
            full_csv = output_dir / "tracking_all.csv"
            viewer_csv = output_dir / "tracking_viewer.csv"
            needs_build = full_csv.is_file() and (
                not viewer_csv.is_file()
                or viewer_csv.stat().st_mtime < full_csv.stat().st_mtime
            )
            if needs_build:
                script = self.project / "scripts/10_make_viewer_csv.py"
                command = [
                    str(self.python),
                    "-u",
                    str(script),
                    "--input",
                    str(full_csv),
                    "--output",
                    str(viewer_csv),
                ]
                self._run_task(
                    "Viewer dosyası hazırlanıyor",
                    command,
                    cwd=self.project,
                    output=output_dir,
                    on_success=lambda: os.startfile(str(viewer)),
                )
                return
        os.startfile(str(viewer))

    def start_health_check(self) -> None:
        code = (
            "import importlib.util as u, sys\n"
            "names=['cv2','torch','ultralytics','pandas','numpy','openpyxl',"
            "'tqdm','requests','playwright','imageio_ffmpeg']\n"
            "print('Python:', sys.version.replace('\\\\n',' '))\n"
            "for n in names: print(f'{n}:', 'OK' if u.find_spec(n) else 'EKSİK')\n"
            "import torch\n"
            "print('PyTorch:', torch.__version__)\n"
            "print('CUDA kullanılabilir:', torch.cuda.is_available())\n"
            "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')\n"
        )
        self._run_task("Sistem kontrol ediliyor", [str(self.python), "-u", "-c", code])

    def start_install_requirements(self) -> None:
        requirements = self.project / "studio_requirements.txt"
        command = [
            str(self.python), "-u", "-m", "pip", "install", "-r", str(requirements)
        ]
        self._run_task("Python paketleri kuruluyor", command, cwd=self.project)

    def start_install_browser(self) -> None:
        command = [str(self.python), "-u", "-m", "playwright", "install", "chromium"]
        self._run_task("Playwright Chromium kuruluyor", command, cwd=self.project)

    def start_setup_project(self) -> None:
        script = self.project / "scripts/00_setup_project.py"
        command = [str(self.python), "-u", str(script), "--project-root", str(self.project)]
        self._run_task(
            "Proje klasörleri hazırlanıyor",
            command,
            cwd=self.project,
            output=self.project,
            on_success=self.refresh_dashboard,
        )

    def _run_task(
        self,
        title: str,
        command: list[str],
        *,
        cwd: Path | None = None,
        output: Path | None = None,
        on_success: Callable[[], None] | None = None,
        success_message: str = "",
    ) -> None:
        if self.process is not None or (self.worker is not None and self.worker.is_alive()):
            messagebox.showwarning(
                "İşlem devam ediyor",
                f"Önce devam eden işlemi bitirin veya durdurun:\n{self.active_task}",
            )
            return
        self.active_task = title
        self.last_output = output
        self._success_callback = on_success
        self._success_message = success_message
        self.status_var.set(title)
        self.cancel_button.configure(state="normal")
        self.progress.start(12)
        self._append_log("")
        self._append_log(f"--- {title} ---")
        self._append_log(subprocess.list2cmdline(command))
        self.worker = threading.Thread(
            target=self._process_worker,
            args=(command, cwd or self.workspace),
            daemon=True,
        )
        self.worker.start()

    def _process_worker(self, command: list[str], cwd: Path) -> None:
        try:
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                env=environment,
            )
            assert self.process.stdout is not None
            for raw_line in self.process.stdout:
                line = ANSI_ESCAPE_RE.sub(
                    "", repair_mojibake(raw_line.rstrip("\r\n"))
                )
                if line:
                    self.root.after(0, self._append_log, line)
            return_code = self.process.wait()
            self.root.after(0, self._task_finished, return_code)
        except Exception as exc:
            self.root.after(0, self._task_failed, str(exc))

    def _task_finished(self, return_code: int) -> None:
        title = self.active_task
        self.process = None
        self.worker = None
        self.progress.stop()
        self.cancel_button.configure(state="disabled")
        callback = self._success_callback
        self._success_callback = None
        success_message = self._success_message
        self._success_message = ""
        if return_code == 0:
            self.status_var.set(f"Tamamlandı: {title}")
            self._append_log(f"--- Tamamlandı: {title} ---")
            if callback:
                callback()
            self.refresh_dashboard()
            if success_message:
                messagebox.showinfo("Tamamlandı", success_message)
        else:
            self.status_var.set(f"Hata kodu {return_code}: {title}")
            self._append_log(f"--- İşlem hata kodu {return_code} ile durdu ---")
            if return_code not in (130, 3221225786):
                messagebox.showerror(
                    "İşlem başarısız",
                    f"{title}\n\nHata kodu: {return_code}\nAyrıntılar işlem günlüğünde.",
                )
        self.active_task = ""

    def _task_failed(self, error: str) -> None:
        self.process = None
        self.worker = None
        self.progress.stop()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("İşlem başlatılamadı")
        self._append_log(f"HATA: {error}")
        self.active_task = ""
        messagebox.showerror("İşlem başlatılamadı", error)

    def _cancel_task(self) -> None:
        process = self.process
        if process is None:
            return
        if not messagebox.askyesno("İşlemi durdur", f"{self.active_task} durdurulsun mu?"):
            return
        self.status_var.set("İşlem durduruluyor...")
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(
                "İşlem devam ediyor",
                "Devam eden işlem durdurulup uygulama kapatılsın mı?",
            ):
                return
            process = self.process
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW,
                    check=False,
                )
            except OSError:
                pass
        if hasattr(self, "tracking_player"):
            self.tracking_player.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SahaAIStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
