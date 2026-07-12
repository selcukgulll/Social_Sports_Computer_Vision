#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saha AI Studio için bellek dostu tracking video inceleyicisi."""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import queue
import sqlite3
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Iterable, Iterator

import cv2
from PIL import Image, ImageTk


VIDEO_TYPES = [
    ("Video dosyaları", "*.mp4 *.mov *.avi *.mkv *.webm"),
    ("Tüm dosyalar", "*.*"),
]


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolean(value: object) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "evet"})


def _natural_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


class TrackingIndex:
    """tracking_all.csv dosyasını düşük bellekle sorgulanabilir hale getirir."""

    SCHEMA_VERSION = "2"
    INSERT_SQL = """
        INSERT INTO observations (
            frame_idx, sample_idx, time_sec, stable_id, raw_id, entity, variant,
            team, det_conf, position_conf, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            image_x, image_y, source, match_type, is_interpolated,
            missing_gap_sec, det_source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.csv_path = self.output_dir / "tracking_all.csv"
        self.summary_path = self.output_dir / "summary.json"
        self.db_path = self.output_dir / ".tracking_review.sqlite"

    def ensure(self) -> None:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"tracking_all.csv bulunamadı:\n{self.csv_path}")
        signature = self._source_signature()
        if self._cache_is_current(signature):
            return
        self._build(signature)

    def read_summary(self) -> dict:
        if not self.summary_path.is_file():
            return {}
        try:
            with self.summary_path.open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def analyzed_frames(self) -> list[int]:
        with self._connect() as connection:
            return [
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT frame_idx FROM observations ORDER BY frame_idx"
                )
            ]

    def stable_ids(self) -> list[str]:
        with self._connect() as connection:
            values = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT stable_id
                    FROM observations
                    WHERE stable_id <> '' AND entity = 'player'
                    """
                )
            ]
        return sorted(values, key=_natural_key)

    def rows_at(self, frame_idx: int, variant: str) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT *
                FROM observations
                WHERE frame_idx = ? AND variant = ?
                ORDER BY CASE WHEN entity = 'ball' THEN 1 ELSE 0 END, stable_id
                """,
                (frame_idx, variant),
            ).fetchall()
        return [dict(row) for row in rows]

    def trail(
        self,
        frame_idx: int,
        stable_id: str,
        entity: str,
        variant: str,
        limit: int = 12,
    ) -> list[tuple[float, float]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT image_x, image_y
                FROM observations
                WHERE frame_idx <= ? AND stable_id = ? AND entity = ? AND variant = ?
                  AND image_x IS NOT NULL AND image_y IS NOT NULL
                ORDER BY frame_idx DESC
                LIMIT ?
                """,
                (frame_idx, stable_id, entity, variant, limit),
            ).fetchall()
        return [(float(x), float(y)) for x, y in reversed(rows)]

    def issue_frames(self, expected_players: int) -> list[int]:
        """Kesin hata değil, insan incelemesini hızlandıran aday kareler."""
        expected_floor = max(1, expected_players - 2)
        with self._connect() as connection:
            min_frame = connection.execute(
                "SELECT COALESCE(MIN(frame_idx), 0) FROM observations"
            ).fetchone()[0]
            rows = connection.execute(
                """
                WITH raw_counts AS (
                    SELECT frame_idx,
                           SUM(CASE WHEN entity = 'player' THEN 1 ELSE 0 END) players
                    FROM observations
                    WHERE variant = 'raw'
                    GROUP BY frame_idx
                ),
                event_frames AS (
                    SELECT DISTINCT frame_idx
                    FROM observations
                    WHERE
                        (variant = 'interpolated' AND is_interpolated = 1)
                        OR missing_gap_sec > 0.25
                        OR (match_type = 'new' AND frame_idx > ?)
                        OR (entity = 'player' AND det_conf IS NOT NULL AND det_conf < 0.20)
                )
                SELECT frame_idx FROM event_frames
                UNION
                SELECT frame_idx FROM raw_counts WHERE players < ?
                ORDER BY frame_idx
                """,
                (min_frame, expected_floor),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def frame_diagnostics(self, frame_idx: int, expected_players: int) -> dict:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN variant='raw' AND entity='player' THEN 1 ELSE 0 END)
                        AS raw_players,
                    SUM(CASE WHEN variant='raw' AND entity='ball' THEN 1 ELSE 0 END)
                        AS raw_balls,
                    SUM(CASE WHEN variant='interpolated' AND is_interpolated=1
                        THEN 1 ELSE 0 END) AS filled,
                    GROUP_CONCAT(CASE WHEN variant='interpolated'
                        AND is_interpolated=1 AND entity='player'
                        THEN stable_id END) AS filled_ids,
                    SUM(CASE WHEN variant='raw' AND match_type='new' THEN 1 ELSE 0 END)
                        AS new_tracks,
                    GROUP_CONCAT(CASE WHEN variant='raw' AND match_type='new'
                        THEN stable_id END) AS new_ids,
                    SUM(CASE WHEN variant='raw' AND entity='player'
                        AND det_conf IS NOT NULL AND det_conf < 0.20 THEN 1 ELSE 0 END)
                        AS low_conf,
                    GROUP_CONCAT(CASE WHEN variant='raw' AND entity='player'
                        AND det_conf IS NOT NULL AND det_conf < 0.20
                        THEN stable_id END) AS low_conf_ids,
                    MAX(COALESCE(missing_gap_sec, 0)) AS max_gap
                FROM observations
                WHERE frame_idx = ?
                """,
                (frame_idx,),
            ).fetchone()
        values = dict(row) if row else {}
        for key in ("raw_players", "raw_balls", "filled", "new_tracks", "low_conf"):
            values[key] = int(values.get(key) or 0)
        values["max_gap"] = float(values.get("max_gap") or 0.0)
        for key in ("filled_ids", "new_ids", "low_conf_ids"):
            values[key] = str(values.get(key) or "")
        values["missing_players"] = max(
            0, expected_players - int(values["raw_players"])
        )
        return values

    def _source_signature(self) -> str:
        stat = self.csv_path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _cache_is_current(self, signature: str) -> bool:
        if not self.db_path.is_file():
            return False
        try:
            with self._connect() as connection:
                values = dict(connection.execute("SELECT key, value FROM metadata"))
            return (
                values.get("schema_version") == self.SCHEMA_VERSION
                and values.get("source_signature") == signature
            )
        except (OSError, sqlite3.DatabaseError):
            return False

    def _build(self, signature: str) -> None:
        temporary = self.db_path.with_name(
            f"{self.db_path.name}.{os.getpid()}.{threading.get_ident()}.building"
        )
        connection: sqlite3.Connection | None = None
        try:
            temporary.unlink(missing_ok=True)
            connection = sqlite3.connect(temporary)
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE observations (
                    frame_idx INTEGER NOT NULL,
                    sample_idx INTEGER NOT NULL,
                    time_sec REAL NOT NULL,
                    stable_id TEXT NOT NULL,
                    raw_id TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    team TEXT NOT NULL,
                    det_conf REAL,
                    position_conf REAL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    image_x REAL,
                    image_y REAL,
                    source TEXT NOT NULL,
                    match_type TEXT NOT NULL,
                    is_interpolated INTEGER NOT NULL,
                    missing_gap_sec REAL NOT NULL,
                    det_source TEXT NOT NULL
                );
                """
            )
            batch: list[tuple] = []
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"frame_idx", "stable_id", "variant", "entity"}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(
                        "tracking_all.csv beklenen kolonları içermiyor: "
                        + ", ".join(sorted(missing))
                    )
                for item in reader:
                    batch.append(self._compact_row(item))
                    if len(batch) >= 5000:
                        connection.executemany(self.INSERT_SQL, batch)
                        batch.clear()
                if batch:
                    connection.executemany(self.INSERT_SQL, batch)
            connection.executescript(
                """
                CREATE INDEX idx_observations_frame_variant
                    ON observations(frame_idx, variant);
                CREATE INDEX idx_observations_id_frame
                    ON observations(stable_id, frame_idx, variant);
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", self.SCHEMA_VERSION),
                    ("source_signature", signature),
                ),
            )
            connection.commit()
            connection.close()
            connection = None
            self._install_built_cache(temporary)
        except Exception:
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            temporary.unlink(missing_ok=True)
            raise

    def _install_built_cache(self, temporary: Path) -> None:
        """Yeni index dosyasını Windows dosya kilitlerine dayanıklı şekilde kur."""
        try:
            os.replace(temporary, self.db_path)
            return
        except PermissionError:
            pass
        except OSError as exc:
            if getattr(exc, "winerror", None) != 5:
                raise

        session_path = self.db_path.with_name(
            f"{self.db_path.name}.{os.getpid()}.{threading.get_ident()}.session"
        )
        session_path.unlink(missing_ok=True)
        os.replace(temporary, session_path)
        self.db_path = session_path

    @staticmethod
    def _compact_row(item: dict[str, str]) -> tuple:
        return (
            _integer(item.get("frame_idx")),
            _integer(item.get("sample_idx")),
            _float(item.get("time_sec")),
            str(item.get("stable_id") or ""),
            str(item.get("raw_id") or ""),
            str(item.get("entity") or ""),
            str(item.get("variant") or "raw"),
            str(item.get("team") or ""),
            _optional_float(item.get("det_conf")),
            _optional_float(item.get("position_conf")),
            _optional_float(item.get("bbox_x1")),
            _optional_float(item.get("bbox_y1")),
            _optional_float(item.get("bbox_x2")),
            _optional_float(item.get("bbox_y2")),
            _optional_float(item.get("image_x")),
            _optional_float(item.get("image_y")),
            str(item.get("source") or ""),
            str(item.get("match_type") or ""),
            _boolean(item.get("is_interpolated")),
            _float(item.get("missing_gap_sec")),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=20)
        try:
            yield connection
        finally:
            connection.close()


class TrackingVideoPlayer(ttk.Frame):
    """Video ve tracking sonuçlarını aynı kare üzerinde gösterir."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        output_getter: Callable[[], str] | None = None,
        video_getter: Callable[[], str] | None = None,
        on_output_changed: Callable[[str], None] | None = None,
        colors: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.output_getter = output_getter
        self.video_getter = video_getter
        self.on_output_changed = on_output_changed
        self.colors = colors or {}

        self.index: TrackingIndex | None = None
        self.summary: dict = {}
        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.analyzed_frames: list[int] = []
        self.issue_frame_values: list[int] = []
        self.ids: list[str] = []
        self.fps = 30.0
        self.video_frame_count = 0
        self.review_max_frame = 0
        self.current_frame = 0
        self.current_overlay_frame: int | None = None
        self.playing = False
        self.after_id: str | None = None
        self._load_generation = 0
        self._load_pending: int | None = None
        self._load_queue: queue.Queue[tuple] = queue.Queue()
        self._photo: ImageTk.PhotoImage | None = None
        self._last_bgr = None
        self._last_render_key: tuple | None = None
        self._frame_cache: OrderedDict[int, object] = OrderedDict()
        self._rows_cache: OrderedDict[tuple[int, str], list[dict]] = OrderedDict()
        self._trail_cache: OrderedDict[
            tuple[int, str, str, str], list[tuple[float, float]]
        ] = OrderedDict()

        self.output_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Bir tracking çıktı klasörü yükleyin."
        )
        self.position_var = tk.StringVar(value="Kare - / -")
        self.overlay_var = tk.StringVar(value="Analiz karesi: -")
        self.variant_var = tk.StringVar(value="raw")
        self.hold_overlay_var = tk.BooleanVar(value=True)
        self.show_trails_var = tk.BooleanVar(value=True)
        self.selected_id_var = tk.StringVar(value="Tümü")
        self.speed_var = tk.StringVar(value="1.0x")
        self.diagnostic_var = tk.StringVar(
            value="Kopma adayları burada açıklanacak."
        )
        self.slider_var = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(1, weight=1)
        ttk.Button(
            toolbar,
            text="Çıktıyı yükle",
            command=self.choose_output,
        ).grid(row=0, column=0, padx=(0, 7))
        ttk.Entry(toolbar, textvariable=self.output_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            toolbar,
            text="Studio seçimlerini yükle",
            command=self.load_from_studio,
        ).grid(row=0, column=2, padx=7)
        ttk.Button(
            toolbar,
            text="Videoyu değiştir",
            command=self.choose_video,
        ).grid(row=0, column=3)
        ttk.Label(
            toolbar,
            textvariable=self.video_var,
            foreground=self.colors.get("muted", "#69758A"),
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew")

        video_panel = ttk.Frame(paned)
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            video_panel,
            bg="#070B12",
            highlightthickness=1,
            highlightbackground=self.colors.get("border", "#DDE4EF"),
            width=720,
            height=410,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render(force=True))
        paned.add(video_panel, weight=4)

        side = ttk.Frame(paned, padding=(10, 0, 0, 0))
        side.rowconfigure(4, weight=1)
        side.columnconfigure(0, weight=1)
        ttk.Label(
            side,
            text="Kare tanısı",
            font=("Segoe UI Semibold", 11),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            side,
            textvariable=self.diagnostic_var,
            wraplength=315,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(4, 10))

        filters = ttk.Frame(side)
        filters.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(filters, text="Görünüm").pack(side="left")
        variant = ttk.Combobox(
            filters,
            textvariable=self.variant_var,
            values=("raw", "interpolated"),
            state="readonly",
            width=13,
        )
        variant.pack(side="left", padx=(6, 12))
        variant.bind("<<ComboboxSelected>>", lambda _event: self._render(force=True))
        ttk.Label(filters, text="ID").pack(side="left")
        self.id_combo = ttk.Combobox(
            filters,
            textvariable=self.selected_id_var,
            values=("Tümü",),
            state="readonly",
            width=10,
        )
        self.id_combo.pack(side="left", padx=(6, 0))
        self.id_combo.bind("<<ComboboxSelected>>", lambda _event: self._render(force=True))

        toggles = ttk.Frame(side)
        toggles.grid(row=3, column=0, sticky="w", pady=(0, 8))
        ttk.Checkbutton(
            toggles,
            text="Ara karede son kutuyu tut",
            variable=self.hold_overlay_var,
            command=lambda: self._render(force=True),
        ).pack(side="left")
        ttk.Checkbutton(
            toggles,
            text="İzleri göster",
            variable=self.show_trails_var,
            command=lambda: self._render(force=True),
        ).pack(side="left", padx=(10, 0))

        columns = ("id", "varlık", "takım", "güven", "eşleşme")
        self.table = ttk.Treeview(
            side,
            columns=columns,
            show="headings",
            height=12,
        )
        headings = {
            "id": "ID",
            "varlık": "Varlık",
            "takım": "Takım",
            "güven": "Güven",
            "eşleşme": "Kaynak",
        }
        widths = {"id": 45, "varlık": 60, "takım": 65, "güven": 55, "eşleşme": 100}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="center")
        self.table.grid(row=4, column=0, sticky="nsew")
        self.table.bind("<<TreeviewSelect>>", self._select_table_id)
        paned.add(side, weight=2)

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        controls.columnconfigure(8, weight=1)
        ttk.Button(controls, text="⏮ Analiz", command=self.previous_analyzed).grid(
            row=0, column=0, padx=(0, 4)
        )
        ttk.Button(controls, text="◀ Kare", command=lambda: self.step(-1)).grid(
            row=0, column=1, padx=4
        )
        self.play_button = ttk.Button(controls, text="▶ Oynat", command=self.toggle_play)
        self.play_button.grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Kare ▶", command=lambda: self.step(1)).grid(
            row=0, column=3, padx=4
        )
        ttk.Button(controls, text="Analiz ⏭", command=self.next_analyzed).grid(
            row=0, column=4, padx=4
        )
        ttk.Button(
            controls,
            text="Sonraki kopma adayı",
            command=self.next_issue,
        ).grid(row=0, column=5, padx=(12, 4))
        ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=("0.25x", "0.5x", "1.0x", "2.0x"),
            state="readonly",
            width=7,
        ).grid(row=0, column=6, padx=4)
        ttk.Label(controls, textvariable=self.position_var).grid(
            row=0, column=8, sticky="e"
        )

        self.slider = ttk.Scale(
            self,
            from_=0,
            to=1,
            variable=self.slider_var,
            command=self._slider_changed,
        )
        self.slider.grid(row=3, column=0, sticky="ew", pady=(7, 0))

        footer = ttk.Frame(self)
        footer.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(footer, textvariable=self.overlay_var).grid(
            row=0, column=1, sticky="e"
        )

    def choose_output(self) -> None:
        initial = self.output_var.get().strip()
        selected = filedialog.askdirectory(
            title="tracking_all.csv içeren çıktı klasörünü seç",
            initialdir=initial if Path(initial).is_dir() else None,
        )
        if selected:
            self.load_session(Path(selected))

    def choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Tracking ile eşleşen videoyu seç",
            filetypes=VIDEO_TYPES,
        )
        if selected:
            self.video_var.set(selected)
            output = self.output_var.get().strip()
            if output:
                self.load_session(Path(output), Path(selected))

    def load_from_studio(self) -> None:
        output_text = self.output_getter().strip() if self.output_getter else ""
        video_text = self.video_getter().strip() if self.video_getter else ""
        if not output_text:
            messagebox.showwarning(
                "Çıktı seçilmedi",
                "Önce Tracking Çalıştır bölümünde çıktı klasörünü seçin.",
            )
            return
        self.load_session(
            Path(output_text),
            Path(video_text) if video_text else None,
        )

    def load_session(
        self,
        output_dir: Path,
        video_path: Path | None = None,
    ) -> None:
        self.pause()
        output_dir = output_dir.expanduser().resolve()
        self.output_var.set(str(output_dir))
        self.status_var.set("Tracking indeksi hazırlanıyor…")
        self._load_generation += 1
        generation = self._load_generation
        self._load_pending = generation
        fallback_video: Path | None = None
        if self.video_getter:
            fallback_text = self.video_getter().strip()
            if fallback_text:
                fallback_video = Path(fallback_text)

        def worker() -> None:
            try:
                index = TrackingIndex(output_dir)
                index.ensure()
                summary = index.read_summary()
                frames = index.analyzed_frames()
                ids = index.stable_ids()
                if not frames:
                    raise ValueError("CSV içinde görüntülenecek tracking satırı yok.")
                chosen_video = video_path
                if chosen_video is None:
                    summary_video = str(summary.get("video") or "").strip()
                    if summary_video:
                        chosen_video = Path(summary_video)
                if chosen_video is None:
                    chosen_video = fallback_video
                if chosen_video is None or not chosen_video.expanduser().is_file():
                    raise FileNotFoundError(
                        "Tracking videosu bulunamadı. “Videoyu değiştir” ile seçin."
                    )
                expected = max(1, _integer(summary.get("max_players"), 14))
                issues = index.issue_frames(expected)
                self._load_queue.put(
                    (
                        "success",
                        generation,
                        index,
                        summary,
                        frames,
                        ids,
                        chosen_video.expanduser().resolve(),
                        issues,
                    )
                )
            except Exception as exc:
                self._load_queue.put(("error", generation, str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(30, self._poll_load_queue)

    def _poll_load_queue(self) -> None:
        try:
            while True:
                result = self._load_queue.get_nowait()
                kind, generation, *payload = result
                if kind == "success":
                    self._finish_load(
                        generation,
                        *payload,
                    )
                else:
                    self._load_failed(generation, str(payload[0]))
                if generation == self._load_pending:
                    self._load_pending = None
        except queue.Empty:
            pass
        if self._load_pending is not None:
            self.after(50, self._poll_load_queue)

    def _finish_load(
        self,
        generation: int,
        index: TrackingIndex,
        summary: dict,
        frames: list[int],
        ids: list[str],
        video_path: Path,
        issues: list[int],
    ) -> None:
        if generation != self._load_generation:
            return
        self._release_capture()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            self._load_failed(generation, f"Video açılamadı:\n{video_path}")
            return

        self.index = index
        self.summary = summary
        self.analyzed_frames = frames
        self.issue_frame_values = issues
        self.ids = ids
        self.capture = capture
        self.video_path = video_path
        self.fps = capture.get(cv2.CAP_PROP_FPS) or _float(summary.get("fps"), 30.0)
        if self.fps <= 0:
            self.fps = 30.0
        self.video_frame_count = max(
            1, _integer(capture.get(cv2.CAP_PROP_FRAME_COUNT), max(frames) + 1)
        )
        self.review_max_frame = min(self.video_frame_count - 1, max(frames))
        self.slider.configure(to=max(1, self.review_max_frame))
        self.id_combo.configure(values=("Tümü", *ids))
        self.selected_id_var.set("Tümü")
        self.video_var.set(str(video_path))
        self._frame_cache.clear()
        self._rows_cache.clear()
        self._trail_cache.clear()
        self.current_frame = min(frames)
        self.slider_var.set(self.current_frame)
        self.status_var.set(
            f"{len(frames)} analiz karesi · {len(ids)} track ID · "
            f"{len(issues)} inceleme adayı"
        )
        if self.on_output_changed:
            self.on_output_changed(str(index.output_dir))
        self._render(force=True)

    def _load_failed(self, generation: int, error: str) -> None:
        if generation != self._load_generation:
            return
        self.status_var.set("Yükleme başarısız.")
        messagebox.showerror("Frame inceleyici", error)

    def _slider_changed(self, value: str) -> None:
        if not self.capture:
            return
        frame_idx = max(0, min(self.review_max_frame, int(round(float(value)))))
        if frame_idx == self.current_frame:
            return
        self.current_frame = frame_idx
        self._render()

    def step(self, amount: int) -> None:
        self.pause()
        self.seek(self.current_frame + amount)

    def seek(self, frame_idx: int) -> None:
        if not self.capture:
            return
        self.current_frame = max(0, min(self.review_max_frame, int(frame_idx)))
        self.slider_var.set(self.current_frame)
        self._render()

    def previous_analyzed(self) -> None:
        self.pause()
        position = bisect.bisect_left(self.analyzed_frames, self.current_frame) - 1
        if position >= 0:
            self.seek(self.analyzed_frames[position])

    def next_analyzed(self) -> None:
        self.pause()
        position = bisect.bisect_right(self.analyzed_frames, self.current_frame)
        if position < len(self.analyzed_frames):
            self.seek(self.analyzed_frames[position])

    def next_issue(self) -> None:
        self.pause()
        if not self.issue_frame_values:
            messagebox.showinfo(
                "Kopma adayı",
                "Bu çıktı için otomatik bir kopma adayı bulunmadı.",
            )
            return
        position = bisect.bisect_right(self.issue_frame_values, self.current_frame)
        if position >= len(self.issue_frame_values):
            position = 0
        self.seek(self.issue_frame_values[position])

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
            return
        if not self.capture:
            return
        self.playing = True
        self.play_button.configure(text="⏸ Duraklat")
        self._play_tick()

    def pause(self) -> None:
        self.playing = False
        self.play_button.configure(text="▶ Oynat")
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def _play_tick(self) -> None:
        if not self.playing:
            return
        if self.current_frame >= self.review_max_frame:
            self.pause()
            return
        self.current_frame += 1
        self.slider_var.set(self.current_frame)
        self._render()
        speed = _float(self.speed_var.get().rstrip("x"), 1.0)
        delay = max(1, int(1000.0 / (self.fps * max(0.1, speed))))
        self.after_id = self.after(delay, self._play_tick)

    def _overlay_frame(self) -> int | None:
        if not self.analyzed_frames:
            return None
        position = bisect.bisect_left(self.analyzed_frames, self.current_frame)
        if (
            position < len(self.analyzed_frames)
            and self.analyzed_frames[position] == self.current_frame
        ):
            return self.current_frame
        if not self.hold_overlay_var.get() or position == 0:
            return None
        return self.analyzed_frames[position - 1]

    def _render(self, force: bool = False) -> None:
        if not self.capture:
            self._draw_empty()
            return
        overlay_frame = self._overlay_frame()
        key = (
            self.current_frame,
            overlay_frame,
            self.variant_var.get(),
            self.selected_id_var.get(),
            self.show_trails_var.get(),
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )
        if not force and key == self._last_render_key:
            return
        self._last_render_key = key

        frame = self._read_frame(self.current_frame)
        if frame is None:
            self.status_var.set(f"Video karesi okunamadı: {self.current_frame}")
            return
        image = frame.copy()
        rows: list[dict] = []
        if overlay_frame is not None and self.index:
            rows = self._rows_at(overlay_frame, self.variant_var.get())
            self._draw_overlays(image, rows, overlay_frame)
        self._display_image(image)
        self._update_table(rows)
        self._update_diagnostics(overlay_frame)

        seconds = self.current_frame / self.fps
        duration = self.review_max_frame / self.fps
        self.position_var.set(
            f"Kare {self.current_frame}/{self.review_max_frame} · "
            f"{self._clock(seconds)} / {self._clock(duration)}"
        )
        if overlay_frame is None:
            self.overlay_var.set("Analiz karesi: yok")
        elif overlay_frame == self.current_frame:
            self.overlay_var.set(f"Analiz karesi: {overlay_frame} (gerçek)")
        else:
            self.overlay_var.set(
                f"Analiz karesi: {overlay_frame} (ara karede tutuluyor)"
            )
        self.current_overlay_frame = overlay_frame

    def _read_frame(self, frame_idx: int):
        cached = self._frame_cache.get(frame_idx)
        if cached is not None:
            self._frame_cache.move_to_end(frame_idx)
            return cached
        if not self.capture:
            return None
        position = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if position != frame_idx:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.capture.read()
        if not ok:
            return None
        self._frame_cache[frame_idx] = frame
        while len(self._frame_cache) > 6:
            self._frame_cache.popitem(last=False)
        return frame

    def _rows_at(self, frame_idx: int, variant: str) -> list[dict]:
        key = (frame_idx, variant)
        cached = self._rows_cache.get(key)
        if cached is not None:
            self._rows_cache.move_to_end(key)
            return cached
        assert self.index is not None
        rows = self.index.rows_at(frame_idx, variant)
        self._rows_cache[key] = rows
        while len(self._rows_cache) > 60:
            self._rows_cache.popitem(last=False)
        return rows

    def _draw_overlays(
        self,
        image,
        rows: Iterable[dict],
        overlay_frame: int,
    ) -> None:
        selected = self.selected_id_var.get()
        for row in rows:
            stable_id = str(row.get("stable_id") or "?")
            entity = str(row.get("entity") or "")
            highlighted = selected == "Tümü" or (
                entity == "player" and selected == stable_id
            )
            color = self._id_color(stable_id, entity)
            if not highlighted:
                color = tuple(int(channel * 0.35) for channel in color)
            x1, y1, x2, y2 = (
                row.get("bbox_x1"),
                row.get("bbox_y1"),
                row.get("bbox_x2"),
                row.get("bbox_y2"),
            )
            if None not in (x1, y1, x2, y2):
                point1 = (int(round(x1)), int(round(y1)))
                point2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(image, point1, point2, color, 3 if highlighted else 1)
                confidence = row.get("det_conf")
                conf_text = f" {float(confidence):.2f}" if confidence is not None else ""
                source = str(row.get("match_type") or row.get("source") or "")
                label = f"ID {stable_id}{conf_text} {source}".strip()
                top = max(18, point1[1] - 7)
                cv2.putText(
                    image,
                    label,
                    (point1[0], top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    label,
                    (point1[0], top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if (
                highlighted
                and self.show_trails_var.get()
                and self.index
                and stable_id not in {"", "?"}
            ):
                points = self._trail_at(
                    overlay_frame,
                    stable_id,
                    entity,
                    self.variant_var.get(),
                )
                for start, end in zip(points, points[1:]):
                    cv2.line(
                        image,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        color,
                        2,
                        cv2.LINE_AA,
                    )

    def _trail_at(
        self,
        frame_idx: int,
        stable_id: str,
        entity: str,
        variant: str,
    ) -> list[tuple[float, float]]:
        key = (frame_idx, stable_id, entity, variant)
        cached = self._trail_cache.get(key)
        if cached is not None:
            self._trail_cache.move_to_end(key)
            return cached
        assert self.index is not None
        points = self.index.trail(frame_idx, stable_id, entity, variant)
        self._trail_cache[key] = points
        while len(self._trail_cache) > 300:
            self._trail_cache.popitem(last=False)
        return points

    def _display_image(self, bgr_image) -> None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        canvas_width = max(100, self.canvas.winfo_width() - 4)
        canvas_height = max(100, self.canvas.winfo_height() - 4)
        scale = min(canvas_width / image.width, canvas_height / image.height)
        size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self._photo,
            anchor="center",
        )

    def _draw_empty(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width() // 2),
            max(1, self.canvas.winfo_height() // 2),
            text="Tracking çıktısını yüklediğinizde video burada açılacak.",
            fill="#9AA8BE",
            font=("Segoe UI", 12),
        )

    def _update_table(self, rows: list[dict]) -> None:
        selected = self.selected_id_var.get()
        self.table.delete(*self.table.get_children())
        for index, row in enumerate(rows):
            confidence = row.get("det_conf")
            confidence_text = (
                f"{float(confidence):.2f}" if confidence is not None else "-"
            )
            stable_id = str(row.get("stable_id") or "-")
            item = self.table.insert(
                "",
                "end",
                iid=f"row-{index}",
                values=(
                    stable_id,
                    row.get("entity") or "-",
                    row.get("team") or "-",
                    confidence_text,
                    row.get("match_type") or row.get("source") or "-",
                ),
            )
            if stable_id == selected:
                self.table.selection_set(item)

    def _update_diagnostics(self, frame_idx: int | None) -> None:
        if frame_idx is None or not self.index:
            self.diagnostic_var.set(
                "Bu ham video karesi analiz edilmemiş. "
                "“Ara karede son kutuyu tut” seçeneğiyle son sonucu görebilirsiniz."
            )
            return
        expected = max(1, _integer(self.summary.get("max_players"), 14))
        values = self.index.frame_diagnostics(frame_idx, expected)
        parts = [
            f"Ham sonuç: {values['raw_players']} oyuncu, {values['raw_balls']} top."
        ]
        if values["missing_players"]:
            parts.append(
                f"Hedef sayıya göre {values['missing_players']} oyuncu eksik; "
                "bu kare bir kopma/kaçırma adayıdır."
            )
        if values["new_tracks"]:
            parts.append(
                f"{values['new_tracks']} yeni track başlatılmış"
                + (f" (ID: {values['new_ids']})" if values["new_ids"] else "")
                + "."
            )
        if values["filled"]:
            parts.append(
                f"{values['filled']} nesne boşluk doldurmayla üretilmiş"
                + (
                    f" (oyuncu ID: {values['filled_ids']})"
                    if values["filled_ids"]
                    else ""
                )
                + "."
            )
        if values["low_conf"]:
            parts.append(
                f"{values['low_conf']} düşük güvenli oyuncu kutusu var"
                + (
                    f" (ID: {values['low_conf_ids']})"
                    if values["low_conf_ids"]
                    else ""
                )
                + "."
            )
        if values["max_gap"] > 0:
            parts.append(f"En uzun eksik süre: {values['max_gap']:.2f} sn.")
        if len(parts) == 1:
            parts.append("Belirgin otomatik kopma işareti yok.")
        self.diagnostic_var.set(" ".join(parts))

    def _select_table_id(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection:
            return
        values = self.table.item(selection[0], "values")
        if values and str(values[1]) == "player":
            self.selected_id_var.set(str(values[0]))
            self._render(force=True)

    @staticmethod
    def _id_color(stable_id: str, entity: str) -> tuple[int, int, int]:
        if entity == "ball":
            return (20, 210, 255)
        seed = sum((index + 1) * ord(char) for index, char in enumerate(stable_id))
        hue = seed % 180
        hsv = cv2.cvtColor(
            ImageColorArray([[[hue, 210, 245]]]),
            cv2.COLOR_HSV2BGR,
        )
        return tuple(int(value) for value in hsv[0, 0])

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def close(self) -> None:
        self._load_generation += 1
        self._load_pending = None
        self.pause()
        self._release_capture()
        self._frame_cache.clear()
        self._rows_cache.clear()
        self._trail_cache.clear()

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def ImageColorArray(value):
    """OpenCV renk dönüşümü için NumPy'ı cv2 üzerinden tembelce kullan."""
    import numpy as np

    return np.array(value, dtype="uint8")
