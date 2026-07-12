#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Komut satırı kullanmadan saha tracking pipeline'ını çalıştıran Windows arayüzü."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    BooleanVar,
    END,
    LEFT,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter import scrolledtext, ttk


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def discover_workspace() -> Path:
    """Kaynak dosyanın veya paketlenmiş EXE'nin yanından repo kökünü bul."""
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
            tracker = candidate / "saha_ai_project/app/process_video_to_csv_v8.py"
            if tracker.is_file():
                return candidate
    raise FileNotFoundError(
        "saha_ai_project/app/process_video_to_csv_v8.py bulunamadı. "
        "EXE'yi sosyaltactics klasöründe tutun."
    )


def find_python() -> Path:
    """Tracking scriptini çalıştıracak, paketlerin kurulu olduğu Python'ı bul."""
    if not getattr(sys, "frozen", False):
        current = Path(sys.executable).resolve()
        if current.name.lower() in {"python.exe", "pythonw.exe", "python"}:
            return current

    for executable_name in ("python.exe", "python"):
        found = shutil.which(executable_name)
        if found:
            return Path(found).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        programs = Path(local_app_data) / "Programs/Python"
        if programs.is_dir():
            candidates = sorted(
                programs.glob("Python*/python.exe"),
                key=lambda path: path.parent.name,
                reverse=True,
            )
            if candidates:
                return candidates[0].resolve()

    raise FileNotFoundError(
        "Python bulunamadı. Python 3.12'nin kurulu ve PATH üzerinde olduğundan emin olun."
    )


def build_tracking_command(
    *,
    python_exe: Path,
    project_root: Path,
    video_path: Path,
    output_dir: Path,
    model_path: Path,
    sample_sec: float,
    confidence: float,
    max_players: int,
    test_mode: bool,
    calibration_mode: str,
    force_calibration: bool,
    use_sahi: bool,
) -> list[str]:
    tracker_script = project_root / "app/process_video_to_csv_v8.py"
    mapper = "homography" if calibration_mode == "four_points" else "poly2"
    command = [
        str(python_exe),
        "-u",
        str(tracker_script),
        "--video",
        str(video_path),
        "--out",
        str(output_dir),
        "--model",
        str(model_path),
        "--sample-sec",
        str(sample_sec),
        "--conf",
        str(confidence),
        "--yolo-classes",
        "0,1",
        "--max-players",
        str(max_players),
        "--calibration-mode",
        calibration_mode,
        "--field-mapper",
        mapper,
    ]
    if test_mode:
        command.extend(["--duration", "30"])
    if force_calibration:
        command.append("--force-calibration")
    if use_sahi:
        command.extend(
            [
                "--sahi",
                "--sahi-regions",
                "ball",
                "--sahi-ball-always",
            ]
        )
    return command


class TrackingApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Saha Tracking")
        self.root.geometry("900x720")
        self.root.minsize(780, 640)

        try:
            self.workspace = discover_workspace()
            self.project_root = self.workspace / "saha_ai_project"
            self.python_exe = find_python()
        except (FileNotFoundError, OSError) as exc:
            messagebox.showerror("Başlatılamadı", str(exc))
            self.root.after(10, self.root.destroy)
            return

        default_model = (
            self.project_root
            / "models/trained/player_ball_v1/weights/best.pt"
        )
        self.video_var = StringVar()
        self.output_var = StringVar()
        self.model_var = StringVar(value=str(default_model))
        self.mode_var = StringVar(value="test")
        self.sample_var = StringVar(value="0.2")
        self.conf_var = StringVar(value="0.12")
        self.players_var = StringVar(value="14")
        self.calibration_var = StringVar(value="four_points")
        self.force_calibration_var = BooleanVar(value=False)
        self.sahi_var = BooleanVar(value=False)
        self.status_var = StringVar(value="Video seçerek başlayın.")

        self.process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.last_output_dir: Path | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(7, weight=1)

        ttk.Label(main, text="Video").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.video_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(main, text="Seç…", command=self._choose_video).grid(
            row=0, column=2, pady=5
        )

        ttk.Label(main, text="Model").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.model_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(main, text="Seç…", command=self._choose_model).grid(
            row=1, column=2, pady=5
        )

        ttk.Label(main, text="Çıktı").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.output_var).grid(
            row=2, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(main, text="Seç…", command=self._choose_output).grid(
            row=2, column=2, pady=5
        )

        settings = ttk.LabelFrame(main, text="Ayarlar", padding=12)
        settings.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        settings.columnconfigure(5, weight=1)

        ttk.Radiobutton(
            settings,
            text="İlk 30 saniyeyi dene",
            variable=self.mode_var,
            value="test",
            command=self._sync_mode_defaults,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(
            settings,
            text="Tüm videoyu işle",
            variable=self.mode_var,
            value="full",
            command=self._sync_mode_defaults,
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=(18, 0))

        ttk.Label(settings, text="Örnekleme (sn)").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(settings, textvariable=self.sample_var, width=9).grid(
            row=1, column=1, sticky="w", padx=(8, 20), pady=(12, 0)
        )
        ttk.Label(settings, text="Confidence").grid(
            row=1, column=2, sticky="w", pady=(12, 0)
        )
        ttk.Entry(settings, textvariable=self.conf_var, width=9).grid(
            row=1, column=3, sticky="w", padx=(8, 20), pady=(12, 0)
        )
        ttk.Label(settings, text="Maks. oyuncu").grid(
            row=1, column=4, sticky="w", pady=(12, 0)
        )
        ttk.Entry(settings, textvariable=self.players_var, width=9).grid(
            row=1, column=5, sticky="w", padx=(8, 0), pady=(12, 0)
        )

        ttk.Label(settings, text="Kalibrasyon").grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )
        calibration_combo = ttk.Combobox(
            settings,
            textvariable=self.calibration_var,
            state="readonly",
            width=23,
            values=("four_points", "line"),
        )
        calibration_combo.grid(
            row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(12, 0)
        )

        ttk.Checkbutton(
            settings,
            text="Kalibrasyonu yeniden yap",
            variable=self.force_calibration_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            settings,
            text="Top için SAHI kullan (daha yavaş)",
            variable=self.sahi_var,
        ).grid(row=3, column=3, columnspan=3, sticky="w", pady=(12, 0))

        note = (
            "four_points: sol üst → sağ üst → sağ alt → sol alt, ardından Enter. "
            "İlk testten sonra aynı çıktı klasörüyle tam videoyu çalıştırırsanız "
            "kalibrasyon otomatik yeniden kullanılır."
        )
        ttk.Label(settings, text=note, wraplength=820).grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(12, 0)
        )

        actions = ttk.Frame(main)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.start_button = ttk.Button(
            actions, text="İşlemeyi Başlat", command=self._start
        )
        self.start_button.pack(side=LEFT)
        self.cancel_button = ttk.Button(
            actions, text="İptal", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side=LEFT, padx=8)
        self.viewer_button = ttk.Button(
            actions, text="Viewer'ı Aç", command=self._open_viewer
        )
        self.viewer_button.pack(side=LEFT, padx=(20, 8))
        self.folder_button = ttk.Button(
            actions, text="Çıktı Klasörünü Aç", command=self._open_output
        )
        self.folder_button.pack(side=LEFT)

        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(15, 5)
        )
        ttk.Label(main, textvariable=self.status_var).grid(
            row=6, column=0, columnspan=3, sticky="w"
        )

        self.log = scrolledtext.ScrolledText(
            main, height=16, state="disabled", wrap="word"
        )
        self.log.grid(
            row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0)
        )

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="İşlenecek videoyu seçin",
            filetypes=[
                ("Video dosyaları", "*.mp4 *.mov *.avi *.mkv"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not selected:
            return
        video = Path(selected).resolve()
        self.video_var.set(str(video))
        default_output = (
            self.project_root / "experiments/outputs" / f"{video.stem}_tracked"
        )
        self.output_var.set(str(default_output))

    def _choose_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="YOLO modelini seçin",
            filetypes=[("PyTorch modeli", "*.pt"), ("Tüm dosyalar", "*.*")],
        )
        if selected:
            self.model_var.set(selected)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Çıktı klasörünü seçin")
        if selected:
            self.output_var.set(selected)

    def _sync_mode_defaults(self) -> None:
        self.sample_var.set("0.2" if self.mode_var.get() == "test" else "0.1")

    def _validated_settings(self) -> dict[str, object]:
        video = Path(self.video_var.get().strip()).expanduser()
        model = Path(self.model_var.get().strip()).expanduser()
        output_text = self.output_var.get().strip()
        if not video.is_file():
            raise ValueError("Geçerli bir video seçin.")
        if not model.is_file():
            raise ValueError("Geçerli bir .pt model dosyası seçin.")
        if not output_text:
            raise ValueError("Çıktı klasörü seçin.")
        output = Path(output_text).expanduser()

        try:
            sample_sec = float(self.sample_var.get())
            confidence = float(self.conf_var.get())
            max_players = int(self.players_var.get())
        except ValueError as exc:
            raise ValueError(
                "Örnekleme, confidence ve maksimum oyuncu değerlerini kontrol edin."
            ) from exc
        if sample_sec <= 0:
            raise ValueError("Örnekleme süresi sıfırdan büyük olmalıdır.")
        if not 0 < confidence <= 1:
            raise ValueError("Confidence 0 ile 1 arasında olmalıdır.")
        if max_players <= 0:
            raise ValueError("Maksimum oyuncu sayısı pozitif olmalıdır.")

        return {
            "video_path": video.resolve(),
            "model_path": model.resolve(),
            "output_dir": output.resolve(),
            "sample_sec": sample_sec,
            "confidence": confidence,
            "max_players": max_players,
        }

    def _start(self) -> None:
        if self.process is not None:
            return
        try:
            settings = self._validated_settings()
            output_dir = settings["output_dir"]
            assert isinstance(output_dir, Path)
            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_tracking_command(
                python_exe=self.python_exe,
                project_root=self.project_root,
                video_path=settings["video_path"],
                output_dir=output_dir,
                model_path=settings["model_path"],
                sample_sec=settings["sample_sec"],
                confidence=settings["confidence"],
                max_players=settings["max_players"],
                test_mode=self.mode_var.get() == "test",
                calibration_mode=self.calibration_var.get(),
                force_calibration=self.force_calibration_var.get(),
                use_sahi=self.sahi_var.get(),
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Başlatılamadı", str(exc))
            return

        self.last_output_dir = output_dir
        self._set_running(True)
        self._clear_log()
        self._append_log("İşlem başlatılıyor…")
        self._append_log(f"Video: {settings['video_path']}")
        self._append_log(f"Model: {settings['model_path']}")
        self._append_log(f"Çıktı: {output_dir}")

        self.worker = threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
        )
        self.worker.start()

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.root.after(0, self._append_log, line.rstrip())
            return_code = self.process.wait()
            self.root.after(0, self._process_finished, return_code)
        except Exception as exc:
            self.root.after(0, self._process_failed, str(exc))

    def _process_finished(self, return_code: int) -> None:
        self.process = None
        self._set_running(False)
        output_csv = (
            self.last_output_dir / "tracking_all.csv"
            if self.last_output_dir is not None
            else None
        )
        if return_code == 0 and output_csv is not None and output_csv.is_file():
            self.status_var.set("Tamamlandı. Viewer ile sonucu açabilirsiniz.")
            self._append_log(f"Tamamlandı: {output_csv}")
            messagebox.showinfo(
                "Tamamlandı",
                "Tracking tamamlandı.\n\n"
                "Viewer'ı Aç düğmesine basıp tracking_all.csv dosyasını seçin.",
            )
        elif return_code == 0:
            self.status_var.set("İşlem tamamlandı fakat CSV bulunamadı.")
            messagebox.showwarning(
                "CSV bulunamadı",
                "İşlem sona erdi ancak tracking_all.csv bulunamadı. Logu kontrol edin.",
            )
        else:
            self.status_var.set(f"İşlem hata koduyla durdu: {return_code}")
            messagebox.showerror(
                "İşlem başarısız",
                f"Tracking hata kodu {return_code} ile durdu. Logu kontrol edin.",
            )

    def _process_failed(self, error: str) -> None:
        self.process = None
        self._set_running(False)
        self.status_var.set("İşlem başlatılamadı.")
        self._append_log(f"HATA: {error}")
        messagebox.showerror("İşlem başlatılamadı", error)

    def _cancel(self) -> None:
        if self.process is None:
            return
        if messagebox.askyesno("İptal", "Devam eden tracking işlemi durdurulsun mu?"):
            try:
                self.process.terminate()
            except OSError:
                pass
            self.status_var.set("İptal ediliyor…")

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start(12)
            self.status_var.set(
                "İşleniyor… Kalibrasyon penceresi açılırsa saha noktalarını ayarlayın."
            )
        else:
            self.progress.stop()

    def _open_viewer(self) -> None:
        viewer = self.project_root / "app/saha_csv_viewer.html"
        if not viewer.is_file():
            messagebox.showerror("Viewer bulunamadı", str(viewer))
            return
        os.startfile(str(viewer))

    def _open_output(self) -> None:
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showwarning("Çıktı yok", "Önce bir çıktı klasörü seçin.")
            return
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text + "\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", END)
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(
                "İşlem devam ediyor",
                "Tracking işlemi durdurulup pencere kapatılsın mı?",
            ):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        self.root.destroy()


def main() -> None:
    root = Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except Exception:
        pass
    TrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
