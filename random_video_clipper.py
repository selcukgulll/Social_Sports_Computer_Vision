#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bir videodan rastgele başlangıç noktalarına sahip sabit süreli klipler üretir.

Kurulum:
    python -m pip install imageio-ffmpeg

Çalıştırma:
    python random_video_clipper.py

Not:
    Kliplerin çakışmasına izin verilir. Örneğin 30 dakikalık bir kaynaktan
    100 adet 2 dakikalık klip üretilebilir.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import scrolledtext, ttk
from typing import Callable

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DURATION_PATTERN = re.compile(
    r"Duration:\s*(\d{1,3}):(\d{2}):(\d{2}(?:\.\d+)?)"
)


class ClipCancelled(Exception):
    """Kullanıcı işlemi iptal etti."""


def find_ffmpeg() -> str:
    """Sistem FFmpeg'ini veya imageio-ffmpeg ile gelen yürütülebilir dosyayı bul."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise RuntimeError(
        "FFmpeg bulunamadı. Şu komutla gerekli paketi kurun:\n"
        "python -m pip install imageio-ffmpeg"
    )


def probe_duration(ffmpeg_exe: str, video_path: Path) -> float:
    """FFmpeg çıktısından kaynak videonun süresini saniye olarak oku."""
    completed = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(video_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    match = DURATION_PATTERN.search(completed.stderr)
    if not match:
        raise RuntimeError(
            "Video süresi okunamadı. Dosyanın geçerli ve desteklenen bir video "
            "olduğundan emin olun."
        )
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_filename_time(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"


def make_unique_run_directory(base_dir: Path, source_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{source_stem}_random_clips_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{source_stem}_random_clips_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def build_ffmpeg_command(
    ffmpeg_exe: str,
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    fast_copy: bool,
) -> list[str]:
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if fast_copy:
        command.extend(
            [
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
            ]
        )
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
        )
    command.extend(["-movflags", "+faststart", str(output_path)])
    return command


def run_ffmpeg(
    command: list[str],
    cancel_event: threading.Event,
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    while process.poll() is None:
        if cancel_event.wait(0.1):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            process.communicate()
            raise ClipCancelled

    _, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.strip() or f"FFmpeg çıkış kodu: {process.returncode}"
        raise RuntimeError(detail)


def create_random_clips(
    *,
    ffmpeg_exe: str,
    source_path: Path,
    output_base_dir: Path,
    source_duration: float,
    clip_duration: float,
    clip_count: int,
    seed: int,
    fast_copy: bool,
    cancel_event: threading.Event,
    progress_callback: Callable[[int, int, float, Path], None],
) -> tuple[Path, int, bool]:
    if clip_duration > source_duration:
        raise ValueError(
            f"Klip süresi ({format_clock(clip_duration)}) kaynak videodan "
            f"({format_clock(source_duration)}) uzun olamaz."
        )

    # Küçük pay, FFmpeg süre bilgisindeki yuvarlama nedeniyle son klibin kısa
    # kalmasını engeller.
    max_start = max(0.0, source_duration - clip_duration - 0.05)
    rng = random.Random(seed)
    starts = [rng.uniform(0.0, max_start) for _ in range(clip_count)]

    run_dir = make_unique_run_directory(output_base_dir, source_path.stem)
    clips: list[dict[str, object]] = []
    cancelled = False

    try:
        for index, start_seconds in enumerate(starts, start=1):
            if cancel_event.is_set():
                raise ClipCancelled

            start_label = format_filename_time(start_seconds)
            output_path = run_dir / (
                f"{source_path.stem}_clip_{index:04d}_start_{start_label}.mp4"
            )
            command = build_ffmpeg_command(
                ffmpeg_exe,
                source_path,
                output_path,
                start_seconds,
                clip_duration,
                fast_copy,
            )
            try:
                run_ffmpeg(command, cancel_event)
            except Exception:
                output_path.unlink(missing_ok=True)
                raise

            clips.append(
                {
                    "index": index,
                    "file": output_path.name,
                    "start_seconds": round(start_seconds, 3),
                    "start_time": format_clock(start_seconds),
                    "duration_seconds": round(clip_duration, 3),
                }
            )
            progress_callback(index, clip_count, start_seconds, output_path)
    except ClipCancelled:
        cancelled = True
    finally:
        manifest = {
            "source": str(source_path.resolve()),
            "source_duration_seconds": round(source_duration, 3),
            "clip_duration_seconds": round(clip_duration, 3),
            "requested_clip_count": clip_count,
            "completed_clip_count": len(clips),
            "seed": seed,
            "mode": "fast_stream_copy" if fast_copy else "precise_reencode",
            "cancelled": cancelled,
            "clips": clips,
        }
        manifest_path = run_dir / "clips_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return run_dir, len(clips), cancelled


class VideoClipperApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Rastgele Video Parçalayıcı")
        self.root.geometry("760x570")
        self.root.minsize(700, 520)

        self.source_var = StringVar()
        self.output_var = StringVar()
        self.count_var = StringVar(value="100")
        self.minutes_var = StringVar(value="2")
        self.seconds_var = StringVar(value="0")
        self.seed_var = StringVar()
        self.fast_copy_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Bir MP4 dosyası seçerek başlayın.")

        self.cancel_event = threading.Event()
        self.running = False
        self.close_when_finished = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Kaynak MP4").grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.source_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(frame, text="Seç…", command=self._choose_source).grid(
            row=0, column=2, pady=5
        )

        ttk.Label(frame, text="Çıktı klasörü").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(frame, text="Seç…", command=self._choose_output).grid(
            row=1, column=2, pady=5
        )

        options = ttk.LabelFrame(frame, text="Klip ayarları", padding=12)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 8))

        ttk.Label(options, text="Klip adedi").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.count_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(8, 25)
        )
        ttk.Label(options, text="Süre").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.minutes_var, width=8).grid(
            row=0, column=3, sticky="w", padx=(8, 4)
        )
        ttk.Label(options, text="dakika").grid(row=0, column=4, sticky="w")
        ttk.Entry(options, textvariable=self.seconds_var, width=8).grid(
            row=0, column=5, sticky="w", padx=(12, 4)
        )
        ttk.Label(options, text="saniye").grid(row=0, column=6, sticky="w")

        ttk.Label(options, text="Random seed (isteğe bağlı)").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(12, 0)
        )
        ttk.Entry(options, textvariable=self.seed_var, width=18).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(12, 0)
        )
        ttk.Checkbutton(
            options,
            text="Hızlı kesim (yeniden kodlama yapma)",
            variable=self.fast_copy_var,
        ).grid(row=2, column=0, columnspan=7, sticky="w", pady=(12, 0))

        note = (
            "Hızlı kesim ses ve görüntüyü korur; başlangıç noktası en yakın ana "
            "kareye birkaç saniye kayabilir. Kutuyu kapatırsanız kesimler daha "
            "hassas olur fakat işlem çok daha uzun sürer."
        )
        ttk.Label(frame, text=note, wraplength=700).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 12)
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.start_button = ttk.Button(
            buttons, text="Klipleri Oluştur", command=self._start
        )
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(
            buttons, text="İptal", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(16, 6)
        )
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=6, column=0, columnspan=3, sticky="w"
        )

        self.log = scrolledtext.ScrolledText(
            frame, height=12, state="disabled", wrap="word"
        )
        self.log.grid(
            row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0)
        )
        frame.rowconfigure(7, weight=1)

    def _choose_source(self) -> None:
        selected = filedialog.askopenfilename(
            title="Kaynak MP4 dosyasını seçin",
            filetypes=[
                ("MP4 videoları", "*.mp4"),
                ("Video dosyaları", "*.mp4 *.mov *.mkv *.avi"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if not selected:
            return
        source_path = Path(selected)
        self.source_var.set(str(source_path))
        if not self.output_var.get().strip():
            self.output_var.set(str(source_path.parent))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Çıktı klasörünü seçin")
        if selected:
            self.output_var.set(selected)

    def _read_inputs(self) -> tuple[Path, Path, int, float, int]:
        source_path = Path(self.source_var.get().strip()).expanduser()
        if not source_path.is_file():
            raise ValueError("Geçerli bir kaynak video seçin.")

        output_text = self.output_var.get().strip()
        if not output_text:
            raise ValueError("Bir çıktı klasörü seçin.")
        output_dir = Path(output_text).expanduser()

        try:
            clip_count = int(self.count_var.get())
        except ValueError as exc:
            raise ValueError("Klip adedi tam sayı olmalıdır.") from exc
        if not 1 <= clip_count <= 10000:
            raise ValueError("Klip adedi 1 ile 10000 arasında olmalıdır.")

        try:
            minutes = float(self.minutes_var.get() or 0)
            seconds = float(self.seconds_var.get() or 0)
        except ValueError as exc:
            raise ValueError("Dakika ve saniye sayısal olmalıdır.") from exc
        clip_duration = minutes * 60 + seconds
        if minutes < 0 or seconds < 0 or clip_duration <= 0:
            raise ValueError("Klip süresi sıfırdan büyük olmalıdır.")

        seed_text = self.seed_var.get().strip()
        try:
            seed = int(seed_text) if seed_text else random.SystemRandom().randrange(
                0, 2**63
            )
        except ValueError as exc:
            raise ValueError("Random seed tam sayı olmalıdır.") from exc

        return source_path, output_dir, clip_count, clip_duration, seed

    def _start(self) -> None:
        try:
            source_path, output_dir, clip_count, clip_duration, seed = (
                self._read_inputs()
            )
            ffmpeg_exe = find_ffmpeg()
            source_duration = probe_duration(ffmpeg_exe, source_path)
            if clip_duration > source_duration:
                raise ValueError(
                    f"İstenen klip süresi ({format_clock(clip_duration)}) kaynak "
                    f"videodan ({format_clock(source_duration)}) uzun."
                )
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Başlatılamadı", str(exc))
            return

        estimated_bytes = (
            source_path.stat().st_size
            * clip_duration
            * clip_count
            / source_duration
        )
        estimated_gb = estimated_bytes / (1024**3)
        confirmation = (
            f"Kaynak süre: {format_clock(source_duration)}\n"
            f"Klip: {clip_count} × {format_clock(clip_duration)}\n"
            f"Tahmini çıktı boyutu: {estimated_gb:.2f} GB\n\n"
            "Kliplerin birbiriyle çakışmasına izin verilecek. Devam edilsin mi?"
        )
        if not messagebox.askyesno("Klipleri oluştur", confirmation):
            return

        self.running = True
        self.close_when_finished = False
        self.cancel_event.clear()
        self.progress.configure(maximum=clip_count, value=0)
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("Klipler hazırlanıyor…")
        self._append_log(
            f"Kaynak: {source_path}\n"
            f"Süre: {format_clock(source_duration)}, seed: {seed}\n"
        )

        worker = threading.Thread(
            target=self._worker,
            kwargs={
                "ffmpeg_exe": ffmpeg_exe,
                "source_path": source_path,
                "output_dir": output_dir,
                "source_duration": source_duration,
                "clip_duration": clip_duration,
                "clip_count": clip_count,
                "seed": seed,
                "fast_copy": self.fast_copy_var.get(),
            },
            daemon=True,
        )
        worker.start()

    def _worker(
        self,
        *,
        ffmpeg_exe: str,
        source_path: Path,
        output_dir: Path,
        source_duration: float,
        clip_duration: float,
        clip_count: int,
        seed: int,
        fast_copy: bool,
    ) -> None:
        try:
            run_dir, completed_count, cancelled = create_random_clips(
                ffmpeg_exe=ffmpeg_exe,
                source_path=source_path,
                output_base_dir=output_dir,
                source_duration=source_duration,
                clip_duration=clip_duration,
                clip_count=clip_count,
                seed=seed,
                fast_copy=fast_copy,
                cancel_event=self.cancel_event,
                progress_callback=self._report_progress,
            )
            self.root.after(
                0,
                self._finish,
                run_dir,
                completed_count,
                cancelled,
                None,
            )
        except Exception as exc:
            self.root.after(0, self._finish, None, 0, False, str(exc))

    def _report_progress(
        self,
        current: int,
        total: int,
        start_seconds: float,
        output_path: Path,
    ) -> None:
        def update() -> None:
            self.progress.configure(value=current)
            self.status_var.set(
                f"{current}/{total} tamamlandı — başlangıç "
                f"{format_clock(start_seconds)}"
            )
            self._append_log(f"{current:04d}: {output_path.name}")

        self.root.after(0, update)

    def _finish(
        self,
        run_dir: Path | None,
        completed_count: int,
        cancelled: bool,
        error: str | None,
    ) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

        if error:
            self.status_var.set("İşlem hata nedeniyle durdu.")
            self._append_log(f"HATA: {error}")
            messagebox.showerror("Klipler oluşturulamadı", error)
        elif cancelled:
            self.status_var.set(f"İptal edildi. {completed_count} klip tamamlandı.")
            self._append_log(f"İptal edildi. Çıktı: {run_dir}")
            messagebox.showinfo(
                "İşlem iptal edildi",
                f"{completed_count} klip tamamlandı.\n\nÇıktı:\n{run_dir}",
            )
        else:
            self.status_var.set(f"Tamamlandı: {completed_count} klip oluşturuldu.")
            self._append_log(f"Tamamlandı. Çıktı: {run_dir}")
            messagebox.showinfo(
                "Tamamlandı",
                f"{completed_count} klip oluşturuldu.\n\nÇıktı:\n{run_dir}",
            )

        if self.close_when_finished:
            self.root.destroy()

    def _cancel(self) -> None:
        if self.running:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status_var.set("İptal ediliyor…")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if not self.running:
            self.root.destroy()
            return
        if messagebox.askyesno(
            "İşlem devam ediyor",
            "Klip oluşturma işlemi iptal edilip pencere kapatılsın mı?",
        ):
            self.close_when_finished = True
            self._cancel()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bir videodan rastgele başlangıçlı sabit süreli klipler üretir."
    )
    parser.add_argument("--source", type=Path, help="Kaynak video")
    parser.add_argument("--output", type=Path, help="Çıktı ana klasörü")
    parser.add_argument("--count", type=int, default=10, help="Klip adedi")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=120.0,
        help="Her klibin süresi, saniye",
    )
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Hızlı stream copy yerine hassas yeniden kodlama kullan",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    if args.source is None:
        return -1
    source = args.source.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.parent.resolve()
    )
    if not source.is_file():
        print(f"HATA: Kaynak video bulunamadı: {source}")
        return 2
    if args.count <= 0 or args.duration_sec <= 0:
        print("HATA: Klip adedi ve süresi sıfırdan büyük olmalıdır.")
        return 2

    output.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = find_ffmpeg()
    source_duration = probe_duration(ffmpeg_exe, source)
    print(f"Kaynak: {source}")
    print(f"Kaynak süresi: {format_clock(source_duration)}")

    def progress(index: int, total: int, start: float, path: Path) -> None:
        print(
            f"[{index}/{total}] başlangıç={format_clock(start)} "
            f"dosya={path.name}",
            flush=True,
        )

    run_dir, completed, cancelled = create_random_clips(
        ffmpeg_exe=ffmpeg_exe,
        source_path=source,
        output_base_dir=output,
        source_duration=source_duration,
        clip_duration=float(args.duration_sec),
        clip_count=int(args.count),
        seed=int(args.seed),
        fast_copy=not args.precise,
        cancel_event=threading.Event(),
        progress_callback=progress,
    )
    print(f"Tamamlanan klip: {completed}")
    print(f"Çıktı klasörü: {run_dir}")
    return 130 if cancelled else 0


def main() -> None:
    args = build_cli_parser().parse_args()
    cli_result = run_cli(args)
    if cli_result >= 0:
        raise SystemExit(cli_result)

    root = Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except Exception:
        pass
    VideoClipperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
