"""PitchCalibStudio'yu konsol penceresi açmadan başlatır."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox


ROOT = Path(__file__).resolve().parent
PORT = 8502
URL = f"http://127.0.0.1:{PORT}"


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> None:
    if healthy():
        webbrowser.open(URL)
        return
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    if pythonw.exists():
        python = pythonw
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [
            str(python),
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless=true",
            f"--server.port={PORT}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    for _ in range(40):
        if process.poll() is not None:
            messagebox.showerror(
                "PitchCalibStudio",
                "Uygulama başlatılamadı. Önce requirements.txt kurulumunu kontrol edin.",
            )
            return
        if healthy():
            webbrowser.open(URL)
            return
        time.sleep(0.25)
    messagebox.showerror(
        "PitchCalibStudio",
        f"Sunucu zamanında yanıt vermedi. Adres: {URL}",
    )


if __name__ == "__main__":
    main()
