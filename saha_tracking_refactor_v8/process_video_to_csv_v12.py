#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SosyalHalisaha / halı saha video -> saha koordinatı CSV/XLSX üretici

Bu dosya sadece görüntü işleme + CSV/XLSX üretir.
Viewer ayrı dosyadır: saha_csv_viewer.html

Kurulum:
    pip install ultralytics opencv-python pandas numpy openpyxl tqdm

Temel kullanım:
    python process_video_to_csv.py

Direkt video vererek:
    python process_video_to_csv.py --video "mac.mp4" --out "output" --sample-sec 0.1 --model yolo11s.pt

Önemli fikirler:
- 4 köşe görünmüyorsa çizgi bazlı kalibrasyon kullanır.
- Görünen saha çizgilerinden homography hesaplar.
- Oyuncular için stable_id üretir.
- Forma/şort renk histogramı ile basit appearance signature tutar.
- Oyuncu kaybolursa son pozisyon + hız tahmini + appearance ile geri bağlamaya çalışır.
- Kısa kayıpları interpolate ederek tracks_interpolated.csv üretir.
- Top için ayrı raw/interpolated CSV üretir. COCO top tespiti zayıf olabilir; iyi sonuç için ileride özel ball model önerilir.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# Normalleştirilmiş küçük saha modeli.
# İstersen gerçek ölçüye yakın değerler verebilirsin; viewer aynı değerlerle çizer.
PITCH_LENGTH = 100.0
PITCH_WIDTH = 60.0
PENALTY_DEPTH = 16.0
PENALTY_WIDTH = 36.0
GOALBOX_DEPTH = 6.0
GOALBOX_WIDTH = 18.0


@dataclass
class FieldLineDef:
    key: str
    label: str
    # Homojen saha doğrusu: a*x + b*y + c = 0
    coeff: tuple[float, float, float]
    color_bgr: tuple[int, int, int]


LINE_DEFS: dict[str, FieldLineDef] = {
    "1": FieldLineDef("top_touch", "UST TAC / UST KENAR (y=0)", (0.0, 1.0, 0.0), (0, 255, 255)),
    "2": FieldLineDef("bottom_touch", "ALT TAC / ALT KENAR (y=W)", (0.0, 1.0, -PITCH_WIDTH), (0, 255, 255)),
    "3": FieldLineDef("left_goal", "SOL KALE CIZGISI (x=0)", (1.0, 0.0, 0.0), (255, 0, 0)),
    "4": FieldLineDef("right_goal", "SAG KALE CIZGISI (x=L)", (1.0, 0.0, -PITCH_LENGTH), (255, 0, 0)),
    "5": FieldLineDef("halfway", "ORTA CIZGI (x=L/2)", (1.0, 0.0, -PITCH_LENGTH / 2.0), (255, 255, 0)),
    "6": FieldLineDef("left_pen_front", "SOL CEZA ON CIZGISI", (1.0, 0.0, -PENALTY_DEPTH), (0, 180, 255)),
    "7": FieldLineDef("right_pen_front", "SAG CEZA ON CIZGISI", (1.0, 0.0, -(PITCH_LENGTH - PENALTY_DEPTH)), (0, 180, 255)),
    "8": FieldLineDef("pen_top", "CEZA SAHASI UST YATAY", (0.0, 1.0, -(PITCH_WIDTH - PENALTY_WIDTH) / 2.0), (180, 255, 0)),
    "9": FieldLineDef("pen_bottom", "CEZA SAHASI ALT YATAY", (0.0, 1.0, -(PITCH_WIDTH + PENALTY_WIDTH) / 2.0), (180, 255, 0)),
    "q": FieldLineDef("left_goalbox_front", "SOL KALE ALANI ON", (1.0, 0.0, -GOALBOX_DEPTH), (255, 180, 0)),
    "w": FieldLineDef("right_goalbox_front", "SAG KALE ALANI ON", (1.0, 0.0, -(PITCH_LENGTH - GOALBOX_DEPTH)), (255, 180, 0)),
    "e": FieldLineDef("goalbox_top", "KALE ALANI UST YATAY", (0.0, 1.0, -(PITCH_WIDTH - GOALBOX_WIDTH) / 2.0), (180, 180, 255)),
    "r": FieldLineDef("goalbox_bottom", "KALE ALANI ALT YATAY", (0.0, 1.0, -(PITCH_WIDTH + GOALBOX_WIDTH) / 2.0), (180, 180, 255)),
}


def choose_file_gui(title: str, patterns: list[tuple[str, str]]) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=patterns)
        root.destroy()
        return path or None
    except Exception:
        return None


def choose_dir_gui(title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or None
    except Exception:
        return None


def ensure_out_dir(video_path: Path, out_arg: str | None) -> Path:
    if out_arg:
        out = Path(out_arg)
    else:
        out = video_path.with_suffix("").parent / f"{video_path.stem}_tracking_output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_frame_at(video_path: str | Path, time_sec: float = 0.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Frame okunamadı.")
    return frame


def apply_lens_correction(frame: np.ndarray, args=None) -> np.ndarray:
    """
    Basit radial lens düzeltmesi.

    Not: Gerçek balık-gözü düzeltmesi için kamera kalibrasyonu en doğru yoldur.
    Bu pratik ayar, halı saha videosunda çizgiler hafif eğri/yassı görünüyorsa deneme amaçlıdır.

    Örnek:
        --undistort-k1 -0.20
        --undistort-k1 0.20

    Hangisinin iyi geldiği kameraya göre değişir; calibration_debug.jpg ile kontrol et.
    """
    if args is None:
        return frame
    k1 = float(getattr(args, "undistort_k1", 0.0) or 0.0)
    k2 = float(getattr(args, "undistort_k2", 0.0) or 0.0)
    k3 = float(getattr(args, "undistort_k3", 0.0) or 0.0)
    if abs(k1) < 1e-12 and abs(k2) < 1e-12 and abs(k3) < 1e-12:
        return frame

    h, w = frame.shape[:2]
    focal = float(getattr(args, "undistort_focal", 0.0) or 0.0)
    if focal <= 0:
        focal = max(w, h)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
    return cv2.undistort(frame, K, D, None, K)


def read_corrected_frame_at(video_path: str | Path, time_sec: float = 0.0, args=None) -> np.ndarray:
    return apply_lens_correction(read_frame_at(video_path, time_sec), args)


def line_from_two_points(p1: tuple[float, float], p2: tuple[float, float]) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    l = np.cross(np.array([x1, y1, 1.0], dtype=np.float64), np.array([x2, y2, 1.0], dtype=np.float64))
    n = math.hypot(l[0], l[1])
    if n < 1e-9:
        return l
    return l / n


def intersection_of_lines(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float] | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def field_line_array(coeff: tuple[float, float, float]) -> np.ndarray:
    a, b, c = coeff
    n = math.hypot(a, b)
    return np.array([a / n, b / n, c / n], dtype=np.float64)


def solve_homography_from_lines(lines: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Kullanıcının çizdiği görüntü çizgilerinin bilinen saha çizgileriyle eşleşmesinden
    kesişim noktaları çıkarır ve image->field homography hesaplar.
    """
    img_h, img_w = image_shape[:2]
    img_lines = []
    for ln in lines:
        p1 = tuple(ln["p1"])
        p2 = tuple(ln["p2"])
        line_key = ln["line_key"]
        l_img = line_from_two_points(p1, p2)
        l_field = field_line_array(LINE_DEFS[line_key].coeff)
        img_lines.append({**ln, "l_img": l_img, "l_field": l_field})

    image_pts: list[tuple[float, float]] = []
    field_pts: list[tuple[float, float]] = []
    used_pairs = []

    for i in range(len(img_lines)):
        for j in range(i + 1, len(img_lines)):
            fi = img_lines[i]["l_field"]
            fj = img_lines[j]["l_field"]
            ii = img_lines[i]["l_img"]
            ij = img_lines[j]["l_img"]
            fpt = intersection_of_lines(fi, fj)
            ipt = intersection_of_lines(ii, ij)
            if fpt is None or ipt is None:
                continue
            fx, fy = fpt
            ix, iy = ipt
            # Saha modelinin çok dışındaki kesişimleri alma.
            if not (-5 <= fx <= PITCH_LENGTH + 5 and -5 <= fy <= PITCH_WIDTH + 5):
                continue
            # Görüntüde hayali kesişim biraz dışarıda kalabilir, ama aşırı uzaksa alma.
            if not (-img_w * 2 <= ix <= img_w * 3 and -img_h * 2 <= iy <= img_h * 3):
                continue
            image_pts.append((ix, iy))
            field_pts.append((fx, fy))
            used_pairs.append({
                "image_point": [ix, iy],
                "field_point": [fx, fy],
                "from_lines": [img_lines[i]["line_key"], img_lines[j]["line_key"]],
            })

    if len(image_pts) < 4:
        raise RuntimeError(
            f"Homography için en az 4 kesişim noktası gerekli, {len(image_pts)} bulundu. "
            "En az iki dikey ve iki yatay saha çizgisi çiz; örn. sağ kale çizgisi + sağ ceza ön çizgisi + üst/alt taç veya ceza yatayları."
        )

    H, mask = cv2.findHomography(
        np.array(image_pts, dtype=np.float32),
        np.array(field_pts, dtype=np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if H is None:
        raise RuntimeError("Homography hesaplanamadı. Çizgiler yanlış etiketlenmiş olabilir.")
    return H, used_pairs


def draw_line_segment(canvas: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int], label: str):
    p1i = (int(round(p1[0])), int(round(p1[1])))
    p2i = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(canvas, p1i, p2i, color, 2)
    mx = int((p1i[0] + p2i[0]) / 2)
    my = int((p1i[1] + p2i[1]) / 2)
    cv2.putText(canvas, label[:18], (mx + 5, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def line_based_calibration(frame: np.ndarray, out_dir: Path, video_path: Path | None = None, args=None) -> dict[str, Any]:
    """
    Curve-aware kalibrasyon ekranı.

    Ana saha kenarları artık 3 handle'lıdır:
    - iki uç nokta
    - bir orta kare handle

    Bu, balık gözü / yassı saha çizgilerine düz homography'den daha iyi tepki verebilir.
    Homography yine fallback olarak kaydedilir.
    """
    print()
    print("=== Curve-aware Kalibrasyon Editörü ===")
    print("Ana saha sınırları 3 handle'lı: sol/sağ kısa kenar + üst/alt uzun kenar. Uçları köşeye kadar götürmek zorunda değilsin; sistem çizgileri uzatıp köşeleri kesişimden bulur.")
    print("N/Space: sonraki sahne | P: önceki sahne | Enter/S: kaydet | R: reset | Esc: iptal")
    print("Futbol sahasında kaleler sol/sağ kısa kenarlardadır. SOL/SAG KALE CIZGISI kısa kenar; UST/ALT TAC CIZGISI uzun kenardır.")
    print("Orta çizgi ve iki ceza sahası ön çizgisi de manuel ayarlanabilir. Sistem sol/sağ ve üst/alt yönünü görüntü geometrisinden otomatik tahmin etmeye çalışır.")
    print()

    max_w, max_h = 1500, 900
    current_time = float(getattr(args, "calibration_time", 0.0) or 0.0)
    step_sec = float(getattr(args, "calibration_step_sec", 5.0) or 5.0)

    duration = 0.0
    if video_path is not None:
        cap_meta = cv2.VideoCapture(str(video_path))
        fps_meta = float(cap_meta.get(cv2.CAP_PROP_FPS) or 0.0)
        cnt_meta = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap_meta.release()
        if fps_meta > 0 and cnt_meta > 0:
            duration = cnt_meta / fps_meta

    state = {"frame": frame, "disp_base": None, "scale": 1.0, "w": frame.shape[1], "h": frame.shape[0], "items": None}

    def make_defaults(w: int, h: int) -> dict[str, dict[str, Any]]:
        py1 = (PITCH_WIDTH - PENALTY_WIDTH) / 2.0
        py2 = (PITCH_WIDTH + PENALTY_WIDTH) / 2.0
        # Ekran üzerinde yaklaşık yerler. Kullanıcı uçları köşeye kadar götürmek zorunda değil;
        # solver ana çizgileri uzatıp kesiştirerek köşeleri kendi bulur.
        return {
            "sol_tac": {"label": "SOL KALE CIZGISI / SOL KISA KENAR", "type": "curve", "color": (0, 255, 255), "p1": [0.18*w, 0.16*h], "pm": [0.10*w, 0.54*h], "p2": [0.05*w, 0.92*h]},
            "sag_tac": {"label": "SAG KALE CIZGISI / SAG KISA KENAR", "type": "curve", "color": (0, 200, 255), "p1": [0.82*w, 0.16*h], "pm": [0.90*w, 0.54*h], "p2": [0.95*w, 0.92*h]},
            "yukari_korner": {"label": "UST TAC CIZGISI / UST UZUN KENAR", "type": "curve", "color": (255, 200, 0), "p1": [0.18*w, 0.20*h], "pm": [0.50*w, 0.16*h], "p2": [0.82*w, 0.20*h]},
            "asagi_korner": {"label": "ALT TAC CIZGISI / ALT UZUN KENAR", "type": "curve", "color": (255, 160, 0), "p1": [0.08*w, 0.84*h], "pm": [0.50*w, 0.90*h], "p2": [0.92*w, 0.84*h]},

            # İç referans çizgileri: poly mapper'a ek kısıt sağlar.
            "orta_cizgi": {"label": "ORTA CIZGI", "type": "line", "color": (255, 255, 255), "p1": [0.50*w, 0.18*h], "p2": [0.50*w, 0.86*h]},
            "sol_ceza": {"label": "SOL CEZA SAHASI ON CIZGISI", "type": "line", "color": (180, 255, 180), "p1": [0.28*w, 0.34*h], "p2": [0.34*w, 0.72*h]},
            "sag_ceza": {"label": "SAG CEZA SAHASI ON CIZGISI", "type": "line", "color": (140, 255, 140), "p1": [0.74*w, 0.30*h], "p2": [0.70*w, 0.72*h]},

            "sol_kale_agzi": {"label": "SOL KALE / KALE AGZI", "type": "line", "color": (0, 255, 0), "p1": [0.08*w, 0.44*h], "p2": [0.08*w, 0.56*h]},
            "sag_kale_agzi": {"label": "SAG KALE / KALE AGZI", "type": "line", "color": (0, 220, 0), "p1": [0.92*w, 0.44*h], "p2": [0.92*w, 0.56*h]},
            "top": {"label": "Top", "type": "point", "color": (0, 0, 255), "p": [0.50*w, 0.55*h]},
        }

    def set_frame(new_frame: np.ndarray, reset_items: bool = True):
        h, w = new_frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        state["frame"] = new_frame
        state["scale"] = scale
        state["w"] = w
        state["h"] = h
        state["disp_base"] = cv2.resize(new_frame, (int(w * scale), int(h * scale)))
        if reset_items or state.get("items") is None:
            state["items"] = make_defaults(w, h)

    set_frame(frame, reset_items=True)
    drag = {"item": None, "part": None}
    handle_r = 10

    def orig_to_scaled(p):
        s = state["scale"]
        return int(round(p[0] * s)), int(round(p[1] * s))

    def scaled_to_orig(x, y):
        s = state["scale"]
        return x / s, y / s

    def dist2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def put_label(canvas, label, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        th = 2
        (tw, txt_h), base = cv2.getTextSize(label, font, fs, th)
        x = max(2, min(int(x), canvas.shape[1] - tw - 8))
        y = max(txt_h + 8, min(int(y), canvas.shape[0] - 8))
        cv2.rectangle(canvas, (x - 3, y - txt_h - 7), (x + tw + 5, y + base + 3), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x, y), font, fs, color, th, cv2.LINE_AA)

    def draw_handle(canvas, pt, color, middle=False):
        if middle:
            cv2.rectangle(canvas, (pt[0]-handle_r, pt[1]-handle_r), (pt[0]+handle_r, pt[1]+handle_r), (255, 255, 255), -1)
            cv2.rectangle(canvas, (pt[0]-handle_r+3, pt[1]-handle_r+3), (pt[0]+handle_r-3, pt[1]+handle_r-3), color, -1)
            cv2.rectangle(canvas, (pt[0]-handle_r, pt[1]-handle_r), (pt[0]+handle_r, pt[1]+handle_r), (0, 0, 0), 1)
        else:
            cv2.circle(canvas, pt, handle_r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, handle_r - 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, handle_r, (0, 0, 0), 1, cv2.LINE_AA)

    def curve_points_scaled(item, n=50):
        return [orig_to_scaled(quad_bezier(item["p1"], item["pm"], item["p2"], float(t))) for t in np.linspace(0, 1, n)]

    def redraw():
        canvas = state["disp_base"].copy()
        for key in ["sol_tac", "sag_tac", "yukari_korner", "asagi_korner", "orta_cizgi", "sol_ceza", "sag_ceza", "sol_kale_agzi", "sag_kale_agzi", "top"]:
            item = state["items"][key]
            color = tuple(int(c) for c in item["color"])
            if item["type"] == "curve":
                pts = np.array(curve_points_scaled(item), dtype=np.int32)
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
                p1 = orig_to_scaled(item["p1"]); pm = orig_to_scaled(item["pm"]); p2 = orig_to_scaled(item["p2"])
                cv2.line(canvas, p1, pm, (120, 120, 120), 1, cv2.LINE_AA)
                cv2.line(canvas, pm, p2, (120, 120, 120), 1, cv2.LINE_AA)
                draw_handle(canvas, p1, color)
                draw_handle(canvas, pm, color, middle=True)
                draw_handle(canvas, p2, color)
                mid = orig_to_scaled(quad_bezier(item["p1"], item["pm"], item["p2"], 0.5))
                put_label(canvas, item["label"], mid[0] + 8, mid[1] - 10, color)
            elif item["type"] == "line":
                p1 = orig_to_scaled(item["p1"]); p2 = orig_to_scaled(item["p2"])
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
                draw_handle(canvas, p1, color); draw_handle(canvas, p2, color)
                put_label(canvas, item["label"], (p1[0]+p2[0])//2 + 8, (p1[1]+p2[1])//2 - 10, color)
            else:
                p = orig_to_scaled(item["p"])
                cv2.circle(canvas, p, 8, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 12, (255, 255, 255), 1, cv2.LINE_AA)
                put_label(canvas, item["label"], p[0] + 14, p[1] - 8, color)

        mapper_name = str(getattr(args, "field_mapper", "poly2") if args is not None else "poly2")
        info = f"t={current_time:.1f}s | mapper={mapper_name} | N/Space sonraki | P onceki | Enter/S kaydet | R reset"
        cv2.putText(canvas, info, (12, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, info, (12, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
        return canvas

    def load_scene(delta: float):
        nonlocal current_time, drag
        if video_path is None:
            return
        current_time = max(0.0, current_time + delta)
        if duration > 0:
            current_time = min(current_time, max(0.0, duration - 0.05))
        new_frame = read_corrected_frame_at(video_path, current_time, args)
        set_frame(new_frame, reset_items=True)
        drag = {"item": None, "part": None}
        try:
            cv2.resizeWindow(win, state["disp_base"].shape[1], state["disp_base"].shape[0])
        except Exception:
            pass

    def mouse_cb(event, x, y, flags, param):
        nonlocal drag
        items = state["items"]; w = state["w"]; h = state["h"]; s = state["scale"]
        pick_thresh_orig = max(18 / s, 12)
        ox, oy = scaled_to_orig(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            best = None; best_d = pick_thresh_orig ** 2
            for key, item in items.items():
                if item["type"] == "curve":
                    parts = ("p1", "pm", "p2")
                elif item["type"] == "line":
                    parts = ("p1", "p2")
                else:
                    parts = ("p",)
                for part in parts:
                    d = dist2((ox, oy), tuple(item[part]))
                    if d < best_d:
                        best_d = d; best = (key, part)
            if best:
                drag = {"item": best[0], "part": best[1]}
        elif event == cv2.EVENT_MOUSEMOVE and drag["item"] is not None:
            key = drag["item"]; part = drag["part"]
            ox = float(max(0, min(ox, w - 1))); oy = float(max(0, min(oy, h - 1)))
            items[key][part] = [ox, oy]
        elif event == cv2.EVENT_LBUTTONUP:
            drag = {"item": None, "part": None}

    def solve_from_items():
        items = state["items"]

        def item_center(it: dict[str, Any]) -> tuple[float, float]:
            if it["type"] == "curve":
                pts = np.asarray([it["p1"], it["pm"], it["p2"]], dtype=np.float64)
                return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            if it["type"] == "line":
                return float((it["p1"][0] + it["p2"][0]) / 2.0), float((it["p1"][1] + it["p2"][1]) / 2.0)
            p = it.get("p", [0.0, 0.0])
            return float(p[0]), float(p[1])

        def line_item_to_line(it: dict[str, Any]):
            return line_from_two_points(tuple(it["p1"]), tuple(it["p2"]))

        def extended_curve_control(item: dict[str, Any], c1: tuple[float, float], c2: tuple[float, float]) -> list[float]:
            p1 = np.asarray(item["p1"], dtype=np.float64)
            p2 = np.asarray(item["p2"], dtype=np.float64)
            pm = np.asarray(item["pm"], dtype=np.float64)
            c1a = np.asarray(c1, dtype=np.float64)
            c2a = np.asarray(c2, dtype=np.float64)
            old_mid = 0.5 * (p1 + p2)
            new_mid = 0.5 * (c1a + c2a)
            old_len = max(1.0, float(np.linalg.norm(p2 - p1)))
            new_len = max(1.0, float(np.linalg.norm(c2a - c1a)))
            scale = float(getattr(args, "curve_extension_curvature_scale", 1.0) if args is not None else 1.0)
            if bool(getattr(args, "curve_extension_auto_scale", False) if args is not None else False):
                scale *= max(0.5, min(2.0, new_len / old_len))
            dev = pm - old_mid
            ctrl = new_mid + scale * dev
            return [float(ctrl[0]), float(ctrl[1])]

        def sample_extended_curve(item: dict[str, Any], c1: tuple[float, float], c2: tuple[float, float], n: int) -> list[tuple[float, float]]:
            ctrl = extended_curve_control(item, c1, c2)
            return [quad_bezier(c1, ctrl, c2, float(t)) for t in np.linspace(0, 1, n)]

        short_edges = [("sol_tac", items["sol_tac"]), ("sag_tac", items["sag_tac"])]
        long_edges = [("yukari_korner", items["yukari_korner"]), ("asagi_korner", items["asagi_korner"])]
        short_edges_sorted = sorted(short_edges, key=lambda kv: item_center(kv[1])[0])
        long_edges_sorted = sorted(long_edges, key=lambda kv: item_center(kv[1])[1])

        left_key, left_item = short_edges_sorted[0]
        right_key, right_item = short_edges_sorted[-1]
        top_key, top_item = long_edges_sorted[0]
        bottom_key, bottom_item = long_edges_sorted[-1]

        img_left = line_item_to_line(left_item)
        img_right = line_item_to_line(right_item)
        img_top = line_item_to_line(top_item)
        img_bottom = line_item_to_line(bottom_item)

        image_corners = [
            intersection_of_lines(img_left, img_top),
            intersection_of_lines(img_right, img_top),
            intersection_of_lines(img_right, img_bottom),
            intersection_of_lines(img_left, img_bottom),
        ]
        if any(p is None for p in image_corners):
            raise RuntimeError("Ana çizgilerden köşe noktaları çıkarılamadı.")

        tl, trc, br, bl = image_corners
        field_corners = [(0.0, 0.0), (PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH), (0.0, PITCH_WIDTH)]
        H, _ = cv2.findHomography(np.array(image_corners, dtype=np.float32), np.array(field_corners, dtype=np.float32))
        if H is None:
            raise RuntimeError("Homography hesaplanamadı.")

        image_pts: list[tuple[float, float]] = []
        field_pts: list[tuple[float, float]] = []
        samples = max(7, int(getattr(args, "curve_samples", 15) if args is not None else 15))

        # Kullanıcının çizdiği kısa boundary segmentini tam saha kenarı sanma.
        # Önce çizgileri kesişim köşelerine uzat, sonra poly sample'ları o tam kenarlardan üret.
        left_curve = sample_extended_curve(left_item, tl, bl, samples)
        right_curve = sample_extended_curve(right_item, trc, br, samples)
        top_curve = sample_extended_curve(top_item, tl, trc, samples)
        bottom_curve = sample_extended_curve(bottom_item, bl, br, samples)

        for i, tt in enumerate(np.linspace(0, 1, samples)):
            tt = float(tt)
            image_pts.append(left_curve[i])
            field_pts.append((0.0, PITCH_WIDTH * tt))

            image_pts.append(right_curve[i])
            field_pts.append((PITCH_LENGTH, PITCH_WIDTH * tt))

            image_pts.append(top_curve[i])
            field_pts.append((PITCH_LENGTH * tt, 0.0))

            image_pts.append(bottom_curve[i])
            field_pts.append((PITCH_LENGTH * tt, PITCH_WIDTH))

        # İç çizgiler görünen segmenttir; uçları tam taç/kale kesişimi sayılmaz.
        # Homography ile segment üzerindeki yaklaşık y'yi tahmin edip sadece x=orta/ceza constraint'i veriyoruz.
        def add_visible_constant_x_line(item_key: str, field_x_const: float):
            if item_key not in items:
                return
            it = items[item_key]
            for t in np.linspace(0, 1, max(5, samples // 2)):
                tt = float(t)
                ix = (1.0 - tt) * it["p1"][0] + tt * it["p2"][0]
                iy = (1.0 - tt) * it["p1"][1] + tt * it["p2"][1]
                hx, hy = image_to_field_point(H, ix, iy)
                if not np.isfinite(hy):
                    continue
                hy = float(max(0.0, min(PITCH_WIDTH, hy)))
                image_pts.append((float(ix), float(iy)))
                field_pts.append((float(field_x_const), hy))

        add_visible_constant_x_line("orta_cizgi", PITCH_LENGTH / 2.0)
        add_visible_constant_x_line("sol_ceza", PENALTY_DEPTH)
        add_visible_constant_x_line("sag_ceza", PITCH_LENGTH - PENALTY_DEPTH)

        choice = str(getattr(args, "field_mapper", "poly2") if args is not None else "poly2").lower()
        degree = 3 if choice == "poly3" else 2
        poly_mapper = fit_poly_mapper(image_pts, field_pts, state["frame"].shape, degree=degree)

        rmse = float(poly_mapper.get("fit_rmse", 0.0))
        max_err = float(poly_mapper.get("fit_max_err", 0.0))
        max_rmse = float(getattr(args, "calib_max_rmse", 6.0) if args is not None else 6.0)
        max_maxerr = float(getattr(args, "calib_max_err", 18.0) if args is not None else 18.0)
        bad_poly = rmse > max_rmse or max_err > max_maxerr
        poly_mapper["bad_calibration"] = bool(bad_poly)
        poly_mapper["quality_gate_rmse"] = max_rmse
        poly_mapper["quality_gate_max_err"] = max_maxerr
        if bad_poly:
            print(f"UYARI: Poly calibration kötü görünüyor. rmse={rmse:.2f}m max={max_err:.2f}m. "
                  f"{'Homography fallback kullanılacak.' if getattr(args, 'auto_fallback_bad_calib', True) else 'Yine de poly kullanılacak.'}")

        used_pairs = []
        names = [f"{left_key}+{top_key}", f"{right_key}+{top_key}", f"{right_key}+{bottom_key}", f"{left_key}+{bottom_key}"]
        for name, ip, fp in zip(names, image_corners, field_corners):
            used_pairs.append({"image_point": [float(ip[0]), float(ip[1])], "field_point": [float(fp[0]), float(fp[1])], "from_lines": [name]})

        inferred_roles = {
            "left_short_edge": left_key,
            "right_short_edge": right_key,
            "top_long_edge": top_key,
            "bottom_long_edge": bottom_key,
            "poly_bad_calibration": bool(bad_poly),
        }
        return H, poly_mapper, used_pairs, inferred_roles

    win = "Kalibrasyon - Curve Aware"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, state["disp_base"].shape[1], state["disp_base"].shape[0])
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Kalibrasyon iptal edildi.")
        if key in (ord("n"), ord("N"), 32, 83):
            load_scene(step_sec)
        elif key in (ord("p"), ord("P"), 81):
            load_scene(-step_sec)
        elif key in (ord("r"), ord("R")):
            state["items"] = make_defaults(state["w"], state["h"]); drag = {"item": None, "part": None}
        elif key in (ord("s"), ord("S"), 13, 10):
            try:
                H, poly_mapper, used_pairs, inferred_roles = solve_from_items()
                break
            except Exception as e:
                print(f"Kalibrasyon çözülemedi: {e}")

    cv2.destroyWindow(win)

    items = state["items"]
    choice = str(getattr(args, "field_mapper", "poly2") if args is not None else "poly2").lower()
    auto_fallback = bool(getattr(args, "auto_fallback_bad_calib", True) if args is not None else True)
    if choice in ("poly2", "poly3") and auto_fallback and bool(poly_mapper.get("bad_calibration", False)):
        mapping_type = "homography"
    else:
        mapping_type = choice if choice in ("poly2", "poly3") else "homography"
    calib = {
        "mode": "curve_aware_drag_lines_v12",
        "mapping_type": mapping_type,
        "calibration_time_sec": current_time,
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "coordinate_convention": "x=0 sol kale cizgisi, x=L sag kale cizgisi, y=0 ust tac cizgisi, y=W alt tac cizgisi",
        "penalty_depth": PENALTY_DEPTH,
        "penalty_width": PENALTY_WIDTH,
        "goalbox_depth": GOALBOX_DEPTH,
        "goalbox_width": GOALBOX_WIDTH,
        "image_to_field_H": H.tolist(),
        "image_to_field_poly": poly_mapper,
        "ui_items": items,
        "goal_guides": {"sol_kale_agzi": {"p1": items["sol_kale_agzi"]["p1"], "p2": items["sol_kale_agzi"]["p2"]}, "sag_kale_agzi": {"p1": items["sag_kale_agzi"]["p1"], "p2": items["sag_kale_agzi"]["p2"]}},
        "inferred_roles": inferred_roles,
        "ball_marker": items["top"]["p"],
        "intersection_points_used": used_pairs,
        "lens_correction": {
            "undistort_k1": float(getattr(args, "undistort_k1", 0.0) or 0.0) if args is not None else 0.0,
            "undistort_k2": float(getattr(args, "undistort_k2", 0.0) or 0.0) if args is not None else 0.0,
            "undistort_k3": float(getattr(args, "undistort_k3", 0.0) or 0.0) if args is not None else 0.0,
            "undistort_focal": float(getattr(args, "undistort_focal", 0.0) or 0.0) if args is not None else 0.0,
        },
        "created_at_unix": time.time(),
    }
    save_calibration_debug(state["frame"], calib, out_dir)
    return calib

def four_point_calibration(frame: np.ndarray, out_dir: Path) -> dict[str, Any]:
    """Fallback: oyun alanının dört köşesi görünüyorsa kullanılır."""
    max_w, max_h = 1280, 760
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    points: list[tuple[float, float]] = []
    names = ["SOL-UST", "SAG-UST", "SAG-ALT", "SOL-ALT"]

    def redraw():
        canvas = disp.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(canvas, "4 KÖSE KALIBRASYON", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        msg = f"Siradaki: {names[len(points)] if len(points) < 4 else 'TAMAM'} | Sag tik/u geri | Enter onay"
        cv2.putText(canvas, msg, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, p in enumerate(points):
            x, y = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(canvas, (x, y), 6, (0, 180, 255), -1)
            cv2.putText(canvas, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
        if len(points) >= 2:
            pts = np.array([[int(x * scale), int(y * scale)] for x, y in points], np.int32)
            cv2.polylines(canvas, [pts], len(points) == 4, (0, 180, 255), 2)
        return canvas

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "Kalibrasyon - 4 kose"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Kalibrasyon iptal edildi.")
        if key in (ord("u"), ord("z")) and points:
            points.pop()
        if key in (13, 10) and len(points) == 4:
            break
    cv2.destroyWindow(win)

    image_pts = np.float32(points)
    field_pts = np.float32([[0, 0], [PITCH_LENGTH, 0], [PITCH_LENGTH, PITCH_WIDTH], [0, PITCH_WIDTH]])
    H, _ = cv2.findHomography(image_pts, field_pts)
    if H is None:
        raise RuntimeError("4 nokta homography hesaplanamadı.")
    calib = {
        "mode": "four_points",
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "coordinate_convention": "x=0 sol kale cizgisi, x=L sag kale cizgisi, y=0 ust tac cizgisi, y=W alt tac cizgisi",
        "penalty_depth": PENALTY_DEPTH,
        "penalty_width": PENALTY_WIDTH,
        "goalbox_depth": GOALBOX_DEPTH,
        "goalbox_width": GOALBOX_WIDTH,
        "image_to_field_H": H.tolist(),
        "clicked_points": points,
        "created_at_unix": time.time(),
    }
    save_calibration_debug(frame, calib, out_dir)
    return calib


def save_calibration_debug(frame: np.ndarray, calib: dict[str, Any], out_dir: Path):
    debug = frame.copy()
    if calib["mode"] == "line_based":
        for ln in calib.get("drawn_lines", []):
            ld = LINE_DEFS[ln["line_key"]]
            draw_line_segment(debug, tuple(ln["p1"]), tuple(ln["p2"]), ld.color_bgr, ld.key)
        for p in calib.get("intersection_points_used", []):
            ix, iy = p["image_point"]
            cv2.circle(debug, (int(ix), int(iy)), 5, (0, 0, 255), -1)
    elif str(calib["mode"]).startswith("simple_drag_lines") or str(calib["mode"]).startswith("curve_aware"):
        for key, item in calib.get("ui_items", {}).items():
            color = tuple(int(c) for c in item.get("color", (0, 255, 255)))
            label = item.get("label", key)
            if item.get("type") == "curve":
                pts = [quad_bezier(item["p1"], item["pm"], item["p2"], float(t)) for t in np.linspace(0, 1, 50)]
                pts_i = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
                cv2.polylines(debug, [pts_i], False, color, 2, cv2.LINE_AA)
                mid = quad_bezier(item["p1"], item["pm"], item["p2"], 0.5)
                cv2.putText(debug, label, (int(mid[0]) + 10, int(mid[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            elif item.get("type") == "line":
                draw_line_segment(debug, tuple(item["p1"]), tuple(item["p2"]), color, label)
            elif item.get("type") == "point":
                p = item.get("p", [0, 0])
                cv2.circle(debug, (int(p[0]), int(p[1])), 8, color, -1)
                cv2.putText(debug, label, (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        for p in calib.get("intersection_points_used", []):
            ix, iy = p["image_point"]
            cv2.circle(debug, (int(ix), int(iy)), 5, (0, 0, 255), -1)
    else:
        for p in calib.get("clicked_points", []):
            cv2.circle(debug, (int(p[0]), int(p[1])), 6, (0, 180, 255), -1)

    mapping = calib.get("mapping_type", "homography")
    poly = calib.get("image_to_field_poly") or {}
    msg = f"mapping={mapping}"
    if poly:
        msg += f" rmse={poly.get('fit_rmse', 0):.3f} max={poly.get('fit_max_err', 0):.3f}"
    cv2.putText(debug, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(debug, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "calibration_debug.jpg"), debug)

def load_or_create_calibration(video_path: Path, out_dir: Path, args) -> dict[str, Any]:
    if args.calibration:
        with open(args.calibration, "r", encoding="utf-8") as f:
            return json.load(f)

    calib_path = out_dir / "calibration.json"
    if calib_path.exists() and not args.force_calibration:
        print(f"Var olan kalibrasyon kullanılacak: {calib_path}")
        with open(calib_path, "r", encoding="utf-8") as f:
            return json.load(f)

    frame = read_corrected_frame_at(video_path, args.calibration_time, args)
    if args.calibration_mode == "four_points":
        calib = four_point_calibration(frame, out_dir)
    else:
        calib = line_based_calibration(frame, out_dir, video_path=video_path, args=args)

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"Kalibrasyon kaydedildi: {calib_path}")
    return calib


def quad_bezier(p1, pm, p2, t: float) -> tuple[float, float]:
    u = 1.0 - t
    x = u * u * p1[0] + 2.0 * u * t * pm[0] + t * t * p2[0]
    y = u * u * p1[1] + 2.0 * u * t * pm[1] + t * t * p2[1]
    return float(x), float(y)


def poly_feature_vector(x: float, y: float, w: float, h: float, degree: int = 2) -> np.ndarray:
    xn = (float(x) / max(1.0, float(w)) - 0.5) * 2.0
    yn = (float(y) / max(1.0, float(h)) - 0.5) * 2.0
    feats = [1.0, xn, yn, xn * yn, xn * xn, yn * yn]
    if degree >= 3:
        feats += [xn ** 3, yn ** 3, (xn ** 2) * yn, xn * (yn ** 2)]
    return np.asarray(feats, dtype=np.float64)


def fit_poly_mapper(image_points: list[tuple[float, float]], field_points: list[tuple[float, float]], image_shape, degree: int = 2) -> dict[str, Any]:
    h, w = image_shape[:2]
    A = np.vstack([poly_feature_vector(x, y, w, h, degree) for x, y in image_points])
    bx = np.asarray([p[0] for p in field_points], dtype=np.float64)
    by = np.asarray([p[1] for p in field_points], dtype=np.float64)
    coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)
    pred_x = A @ coef_x
    pred_y = A @ coef_y
    err = np.sqrt((pred_x - bx) ** 2 + (pred_y - by) ** 2)
    return {
        "type": f"poly{degree}",
        "degree": int(degree),
        "image_w": int(w),
        "image_h": int(h),
        "coef_x": coef_x.tolist(),
        "coef_y": coef_y.tolist(),
        "fit_rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
        "fit_max_err": float(np.max(err)) if len(err) else 0.0,
        "num_points": int(len(image_points)),
    }


def build_field_mapper(calib: dict[str, Any]) -> Any:
    if calib.get("mapping_type", "").startswith("poly") and calib.get("image_to_field_poly"):
        return calib["image_to_field_poly"]
    return np.asarray(calib["image_to_field_H"], dtype=np.float64)


def image_to_field_point(mapper: Any, x: float, y: float) -> tuple[float, float]:
    if isinstance(mapper, dict) and str(mapper.get("type", "")).startswith("poly"):
        degree = int(mapper.get("degree", 2))
        w = float(mapper.get("image_w", 1))
        h = float(mapper.get("image_h", 1))
        feats = poly_feature_vector(x, y, w, h, degree)
        fx = float(feats @ np.asarray(mapper["coef_x"], dtype=np.float64))
        fy = float(feats @ np.asarray(mapper["coef_y"], dtype=np.float64))
        return fx, fy

    pts = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, np.asarray(mapper, dtype=np.float64))[0, 0]
    return float(out[0]), float(out[1])


def bbox_clip(x1, y1, x2, y2, shape):
    h, w = shape[:2]
    x1 = max(0, min(int(round(x1)), w - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def hsv_hist(crop: np.ndarray, bins_h: int = 16, bins_s: int = 8) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Çok karanlık/doygun olmayan pikselleri zayıflat; ama tamamen maskeleme yapma.
    hist = cv2.calcHist([hsv], [0, 1], None, [bins_h, bins_s], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def hist_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.45
    try:
        d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        if math.isnan(d):
            return 0.45
        return float(max(0.0, min(1.0, d)))
    except Exception:
        return 0.45


def extract_appearance(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    clipped = bbox_clip(*bbox, frame.shape)
    empty = {
        "upper_hist": None,
        "lower_hist": None,
        "mid_hist": None,
        "shoe_hist": None,
        "head_hist": None,
        "team_hint": "unknown",
        "orange_ratio": 0.0,
        "dark_ratio": 0.0,
        "skin_ratio": 0.0,
        "head_dark_ratio": 0.0,
    }
    if clipped is None:
        return empty
    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return empty

    h = crop.shape[0]
    upper = crop[int(0.14 * h): int(0.42 * h), :]
    mid = crop[int(0.40 * h): int(0.68 * h), :]
    lower = crop[int(0.58 * h): int(0.86 * h), :]
    shoes = crop[int(0.84 * h): int(0.98 * h), :]
    head = crop[int(0.00 * h): int(0.20 * h), :]

    upper_hist = hsv_hist(upper)
    lower_hist = hsv_hist(lower)
    mid_hist = hsv_hist(mid)
    shoe_hist = hsv_hist(shoes)
    head_hist = hsv_hist(head)

    hsv = cv2.cvtColor(upper if upper.size else crop, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    orange_mask = (((Hc >= 3) & (Hc <= 33)) | ((Hc >= 170) & (Hc <= 179))) & (Sc > 70) & (Vc > 70)
    dark_mask = (Vc < 85) & (Sc < 160)
    total = max(1, Hc.size)
    orange_ratio = float(np.count_nonzero(orange_mask) / total)
    dark_ratio = float(np.count_nonzero(dark_mask) / total)

    if head.size:
        head_hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        Hh, Sh, Vh = head_hsv[:, :, 0], head_hsv[:, :, 1], head_hsv[:, :, 2]
        skin_mask = (
            (((Hh >= 0) & (Hh <= 25)) | ((Hh >= 160) & (Hh <= 179)))
            & (Sh >= 25) & (Sh <= 180)
            & (Vh >= 45)
        )
        skin_ratio = float(np.count_nonzero(skin_mask) / max(1, Hh.size))
        head_dark_mask = (Vh < 70) & (Sh < 120)
        head_dark_ratio = float(np.count_nonzero(head_dark_mask) / max(1, Hh.size))
    else:
        skin_ratio = 0.0
        head_dark_ratio = 0.0

    if orange_ratio > 0.10:
        team = "orange"
    elif dark_ratio > 0.30:
        team = "dark"
    else:
        team = "unknown"

    return {
        "upper_hist": upper_hist,
        "lower_hist": lower_hist,
        "mid_hist": mid_hist,
        "shoe_hist": shoe_hist,
        "head_hist": head_hist,
        "team_hint": team,
        "orange_ratio": orange_ratio,
        "dark_ratio": dark_ratio,
        "skin_ratio": skin_ratio,
        "head_dark_ratio": head_dark_ratio,
    }


@dataclass
class TrackState:
    stable_id: int
    cls_name: str
    x: float
    y: float
    image_x: float
    image_y: float
    last_seen_idx: int
    last_seen_time: float
    vx: float = 0.0
    vy: float = 0.0
    raw_id: int | None = None
    team_votes: dict[str, int] = field(default_factory=dict)
    upper_hist: np.ndarray | None = None
    lower_hist: np.ndarray | None = None
    mid_hist: np.ndarray | None = None
    shoe_hist: np.ndarray | None = None
    head_hist: np.ndarray | None = None
    skin_ratio: float = 0.0
    head_dark_ratio: float = 0.0
    bbox_h: float = 0.0
    bbox_w: float = 0.0
    hits: int = 1

    @property
    def team(self) -> str:
        if not self.team_votes:
            return "unknown"
        return max(self.team_votes.items(), key=lambda kv: kv[1])[0]

    def predict(self, t: float) -> tuple[float, float]:
        dt = max(0.0, t - self.last_seen_time)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def update_with_det(self, det: dict[str, Any], sample_idx: int, time_sec: float):
        dt = max(1e-6, time_sec - self.last_seen_time)
        new_vx = (det["field_x"] - self.x) / dt
        new_vy = (det["field_y"] - self.y) / dt
        # Aşırı sıçramalarda hız filtresini yumuşat.
        speed = math.hypot(new_vx, new_vy)
        if speed < 14.0:
            self.vx = 0.65 * self.vx + 0.35 * new_vx
            self.vy = 0.65 * self.vy + 0.35 * new_vy
        self.x = det["field_x"]
        self.y = det["field_y"]
        self.image_x = det["image_x"]
        self.image_y = det["image_y"]
        self.last_seen_idx = sample_idx
        self.last_seen_time = time_sec
        self.raw_id = det.get("raw_id")
        team = det.get("team_hint", "unknown")
        self.team_votes[team] = self.team_votes.get(team, 0) + 1
        self.bbox_w = 0.85 * self.bbox_w + 0.15 * det.get("bbox_w", self.bbox_w)
        self.bbox_h = 0.85 * self.bbox_h + 0.15 * det.get("bbox_h", self.bbox_h)
        uh = det.get("upper_hist")
        lh = det.get("lower_hist")
        mh = det.get("mid_hist")
        sh = det.get("shoe_hist")
        hh = det.get("head_hist")
        if uh is not None:
            if self.upper_hist is None:
                self.upper_hist = uh.copy()
            else:
                self.upper_hist = cv2.normalize(0.85 * self.upper_hist + 0.15 * uh, None).flatten().astype(np.float32)
        if lh is not None:
            if self.lower_hist is None:
                self.lower_hist = lh.copy()
            else:
                self.lower_hist = cv2.normalize(0.85 * self.lower_hist + 0.15 * lh, None).flatten().astype(np.float32)
        if mh is not None:
            if self.mid_hist is None:
                self.mid_hist = mh.copy()
            else:
                self.mid_hist = cv2.normalize(0.85 * self.mid_hist + 0.15 * mh, None).flatten().astype(np.float32)
        if sh is not None:
            if self.shoe_hist is None:
                self.shoe_hist = sh.copy()
            else:
                self.shoe_hist = cv2.normalize(0.80 * self.shoe_hist + 0.20 * sh, None).flatten().astype(np.float32)
        if hh is not None:
            if self.head_hist is None:
                self.head_hist = hh.copy()
            else:
                self.head_hist = cv2.normalize(0.85 * self.head_hist + 0.15 * hh, None).flatten().astype(np.float32)
        self.skin_ratio = 0.85 * float(self.skin_ratio) + 0.15 * float(det.get("skin_ratio", self.skin_ratio))
        self.head_dark_ratio = 0.85 * float(self.head_dark_ratio) + 0.15 * float(det.get("head_dark_ratio", self.head_dark_ratio))
        self.hits += 1


class AppearanceStableTracker:
    def __init__(
        self,
        max_lost_sec: float = 5.0,
        base_search_radius: float = 4.0,
        radius_growth_per_sec: float = 3.0,
        max_tracks: int = 0,
        object_name: str = "person",
    ):
        self.max_lost_sec = max_lost_sec
        self.base_search_radius = base_search_radius
        self.radius_growth_per_sec = radius_growth_per_sec
        self.max_tracks = max_tracks
        self.object_name = object_name
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}
        self.raw_to_stable: dict[int, int] = {}

    def _new_track(self, det: dict[str, Any], sample_idx: int, time_sec: float) -> int | None:
        if self.max_tracks and len(self.tracks) >= self.max_tracks:
            return None
        sid = self.next_id
        self.next_id += 1
        votes = {det.get("team_hint", "unknown"): 1}
        tr = TrackState(
            stable_id=sid,
            cls_name=det.get("cls_name", self.object_name),
            x=det["field_x"],
            y=det["field_y"],
            image_x=det["image_x"],
            image_y=det["image_y"],
            last_seen_idx=sample_idx,
            last_seen_time=time_sec,
            raw_id=det.get("raw_id"),
            team_votes=votes,
            upper_hist=det.get("upper_hist"),
            lower_hist=det.get("lower_hist"),
            mid_hist=det.get("mid_hist"),
            shoe_hist=det.get("shoe_hist"),
            head_hist=det.get("head_hist"),
            skin_ratio=float(det.get("skin_ratio", 0.0)),
            head_dark_ratio=float(det.get("head_dark_ratio", 0.0)),
            bbox_w=det.get("bbox_w", 0.0),
            bbox_h=det.get("bbox_h", 0.0),
        )
        self.tracks[sid] = tr
        rid = det.get("raw_id")
        if rid is not None:
            self.raw_to_stable[int(rid)] = sid
        return sid

    def _cost(self, tr: TrackState, det: dict[str, Any], time_sec: float) -> tuple[float, dict[str, float]]:
        dt_lost = max(0.0, time_sec - tr.last_seen_time)
        if dt_lost > self.max_lost_sec:
            return 999.0, {}
        px, py = tr.predict(time_sec)
        pred_dist = math.hypot(det["field_x"] - px, det["field_y"] - py)
        last_dist = math.hypot(det["field_x"] - tr.x, det["field_y"] - tr.y)
        speed = math.hypot(tr.vx, tr.vy)
        radius = self.base_search_radius + self.radius_growth_per_sec * dt_lost + 0.35 * speed * dt_lost
        radius = max(radius, self.base_search_radius)
        if pred_dist > radius:
            return 999.0, {"pred_dist": pred_dist, "radius": radius}

        app_u = hist_distance(tr.upper_hist, det.get("upper_hist"))
        app_l = hist_distance(tr.lower_hist, det.get("lower_hist"))
        app_m = hist_distance(tr.mid_hist, det.get("mid_hist"))
        app_shoe = hist_distance(tr.shoe_hist, det.get("shoe_hist"))
        app_head = hist_distance(tr.head_hist, det.get("head_hist"))
        skin_pen = min(1.0, abs(float(tr.skin_ratio) - float(det.get("skin_ratio", 0.0))) * 2.0)
        head_dark_pen = min(1.0, abs(float(tr.head_dark_ratio) - float(det.get("head_dark_ratio", 0.0))) * 2.0)
        app = 0.42 * app_u + 0.20 * app_m + 0.18 * app_l + 0.10 * app_shoe + 0.06 * app_head + 0.02 * skin_pen + 0.02 * head_dark_pen
        team_pen = 0.0
        det_team = det.get("team_hint", "unknown")
        if tr.team != "unknown" and det_team != "unknown" and tr.team != det_team:
            team_pen = 1.0

        bbox_pen = 0.0
        if tr.bbox_h > 1 and det.get("bbox_h", 0) > 1:
            ratio = det["bbox_h"] / max(1.0, tr.bbox_h)
            bbox_pen = min(1.0, abs(math.log(max(0.1, min(10.0, ratio)))))

        # Maliyet 0 iyi, 1 civarı eşik.
        cost = (
            0.42 * min(1.5, pred_dist / radius)
            + 0.18 * min(1.5, last_dist / (radius + 1e-6))
            + 0.25 * app
            + 0.10 * team_pen
            + 0.05 * bbox_pen
        )
        return float(cost), {
            "pred_dist": pred_dist,
            "last_dist": last_dist,
            "radius": radius,
            "appearance_dist": app,
            "skin_penalty": skin_pen,
            "head_dark_penalty": head_dark_pen,
            "team_penalty": team_pen,
            "bbox_penalty": bbox_pen,
        }

    def update(self, detections: list[dict[str, Any]], sample_idx: int, time_sec: float) -> list[dict[str, Any]]:
        if not detections:
            return []
        assigned: list[dict[str, Any]] = []
        used_sids: set[int] = set()
        used_dets: set[int] = set()

        # 1) Ham tracker ID devam ediyorsa öncelik ver; ama cost çok kötüyse bağlama.
        for di, det in enumerate(detections):
            rid = det.get("raw_id")
            if rid is None:
                continue
            sid = self.raw_to_stable.get(int(rid))
            if sid is None or sid in used_sids or sid not in self.tracks:
                continue
            tr = self.tracks[sid]
            cost, parts = self._cost(tr, det, time_sec)
            if cost <= 0.95:
                tr.update_with_det(det, sample_idx, time_sec)
                det.update({
                    "stable_id": sid,
                    "team": tr.team,
                    "match_type": "raw_id",
                    "reid_conf": float(max(0.0, 1.0 - cost)),
                    "position_conf": float(max(0.05, min(1.0, det.get("det_conf", 0.5) * (1.0 - 0.25 * cost)))),
                    **{f"cost_{k}": v for k, v in parts.items()},
                })
                assigned.append(det)
                used_sids.add(sid)
                used_dets.add(di)

        # 2) Kalanları lost/active tracklere appearance + hareket ile bağla.
        candidates = []
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            for sid, tr in self.tracks.items():
                if sid in used_sids:
                    continue
                cost, parts = self._cost(tr, det, time_sec)
                if cost < 0.88:
                    candidates.append((cost, di, sid, parts))
        candidates.sort(key=lambda x: x[0])
        for cost, di, sid, parts in candidates:
            if di in used_dets or sid in used_sids:
                continue
            det = detections[di]
            tr = self.tracks[sid]
            tr.update_with_det(det, sample_idx, time_sec)
            rid = det.get("raw_id")
            if rid is not None:
                self.raw_to_stable[int(rid)] = sid
            det.update({
                "stable_id": sid,
                "team": tr.team,
                "match_type": "reid",
                "reid_conf": float(max(0.0, 1.0 - cost)),
                "position_conf": float(max(0.05, min(1.0, det.get("det_conf", 0.5) * (1.0 - 0.35 * cost)))),
                **{f"cost_{k}": v for k, v in parts.items()},
            })
            assigned.append(det)
            used_dets.add(di)
            used_sids.add(sid)

        # 3) Hala eşleşmeyenler yeni stable_id alır.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            sid = self._new_track(det, sample_idx, time_sec)
            if sid is None:
                # Max track sınırı yüzünden ignore.
                continue
            tr = self.tracks[sid]
            det.update({
                "stable_id": sid,
                "team": tr.team,
                "match_type": "new",
                "reid_conf": 1.0,
                "position_conf": float(det.get("det_conf", 0.5)),
                "cost_pred_dist": 0.0,
                "cost_last_dist": 0.0,
                "cost_radius": 0.0,
                "cost_appearance_dist": 0.0,
                "cost_team_penalty": 0.0,
                "cost_bbox_penalty": 0.0,
            })
            assigned.append(det)
        return assigned


def class_name_from_result(model: Any, cls_id: int) -> str:
    try:
        names = model.names
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        return str(names[cls_id])
    except Exception:
        return str(cls_id)



def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(1e-9, area_a + area_b - inter))


def make_slices_for_roi(
    frame_w: int,
    frame_h: int,
    roi: tuple[int, int, int, int],
    slice_size: int,
    overlap: float,
    max_slices: int = 999,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(x1 + 1, min(int(x2), frame_w))
    y2 = max(y1 + 1, min(int(y2), frame_h))
    slice_size = max(128, int(slice_size))
    step = max(64, int(slice_size * (1.0 - float(overlap))))

    xs = list(range(x1, max(x1 + 1, x2 - slice_size + 1), step))
    ys = list(range(y1, max(y1 + 1, y2 - slice_size + 1), step))
    if not xs or xs[-1] + slice_size < x2:
        xs.append(max(x1, x2 - slice_size))
    if not ys or ys[-1] + slice_size < y2:
        ys.append(max(y1, y2 - slice_size))

    slices = []
    seen = set()
    for yy in ys:
        for xx in xs:
            sx1 = max(0, min(xx, frame_w - 1))
            sy1 = max(0, min(yy, frame_h - 1))
            sx2 = min(frame_w, sx1 + slice_size)
            sy2 = min(frame_h, sy1 + slice_size)
            # Eğer ROI slice'tan küçükse alanı ROI ile sınırla.
            sx1 = max(x1, sx1)
            sy1 = max(y1, sy1)
            sx2 = min(x2, sx2)
            sy2 = min(y2, sy2)
            if sx2 - sx1 < 80 or sy2 - sy1 < 80:
                continue
            key = (sx1, sy1, sx2, sy2)
            if key in seen:
                continue
            seen.add(key)
            slices.append(key)
            if len(slices) >= max_slices:
                return slices
    return slices


def result_to_detections(
    model: Any,
    result: Any,
    frame: np.ndarray,
    mapper: Any,
    args,
    det_source: str = "full",
    offset_xy: tuple[int, int] = (0, 0),
    slice_id: str = "",
    conf_override: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    persons: list[dict[str, Any]] = []
    balls: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return persons, balls

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None and hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf) if boxes.conf is not None else np.ones(len(xyxy))
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None and hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls, dtype=int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes.id, "cpu") else np.asarray(boxes.id, dtype=int)

    ox, oy = offset_xy
    min_conf = float(args.conf if conf_override is None else conf_override)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 += ox
        x2 += ox
        y1 += oy
        y2 += oy
        conf = float(confs[i])
        cls_id = int(clss[i])
        cls_name = class_name_from_result(model, cls_id).lower()
        raw_id = int(ids[i]) if ids is not None and i < len(ids) else None
        if conf < min_conf:
            continue

        is_person = cls_id == 0 or cls_name == "person"
        is_ball = cls_id == 32 or "ball" in cls_name
        if not is_person and not is_ball:
            continue

        # Top için yanlış pozitif azaltma: çok uzun/çok büyük bbox'u zayıflat.
        if is_ball:
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            ratio = max(bw / bh, bh / bw)
            if ratio > float(getattr(args, "ball_max_aspect", 2.2)):
                continue
            if max(bw, bh) > float(getattr(args, "ball_max_px", 80.0)):
                continue

        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        fx, fy = image_to_field_point(mapper, foot_x, foot_y)
        if not (-15 <= fx <= PITCH_LENGTH + 15 and -15 <= fy <= PITCH_WIDTH + 15):
            continue

        app = extract_appearance(frame, (x1, y1, x2, y2)) if is_person else {
            "upper_hist": None,
            "lower_hist": None,
            "team_hint": "ball",
            "orange_ratio": 0.0,
            "dark_ratio": 0.0,
        }
        det = {
            "raw_id": raw_id,
            "cls_id": cls_id,
            "cls_name": "person" if is_person else "ball",
            "det_conf": conf,
            "image_x": foot_x,
            "image_y": foot_y,
            "field_x": fx,
            "field_y": fy,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
            "upper_hist": app["upper_hist"],
            "lower_hist": app["lower_hist"],
            "team_hint": app["team_hint"],
            "orange_ratio": app["orange_ratio"],
            "dark_ratio": app["dark_ratio"],
            "mid_hist": app.get("mid_hist"),
            "shoe_hist": app.get("shoe_hist"),
            "head_hist": app.get("head_hist"),
            "skin_ratio": app.get("skin_ratio", 0.0),
            "head_dark_ratio": app.get("head_dark_ratio", 0.0),
            "det_source": det_source,
            "slice_id": slice_id,
            "merge_count": 1,
        }
        if is_person:
            persons.append(det)
        else:
            balls.append(det)
    return persons, balls


def merge_detections(det_list: list[dict[str, Any]], iou_thr: float = 0.45, foot_thr_px: float = 28.0) -> list[dict[str, Any]]:
    """
    Full-frame + slice duplicate temizliği.
    IoU + ayak noktası yakınlığı ile aynı objeyi birleştirir.
    """
    if not det_list:
        return []
    dets = sorted(det_list, key=lambda d: float(d.get("det_conf", 0.0)), reverse=True)
    merged: list[dict[str, Any]] = []

    for det in dets:
        box = (det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"])
        foot = (det["image_x"], det["image_y"])
        matched = False
        for m in merged:
            if det.get("cls_name") != m.get("cls_name"):
                continue
            mbox = (m["bbox_x1"], m["bbox_y1"], m["bbox_x2"], m["bbox_y2"])
            mfoot = (m["image_x"], m["image_y"])
            iou = bbox_iou(box, mbox)
            fdist = math.hypot(foot[0] - mfoot[0], foot[1] - mfoot[1])
            if iou >= iou_thr or fdist <= foot_thr_px:
                # Confidence yüksek olan temel kalsın; bbox'u weighted average ile biraz iyileştir.
                c1 = float(m.get("det_conf", 0.0))
                c2 = float(det.get("det_conf", 0.0))
                if c2 > c1:
                    keep_raw = {k: m.get(k) for k in ("raw_id", "match_type") if k in m}
                    m.update(det)
                    for k, v in keep_raw.items():
                        # Full tracker raw_id varsa koru.
                        if v is not None and m.get(k) is None:
                            m[k] = v
                w1 = max(1e-6, c1)
                w2 = max(1e-6, c2)
                denom = w1 + w2
                for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "image_x", "image_y", "field_x", "field_y"):
                    try:
                        m[k] = (float(m[k]) * w1 + float(det[k]) * w2) / denom
                    except Exception:
                        pass
                m["bbox_w"] = float(m["bbox_x2"] - m["bbox_x1"])
                m["bbox_h"] = float(m["bbox_y2"] - m["bbox_y1"])
                m["det_conf"] = max(c1, c2)
                m["merge_count"] = int(m.get("merge_count", 1)) + int(det.get("merge_count", 1))
                srcs = set(str(m.get("det_source", "")).split("+")) | set(str(det.get("det_source", "")).split("+"))
                m["det_source"] = "+".join(sorted([s for s in srcs if s]))
                if det.get("slice_id"):
                    old = str(m.get("slice_id", ""))
                    m["slice_id"] = (old + ";" if old else "") + str(det.get("slice_id"))
                matched = True
                break
        if not matched:
            merged.append(dict(det))
    return merged


def run_sliced_inference(
    model: Any,
    frame: np.ndarray,
    mapper: Any,
    args,
    full_persons: list[dict[str, Any]],
    full_balls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Mini-SAHi yaklaşımı: SAHI paketine bağımlı olmadan seçili bölgelerde tiled inference.
    """
    h, w = frame.shape[:2]
    rois: list[tuple[str, tuple[int, int, int, int], int]] = []

    regions = set([r.strip().lower() for r in str(getattr(args, "sahi_regions", "far,ball")).split(",") if r.strip()])
    base_size = int(getattr(args, "sahi_slice_size", 640))
    ball_size = int(getattr(args, "sahi_ball_slice_size", 384))

    if "all" in regions:
        rois.append(("all", (0, 0, w, h), base_size))
    if "far" in regions:
        # Üst/uzak saha kısmı genelde küçük oyuncu problemi yaratıyor.
        rois.append(("far", (0, 0, w, int(h * float(getattr(args, "sahi_far_ymax", 0.58)))), base_size))
    if "right_box" in regions or "box" in regions:
        rois.append(("right_box", (int(w * 0.55), int(h * 0.12), w, h), base_size))
    if "left_box" in regions or "box" in regions:
        rois.append(("left_box", (0, int(h * 0.05), int(w * 0.45), int(h * 0.60)), base_size))
    if "ball" in regions and (not full_balls or bool(getattr(args, "sahi_ball_always", False))):
        # Top kayıpsa tüm sahada küçük slice; pahalı olmasın diye max_slices limitli.
        rois.append(("ball_search", (0, 0, w, h), ball_size))

    # Duplicate ROI temizliği
    unique = []
    seen = set()
    for name, roi, ssize in rois:
        key = (name, roi, ssize)
        if key not in seen:
            seen.add(key)
            unique.append((name, roi, ssize))
    rois = unique

    all_persons = list(full_persons)
    all_balls = list(full_balls)
    max_slices_total = int(getattr(args, "sahi_max_slices", 24))
    used_slices = 0
    conf = float(getattr(args, "sahi_conf", 0.0) or getattr(args, "conf", 0.18))
    overlap = float(getattr(args, "sahi_overlap", 0.25))

    for region_name, roi, ssize in rois:
        remaining = max(0, max_slices_total - used_slices)
        if remaining <= 0:
            break
        slices = make_slices_for_roi(w, h, roi, ssize, overlap, max_slices=remaining)
        for si, (x1, y1, x2, y2) in enumerate(slices):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                res_list = model.predict(
                    crop,
                    conf=conf,
                    iou=float(getattr(args, "iou", 0.45)),
                    classes=[0, 32],
                    verbose=False,
                )
            except TypeError:
                res_list = model.predict(crop, conf=conf, verbose=False)
            if not res_list:
                continue
            sid = f"{region_name}_{used_slices:03d}_{x1}_{y1}_{x2}_{y2}"
            ps, bs = result_to_detections(
                model,
                res_list[0],
                frame,
                mapper,
                args,
                det_source="slice",
                offset_xy=(x1, y1),
                slice_id=sid,
                conf_override=conf,
            )
            all_persons.extend(ps)
            all_balls.extend(bs)
            used_slices += 1
            if used_slices >= max_slices_total:
                break

    persons = merge_detections(all_persons, iou_thr=float(getattr(args, "sahi_merge_iou", 0.45)), foot_thr_px=float(getattr(args, "sahi_merge_foot_px", 28.0)))
    balls = merge_detections(all_balls, iou_thr=float(getattr(args, "sahi_merge_iou", 0.35)), foot_thr_px=float(getattr(args, "sahi_ball_merge_foot_px", 16.0)))
    return persons, balls


def parse_yolo_results(model: Any, result: Any, frame: np.ndarray, H: Any, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return result_to_detections(model, result, frame, H, args, det_source="full", offset_xy=(0, 0), slice_id="", conf_override=args.conf)



def sanitize_for_csv(det: dict[str, Any], sample_idx: int, frame_idx: int, time_sec: float) -> dict[str, Any]:
    skip = {"upper_hist", "lower_hist", "mid_hist", "shoe_hist", "head_hist"}
    row = {k: v for k, v in det.items() if k not in skip}
    src = det.get("source")
    if not src:
        src = "detected" if det.get("match_type") in ("raw_id", "new") else "reid_recovered"
    row.update({
        "sample_idx": sample_idx,
        "frame_idx": frame_idx,
        "time_sec": round(float(time_sec), 6),
        "source": src,
        "is_interpolated": False,
        "missing_gap_sec": float(det.get("missing_gap_sec", 0.0)),
    })
    # JSON/CSV için np scalar temizliği.
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
    return row


def interpolate_entity_csv(
    df: pd.DataFrame,
    id_col: str,
    sample_sec: float,
    max_gap_sec: float,
    entity_name: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.copy()
    df = df.sort_values([id_col, "sample_idx"])
    # Aynı id-time duplicate varsa confidence yüksek olanı tut.
    if "position_conf" in df.columns:
        df = df.sort_values([id_col, "sample_idx", "position_conf"], ascending=[True, True, False])
    df = df.drop_duplicates([id_col, "sample_idx"], keep="first")

    rows: list[dict[str, Any]] = []
    numeric_cols = [
        "time_sec", "frame_idx", "image_x", "image_y", "field_x", "field_y",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_w", "bbox_h",
        "det_conf", "position_conf", "orange_ratio", "dark_ratio",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for ent_id, g in df.groupby(id_col):
        g = g.sort_values("sample_idx")
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            rows.append(a)
            gap_idx = int(b["sample_idx"]) - int(a["sample_idx"])
            gap_sec = gap_idx * sample_sec
            if gap_idx <= 1:
                continue
            if gap_sec <= max_gap_sec + 1e-9:
                for k in range(1, gap_idx):
                    alpha = k / gap_idx
                    r = dict(a)
                    r["sample_idx"] = int(a["sample_idx"]) + k
                    r["time_sec"] = float(a["time_sec"]) + alpha * (float(b["time_sec"]) - float(a["time_sec"]))
                    for c in numeric_cols:
                        if c in ("sample_idx",):
                            continue
                        va = a.get(c, np.nan)
                        vb = b.get(c, np.nan)
                        try:
                            if pd.notna(va) and pd.notna(vb):
                                r[c] = float(va) + alpha * (float(vb) - float(va))
                        except Exception:
                            pass
                    r["source"] = "interpolated"
                    r["is_interpolated"] = True
                    r["missing_gap_sec"] = float(gap_sec)
                    # Aranın ortasında güven düşer.
                    base_conf = min(float(a.get("position_conf", a.get("det_conf", 0.5)) or 0.5), float(b.get("position_conf", b.get("det_conf", 0.5)) or 0.5))
                    middle_penalty = 1.0 - 0.55 * min(alpha, 1 - alpha) * 2.0
                    r["position_conf"] = max(0.15, base_conf * middle_penalty)
                    if "match_type" in r:
                        r["match_type"] = "interpolated"
                    rows.append(r)
        rows.append(recs[-1])

    out = pd.DataFrame(rows)
    out = out.sort_values(["sample_idx", id_col]).reset_index(drop=True)
    out["entity"] = entity_name
    return out


def draw_pitch_canvas(width: int = 1000, height: int = 600) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (35, 115, 45)
    sx = width / PITCH_LENGTH
    sy = height / PITCH_WIDTH

    def pt(x, y):
        return int(round(x * sx)), int(round(y * sy))

    white = (245, 245, 245)
    cv2.rectangle(canvas, pt(0, 0), pt(PITCH_LENGTH, PITCH_WIDTH), white, 3)
    cv2.line(canvas, pt(PITCH_LENGTH / 2, 0), pt(PITCH_LENGTH / 2, PITCH_WIDTH), white, 2)
    cv2.circle(canvas, pt(PITCH_LENGTH / 2, PITCH_WIDTH / 2), int(9 * sx), white, 2)
    py1 = (PITCH_WIDTH - PENALTY_WIDTH) / 2
    py2 = (PITCH_WIDTH + PENALTY_WIDTH) / 2
    cv2.rectangle(canvas, pt(0, py1), pt(PENALTY_DEPTH, py2), white, 2)
    cv2.rectangle(canvas, pt(PITCH_LENGTH - PENALTY_DEPTH, py1), pt(PITCH_LENGTH, py2), white, 2)
    gy1 = (PITCH_WIDTH - GOALBOX_WIDTH) / 2
    gy2 = (PITCH_WIDTH + GOALBOX_WIDTH) / 2
    cv2.rectangle(canvas, pt(0, gy1), pt(GOALBOX_DEPTH, gy2), white, 2)
    cv2.rectangle(canvas, pt(PITCH_LENGTH - GOALBOX_DEPTH, gy1), pt(PITCH_LENGTH, gy2), white, 2)
    return canvas


def correct_manual_players_on_pitch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Manual correction UI with an OUTSIDE margin.

    V8'de saha dışında kalan oyuncular görünmüyordu. Burada pitch etrafında ekstra
    margin var; field_x/field_y negatif veya 100/60 dışı olsa bile noktayı
    kırmızı çerçeveli olarak gösterebildiği kadar gösterir. Böylece saha dışından
    tutup saha içine sürükleyebilirsin.
    """
    if not rows:
        return rows

    W, Hh = 1100, 720
    margin_m = 18.0
    sx = W / (PITCH_LENGTH + 2.0 * margin_m)
    sy = Hh / (PITCH_WIDTH + 2.0 * margin_m)
    drag = {"idx": None}

    def f_to_canvas(fx: float, fy: float, clamp: bool = True):
        x = int(round((float(fx) + margin_m) * sx))
        y = int(round((float(fy) + margin_m) * sy))
        if clamp:
            x = max(8, min(W - 8, x))
            y = max(8, min(Hh - 8, y))
        return x, y

    def c_to_field(x: float, y: float):
        fx = x / sx - margin_m
        fy = y / sy - margin_m
        # Son düzeltme ekranında kaydettiğin noktayı saha içine alıyoruz.
        # Bu, "dışarıda bulsa bile içeri çekebilme" isteği için.
        return max(0.0, min(PITCH_LENGTH, fx)), max(0.0, min(PITCH_WIDTH, fy))

    def draw_canvas():
        canvas = np.zeros((Hh, W, 3), dtype=np.uint8)
        canvas[:, :] = (25, 85, 35)
        # Outside margin background
        px1, py1 = f_to_canvas(0, 0, clamp=False)
        px2, py2 = f_to_canvas(PITCH_LENGTH, PITCH_WIDTH, clamp=False)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (35, 115, 45), -1)

        def pt(x, y):
            return f_to_canvas(x, y, clamp=False)

        white = (245, 245, 245)
        cv2.rectangle(canvas, pt(0, 0), pt(PITCH_LENGTH, PITCH_WIDTH), white, 3)
        cv2.line(canvas, pt(PITCH_LENGTH / 2, 0), pt(PITCH_LENGTH / 2, PITCH_WIDTH), white, 2)
        # center circle using average scale
        cv2.circle(canvas, pt(PITCH_LENGTH / 2, PITCH_WIDTH / 2), int(9 * min(sx, sy)), white, 2)

        py1b = (PITCH_WIDTH - PENALTY_WIDTH) / 2
        py2b = (PITCH_WIDTH + PENALTY_WIDTH) / 2
        cv2.rectangle(canvas, pt(0, py1b), pt(PENALTY_DEPTH, py2b), white, 2)
        cv2.rectangle(canvas, pt(PITCH_LENGTH - PENALTY_DEPTH, py1b), pt(PITCH_LENGTH, py2b), white, 2)

        gy1 = (PITCH_WIDTH - GOALBOX_WIDTH) / 2
        gy2 = (PITCH_WIDTH + GOALBOX_WIDTH) / 2
        cv2.rectangle(canvas, pt(0, gy1), pt(GOALBOX_DEPTH, gy2), white, 2)
        cv2.rectangle(canvas, pt(PITCH_LENGTH - GOALBOX_DEPTH, gy1), pt(PITCH_LENGTH, gy2), white, 2)

        return canvas

    def is_outside(row):
        fx = float(row["field_x"])
        fy = float(row["field_y"])
        return fx < 0 or fx > PITCH_LENGTH or fy < 0 or fy > PITCH_WIDTH

    def redraw():
        canvas = draw_canvas()
        outside_count = 0
        for i, r in enumerate(rows):
            out = is_outside(r)
            if out:
                outside_count += 1
            x, y = f_to_canvas(r["field_x"], r["field_y"], clamp=True)
            color = (0, 165, 255) if str(r.get("team_hint", "unknown")) == "orange" else (60, 30, 10)
            outline = (0, 0, 255) if out else (255, 255, 255)
            cv2.circle(canvas, (x, y), 16, outline, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 11, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(int(r["stable_id"])), (x - 7, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        info = f"Kus bakisi duzeltme: disaridakiler KIRMIZI | surukle saha icine | Enter/S kaydet | Esc oldugu gibi | outside={outside_count}"
        cv2.putText(canvas, info, (12, Hh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, info, (12, Hh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
        return canvas

    def mouse_cb(event, x, y, flags, param):
        nonlocal drag
        if event == cv2.EVENT_LBUTTONDOWN:
            best = None
            best_d = 26 ** 2
            for i, r in enumerate(rows):
                px, py = f_to_canvas(r["field_x"], r["field_y"], clamp=True)
                d = (x - px) ** 2 + (y - py) ** 2
                if d < best_d:
                    best_d = d
                    best = i
            if best is not None:
                drag["idx"] = best
        elif event == cv2.EVENT_MOUSEMOVE and drag["idx"] is not None:
            fx, fy = c_to_field(x, y)
            rows[drag["idx"]]["field_x"] = float(fx)
            rows[drag["idx"]]["field_y"] = float(fy)
            rows[drag["idx"]]["manual_field_corrected"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            drag["idx"] = None

    win = "Kus Bakisi Oyuncu Duzeltme"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, Hh)
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key in (ord("s"), ord("S"), 13, 10):
            break
    cv2.destroyWindow(win)
    return rows


def mark_initial_players(frame: np.ndarray, mapper: Any, out_dir: Path, args) -> list[dict[str, Any]]:
    """
    Manuel başlangıç oyuncu kalibrasyonu:
    1) Görüntü üzerinde oyuncu ayak noktalarını 1,2,3... işaretle.
    2) Kuş bakışı sahada çıkan noktaları sürükleyerek düzelt.
    """
    max_players = int(getattr(args, "max_players", 14) or 14)
    max_w, max_h = 1500, 900
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp_base = cv2.resize(frame, (int(w * scale), int(h * scale)))
    points: list[tuple[float, float]] = []

    def redraw():
        canvas = disp_base.copy()
        for i, (ox, oy) in enumerate(points, start=1):
            x, y = int(ox * scale), int(oy * scale)
            cv2.circle(canvas, (x, y), 10, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 12, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        info = f"Manuel oyuncu: sol tik ayak noktasi | sag tik/U geri | Enter/S kus bakisi duzeltme | {len(points)}/{max_players}"
        cv2.putText(canvas, info, (12, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(canvas, info, (12, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
        return canvas

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if max_players <= 0 or len(points) < max_players:
                points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "Manuel Oyuncu Isaretleme"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_base.shape[1], disp_base.shape[0])
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            points = []
            break
        if key in (ord("u"), ord("U"), ord("z"), ord("Z")) and points:
            points.pop()
        if key in (ord("s"), ord("S"), 13, 10):
            break
    cv2.destroyWindow(win)

    rows = []
    for i, (ix, iy) in enumerate(points, start=1):
        fx, fy = image_to_field_point(mapper, ix, iy)
        rows.append({
            "stable_id": i,
            "image_x": float(ix),
            "image_y": float(iy),
            "field_x": float(fx),
            "field_y": float(fy),
            "team_hint": "unknown",
            "team": "unknown",
            "raw_id": None,
            "cls_name": "person",
            "det_conf": 1.0,
            "bbox_x1": max(0.0, ix - 12.0),
            "bbox_y1": max(0.0, iy - 40.0),
            "bbox_x2": min(float(w - 1), ix + 12.0),
            "bbox_y2": min(float(h - 1), iy + 4.0),
            "bbox_w": 24.0,
            "bbox_h": 44.0,
            "upper_hist": None,
            "lower_hist": None,
            "match_type": "manual_seed",
            "reid_conf": 1.0,
            "position_conf": 1.0,
            "manual_field_corrected": False,
        })

    if rows and bool(getattr(args, "manual_pitch_correct", True)):
        rows = correct_manual_players_on_pitch(rows)

    if rows:
        slim_rows = [{k: v for k, v in r.items() if k not in ("upper_hist", "lower_hist")} for r in rows]
        pd.DataFrame(slim_rows).to_csv(out_dir / "manual_initial_players.csv", index=False, encoding="utf-8-sig")
        print(f"Manuel oyuncu kalibrasyonu kaydedildi: {out_dir / 'manual_initial_players.csv'}")
    return rows



def write_outputs(out_dir: Path, players_raw: list[dict[str, Any]], balls_raw: list[dict[str, Any]], args, summary: dict[str, Any]):
    """
    Tek CSV mantığı:
        tracking_all.csv

    Ayrım kolonlarla yapılır:
        entity  = player / ball
        variant = raw / interpolated

    Eski split CSV'ler artık yazılmaz. Aynı output klasöründe eski denemeden kaldıysa silinir.
    """
    players_df = pd.DataFrame(players_raw)
    balls_df = pd.DataFrame(balls_raw)

    if not players_df.empty:
        players_df = players_df.sort_values(["sample_idx", "stable_id"]).reset_index(drop=True)
    if not balls_df.empty:
        balls_df = balls_df.sort_values(["sample_idx", "stable_id"]).reset_index(drop=True)

    players_interp = interpolate_entity_csv(players_df, "stable_id", args.sample_sec, args.player_interp_max_gap, "player") if not players_df.empty else players_df.copy()
    balls_interp = interpolate_entity_csv(balls_df, "stable_id", args.sample_sec, args.ball_interp_max_gap, "ball") if not balls_df.empty else balls_df.copy()

    def tag(df: pd.DataFrame, entity: str, variant: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["entity"] = entity
        out["variant"] = variant
        if "team" not in out.columns:
            out["team"] = entity
        return out

    all_parts = [
        tag(players_df, "player", "raw"),
        tag(players_interp, "player", "interpolated"),
        tag(balls_df, "ball", "raw"),
        tag(balls_interp, "ball", "interpolated"),
    ]
    non_empty = [p for p in all_parts if p is not None and not p.empty]
    if non_empty:
        all_df = pd.concat(non_empty, ignore_index=True, sort=False)
        sort_cols = [c for c in ["variant", "entity", "sample_idx", "stable_id"] if c in all_df.columns]
        all_df = all_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        all_df = pd.DataFrame(columns=["entity", "variant", "sample_idx", "frame_idx", "time_sec", "stable_id", "field_x", "field_y"])

    # Eski bölünmüş CSV'leri temizle ki klasörde kafa karıştırmasın.
    for old_name in ["tracks_raw.csv", "tracks_interpolated.csv", "ball_raw.csv", "ball_interpolated.csv"]:
        old_path = out_dir / old_name
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    all_csv_path = out_dir / "tracking_all.csv"
    all_xlsx_path = out_dir / "tracking_all.xlsx"
    all_df.to_csv(all_csv_path, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(all_xlsx_path, engine="openpyxl") as writer:
            all_df.to_excel(writer, sheet_name="all", index=False)
            tag(players_df, "player", "raw").to_excel(writer, sheet_name="players_raw", index=False)
            tag(players_interp, "player", "interpolated").to_excel(writer, sheet_name="players_interpolated", index=False)
            tag(balls_df, "ball", "raw").to_excel(writer, sheet_name="ball_raw", index=False)
            tag(balls_interp, "ball", "interpolated").to_excel(writer, sheet_name="ball_interpolated", index=False)
    except Exception as e:
        print("Excel yazılamadı, tek CSV yine kaydedildi:", e)

    summary.update({
        "players_raw_rows": int(len(players_df)),
        "players_interpolated_rows": int(len(players_interp)),
        "ball_raw_rows": int(len(balls_df)),
        "ball_interpolated_rows": int(len(balls_interp)),
        "all_rows": int(len(all_df)),
        "tracking_all_csv": str(all_csv_path),
        "tracking_all_xlsx": str(all_xlsx_path),
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return all_df




def recover_lost_track_detections(
    model: Any,
    frame: np.ndarray,
    mapper: Any,
    args,
    tracker: AppearanceStableTracker,
    sample_idx: int,
    time_sec: float,
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "lost_roi_recovery", True)):
        return []
    if sample_idx % max(1, int(getattr(args, "lost_roi_every_n", 1))) != 0:
        return []

    h, w = frame.shape[:2]
    dets: list[dict[str, Any]] = []
    tried = 0

    for sid, tr in sorted(tracker.tracks.items(), key=lambda kv: kv[0]):
        dt = max(0.0, float(time_sec) - float(tr.last_seen_time))
        if dt < float(getattr(args, "lost_roi_min_lost_sec", 0.15)):
            continue
        if dt > float(getattr(args, "player_max_lost", 8.0)):
            continue
        if not bool(getattr(args, "lost_roi_all", False)):
            if tr.image_y > h * float(getattr(args, "lost_roi_far_ymax", 0.45)) and dt < float(getattr(args, "lost_roi_near_min_lost_sec", 0.8)):
                continue

        tried += 1
        if tried > int(getattr(args, "lost_roi_max_tracks", 14)):
            break

        cx = float(tr.image_x)
        cy = float(tr.image_y)
        bw = max(16.0, float(tr.bbox_w or 24.0))
        bh = max(24.0, float(tr.bbox_h or 50.0))
        base = max(float(getattr(args, "lost_roi_min_size", 256)), bw * 5.0, bh * 4.0)
        size = base + dt * float(getattr(args, "lost_roi_growth_px_per_sec", 80.0))
        size = min(float(getattr(args, "lost_roi_max_size", 720)), size)

        x1 = int(max(0, cx - size / 2))
        y1 = int(max(0, cy - size / 2))
        x2 = int(min(w, cx + size / 2))
        y2 = int(min(h, cy + size / 2))
        if x2 - x1 < 96 or y2 - y1 < 96:
            continue

        crop = frame[y1:y2, x1:x2]
        try:
            res_list = model.predict(
                crop,
                conf=float(getattr(args, "recovery_conf", 0.04)),
                iou=float(getattr(args, "iou", 0.45)),
                classes=[0],
                verbose=False,
            )
        except TypeError:
            res_list = model.predict(crop, conf=float(getattr(args, "recovery_conf", 0.04)), verbose=False)

        if not res_list:
            continue

        ps, _ = result_to_detections(
            model,
            res_list[0],
            frame,
            mapper,
            args,
            det_source="lost_roi",
            offset_xy=(x1, y1),
            slice_id=f"lost_sid{sid}_{x1}_{y1}_{x2}_{y2}",
            conf_override=float(getattr(args, "recovery_conf", 0.04)),
        )
        for p in ps:
            p["target_lost_sid"] = int(sid)
        dets.extend(ps)

    return dets


def make_predicted_lost_rows(
    tracker: AppearanceStableTracker,
    assigned: list[dict[str, Any]],
    sample_idx: int,
    frame_idx: int,
    time_sec: float,
    args,
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "emit_lost_predictions", True)):
        return []

    assigned_sids = {int(d.get("stable_id")) for d in assigned if d.get("stable_id") is not None}
    rows: list[dict[str, Any]] = []

    for sid, tr in tracker.tracks.items():
        if sid in assigned_sids:
            continue
        dt = max(0.0, float(time_sec) - float(tr.last_seen_time))
        keep_sec = float(getattr(args, "lost_prediction_sec", 4.0))
        if tr.hits <= 1:
            keep_sec = max(keep_sec, float(getattr(args, "manual_seed_hold_sec", 12.0)))
        if dt <= 0 or dt > keep_sec:
            continue

        px, py = tr.predict(float(time_sec))
        conf = max(0.05, 1.0 - dt / max(0.1, keep_sec))
        det = {
            "stable_id": int(sid),
            "raw_id": None,
            "cls_id": 0,
            "cls_name": "person",
            "det_conf": 0.0,
            "image_x": float(tr.image_x),
            "image_y": float(tr.image_y),
            "field_x": float(px),
            "field_y": float(py),
            "bbox_x1": float(tr.image_x - tr.bbox_w / 2.0),
            "bbox_y1": float(tr.image_y - tr.bbox_h),
            "bbox_x2": float(tr.image_x + tr.bbox_w / 2.0),
            "bbox_y2": float(tr.image_y),
            "bbox_w": float(tr.bbox_w),
            "bbox_h": float(tr.bbox_h),
            "team_hint": tr.team,
            "team": tr.team,
            "orange_ratio": 0.0,
            "dark_ratio": 0.0,
            "match_type": "predicted_lost",
            "reid_conf": 0.0,
            "position_conf": float(conf),
            "source": "predicted_lost",
            "missing_gap_sec": float(dt),
            "det_source": "prediction",
            "merge_count": 0,
        }
        rows.append(sanitize_for_csv(det, sample_idx, frame_idx, time_sec))
    return rows


def process_video(args):
    if YOLO is None:
        raise RuntimeError("ultralytics yüklü değil. Kurulum: pip install ultralytics")

    video_arg = args.video
    if not video_arg:
        video_arg = choose_file_gui("Video seç", [("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
    if not video_arg:
        raise RuntimeError("Video seçilmedi.")
    video_path = Path(video_arg)
    out_dir = ensure_out_dir(video_path, args.out)

    calib = load_or_create_calibration(video_path, out_dir, args)
    field_mapper = build_field_mapper(calib)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    if args.duration and args.duration > 0:
        duration = min(duration, args.duration) if duration > 0 else args.duration
    start_time = max(0.0, args.start)
    if duration <= 0:
        # Fallback: döngü okuyarak yine işler ama progress kötü olur.
        duration = args.duration if args.duration > 0 else 0.0

    print("Model yükleniyor:", args.model)
    model = YOLO(args.model)

    player_tracker = AppearanceStableTracker(
        max_lost_sec=args.player_max_lost,
        base_search_radius=args.player_base_radius,
        radius_growth_per_sec=args.player_radius_growth,
        max_tracks=args.max_players,
        object_name="person",
    )
    ball_tracker = AppearanceStableTracker(
        max_lost_sec=args.ball_max_lost,
        base_search_radius=args.ball_base_radius,
        radius_growth_per_sec=args.ball_radius_growth,
        max_tracks=1,
        object_name="ball",
    )

    players_rows: list[dict[str, Any]] = []
    ball_rows: list[dict[str, Any]] = []
    manual_seed_written = False

    if args.manual_initial_players:
        seed_frame = read_corrected_frame_at(video_path, start_time, args)
        manual_seed_dets = mark_initial_players(seed_frame, field_mapper, out_dir, args)
        if manual_seed_dets:
            assigned_seed = player_tracker.update(manual_seed_dets, sample_idx=0, time_sec=start_time)
            frame_idx0 = int(round(start_time * fps)) if fps > 0 else 0
            for det in assigned_seed:
                row = sanitize_for_csv(det, 0, frame_idx0, float(start_time))
                row["source"] = "manual_seed"
                row["match_type"] = "manual_seed"
                row["position_conf"] = 1.0
                row["reid_conf"] = 1.0
                row["manual_field_corrected"] = bool(det.get("manual_field_corrected", False))
                players_rows.append(row)
            manual_seed_written = True
            print(f"Tracker manuel oyuncu kalibrasyonu ile başlatıldı: {len(manual_seed_dets)} oyuncu")

    sample_times = np.arange(start_time, max(start_time, duration) + 1e-6, args.sample_sec)

    classes = args.yolo_classes
    if classes:
        classes_list = [int(x.strip()) for x in classes.split(",") if x.strip()]
    else:
        classes_list = [0, 32]  # person + sports ball COCO

    pbar = tqdm(enumerate(sample_times), total=len(sample_times), desc="Video işleniyor")
    for sample_idx, tsec in pbar:
        if manual_seed_written and sample_idx == 0:
            # İlk frame manuel kalibre edildi; duplicate otomatik detection yazma.
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, float(tsec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = apply_lens_correction(frame, args)
        frame_idx = int(round(tsec * fps)) if fps > 0 else sample_idx

        try:
            if args.use_ultralytics_track:
                result_list = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    classes=classes_list,
                    verbose=False,
                )
            else:
                # Default: YOLO predict kullan. Kendi stable_id tracker'ımız zaten var.
                # Böylece BoT-SORT/ByteTrack içindeki GMC optical-flow warning'leri tamamen devre dışı kalır.
                result_list = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    classes=classes_list,
                    verbose=False,
                )
        except TypeError:
            # Bazı ultralytics sürümlerinde classes parametre formatı sıkıntı çıkarırsa fallback.
            if args.use_ultralytics_track:
                result_list = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )
            else:
                result_list = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )
        if not result_list:
            continue
        persons, balls = parse_yolo_results(model, result_list[0], frame, field_mapper, args)

        if args.sahi:
            persons, balls = run_sliced_inference(model, frame, field_mapper, args, persons, balls)

        if args.lost_roi_recovery:
            lost_persons = recover_lost_track_detections(model, frame, field_mapper, args, player_tracker, sample_idx, float(tsec))
            if lost_persons:
                persons = merge_detections(
                    list(persons) + lost_persons,
                    iou_thr=float(getattr(args, "sahi_merge_iou", 0.45)),
                    foot_thr_px=float(getattr(args, "sahi_merge_foot_px", 28.0)),
                )

        assigned_players = player_tracker.update(persons, sample_idx, float(tsec))
        assigned_balls = ball_tracker.update(balls, sample_idx, float(tsec))

        for det in assigned_players:
            players_rows.append(sanitize_for_csv(det, sample_idx, frame_idx, float(tsec)))

        for row in make_predicted_lost_rows(player_tracker, assigned_players, sample_idx, frame_idx, float(tsec), args):
            players_rows.append(row)
        for det in assigned_balls:
            row = sanitize_for_csv(det, sample_idx, frame_idx, float(tsec))
            row["team"] = "ball"
            ball_rows.append(row)

        if sample_idx % 20 == 0:
            pbar.set_postfix(players=len(players_rows), ball=len(ball_rows), stable=len(player_tracker.tracks))

    cap.release()

    summary = {
        "video": str(video_path),
        "out_dir": str(out_dir),
        "model": args.model,
        "sample_sec": args.sample_sec,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec_used": float(duration),
        "calibration_mode": calib.get("mode"),
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "max_players": args.max_players,
        "player_max_lost_sec": args.player_max_lost,
        "player_interp_max_gap_sec": args.player_interp_max_gap,
        "ball_interp_max_gap_sec": args.ball_interp_max_gap,
        "undistort_k1": float(args.undistort_k1),
        "undistort_k2": float(args.undistort_k2),
        "undistort_k3": float(args.undistort_k3),
        "undistort_focal": float(args.undistort_focal),
        "manual_initial_players": bool(args.manual_initial_players),
        "manual_pitch_correct": bool(args.manual_pitch_correct),
        "field_mapper": str(args.field_mapper),
        "curve_samples": int(args.curve_samples),
        "sahi": bool(args.sahi),
        "sahi_regions": str(args.sahi_regions),
        "sahi_slice_size": int(args.sahi_slice_size),
        "sahi_ball_slice_size": int(args.sahi_ball_slice_size),
        "sahi_overlap": float(args.sahi_overlap),
        "sahi_max_slices": int(args.sahi_max_slices),
        "use_ultralytics_track": bool(args.use_ultralytics_track),
        "tracker": str(args.tracker),
        "poly_fit_rmse": float((calib.get("image_to_field_poly") or {}).get("fit_rmse", 0.0)),
        "poly_fit_max_err": float((calib.get("image_to_field_poly") or {}).get("fit_max_err", 0.0)),
        "poly_bad_calibration": bool((calib.get("image_to_field_poly") or {}).get("bad_calibration", False)),
        "lost_roi_recovery": bool(args.lost_roi_recovery),
        "recovery_conf": float(args.recovery_conf),
        "emit_lost_predictions": bool(args.emit_lost_predictions),
        "lost_prediction_sec": float(args.lost_prediction_sec),
        "manual_seed_hold_sec": float(args.manual_seed_hold_sec),
        "inferred_roles": calib.get("inferred_roles", {}),
        "created_at_unix": time.time(),
    }
    write_outputs(out_dir, players_rows, ball_rows, args, summary)
    print("\nBitti. Çıktı klasörü:", out_dir)
    print("Ana dosyalar:")
    print(" - tracking_all.csv")
    print(" - tracking_all.xlsx")
    print(" - calibration.json")
    print(" - calibration_debug.jpg")
    print(" - summary.json")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Halı saha video -> tracking CSV/XLSX üretici")
    p.add_argument("--video", default=None, help="MP4/MOV/AVI/MKV video yolu. Boşsa dosya seçtirir.")
    p.add_argument("--out", default=None, help="Çıktı klasörü. Boşsa video yanında otomatik klasör açar.")
    p.add_argument("--model", default="yolo11s.pt", help="YOLO model dosyası. Örn: yolo11s.pt, yolo11m.pt veya kendi eğittiğin model.pt")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Sadece --use-ultralytics-track verilirse kullanılır. Normalde kendi trackerımız var.")
    p.add_argument("--use-ultralytics-track", action="store_true", help="YOLO model.track kullan. Kapalıysa model.predict kullanılır ve GMC warning tamamen engellenir.")
    p.add_argument("--sample-sec", type=float, default=0.1, help="Kaç saniyede bir frame işlensin. 0.1 = 10 FPS")
    p.add_argument("--conf", type=float, default=0.18, help="YOLO confidence eşiği")
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU eşiği")
    p.add_argument("--yolo-classes", default="0,32", help="COCO class filtreleri. 0=person, 32=sports ball")

    # Mini-SAHi / tiled inference: SAHI projesinden esinlenen, dependency gerektirmeyen sliced inference.
    p.add_argument("--sahi", action="store_true", help="Full-frame YOLO + seçili bölgelerde sliced/tiled YOLO birlikte çalışsın")
    p.add_argument("--sahi-regions", default="far,ball", help="Slice bölgeleri: far,ball,box,left_box,right_box,all. Virgülle yazılır")
    p.add_argument("--sahi-slice-size", type=int, default=640, help="Oyuncu/uzak bölge slice boyutu")
    p.add_argument("--sahi-ball-slice-size", type=int, default=384, help="Top arama slice boyutu")
    p.add_argument("--sahi-overlap", type=float, default=0.25, help="Slice overlap oranı")
    p.add_argument("--sahi-conf", type=float, default=0.0, help="Slice detection conf. 0 ise --conf kullanılır")
    p.add_argument("--sahi-max-slices", type=int, default=24, help="Frame başına maksimum slice sayısı")
    p.add_argument("--sahi-far-ymax", type=float, default=0.58, help="far bölgesi görüntünün üst kaç oranına kadar")
    p.add_argument("--sahi-ball-always", action="store_true", help="Top bulunsa bile ball slice search yap")
    p.add_argument("--sahi-merge-iou", type=float, default=0.45, help="Full/slice duplicate merge IoU")
    p.add_argument("--sahi-merge-foot-px", type=float, default=28.0, help="Oyuncu duplicate merge ayak noktası mesafesi px")
    p.add_argument("--sahi-ball-merge-foot-px", type=float, default=16.0, help="Top duplicate merge merkez/ayak noktası mesafesi px")
    p.add_argument("--ball-max-aspect", type=float, default=2.2, help="Top bbox aspect ratio filtresi")
    p.add_argument("--ball-max-px", type=float, default=80.0, help="Top bbox maksimum piksel boyutu filtresi")

    p.add_argument("--start", type=float, default=0.0, help="Başlangıç saniyesi")
    p.add_argument("--duration", type=float, default=0.0, help="Sadece ilk N saniyeyi işle. 0=tüm video")

    p.add_argument("--calibration", default=None, help="Var olan calibration.json yolu")
    p.add_argument("--force-calibration", action="store_true", help="Out klasöründe calibration.json olsa bile yeniden kalibre et")
    p.add_argument("--calibration-mode", choices=["line", "four_points"], default="line", help="line: basit çizgi sürükleme, four_points: 4 köşe")
    p.add_argument("--calibration-time", type=float, default=0.0, help="Kalibrasyon için ilk açılacak frame saniyesi")
    p.add_argument("--calibration-step-sec", type=float, default=5.0, help="Kalibrasyon ekranında N/P ile kaç saniye ileri/geri gidilsin")
    p.add_argument("--field-mapper", choices=["homography", "poly2", "poly3"], default="poly2", help="Saha dönüşümü: homography düz çizgi varsayar; poly2/poly3 yassı/eğri çizgilere daha toleranslıdır")
    p.add_argument("--curve-samples", type=int, default=15, help="Curve-aware kalibrasyonda her ana çizgiden kaç örnek nokta alınsın")
    p.add_argument("--curve-extension-curvature-scale", type=float, default=1.0, help="Kısa çizilen saha kenarını kesişim köşelerine uzatırken eğrilik/yassılık etkisi")
    p.add_argument("--curve-extension-auto-scale", action="store_true", help="Uzatılmış kenar uzunluğuna göre eğrilik etkisini otomatik ölçekle")
    p.add_argument("--calib-max-rmse", type=float, default=6.0, help="Poly mapper RMSE bu değeri aşarsa kötü kalibrasyon say")
    p.add_argument("--calib-max-err", type=float, default=18.0, help="Poly mapper max error bu değeri aşarsa kötü kalibrasyon say")
    p.add_argument("--auto-fallback-bad-calib", action="store_true", default=True, help="Poly kalibrasyon kötü çıkarsa homography fallback kullan")

    p.add_argument("--undistort-k1", type=float, default=0.0, help="Radial lens düzeltme k1. Örn deneme: -0.20 veya 0.20")
    p.add_argument("--undistort-k2", type=float, default=0.0, help="Radial lens düzeltme k2")
    p.add_argument("--undistort-k3", type=float, default=0.0, help="Radial lens düzeltme k3")
    p.add_argument("--undistort-focal", type=float, default=0.0, help="Lens düzeltme focal tahmini. 0=max(width,height)")

    p.add_argument("--max-players", type=int, default=14, help="Beklenen maksimum oyuncu sayısı. 0 = sınırsız")
    p.add_argument("--manual-initial-players", action="store_true", help="İlk frame'de oyuncuları 1,2,3 diye elle işaretleyip tracker stable_id seed et")
    p.add_argument("--manual-pitch-correct", action="store_true", default=True, help="Manuel oyuncu işaretlemeden sonra kuş bakışı noktaları sürükleyerek düzelt")
    p.add_argument("--player-max-lost", type=float, default=5.0, help="Oyuncu kaybolduktan sonra kaç saniye reID aranacak")
    p.add_argument("--player-base-radius", type=float, default=4.0, help="Oyuncu geri bağlama temel arama yarıçapı, saha birimi")
    p.add_argument("--player-radius-growth", type=float, default=3.0, help="Kayıp kaldıkça arama yarıçapı büyümesi, birim/s")
    p.add_argument("--lost-roi-recovery", action="store_true", default=True, help="Kaybolan oyuncu için son konum çevresinde düşük-conf crop/ROI detector çalıştır")
    p.add_argument("--recovery-conf", type=float, default=0.04, help="Lost ROI recovery için düşük confidence eşiği")
    p.add_argument("--lost-roi-min-lost-sec", type=float, default=0.15)
    p.add_argument("--lost-roi-near-min-lost-sec", type=float, default=0.8)
    p.add_argument("--lost-roi-every-n", type=int, default=1)
    p.add_argument("--lost-roi-max-tracks", type=int, default=14)
    p.add_argument("--lost-roi-far-ymax", type=float, default=0.45, help="Görüntünün üst/uzak bölgesi eşiği")
    p.add_argument("--lost-roi-all", action="store_true", help="Sadece uzak oyuncular değil tüm kayıp trackler için ROI ara")
    p.add_argument("--lost-roi-min-size", type=float, default=256)
    p.add_argument("--lost-roi-max-size", type=float, default=720)
    p.add_argument("--lost-roi-growth-px-per-sec", type=float, default=80)
    p.add_argument("--emit-lost-predictions", action="store_true", default=True, help="Kısa süreli detector kaybında source=predicted_lost satırı yaz")
    p.add_argument("--lost-prediction-sec", type=float, default=4.0)
    p.add_argument("--manual-seed-hold-sec", type=float, default=12.0, help="Manuel seed olup hiç yakalanmayan oyuncuyu kaç sn görünür tutalım")
    p.add_argument("--player-interp-max-gap", type=float, default=3.5, help="Oyuncu için en fazla kaç sn boşluk interpolate edilsin")

    p.add_argument("--ball-max-lost", type=float, default=1.5, help="Top kaybolduktan sonra kaç saniye reID aranacak")
    p.add_argument("--ball-base-radius", type=float, default=7.0, help="Top temel arama yarıçapı")
    p.add_argument("--ball-radius-growth", type=float, default=8.0, help="Top arama yarıçapı büyümesi")
    p.add_argument("--ball-interp-max-gap", type=float, default=1.0, help="Top için en fazla kaç sn boşluk interpolate edilsin")
    return p


def main():
    args = build_argparser().parse_args()
    try:
        process_video(args)
    except KeyboardInterrupt:
        print("\nİptal edildi.")
    except Exception as e:
        print("\nHATA:", e)
        raise


if __name__ == "__main__":
    main()
