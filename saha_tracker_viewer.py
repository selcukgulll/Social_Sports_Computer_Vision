#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP4 -> oyuncu tracking CSV/XLSX + oynatılabilir kuş bakışı saha krokisi HTML

Kurulum:
    pip install ultralytics opencv-python pandas numpy openpyxl tqdm

Çalıştırma:
    python saha_tracker_viewer.py

Opsiyonel:
    python saha_tracker_viewer.py --video mac.mp4 --sample-sec 0.1 --model yolo11s.pt --max-players 14

Not:
- İlk çalıştırmada sahayı kuş bakışına çevirmek için videodaki oyun alanının 4 köşesini tıklatır.
- Tıklama sırası: sol-üst, sağ-üst, sağ-alt, sol-alt.
- Çıktılar: tracks.csv, tracks.xlsx, summary.json, viewer.html
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PITCH_W = 100.0
PITCH_H = 60.0


@dataclass
class StableTrack:
    stable_id: int
    x: float
    y: float
    last_seen: int
    raw_id: int | None = None
    team_votes: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def team(self) -> str:
        if not self.team_votes:
            return "unknown"
        return max(self.team_votes.items(), key=lambda kv: kv[1])[0]


class StableIDManager:
    def __init__(self, max_gap_samples=25, reid_max_dist=9.0):
        self.max_gap_samples = max_gap_samples
        self.reid_max_dist = reid_max_dist
        self.next_id = 1
        self.raw_to_stable: dict[int, int] = {}
        self.tracks: dict[int, StableTrack] = {}

    def _new_track(self, det, sample_idx: int) -> int:
        sid = self.next_id
        self.next_id += 1
        tr = StableTrack(stable_id=sid, x=det["field_x"], y=det["field_y"], last_seen=sample_idx, raw_id=det.get("raw_id"))
        tr.team_votes[det.get("team", "unknown")] += 1
        self.tracks[sid] = tr
        if det.get("raw_id") is not None:
            self.raw_to_stable[int(det["raw_id"])] = sid
        return sid

    def _match_lost_track(self, det, sample_idx: int, used_stable_ids: set[int]) -> int | None:
        best_sid = None
        best_score = 1e9
        det_team = det.get("team", "unknown")
        dx0, dy0 = det["field_x"], det["field_y"]

        for sid, tr in self.tracks.items():
            if sid in used_stable_ids:
                continue
            gap = sample_idx - tr.last_seen
            if gap < 1 or gap > self.max_gap_samples:
                continue

            team_penalty = 0.0
            if tr.team != "unknown" and det_team != "unknown" and tr.team != det_team:
                team_penalty = 4.0

            dist = math.hypot(dx0 - tr.x, dy0 - tr.y)
            score = dist + 0.15 * gap + team_penalty
            if dist <= self.reid_max_dist and score < best_score:
                best_score = score
                best_sid = sid

        return best_sid

    def update(self, detections: list[dict], sample_idx: int) -> list[dict]:
        out = []
        used_stable_ids = set()

        # Önce tracker'ın ham ID'si devam edenleri bağla.
        for det in detections:
            raw_id = det.get("raw_id")
            sid = None
            if raw_id is not None and int(raw_id) in self.raw_to_stable:
                sid = self.raw_to_stable[int(raw_id)]
                if sid in used_stable_ids:
                    sid = None

            if sid is None:
                sid = self._match_lost_track(det, sample_idx, used_stable_ids)

            if sid is None:
                sid = self._new_track(det, sample_idx)
            else:
                if raw_id is not None:
                    self.raw_to_stable[int(raw_id)] = sid

            tr = self.tracks[sid]
            tr.x = det["field_x"]
            tr.y = det["field_y"]
            tr.last_seen = sample_idx
            tr.raw_id = raw_id
            tr.team_votes[det.get("team", "unknown")] += 1

            det["stable_id"] = sid
            det["team"] = tr.team
            used_stable_ids.add(sid)
            out.append(det)

        return out


def choose_video_gui() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="MP4 video seç",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def choose_output_dir_gui(default_name="saha_tracking_output") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Çıktı klasörü seç")
        root.destroy()
        return path or None
    except Exception:
        return None


def read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Video okunamadı veya ilk frame alınamadı.")
    return frame


def click_four_corners(frame: np.ndarray, save_path: Path | None = None):
    """Oyun alanının 4 köşesini alır: sol-üst, sağ-üst, sağ-alt, sol-alt."""
    points = []
    names = ["SOL-UST", "SAG-UST", "SAG-ALT", "SOL-ALT"]
    max_w, max_h = 1280, 760
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))

    def redraw():
        canvas = disp.copy()
        for i, p in enumerate(points):
            x, y = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(canvas, (x, y), 6, (0, 180, 255), -1)
            cv2.putText(canvas, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
        if len(points) >= 2:
            pts = np.array([[int(x * scale), int(y * scale)] for x, y in points], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 180, 255), thickness=2)
        msg1 = "Saha 4 kose tikla: sol-ust, sag-ust, sag-alt, sol-alt"
        msg2 = f"Siradaki: {names[len(points)] if len(points) < 4 else 'TAMAM'} | Geri: sag tik / u | Onay: Enter"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(canvas, msg1, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, msg2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return canvas

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "Kalibrasyon - 4 saha kosesi"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(points) == 4:  # Enter
            break
        if key in (ord("u"), ord("z")) and points:
            points.pop()
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("Kalibrasyon iptal edildi.")
    cv2.destroyWindow(win)

    src = np.array(points, dtype=np.float32)
    dst = np.array([[0, 0], [PITCH_W, 0], [PITCH_W, PITCH_H], [0, PITCH_H]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    if save_path is not None:
        payload = {"points_image_px": src.tolist(), "pitch_w": PITCH_W, "pitch_h": PITCH_H, "H_image_to_pitch": H.tolist()}
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return H


def load_calibration(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["H_image_to_pitch"], dtype=np.float32)


def map_point(H: np.ndarray, x: float, y: float):
    pt = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(mapped[0]), float(mapped[1])


def classify_team_by_color(frame: np.ndarray, bbox, orange_threshold=0.055):
    """
    Basit forma/yelek renk sınıflandırması.
    Orange bib varsa 'orange', yoksa 'dark' döndürür.
    Geliştirmek istersen burayı kendi forma renklerine göre değiştir.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H_img, W_img = frame.shape[:2]
    x1 = max(0, min(W_img - 1, x1)); x2 = max(0, min(W_img - 1, x2))
    y1 = max(0, min(H_img - 1, y1)); y2 = max(0, min(H_img - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    bw, bh = x2 - x1, y2 - y1
    # Baş/bacak yerine üst gövdeyi al.
    cx1 = int(x1 + 0.18 * bw)
    cx2 = int(x1 + 0.82 * bw)
    cy1 = int(y1 + 0.18 * bh)
    cy2 = int(y1 + 0.68 * bh)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # OpenCV HSV: H 0-179. Turuncu/sarı yelek için geniş aralık.
    orange_mask = ((h >= 4) & (h <= 35) & (s >= 65) & (v >= 70))
    score = float(orange_mask.mean())
    if score >= orange_threshold:
        return "orange", score
    return "dark", score


def detections_from_yolo_result(result, frame: np.ndarray, H: np.ndarray, args):
    dets = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    if boxes.id is not None:
        raw_ids = boxes.id.cpu().numpy().astype(int).tolist()
    else:
        raw_ids = [None] * len(xyxy)

    for bbox, conf, raw_id in zip(xyxy, confs, raw_ids):
        x1, y1, x2, y2 = bbox.tolist()
        # Ayak noktası: bbox alt-orta. Kuş bakışı pozisyon için en iyi pratik nokta.
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        field_x, field_y = map_point(H, foot_x, foot_y)

        # Kalibrasyon dışına uçan hatalı tespitleri ele.
        margin = args.field_margin
        if field_x < -margin or field_x > PITCH_W + margin or field_y < -margin or field_y > PITCH_H + margin:
            continue

        team, team_score = classify_team_by_color(frame, bbox, orange_threshold=args.orange_threshold)
        dets.append({
            "raw_id": raw_id,
            "conf": float(conf),
            "bbox_x1": float(x1), "bbox_y1": float(y1), "bbox_x2": float(x2), "bbox_y2": float(y2),
            "foot_px_x": float(foot_x), "foot_px_y": float(foot_y),
            "field_x": float(np.clip(field_x, 0, PITCH_W)),
            "field_y": float(np.clip(field_y, 0, PITCH_H)),
            "team": team,
            "team_color_score": team_score,
        })

    # Çok fazla false positive varsa güven skoruna göre kıs.
    if args.max_players and len(dets) > args.max_players:
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)[:args.max_players]

    return dets


def write_viewer_html(df: pd.DataFrame, out_html: Path):
    slim_cols = ["timestamp", "stable_id", "field_x", "field_y", "team"]
    rows = []
    for r in df[slim_cols].itertuples(index=False):
        team_code = 1 if r.team == "orange" else 0
        rows.append([round(float(r.timestamp), 2), int(r.stable_id), round(float(r.field_x), 3), round(float(r.field_y), 3), team_code])

    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

    html = r'''<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Saha Tracking Viewer</title>
<style>
  body { margin:0; font-family: Arial, sans-serif; background:#111; color:#eee; display:flex; flex-direction:column; align-items:center; }
  .wrap { width:min(1100px, 96vw); margin-top:18px; }
  canvas { width:100%; height:auto; background:#17751f; border-radius:14px; box-shadow:0 8px 26px rgba(0,0,0,.45); }
  .panel { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:12px 0 20px; background:#1d1d1d; padding:12px; border-radius:12px; }
  button, select { background:#2b2b2b; color:#fff; border:1px solid #555; border-radius:8px; padding:8px 12px; font-size:14px; }
  input[type=range] { flex:1; min-width:240px; }
  .pill { padding:6px 10px; background:#2b2b2b; border-radius:999px; font-size:14px; }
  .legend { display:flex; gap:10px; align-items:center; margin-left:auto; }
  .dot { width:13px; height:13px; display:inline-block; border-radius:50%; border:2px solid white; vertical-align:middle; margin-right:5px; }
  .orange { background:#ff8500; } .dark { background:#172027; }
</style>
</head>
<body>
<div class="wrap">
  <canvas id="pitch" width="1000" height="600"></canvas>
  <div class="panel">
    <button id="play">▶ Play</button>
    <input id="slider" type="range" min="0" max="0" value="0" step="1" />
    <span class="pill" id="time">0.00 s</span>
    <label>Hız <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>4</option><option>8</option></select>x</label>
    <div class="legend">
      <span><i class="dot orange"></i>Turuncu</span>
      <span><i class="dot dark"></i>Koyu</span>
    </div>
  </div>
</div>
<script>
const rows = %%DATA%%;
const PITCH_W = 100, PITCH_H = 60;
const canvas = document.getElementById('pitch');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const timeBox = document.getElementById('time');
const speedSel = document.getElementById('speed');

const byTime = new Map();
for (const r of rows) {
  const t = r[0];
  if (!byTime.has(t)) byTime.set(t, []);
  byTime.get(t).push(r);
}
const times = Array.from(byTime.keys()).sort((a,b)=>a-b);
slider.max = Math.max(0, times.length - 1);
let idx = 0, timer = null;

function sx(x){ return 40 + x / PITCH_W * (canvas.width - 80); }
function sy(y){ return 40 + y / PITCH_H * (canvas.height - 80); }

function drawPitch(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const grad = ctx.createLinearGradient(0,0,canvas.width,0);
  grad.addColorStop(0,'#0f5f1b'); grad.addColorStop(.5,'#1f8729'); grad.addColorStop(1,'#0f5f1b');
  ctx.fillStyle = grad; ctx.fillRect(0,0,canvas.width,canvas.height);

  ctx.strokeStyle = 'rgba(255,255,255,.92)';
  ctx.lineWidth = 4;
  ctx.strokeRect(sx(0), sy(0), sx(PITCH_W)-sx(0), sy(PITCH_H)-sy(0));

  // Orta çizgi + santra.
  ctx.beginPath(); ctx.moveTo(sx(50), sy(0)); ctx.lineTo(sx(50), sy(PITCH_H)); ctx.stroke();
  ctx.beginPath(); ctx.arc(sx(50), sy(30), 72, 0, Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.arc(sx(50), sy(30), 4, 0, Math.PI*2); ctx.fillStyle='white'; ctx.fill();

  // Ceza sahaları, kale alanları.
  const boxW=16, boxH=36, sixW=7, sixH=20;
  ctx.strokeRect(sx(0), sy((PITCH_H-boxH)/2), sx(boxW)-sx(0), sy((PITCH_H+boxH)/2)-sy((PITCH_H-boxH)/2));
  ctx.strokeRect(sx(PITCH_W-boxW), sy((PITCH_H-boxH)/2), sx(PITCH_W)-sx(PITCH_W-boxW), sy((PITCH_H+boxH)/2)-sy((PITCH_H-boxH)/2));
  ctx.strokeRect(sx(0), sy((PITCH_H-sixH)/2), sx(sixW)-sx(0), sy((PITCH_H+sixH)/2)-sy((PITCH_H-sixH)/2));
  ctx.strokeRect(sx(PITCH_W-sixW), sy((PITCH_H-sixH)/2), sx(PITCH_W)-sx(PITCH_W-sixW), sy((PITCH_H+sixH)/2)-sy((PITCH_H-sixH)/2));

  // Kaleler.
  ctx.lineWidth = 3;
  ctx.strokeRect(sx(0)-24, sy(24), 24, sy(36)-sy(24));
  ctx.strokeRect(sx(PITCH_W), sy(24), 24, sy(36)-sy(24));
}

function drawPlayers(list){
  for (const r of list) {
    const [t,id,x,y,team] = r;
    const px = sx(x), py = sy(y);
    ctx.beginPath();
    ctx.arc(px, py, 13, 0, Math.PI*2);
    ctx.fillStyle = team === 1 ? '#ff8500' : '#172027';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,.95)';
    ctx.stroke();
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    ctx.fillText(String(id), px, py+0.5);
  }
}

function draw(){
  drawPitch();
  if (times.length === 0) return;
  const t = times[idx];
  drawPlayers(byTime.get(t) || []);
  timeBox.textContent = t.toFixed(2) + ' s';
  slider.value = idx;
}

function step(){
  idx += 1;
  if (idx >= times.length) { idx = times.length - 1; stop(); }
  draw();
}
function play(){
  if (timer || times.length === 0) return;
  playBtn.textContent = '⏸ Pause';
  const dt = times.length > 1 ? Math.max(30, (times[1]-times[0])*1000 / parseFloat(speedSel.value)) : 100;
  timer = setInterval(step, dt);
}
function stop(){
  if (timer) clearInterval(timer);
  timer = null;
  playBtn.textContent = '▶ Play';
}

playBtn.onclick = () => timer ? stop() : play();
slider.oninput = () => { idx = parseInt(slider.value); draw(); };
speedSel.onchange = () => { if (timer) { stop(); play(); } };
document.addEventListener('keydown', (e)=>{ if(e.code==='Space'){ e.preventDefault(); timer ? stop() : play(); } });
draw();
</script>
</body>
</html>
'''.replace("%%DATA%%", data_json)

    out_html.write_text(html, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="MP4 oyuncu tracking + kuş bakışı saha viewer")
    p.add_argument("--video", type=str, default=None, help="MP4 yolu. Verilmezse dosya seçme penceresi açılır.")
    p.add_argument("--out", type=str, default=None, help="Çıktı klasörü. Verilmezse video yanında klasör açılır.")
    p.add_argument("--model", type=str, default="yolo11s.pt", help="Ultralytics YOLO modeli: yolo11s.pt, yolo11m.pt, yolo8s.pt vb.")
    p.add_argument("--tracker", type=str, default="botsort.yaml", help="botsort.yaml veya bytetrack.yaml")
    p.add_argument("--sample-sec", type=float, default=0.1, help="Kaç saniyede bir frame işlenecek. 0.1 = 10 FPS")
    p.add_argument("--conf", type=float, default=0.22, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="YOLO IoU threshold")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    p.add_argument("--max-players", type=int, default=22, help="Bir frame'de tutulacak maksimum kişi. Halı saha için 14 deneyebilirsin.")
    p.add_argument("--orange-threshold", type=float, default=0.055, help="Turuncu yelek renk skoru eşiği")
    p.add_argument("--field-margin", type=float, default=5.0, help="Saha dışı filtre toleransı")
    p.add_argument("--reuse-calibration", action="store_true", help="Output klasöründeki calibration.json varsa tekrar tıklama")
    p.add_argument("--reid-max-dist", type=float, default=9.0, help="ID kaybı sonrası yeniden eşleme mesafesi; saha birimi")
    p.add_argument("--max-gap-samples", type=int, default=25, help="ID kaybı sonrası kaç örnek geriye bakılsın. 25 = 2.5 sn")
    return p.parse_args()


def main():
    args = parse_args()
    if YOLO is None:
        print("HATA: ultralytics kurulu değil. Kurulum: pip install ultralytics opencv-python pandas numpy openpyxl tqdm")
        sys.exit(1)

    video_path = args.video or choose_video_gui()
    if not video_path:
        print("Video seçilmedi.")
        sys.exit(1)
    video_path = str(Path(video_path).expanduser().resolve())
    video_stem = Path(video_path).stem

    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        # GUI klasör seçimi iptal olursa video yanında otomatik klasör aç.
        gui_out = choose_output_dir_gui()
        out_dir = Path(gui_out).expanduser().resolve() if gui_out else Path(video_path).parent / f"{video_stem}_saha_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)

    first = read_first_frame(video_path)
    calib_path = out_dir / "calibration.json"
    if args.reuse_calibration and calib_path.exists():
        H = load_calibration(calib_path)
        print(f"Kalibrasyon yüklendi: {calib_path}")
    else:
        print("Kalibrasyon penceresi açılıyor. Saha köşelerini sırayla tıkla: sol-üst, sağ-üst, sağ-alt, sol-alt.")
        H = click_four_corners(first, save_path=calib_path)
        print(f"Kalibrasyon kaydedildi: {calib_path}")

    print(f"Model yükleniyor: {args.model}")
    model = YOLO(args.model)
    id_manager = StableIDManager(max_gap_samples=args.max_gap_samples, reid_max_dist=args.reid_max_dist)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0

    rows = []
    frame_idx = -1
    next_sample_t = 0.0
    sample_idx = 0
    t0 = time.time()

    pbar_total = int(duration / args.sample_sec) + 1 if duration else None
    pbar = tqdm(total=pbar_total, desc="İşleniyor", unit="sample")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_sec = frame_idx / fps
        if t_sec + 1e-9 < next_sample_t:
            continue

        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=[0],  # person
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        dets = detections_from_yolo_result(result, frame, H, args)
        dets = id_manager.update(dets, sample_idx)

        for det in dets:
            rows.append({
                "video": Path(video_path).name,
                "timestamp": round(t_sec, 3),
                "sample_idx": sample_idx,
                "frame_idx": frame_idx,
                "stable_id": det["stable_id"],
                "raw_track_id": det.get("raw_id"),
                "team": det.get("team", "unknown"),
                "conf": round(det["conf"], 4),
                "field_x": round(det["field_x"], 4),
                "field_y": round(det["field_y"], 4),
                "foot_px_x": round(det["foot_px_x"], 2),
                "foot_px_y": round(det["foot_px_y"], 2),
                "bbox_x1": round(det["bbox_x1"], 2),
                "bbox_y1": round(det["bbox_y1"], 2),
                "bbox_x2": round(det["bbox_x2"], 2),
                "bbox_y2": round(det["bbox_y2"], 2),
                "team_color_score": round(det.get("team_color_score", 0.0), 4),
            })

        sample_idx += 1
        next_sample_t += args.sample_sec
        pbar.update(1)

    pbar.close()
    cap.release()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "tracks.csv"
    xlsx_path = out_dir / "tracks.xlsx"
    html_path = out_dir / "viewer.html"
    summary_path = out_dir / "summary.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
        xlsx_msg = str(xlsx_path)
    except Exception as e:
        xlsx_msg = f"Excel yazılamadı: {e}"

    if not df.empty:
        write_viewer_html(df, html_path)
    else:
        html_path.write_text("<html><body><h2>Hiç oyuncu tespiti çıkmadı.</h2></body></html>", encoding="utf-8")

    summary = {
        "video": video_path,
        "fps": fps,
        "duration_sec": duration,
        "sample_sec": args.sample_sec,
        "processed_samples": sample_idx,
        "rows": int(len(df)),
        "unique_stable_ids": int(df["stable_id"].nunique()) if not df.empty else 0,
        "created_files": {
            "csv": str(csv_path),
            "xlsx": xlsx_msg,
            "html_viewer": str(html_path),
            "calibration": str(calib_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "notes": [
            "field_x/field_y değerleri 0-100 ve 0-60 normalize saha koordinatıdır.",
            "Pozisyon noktası bbox alt-orta, yani oyuncunun ayak noktasıdır.",
            "Takım rengi basit HSV turuncu-yoksa-koyu sınıflandırmasıdır; gerekirse classify_team_by_color fonksiyonunu forma rengine göre değiştir.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBitti.")
    print(f"CSV:        {csv_path}")
    print(f"Excel:      {xlsx_msg}")
    print(f"Viewer:     {html_path}")
    print(f"Summary:    {summary_path}")
    print("Viewer için viewer.html dosyasını tarayıcıda açman yeterli.")


if __name__ == "__main__":
    main()
