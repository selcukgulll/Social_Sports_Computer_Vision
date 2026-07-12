import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


DEFAULT_COLORS = [
    "#16a085", "#e74c3c", "#f1c40f", "#8e44ad", "#3498db",
    "#e67e22", "#2ecc71", "#c0392b", "#1abc9c", "#9b59b6"
]


def load_tracks(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tracks dosyası bulunamadı: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


def load_summary(path):
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        print(f"Uyarı: summary bulunamadı: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_team_map(path):
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        print(f"Uyarı: team map bulunamadı: {path}")
        return {}

    df = pd.read_csv(path)
    result = {}

    for _, row in df.iterrows():
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            continue

        result[entity_id] = {
            "label": str(row.get("label", entity_id)).strip() if not pd.isna(row.get("label", "")) else entity_id,
            "team": str(row.get("team", "")).strip() if not pd.isna(row.get("team", "")) else "",
            "color": str(row.get("color", "")).strip() if not pd.isna(row.get("color", "")) else "",
        }

    return result


def make_team_template(df, out_path):
    players = (
        df[df["entity_type"].eq("player")]
        .groupby("entity_id")
        .agg(
            detections=("time_s", "count"),
            first_s=("time_s", "min"),
            last_s=("time_s", "max"),
            mean_conf=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("detections", ascending=False)
    )

    template = pd.DataFrame({
        "entity_id": players["entity_id"],
        "label": players["entity_id"].astype(str).str.replace("player_", "", regex=False),
        "team": "",
        "color": "",
        "detections": players["detections"],
        "first_s": players["first_s"].round(2),
        "last_s": players["last_s"].round(2),
        "mean_conf": players["mean_conf"].round(3),
    })

    template.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Team template kaydedildi: {out_path}")


def get_video_size(summary, df):
    original = summary.get("original_video", {}) if isinstance(summary, dict) else {}
    width = original.get("width", None)
    height = original.get("height", None)

    if width is None or height is None:
        width = int(max(1, math.ceil(df["x_px"].max()))) if "x_px" in df.columns else 1280
        height = int(max(1, math.ceil(df["y_px"].max()))) if "y_px" in df.columns else 720

    return float(width), float(height)


def prepare_positions(df, summary, field_length, field_width, prefer_field_coords=True):
    df = df.copy()

    has_field = (
        prefer_field_coords
        and "x_field_m" in df.columns
        and "y_field_m" in df.columns
        and df["x_field_m"].notna().any()
        and df["y_field_m"].notna().any()
    )

    if has_field:
        df["plot_x"] = df["x_field_m"].astype(float)
        df["plot_y"] = df["y_field_m"].astype(float)
        coord_mode = "field_calibrated"
    else:
        video_w, video_h = get_video_size(summary, df)
        df["plot_x"] = df["x_px"].astype(float) / video_w * field_length
        df["plot_y"] = df["y_px"].astype(float) / video_h * field_width
        coord_mode = "pixel_normalized_uncalibrated"

    return df, coord_mode


def clean_df(df):
    required = ["time_s", "entity_id", "entity_type", "confidence", "x_px", "y_px"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df = df.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["time_s", "confidence", "x_px", "y_px"])
    df["entity_id"] = df["entity_id"].astype(str)
    df["entity_type"] = df["entity_type"].astype(str)
    return df


def make_color_for_entity(entity_id, team_map):
    item = team_map.get(entity_id, {})
    color = item.get("color", "")
    team = item.get("team", "").lower()

    if color:
        return color

    if team in ["red", "kirmizi", "kırmızı", "home", "a"]:
        return "#e74c3c"
    if team in ["blue", "mavi", "cyan", "turkuaz", "away", "b"]:
        return "#16a085"
    if team in ["keeper", "gk", "kaleci"]:
        return "#f1c40f"

    try:
        num = int("".join([ch for ch in str(entity_id) if ch.isdigit()]) or "0")
    except Exception:
        num = abs(hash(entity_id))

    return DEFAULT_COLORS[num % len(DEFAULT_COLORS)]


def make_label_for_entity(entity_id, team_map):
    item = team_map.get(entity_id, {})
    label = item.get("label", "")
    if label:
        return label

    if entity_id.startswith("player_"):
        return entity_id.replace("player_", "").lstrip("0") or "0"

    return entity_id


def draw_pitch(ax, field_length, field_width):
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, field_length)
    ax.set_ylim(field_width, 0)
    ax.set_facecolor("#4f7f2f")

    line = dict(color="white", linewidth=2, alpha=0.85)

    ax.plot([0, field_length, field_length, 0, 0], [0, 0, field_width, field_width, 0], **line)
    ax.plot([field_length / 2, field_length / 2], [0, field_width], **line)

    center = plt.Circle((field_length / 2, field_width / 2), min(field_length, field_width) * 0.13, fill=False, **line)
    ax.add_patch(center)
    ax.scatter([field_length / 2], [field_width / 2], s=10, c="white")

    box_depth = field_length * 0.16
    box_width = field_width * 0.55
    box_y1 = (field_width - box_width) / 2
    box_y2 = box_y1 + box_width

    small_depth = field_length * 0.06
    small_width = field_width * 0.28
    small_y1 = (field_width - small_width) / 2
    small_y2 = small_y1 + small_width

    ax.plot([0, box_depth, box_depth, 0], [box_y1, box_y1, box_y2, box_y2], **line)
    ax.plot([field_length, field_length - box_depth, field_length - box_depth, field_length],
            [box_y1, box_y1, box_y2, box_y2], **line)

    ax.plot([0, small_depth, small_depth, 0], [small_y1, small_y1, small_y2, small_y2], **line)
    ax.plot([field_length, field_length - small_depth, field_length - small_depth, field_length],
            [small_y1, small_y1, small_y2, small_y2], **line)

    ax.set_xticks([])
    ax.set_yticks([])


class TacticalViewer:
    def __init__(self, df, field_length, field_width, team_map, hold_s=0.6, tail_s=3.0):
        self.df = df.sort_values(["time_s", "entity_id"]).reset_index(drop=True)
        self.field_length = field_length
        self.field_width = field_width
        self.team_map = team_map
        self.hold_s = hold_s
        self.tail_s = tail_s

        self.times = np.array(sorted(self.df["time_s"].unique()), dtype=float)
        self.idx = 0
        self.playing = False

        self.players = self.df[self.df["entity_type"].eq("player")].copy()
        self.balls = self.df[self.df["entity_type"].eq("ball")].copy()

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(bottom=0.18)

        slider_ax = self.fig.add_axes([0.15, 0.075, 0.70, 0.03])
        self.slider = Slider(
            ax=slider_ax,
            label="time",
            valmin=float(self.times.min()),
            valmax=float(self.times.max()),
            valinit=float(self.times.min()),
            valstep=0.2,
        )
        self.slider.on_changed(self.on_slider)

        button_ax = self.fig.add_axes([0.87, 0.058, 0.09, 0.06])
        self.button = Button(button_ax, "Play")
        self.button.on_clicked(self.on_button)

        self.anim = FuncAnimation(self.fig, self.animate, interval=200, blit=False, cache_frame_data=False)

    def on_slider(self, val):
        nearest_idx = int(np.argmin(np.abs(self.times - val)))
        self.idx = nearest_idx
        self.update_plot()

    def on_button(self, event):
        self.playing = not self.playing
        self.button.label.set_text("Pause" if self.playing else "Play")

    def get_latest_entities(self, data, t):
        recent = data[(data["time_s"] <= t) & (data["time_s"] >= t - self.hold_s)]
        if recent.empty:
            return recent

        recent = recent.sort_values("time_s").groupby("entity_id", as_index=False).tail(1)
        return recent

    def get_tail(self, entity_id, t):
        tail = self.df[
            (self.df["entity_id"].eq(entity_id))
            & (self.df["time_s"] <= t)
            & (self.df["time_s"] >= t - self.tail_s)
        ]
        return tail.sort_values("time_s")

    def update_plot(self):
        t = float(self.times[self.idx])

        draw_pitch(self.ax, self.field_length, self.field_width)

        active_players = self.get_latest_entities(self.players, t)
        active_balls = self.get_latest_entities(self.balls, t)

        for _, row in active_players.iterrows():
            entity_id = row["entity_id"]
            x = float(row["plot_x"])
            y = float(row["plot_y"])
            conf = float(row["confidence"])

            color = make_color_for_entity(entity_id, self.team_map)
            label = make_label_for_entity(entity_id, self.team_map)

            tail = self.get_tail(entity_id, t)
            if len(tail) >= 2:
                self.ax.plot(tail["plot_x"], tail["plot_y"], color=color, linewidth=1.4, alpha=0.35)

            self.ax.scatter([x], [y], s=360, c=color, edgecolors="white", linewidths=2.2, zorder=5)
            self.ax.text(
                x,
                y,
                label,
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )

            if conf < 0.45:
                self.ax.scatter([x], [y], s=480, facecolors="none", edgecolors="yellow", linewidths=1.5, zorder=4)

        for _, row in active_balls.iterrows():
            x = float(row["plot_x"])
            y = float(row["plot_y"])
            self.ax.scatter([x], [y], s=90, c="yellow", edgecolors="black", linewidths=1.5, zorder=7)
            self.ax.text(x + 0.8, y, "ball", color="black", fontsize=8, fontweight="bold", zorder=8)

        self.ax.set_title(
            f"t = {t:.1f} s | aktif oyuncu: {len(active_players)} | aktif top: {len(active_balls)}",
            fontsize=12,
            fontweight="bold",
        )

        self.fig.canvas.draw_idle()

    def animate(self, frame):
        if self.playing:
            self.idx = (self.idx + 1) % len(self.times)
            self.slider.set_val(float(self.times[self.idx]))
        return []

    def show(self):
        self.update_plot()
        plt.show()


def print_quality_summary(df, summary):
    duration = df["time_s"].max() - df["time_s"].min()
    frames = df["time_s"].nunique()
    players = df[df["entity_type"].eq("player")]
    balls = df[df["entity_type"].eq("ball")]

    print("\n=== Veri Özeti ===")
    print(f"Süre: {duration:.1f} sn")
    print(f"Zaman noktası: {frames}")
    print(f"Toplam satır: {len(df)}")
    print(f"Oyuncu satırı: {len(players)}")
    print(f"Top satırı: {len(balls)}")
    print(f"Unique player ID: {players['entity_id'].nunique()}")

    per_t = df.groupby("time_s").agg(
        players=("entity_type", lambda s: int((s == "player").sum())),
        balls=("entity_type", lambda s: int((s == "ball").sum())),
    )

    print(f"Ortalama oyuncu/frame: {per_t['players'].mean():.2f}")
    print(f"Maks oyuncu/frame: {per_t['players'].max()}")
    print(f"Min oyuncu/frame: {per_t['players'].min()}")

    if len(players) > 0:
        track_lengths = players.groupby("entity_id")["time_s"].count()
        print(f"Median track uzunluğu: {track_lengths.median():.1f} detection")
        print(f"100+ detection track: {(track_lengths >= 100).sum()}")

    if isinstance(summary, dict) and summary:
        print("\n=== Summary ===")
        print(f"Model: {summary.get('model')}")
        print(f"Tracker: {summary.get('tracker')}")
        print(f"Sample FPS: {summary.get('sample_fps')}")
        print(f"Elapsed dakika: {summary.get('elapsed_minutes')}")
        print(f"Calibration var mı: {summary.get('has_calibration')}")


def main():
    parser = argparse.ArgumentParser(description="Tracks CSV/Parquet dosyasını taktik saha görünümünde izleme uygulaması.")
    parser.add_argument("--tracks", required=True, help="tracks_5fps.csv veya tracks_5fps.parquet")
    parser.add_argument("--summary", default=None, help="summary.json")
    parser.add_argument("--team-map", default=None, help="Opsiyonel team_map.csv")
    parser.add_argument("--make-team-template", default=None, help="Oyuncu ID eşleştirme şablonu CSV çıkarır ve kapanır.")
    parser.add_argument("--field-length", type=float, default=50.0, help="Saha uzunluğu. Kalibrasyon yoksa normalize canvas.")
    parser.add_argument("--field-width", type=float, default=30.0, help="Saha genişliği. Kalibrasyon yoksa normalize canvas.")
    parser.add_argument("--hold-s", type=float, default=0.6, help="Detection kaybolursa kaç saniye son pozisyon gösterilsin.")
    parser.add_argument("--tail-s", type=float, default=3.0, help="Oyuncunun son kaç saniyelik iz çizgisi gösterilsin.")
    parser.add_argument("--no-field-coords", action="store_true", help="Varsa bile x_field_m/y_field_m kullanma.")
    args = parser.parse_args()

    df = load_tracks(args.tracks)
    summary = load_summary(args.summary)
    df = clean_df(df)

    if args.make_team_template:
        make_team_template(df, args.make_team_template)
        return

    df, coord_mode = prepare_positions(
        df,
        summary,
        args.field_length,
        args.field_width,
        prefer_field_coords=not args.no_field_coords,
    )

    team_map = load_team_map(args.team_map)

    print_quality_summary(df, summary)
    print(f"\nKoordinat modu: {coord_mode}")
    if coord_mode == "pixel_normalized_uncalibrated":
        print("Uyarı: Bu görüntü gerçek saha koordinatı değil; piksel koordinatı sahaya normalize edilmiştir.")
        print("Gerçek koşu mesafesi/ısı haritası için tracking'i --calibrate ile tekrar üretmelisin.")

    app = TacticalViewer(
        df=df,
        field_length=args.field_length,
        field_width=args.field_width,
        team_map=team_map,
        hold_s=args.hold_s,
        tail_s=args.tail_s,
    )
    app.show()


if __name__ == "__main__":
    main()
