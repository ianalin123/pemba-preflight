"""Generate demo figures from real D1 traces → assets/*.png.

Usage: uv run --with matplotlib python scripts/make_figures.py
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

HEALTHY = ROOT / "data/d1_cold/left_elbow_joint_off0.70_1788078001.npz"
BROWNOUT = ROOT / "data/d1_cold_bad/left_elbow_joint_off0.70_1788078146.npz"


def load(path):
    d = np.load(path)
    return d["t"], d["target"], d["q"]


def fig_brownout():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, path, title, z in (
        (axes[0], HEALTHY, "Healthy — 3 min earlier", "z = 0.67 · PASS"),
        (axes[1], BROWNOUT, "Battery brown-out (red light)", "z = 20.9 · FAIL"),
    ):
        t, tgt, q = load(path)
        ax.plot(t, tgt, "--", color="#888", lw=1.2, label="commanded")
        ax.plot(t, q, color="#c0392b" if "brown" in title.lower() else "#2471a3",
                lw=1.6, label="measured")
        ax.set_title(f"{title}   ({z})", fontsize=11)
        ax.set_xlabel("t (s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("left elbow q (rad)")
    fig.suptitle("Same joint, same probe, 139 s apart — tier-1 caught the battery dying", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "brownout_detection.png", dpi=160)
    print("saved assets/brownout_detection.png")


def fig_temps():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    series = {}
    for f in sorted((ROOT / "data/d1_cold").glob("*.npz")):
        joint = f.stem.rsplit("_off", 1)[0]
        ts = int(f.stem.split("_")[-1])
        temp = np.load(f)["temp"]
        series.setdefault(joint, []).append((ts, float(np.max(temp))))
    t0 = min(ts for pts in series.values() for ts, _ in pts)
    for joint, pts in sorted(series.items()):
        pts.sort()
        xs = [(ts - t0) / 60 for ts, _ in pts]
        ys = [tmp for _, tmp in pts]
        ax.plot(xs, ys, "o-", ms=3, lw=1, label=joint.replace("_joint", ""))
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("motor temp (°C)")
    ax.set_title("Motor temperatures across the 40-min gantry session (from probe telemetry)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "session_temps.png", dpi=160)
    print("saved assets/session_temps.png")


if __name__ == "__main__":
    fig_brownout()
    fig_temps()
