# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Regenerates charts/*.png from results/*/runs.csv.

usage: uv run charts/make_charts.py
"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DPI = 200
SIZE = (8, 4.5)  # 1600x900: 16:9 for README, Reddit and X previews

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
# one ordered blue ramp: stock -> unlock -> unlock + undervolt
STOCK, UNLOCK, UNLOCK_UV = "#86b6ef", "#2a78d6", "#104281"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": MUTED,
})

WORKLOADS = [
    ("zimage", "Z-Image Turbo\nBF16, ComfyUI", "img"),
    ("sd15", "SD 1.5\nFP16, ComfyUI", "img"),
    ("vllm_prefill", "vLLM prefill\nQwen3-8B BF16", "token"),
]
# session 3 is the cooler, faster stock run, so gains against it are the conservative ones
CELLS = {"stock": ("session3", "off", "off"), "unlock": ("session2", "on", "off"), "unlock_uv": ("session2", "on", "on")}


def load():
    groups = defaultdict(list)
    for s in ("session2", "session3"):
        with open(os.path.join(RESULTS, s, "runs.csv")) as f:
            for r in csv.DictReader(f):
                groups[(r["session"], r["unlock"], r["undervolt"], r["workload"])].append(r)
    out = {}
    for w, _, _ in WORKLOADS:
        for cell, key in CELLS.items():
            rows = groups[key + (w,)]
            out[w, cell] = {
                "throughput": statistics.median(float(r["throughput"]) for r in rows),
                "j_per_output": statistics.median(float(r["joules"]) / (float(r["throughput"]) * float(r["seconds"])) for r in rows),
            }
    return out


def rounded_bar(ax, x, width, height, color, radius_px=4 * DPI / 100, horizontal=False):
    """Bar with a rounded data end and a square baseline, built in pixel space."""
    ax.figure.canvas.draw()
    if horizontal:
        (x0, y0), (x1, y1) = ax.transData.transform([(0, x - width / 2), (height, x + width / 2)])
    else:
        (x0, y0), (x1, y1) = ax.transData.transform([(x - width / 2, 0), (x + width / 2, height)])
    r = min(radius_px, abs(x1 - x0) / 2, abs(y1 - y0) / 2)
    if horizontal:
        verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    else:
        verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1), (x1 - r, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_artist(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", transform=IdentityTransform(), zorder=3))


def frame(title, subtitle, footer):
    fig = plt.figure(figsize=SIZE, dpi=DPI, facecolor=SURFACE)
    fig.text(0.04, 0.93, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.855, subtitle, fontsize=10.5, color=INK_2, va="top")
    fig.text(0.04, 0.035, footer, fontsize=8, color=MUTED, va="bottom")
    return fig


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend(fig, items, y=0.77):
    x = 0.04
    for label, color in items:
        fig.patches.append(Rectangle((x, y - 0.012), 0.014, 0.025, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        t = fig.text(x + 0.02, y, label, fontsize=9.5, color=INK_2, va="center")
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(t.get_window_extent())[1][0] + 0.03


def pct(x):
    return f"{(x - 1) * 100:+.0f}%".replace("-", "\u2212")


def headline(data):
    fig = frame("Undervolting made my RTX 3090 faster",
                "GPU Unlock alone, then GPU Unlock + a +150 MHz undervolt. Speed vs stock, higher is better.",
                "RTX 3090, stock 350 W limit. Median of 10 timed runs per bar. Stock = the faster of two stock sessions.")
    ax = fig.add_axes([0.07, 0.2, 0.9, 0.52])
    style(ax)
    ax.set_ylim(0, 1.85)
    ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["0", "0.5×", "1.0×", "1.5×"])
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels([w[1] for w in WORKLOADS], fontsize=9.5, color=INK)
    width, gap = 0.17, 0.01
    for i, (w, _, _) in enumerate(WORKLOADS):
        base = data[w, "stock"]["throughput"]
        for j, (cell, color) in enumerate([("stock", STOCK), ("unlock", UNLOCK), ("unlock_uv", UNLOCK_UV)]):
            x = i + (j - 1) * (width + gap)
            v = data[w, cell]["throughput"] / base
            rounded_bar(ax, x, width, v, color)
            if cell == "unlock":
                # inside the bar: above it, it would collide with its taller neighbour's label
                ax.text(x, v - 0.05, pct(v), ha="center", va="top", fontsize=9.5, color="#ffffff")
            elif cell == "unlock_uv":
                ax.text(x, v + 0.04, pct(v), ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    legend(fig, [("Stock", STOCK), ("GPU Unlock", UNLOCK), ("GPU Unlock + undervolt", UNLOCK_UV)])
    fig.savefig(os.path.join(HERE, "speedup.png"), dpi=DPI, facecolor=SURFACE)


def energy(data):
    fig = frame("About 40% less energy per output",
                "Energy per image or per prompt token, GPU Unlock + undervolt vs stock. Lower is better.",
                "Joules = average board power × run time. Bars indexed to stock = 100%. Median of 10 timed runs.")
    ax = fig.add_axes([0.07, 0.2, 0.9, 0.52])
    style(ax)
    ax.set_ylim(0, 1.25)
    ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels([w[1] for w in WORKLOADS], fontsize=9.5, color=INK)
    width, gap = 0.22, 0.01
    for i, (w, _, unit) in enumerate(WORKLOADS):
        base = data[w, "stock"]["j_per_output"]
        for j, (cell, color) in enumerate([("stock", STOCK), ("unlock_uv", UNLOCK_UV)]):
            x = i + (j - 0.5) * (width + gap)
            jpo = data[w, cell]["j_per_output"]
            rounded_bar(ax, x, width, jpo / base, color)
            value = f"{jpo:,.0f}\nJ/{unit}" if jpo >= 1 else f"{jpo:.3f}\nJ/{unit}"
            # values sit inside the bars: two bars' labels side by side overlap at this width
            ax.text(x, jpo / base - 0.04, value, ha="center", va="top", fontsize=9,
                    color=INK if cell == "stock" else "#ffffff", linespacing=1.3)
            if cell == "unlock_uv":
                ax.text(x, jpo / base + 0.03, pct(jpo / base), ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    legend(fig, [("Stock", STOCK), ("GPU Unlock + undervolt", UNLOCK_UV)])
    fig.savefig(os.path.join(HERE, "energy.png"), dpi=DPI, facecolor=SURFACE)


def correctness():
    with open(os.path.join(RESULTS, "soak-20261028-1327", "runs.csv")) as f:
        soak = list(csv.DictReader(f))
    session_hashed = []
    for s in ("session1", "session2", "session3"):
        with open(os.path.join(RESULTS, s, "runs.csv")) as f:
            session_hashed += [r for r in csv.DictReader(f) if r["pass"]]
    soak_pass = sum(r["pass"] == "1" for r in soak)
    session_fail = sum(r["pass"] == "0" for r in session_hashed)

    fig = frame("Same bits as stock",
                "Every hashed output matched the stock reference (SHA-256), with and without the unlock and undervolt.",
                "Soak: 2 h of fixed-seed SD 1.5 (batch 4) and Z-Image batches. Image hashes are over decoded pixels.")
    fig.text(0.04, 0.6, f"{soak_pass}/{len(soak)}", fontsize=50, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.47, "soak batches bit-identical", fontsize=12, color=INK_2, va="center")
    fig.text(0.04, 0.32, f"{session_fail}", fontsize=34, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.23, f"hash mismatches in {len(session_hashed)} hashed benchmark runs\n(GEMM, LLM tokens, SD 1.5, Z-Image; 3 sessions)",
             fontsize=10, color=INK_2, va="center", linespacing=1.4)

    ax = fig.add_axes([0.47, 0.13, 0.5, 0.66])
    ax.set_axis_off()
    cols = 30
    for k, (w, label) in enumerate([("sd15", "SD 1.5"), ("zimage", "Z-Image")]):
        rows = [r for r in soak if r["workload"] == w]
        y0 = 0 if k == 0 else -(len(rows) // cols + 3)
        for n, r in enumerate(rows):
            ax.add_patch(Rectangle((n % cols, y0 - n // cols), 0.78, 0.78,
                                   facecolor=GOOD if r["pass"] == "1" else "#d03b3b", edgecolor="none"))
        ok = sum(r["pass"] == "1" for r in rows)
        ax.plot([0.05, 0.3, 0.8], [y0 + 1.4, y0 + 1.05, y0 + 1.85], color=GOOD, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.text(1.3, y0 + 1.4, f"{label}: {ok}/{len(rows)} identical", fontsize=10, color=INK, va="center")
    ax.set_xlim(-0.2, cols)
    ax.set_ylim(-23, 2.5)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(HERE, "correctness.png"), dpi=DPI, facecolor=SURFACE)


if __name__ == "__main__":
    d = load()
    headline(d)
    energy(d)
    correctness()
