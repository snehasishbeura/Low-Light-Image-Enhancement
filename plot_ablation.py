"""
plot_ablation.py
----------------
Reads results/LOL/ablation_study.csv and produces a dual-axis chart:
  - Primary   y-axis : Mean PSNR (dB)  — bar chart
  - Secondary y-axis : Mean SSIM       — line chart with markers

Saves to results/LOL/ablation_curve.png at 300 DPI.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH  = os.path.join("results", "LOL", "ablation_study.csv")
SAVE_PATH = os.path.join("results", "LOL", "ablation_curve.png")

# ------------------------------------------------------------------ #
# Read CSV
# ------------------------------------------------------------------ #
stages, psnr_vals, ssim_vals = [], [], []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stages.append(row["Stage"])
        psnr_vals.append(float(row["Mean PSNR (dB)"]))
        ssim_vals.append(float(row["Mean SSIM"]))

# short x-axis labels for readability
x_labels = [
    "Baseline\n(Raw)",
    "+Illumination\n(LIME)",
    "+Adaptive\nTone",
    "+Detail\nRefine",
    "+Color\nRestore",
]

x = np.arange(len(stages))

# ------------------------------------------------------------------ #
# Figure
# ------------------------------------------------------------------ #
fig, ax1 = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#f9f9f9")
ax1.set_facecolor("#f9f9f9")

# --- bars: PSNR ---
bar_colors = ["#4a90d9", "#4a90d9", "#4a90d9", "#4a90d9", "#4a90d9"]
bars = ax1.bar(
    x, psnr_vals,
    width=0.45,
    color=bar_colors,
    alpha=0.75,
    zorder=2,
    label="Mean PSNR (dB)"
)

# PSNR data labels on top of each bar
for bar, val in zip(bars, psnr_vals):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.25,
        f"{val:.2f}",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#1a5276"
    )

ax1.set_xlabel("Pipeline Stage", fontsize=12, labelpad=10)
ax1.set_ylabel("Mean PSNR (dB)", fontsize=12, color="#1a5276")
ax1.tick_params(axis="y", labelcolor="#1a5276")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_ylim(0, max(psnr_vals) + 3)
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.grid(axis="y", which="major", linestyle="--",
         linewidth=0.7, alpha=0.6, zorder=1)
ax1.grid(axis="y", which="minor", linestyle=":",
         linewidth=0.4, alpha=0.4, zorder=1)

# --- line: SSIM ---
ax2 = ax1.twinx()
ax2.plot(
    x, ssim_vals,
    color="#e74c3c",
    linewidth=2.5,
    marker="o",
    markersize=8,
    markerfacecolor="white",
    markeredgecolor="#e74c3c",
    markeredgewidth=2.5,
    zorder=3,
    label="Mean SSIM"
)

# SSIM data labels above each point
for xi, val in zip(x, ssim_vals):
    ax2.text(
        xi,
        val + 0.018,
        f"{val:.4f}",
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold", color="#922b21"
    )

ax2.set_ylabel("Mean SSIM", fontsize=12, color="#922b21")
ax2.tick_params(axis="y", labelcolor="#922b21")
ax2.set_ylim(0, 1.1)
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# --- combined legend ---
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    fontsize=10,
    framealpha=0.9
)

# --- title ---
plt.title(
    "Ablation Study — Mean PSNR and SSIM at Each Pipeline Stage\n"
    "LOL eval15 Dataset (15 images)",
    fontsize=13, fontweight="bold", pad=15
)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()

print(f"Chart saved to: {os.path.abspath(SAVE_PATH)}")
