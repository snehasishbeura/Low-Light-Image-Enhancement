"""
evaluate_lol.py
---------------
Benchmarks the existing enhancement pipeline on LOL or LOL-v2 Real datasets.

Usage
-----
  python evaluate_lol.py --dataset lol        # LOL eval15        (15 images)
  python evaluate_lol.py --dataset lolv2      # LOL-v2 Real       (100 images)
  python evaluate_lol.py --dataset lolv2syn   # LOL-v2 Synthetic  (100 images)
  python evaluate_lol.py --dataset unlol      # UnLOL test        (paired JPEGs)

Outputs (per dataset)
---------------------
  results/LOL/            or   results/LOLv2_Real/   or   results/UnLOL/
    enhanced/                    - every enhanced image
    metrics.csv                  - per-image PSNR and SSIM
    summary.txt                  - final averaged metrics
    ablation_study.csv           - per-stage mean PSNR, SSIM, delta
    ablation_curve.png           - dual-axis PSNR/SSIM chart

The enhancer never reads the ground-truth image. Metrics are computed
afterwards, against the paired normal-light photo.
"""

import os
import re
import csv
import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import config
from modules.image_io   import load_image
from modules.pipeline   import STAGE_NAMES, enhance, run_stages
from modules.evaluation import evaluate, export_ablation_study

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}

ABLATION_STAGES = STAGE_NAMES


# ------------------------------------------------------------------ #
# Dataset configuration
# ------------------------------------------------------------------ #

def get_dataset_config(dataset):
    if dataset == "lol":
        return {
            "name"       : "LOL",
            "low_dir"    : config.DATASET_LOW,
            "high_dir"   : config.DATASET_HIGH,
            "results_dir": os.path.join("results", "LOL"),
            "pairing"    : "exact",        # filename identical in both folders
        }
    elif dataset == "lolv2":
        return {
            "name"       : "LOL-v2 Real",
            "low_dir"    : config.LOLV2_REAL_LOW,
            "high_dir"   : config.LOLV2_REAL_HIGH,
            "results_dir": os.path.join("results", "LOLv2_Real"),
            "pairing"    : "lolv2",        # low00690.png <-> normal00690.png
        }
    elif dataset == "lolv2syn":
        return {
            "name"       : "LOL-v2 Synthetic",
            "low_dir"    : config.LOLV2_SYN_LOW,
            "high_dir"   : config.LOLV2_SYN_HIGH,
            "results_dir": os.path.join("results", "LOLv2_Synthetic"),
            "pairing"    : "exact",        # same filename in Low/ and Normal/
        }
    elif dataset == "unlol":
        return {
            "name"       : "UnLOL",
            "low_dir"    : config.UNLOL_LOW,
            "high_dir"   : config.UNLOL_HIGH,
            "results_dir": os.path.join("results", "UnLOL"),
            "pairing"    : "exact",        # 0103.jpeg in Low/ and High/
        }
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Use 'lol', 'lolv2', 'lolv2syn', or 'unlol'."
        )


# ------------------------------------------------------------------ #
# Pairing helpers
# ------------------------------------------------------------------ #

def find_gt_exact(filename, gt_dir):
    """LOL: ground-truth has the exact same filename."""
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(gt_dir, filename)
    if os.path.isfile(candidate):
        return candidate
    for alt_ext in SUPPORTED_EXT - {ext.lower()}:
        candidate = os.path.join(gt_dir, stem + alt_ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_gt_lolv2(filename, gt_dir):
    """
    LOL-v2: low00690.png -> normal00690.png
    Replaces the 'low' prefix with 'normal', keeps the zero-padded ID and ext.
    """
    # extract the zero-padded numeric suffix e.g. '00690' from 'low00690.png'
    m = re.match(r"low(\d+)(\.[^.]+)$", filename, re.IGNORECASE)
    if not m:
        return None
    gt_name = f"normal{m.group(1)}{m.group(2).lower()}"
    candidate = os.path.join(gt_dir, gt_name)
    return candidate if os.path.isfile(candidate) else None


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def run_pipeline(image):
    """Final enhanced image. Ground truth is not used."""
    return enhance(image)


def run_ablation_stages(image):
    """Return intermediate outputs at each stage."""
    return run_stages(image)


# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #

def save_ablation_plot(ablation_data, save_path):
    stages    = list(ablation_data.keys())
    psnr_vals = [ablation_data[s][0] for s in stages]
    ssim_vals = [ablation_data[s][1] for s in stages]

    short_labels = {
        "Raw Low-Light": "Baseline\n(Raw)",
        "After Illumination": "+Illumination\n(LIME)",
        "After Adaptive Tone": "+Adaptive\nTone",
        "After Detail Refine": "+Detail\nRefine",
        "After Color Restore": "+Color\nRestore",
        "After Grain Suppress": "+Grain\nSuppress",
    }
    x_labels = [short_labels.get(stage, stage) for stage in stages]
    x = np.arange(len(stages))

    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#f9f9f9")
    ax1.set_facecolor("#f9f9f9")

    bars = ax1.bar(x, psnr_vals, width=0.45, color="#4a90d9",
                   alpha=0.75, zorder=2, label="Mean PSNR (dB)")
    for bar, val in zip(bars, psnr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.25, f"{val:.2f}",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#1a5276")

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

    ax2 = ax1.twinx()
    ax2.plot(x, ssim_vals, color="#e74c3c", linewidth=2.5,
             marker="o", markersize=8, markerfacecolor="white",
             markeredgecolor="#e74c3c", markeredgewidth=2.5,
             zorder=3, label="Mean SSIM")
    for xi, val in zip(x, ssim_vals):
        ax2.text(xi, val + 0.018, f"{val:.4f}",
                 ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color="#922b21")

    ax2.set_ylabel("Mean SSIM", fontsize=12, color="#922b21")
    ax2.tick_params(axis="y", labelcolor="#922b21")
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               fontsize=10, framealpha=0.9)

    dataset_label = os.path.basename(os.path.dirname(save_path))
    plt.title(
        f"Ablation Study - Mean PSNR and SSIM at Each Pipeline Stage\n"
        f"Dataset: {dataset_label}",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ------------------------------------------------------------------ #
# Main evaluation
# ------------------------------------------------------------------ #

def main(dataset):
    cfg          = get_dataset_config(dataset)
    low_dir      = cfg["low_dir"]
    high_dir     = cfg["high_dir"]
    results_dir  = cfg["results_dir"]
    pairing      = cfg["pairing"]
    dataset_name = cfg["name"]

    enhanced_dir  = os.path.join(results_dir, "enhanced")
    csv_path      = os.path.join(results_dir, "metrics.csv")
    summary_path  = os.path.join(results_dir, "summary.txt")
    ablation_csv  = os.path.join(results_dir, "ablation_study.csv")
    ablation_plot = os.path.join(results_dir, "ablation_curve.png")

    os.makedirs(enhanced_dir, exist_ok=True)

    low_files = sorted([
        f for f in os.listdir(low_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not low_files:
        print(f"[ERROR] No images found in: {low_dir}")
        return

    total = len(low_files)

    # no pre-built index needed for lolv2 (direct filename substitution)
    gt_index = None

    rows      = []
    psnr_list = []
    ssim_list = []
    failed    = []

    # per-stage accumulators for ablation
    stage_psnr = {s: [] for s in ABLATION_STAGES}
    stage_ssim = {s: [] for s in ABLATION_STAGES}

    print(f"\n{dataset_name} Evaluation -- {total} images found")
    print(f"{'Image':<20} {'PSNR (dB)':>10} {'SSIM':>8}  Status")
    print("=" * 58)

    for filename in low_files:
        low_path = os.path.join(low_dir, filename)
        stem     = os.path.splitext(filename)[0]

        # --- locate ground-truth ---
        if pairing == "exact":
            gt_path = find_gt_exact(filename, high_dir)
        else:  # lolv2
            gt_path = find_gt_lolv2(filename, high_dir)

        if gt_path is None:
            reason = "ground-truth not found"
            print(f"{filename:<20} {'--':>10} {'--':>8}  [FAILED] {reason}")
            failed.append((filename, reason))
            rows.append([filename, "", "", "FAILED", reason])
            continue

        # --- load ---
        try:
            low_image = load_image(low_path)
            gt_image  = load_image(gt_path)
        except Exception as e:
            reason = str(e)
            print(f"{filename:<20} {'--':>10} {'--':>8}  [FAILED] {reason}")
            failed.append((filename, reason))
            rows.append([filename, "", "", "FAILED", reason])
            continue

        # --- ablation stages ---
        try:
            stages_out = run_ablation_stages(low_image)
        except Exception as e:
            reason = f"pipeline error: {e}"
            print(f"{filename:<20} {'--':>10} {'--':>8}  [FAILED] {reason}")
            failed.append((filename, reason))
            rows.append([filename, "", "", "FAILED", reason])
            continue

        # --- evaluate each stage ---
        for stage, img in zip(ABLATION_STAGES, stages_out):
            gt = gt_image
            if img.shape != gt.shape:
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_AREA)
            try:
                p, s = evaluate(gt, img)
                stage_psnr[stage].append(p)
                stage_ssim[stage].append(s)
            except Exception:
                pass

        # final stage = Color Restore
        enhanced = stages_out[-1]
        gt_eval  = gt_image
        if enhanced.shape != gt_eval.shape:
            gt_eval = cv2.resize(gt_eval,
                                 (enhanced.shape[1], enhanced.shape[0]),
                                 interpolation=cv2.INTER_AREA)

        # --- save enhanced ---
        cv2.imwrite(os.path.join(enhanced_dir, f"{stem}.png"), enhanced)

        # --- final metrics ---
        try:
            psnr, ssim = evaluate(gt_eval, enhanced)
        except Exception as e:
            reason = f"metric error: {e}"
            print(f"{filename:<20} {'--':>10} {'--':>8}  [FAILED] {reason}")
            failed.append((filename, reason))
            rows.append([filename, "", "", "FAILED", reason])
            continue

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        rows.append([filename, f"{psnr:.4f}", f"{ssim:.4f}", "OK", ""])
        print(f"{filename:<20} {psnr:>10.2f} {ssim:>8.4f}  OK")

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    n = len(psnr_list)
    print("=" * 58)

    if n == 0:
        print("[ERROR] No images were successfully evaluated.")
        return

    avg_psnr = sum(psnr_list) / n
    avg_ssim = sum(ssim_list) / n
    min_psnr = min(psnr_list)
    max_psnr = max(psnr_list)
    min_ssim = min(ssim_list)
    max_ssim = max(ssim_list)

    sep = "-" * 44
    for line in [
        "", sep, f"  FINAL EVALUATION SUMMARY — {dataset_name}", sep,
        f"  Total images     : {total}",
        f"  Processed OK     : {n}",
        f"  Failed           : {len(failed)}", sep,
        f"  Avg  PSNR (dB)   : {avg_psnr:.4f}",
        f"  Min  PSNR (dB)   : {min_psnr:.4f}",
        f"  Max  PSNR (dB)   : {max_psnr:.4f}", sep,
        f"  Avg  SSIM        : {avg_ssim:.4f}",
        f"  Min  SSIM        : {min_ssim:.4f}",
        f"  Max  SSIM        : {max_ssim:.4f}", sep,
    ]:
        print(line)

    if failed:
        print("\nFailed images:")
        for fname, reason in failed:
            print(f"  {fname}: {reason}")

    # ------------------------------------------------------------------ #
    # Save metrics CSV
    # ------------------------------------------------------------------ #
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "PSNR", "SSIM"])
        for row in rows:
            if row[3] == "OK":
                writer.writerow([row[0], row[1], row[2]])
        writer.writerow([])
        writer.writerow(["Average", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}"])

    # ------------------------------------------------------------------ #
    # Save summary txt
    # ------------------------------------------------------------------ #
    with open(summary_path, "w") as f:
        f.write("========================================\n")
        f.write(f"{dataset_name.upper()} DATASET EVALUATION\n")
        f.write("========================================\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Test Images: {total}\n\n")
        f.write(f"Average PSNR : {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM : {avg_ssim:.4f}\n\n")
        f.write(f"Minimum PSNR : {min_psnr:.2f} dB\n")
        f.write(f"Maximum PSNR : {max_psnr:.2f} dB\n\n")
        f.write(f"Minimum SSIM : {min_ssim:.4f}\n")
        f.write(f"Maximum SSIM : {max_ssim:.4f}\n\n")
        f.write(f"Successfully processed: {n}/{total}\n")
        f.write("========================================\n")

    # ------------------------------------------------------------------ #
    # Ablation study export
    # ------------------------------------------------------------------ #
    ablation_data = {
        s: (
            sum(stage_psnr[s]) / len(stage_psnr[s]),
            sum(stage_ssim[s]) / len(stage_ssim[s])
        )
        for s in ABLATION_STAGES if stage_psnr[s]
    }
    export_ablation_study(ablation_data, output_path=ablation_csv)
    save_ablation_plot(ablation_data, save_path=ablation_plot)

    # ------------------------------------------------------------------ #
    # Print ablation summary table
    # ------------------------------------------------------------------ #
    sep2 = "-" * 44
    print(f"\n{sep2}")
    print(f"  ABLATION SUMMARY -- {dataset_name}")
    print(sep2)
    print(f"  {'Stage':<22} {'PSNR (dB)':>10} {'SSIM':>8} {'Delta PSNR':>12}")
    print(sep2)
    prev = None
    for stage in ABLATION_STAGES:
        if stage_psnr[stage]:
            mp = sum(stage_psnr[stage]) / len(stage_psnr[stage])
            ms = sum(stage_ssim[stage]) / len(stage_ssim[stage])
            d = mp - prev if prev is not None else None
            delta = f"{d:+.4f}" if d is not None else "  baseline"
            print(f"  {stage:<22} {mp:>10.4f} {ms:>8.4f} {delta:>12}")
            prev = mp
    print(sep2)

    print(f"\nMetrics CSV      : {os.path.abspath(csv_path)}")
    print(f"Summary          : {os.path.abspath(summary_path)}")
    print(f"Ablation CSV     : {os.path.abspath(ablation_csv)}")
    print(f"Ablation chart   : {os.path.abspath(ablation_plot)}")
    print(f"Enhanced images  : {os.path.abspath(enhanced_dir)}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate enhancement pipeline on LOL, LOL-v2, or UnLOL."
    )
    parser.add_argument(
        "--dataset",
        choices=["lol", "lolv2", "lolv2syn", "unlol"],
        default="lol",
        help="Dataset: 'lol' (eval15), 'lolv2' (Real test), 'lolv2syn' (Synthetic test), or 'unlol' (UnLOL test)."
    )
    args = parser.parse_args()
    main(args.dataset)
