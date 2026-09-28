"""
make_comparisons_v2.py
----------------------
Reads a LOL-v2 metrics.csv and saves 3-panel figures:

    LOW-LIGHT INPUT | OUR ENHANCED OUTPUT | GROUND TRUTH

  python make_comparisons_v2.py              # LOL-v2 Real, top 3 by PSNR
  python make_comparisons_v2.py --dataset lolv2syn
                                             # Synthetic: best, median, worst
"""

import os
import re
import csv
import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SUPPORTED_EXT  = {".png", ".jpg", ".jpeg"}
TOP_N          = 3


def dataset_paths(dataset):
    if dataset == "lolv2syn":
        root = os.path.join("results", "LOLv2_Synthetic")
        return {
            "name": "LOL-v2 Synthetic",
            "metrics": os.path.join(root, "metrics.csv"),
            "enhanced": os.path.join(root, "enhanced"),
            "comparisons": os.path.join(root, "comparisons"),
            "low": config.LOLV2_SYN_LOW,
            "high": config.LOLV2_SYN_HIGH,
            "pairing": "exact",
            "eval_flag": "lolv2syn",
        }
    root = os.path.join("results", "LOLv2_Real")
    return {
        "name": "LOL-v2 Real",
        "metrics": os.path.join(root, "metrics.csv"),
        "enhanced": os.path.join(root, "enhanced"),
        "comparisons": os.path.join(root, "comparisons"),
        "low": config.LOLV2_REAL_LOW,
        "high": config.LOLV2_REAL_HIGH,
        "pairing": "lolv2",
        "eval_flag": "lolv2",
    }


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def read_metrics(csv_path):
    """Return list of (filename, psnr, ssim) sorted by PSNR descending."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((
                    row["Image Name"],
                    float(row["PSNR"]),
                    float(row["SSIM"])
                ))
            except (ValueError, KeyError):
                continue   # skip Average row and any malformed rows
    return sorted(rows, key=lambda x: x[1], reverse=True)


def find_gt(low_filename, gt_dir, pairing):
    """Locate the normal-light partner for a low-light filename."""
    if pairing == "exact":
        candidate = os.path.join(gt_dir, low_filename)
        return candidate if os.path.isfile(candidate) else None

    # Real captured: low00763.png -> normal00763.png
    nums = re.findall(r"\d+", low_filename)
    if not nums:
        return None
    target_id = str(int(nums[-1]))
    for name in os.listdir(gt_dir):
        if os.path.splitext(name)[1].lower() not in SUPPORTED_EXT:
            continue
        file_nums = re.findall(r"\d+", name)
        if file_nums and str(int(file_nums[-1])) == target_id:
            return os.path.join(gt_dir, name)
    return None


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_comparison(paths, rank_label, filename, psnr, ssim):
    stem = os.path.splitext(filename)[0]

    low_path      = os.path.join(paths["low"], filename)
    enhanced_path = os.path.join(paths["enhanced"], f"{stem}.png")
    gt_path       = find_gt(filename, paths["high"], paths["pairing"])

    if not os.path.isfile(low_path):
        print(f"  [ERROR] Low-light image not found: {low_path}")
        return False
    if not os.path.isfile(enhanced_path):
        print(f"  [ERROR] Enhanced image not found: {enhanced_path}")
        return False
    if gt_path is None or not os.path.isfile(gt_path):
        print(f"  [ERROR] Ground-truth not found for: {filename}")
        return False

    low_image = cv2.imread(low_path)
    enhanced  = cv2.imread(enhanced_path)
    gt_image  = cv2.imread(gt_path)

    if low_image is None or enhanced is None or gt_image is None:
        print(f"  [ERROR] Could not read images for: {filename}")
        return False

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")

    panels = [
        (low_image, "LOW-LIGHT INPUT",     "#aaaaaa"),
        (enhanced,  "OUR ENHANCED OUTPUT", "#4fc3f7"),
        (gt_image,  "GROUND TRUTH",        "#81c784"),
    ]

    for ax, (img, label, color) in zip(axes, panels):
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(label, color=color, fontsize=13,
                     fontweight="bold", pad=12, fontfamily="monospace")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    fig.suptitle(
        f"{paths['name']}  |  {rank_label}  |  {filename}  |  "
        f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}",
        color="white", fontsize=12, y=1.01
    )

    plt.tight_layout(pad=1.5)

    out_path = os.path.join(paths["comparisons"], f"{stem}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return out_path


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def select_rows(all_results, dataset):
    """Real keeps the historical top-3. Synthetic also shows a middle and a weak frame."""
    if dataset != "lolv2syn":
        return [(f"Rank #{i} by PSNR", row) for i, row in enumerate(all_results[:TOP_N], 1)]

    best = all_results[0]
    worst = all_results[-1]
    middle = all_results[len(all_results) // 2]
    chosen = [
        ("Best PSNR", best),
        ("Median PSNR", middle),
        ("Lowest PSNR", worst),
    ]
    # Drop duplicates if the set is tiny.
    seen = set()
    unique = []
    for label, row in chosen:
        if row[0] in seen:
            continue
        seen.add(row[0])
        unique.append((label, row))
    return unique


def main():
    parser = argparse.ArgumentParser(description="Side-by-side LOL-v2 comparisons.")
    parser.add_argument(
        "--dataset",
        choices=["lolv2", "lolv2syn"],
        default="lolv2",
        help="lolv2 = Real test, lolv2syn = Synthetic test.",
    )
    args = parser.parse_args()
    paths = dataset_paths(args.dataset)
    os.makedirs(paths["comparisons"], exist_ok=True)

    if not os.path.isfile(paths["metrics"]):
        print(f"[ERROR] metrics.csv not found at: {paths['metrics']}")
        print(f"Run: python evaluate_lol.py --dataset {paths['eval_flag']}  first.")
        return

    all_results = read_metrics(paths["metrics"])
    if not all_results:
        print("[ERROR] No valid rows found in metrics.csv.")
        return

    picked = select_rows(all_results, args.dataset)

    print(f"\n{paths['name']} comparisons:\n")
    print(f"{'Role':<16} {'Filename':<22} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("-" * 62)

    for label, (filename, psnr, ssim) in picked:
        result = make_comparison(paths, label, filename, psnr, ssim)
        status = os.path.basename(result) if result else "FAILED"
        print(f"{label:<16} {filename:<22} {psnr:>10.2f} {ssim:>8.4f}  -> {status}")

    print("-" * 62)
    print(f"\nComparisons saved to: {os.path.abspath(paths['comparisons'])}")


if __name__ == "__main__":
    main()
