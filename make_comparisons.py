"""
make_comparisons.py
-------------------
Generates 3-panel comparison figures for 5 representative LOL test images:

    LOW-LIGHT INPUT | OUR ENHANCED OUTPUT | GROUND TRUTH

Saves to results/LOL/comparisons/
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from modules.image_io    import load_image
from modules.pipeline    import enhance
from modules.evaluation  import evaluate

COMPARISON_DIR = os.path.join("results", "LOL", "comparisons")
SUPPORTED_EXT  = {".png", ".jpg", ".jpeg"}

# Same five scenes as the original report, plus the two frames that used to
# stay dark and blotchy (55, 665) so those failures can be checked directly.
SELECTED = ["780.png", "748.png", "1.png", "111.png", "23.png", "55.png", "665.png"]


def find_groundtruth(filename, gt_dir):
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(gt_dir, filename)
    if os.path.isfile(candidate):
        return candidate
    for alt_ext in SUPPORTED_EXT - {ext.lower()}:
        candidate = os.path.join(gt_dir, stem + alt_ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def run_pipeline(image):
    return enhance(image)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_comparison(filename):
    stem     = os.path.splitext(filename)[0]
    low_path = os.path.join(config.DATASET_LOW, filename)
    gt_path  = find_groundtruth(filename, config.DATASET_HIGH)

    if gt_path is None:
        print(f"  [ERROR] Ground-truth not found for {filename} — skipped.")
        return False

    low_image = load_image(low_path)
    gt_image  = load_image(gt_path)
    enhanced  = run_pipeline(low_image)

    if enhanced.shape != gt_image.shape:
        gt_image = cv2.resize(
            gt_image,
            (enhanced.shape[1], enhanced.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    psnr, ssim = evaluate(gt_image, enhanced)

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")

    panels = [
        (low_image, "LOW-LIGHT INPUT",       "#aaaaaa"),
        (enhanced,  "OUR ENHANCED OUTPUT",   "#4fc3f7"),
        (gt_image,  "GROUND TRUTH",          "#81c784"),
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
        f"Image: {filename}    |    PSNR: {psnr:.2f} dB    |    SSIM: {ssim:.4f}",
        color="white", fontsize=12, y=1.01
    )

    plt.tight_layout(pad=1.5)

    out_path = os.path.join(COMPARISON_DIR, f"{stem}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return out_path, psnr, ssim


def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("\nGenerating comparison figures...\n")
    print(f"{'#':<4} {'Filename':<12} {'Role':<18} {'PSNR':>9} {'SSIM':>8}  Saved")
    print("-" * 72)

    roles = {
        "780.png": "Night street",
        "748.png": "Indoor shelf",
        "1.png":   "Indoor shelf",
        "111.png": "Indoor room",
        "23.png":  "Very dark room",
        "55.png":  "Very dark room",
        "665.png": "Very dark room",
    }

    for i, filename in enumerate(SELECTED, 1):
        result = make_comparison(filename)
        if result:
            out_path, psnr, ssim = result
            print(f"{i:<4} {filename:<12} {roles[filename]:<18} "
                  f"{psnr:>9.2f} {ssim:>8.4f}  {os.path.basename(out_path)}")

    print("-" * 72)
    print(f"\nAll comparisons saved to: {os.path.abspath(COMPARISON_DIR)}")
    print("\nSelected files:")
    for i, f in enumerate(SELECTED, 1):
        print(f"  {i}. {f}  ({roles[f]})")


if __name__ == "__main__":
    main()
