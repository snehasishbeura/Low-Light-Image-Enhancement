"""
visualize.py
------------
Processes the first LOL test image through the existing pipeline and saves:

    results/
    └── LOL/
        ├── enhanced/
        │   └── 1.png
        └── comparisons/
            └── 1_comparison.png

The comparison shows:
    Low-light input | Enhanced output | Ground-truth

"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from modules.image_io   import load_image
from modules.pipeline   import enhance
from modules.evaluation import evaluate

ENHANCED_DIR    = os.path.join("results", "LOL", "enhanced")
COMPARISON_DIR  = os.path.join("results", "LOL", "comparisons")
SUPPORTED_EXT   = {".png", ".jpg", ".jpeg"}


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


def main():
    os.makedirs(ENHANCED_DIR,   exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    low_files = sorted([
        f for f in os.listdir(config.DATASET_LOW)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not low_files:
        print(f"[ERROR] No images found in: {config.DATASET_LOW}")
        return

    filename = low_files[0]
    stem     = os.path.splitext(filename)[0]
    low_path = os.path.join(config.DATASET_LOW, filename)
    gt_path  = find_groundtruth(filename, config.DATASET_HIGH)

    if gt_path is None:
        print(f"[ERROR] Ground-truth not found for: {filename}")
        return

    # --- load ---
    low_image = load_image(low_path)
    gt_image  = load_image(gt_path)

    # --- run existing pipeline (unchanged) ---
    enhanced = run_pipeline(low_image)

    # --- dimension guard ---
    if enhanced.shape != gt_image.shape:
        gt_image = cv2.resize(
            gt_image,
            (enhanced.shape[1], enhanced.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    # --- metrics ---
    psnr, ssim = evaluate(gt_image, enhanced)

    # ------------------------------------------------------------------ #
    # 1. Save enhanced image
    # ------------------------------------------------------------------ #
    enhanced_path = os.path.join(ENHANCED_DIR, f"{stem}.png")
    cv2.imwrite(enhanced_path, enhanced)

    # ------------------------------------------------------------------ #
    # 2. Save 3-panel comparison
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#111111")

    panels = [
        (low_image,  "Low-light Input",              "#888888"),
        (enhanced,   f"Enhanced Output\nPSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}", "#4fc3f7"),
        (gt_image,   "Ground-truth",                 "#81c784"),
    ]

    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(title, color=color, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(
        f"Low-Light Image Enhancement  ·  {filename}",
        color="white", fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    comparison_path = os.path.join(COMPARISON_DIR, f"{stem}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # ------------------------------------------------------------------ #
    # 3. Report
    # ------------------------------------------------------------------ #
    print(f"Input            : {filename}")
    print(f"Ground-truth     : {os.path.basename(gt_path)}")
    print(f"Enhanced saved   : {enhanced_path}")
    print(f"Comparison saved : {comparison_path}")
    print(f"PSNR             : {psnr:.2f} dB")
    print(f"SSIM             : {ssim:.4f}")
    print(f"Status           : SUCCESS")


if __name__ == "__main__":
    main()
