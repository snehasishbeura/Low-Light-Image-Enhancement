"""
Enhance every image in the input folder and save each pipeline stage.

    python main.py

Ground truth is not required and is not read. For PSNR/SSIM on LOL or
LOL-v2, run evaluate_lol.py after the dataset paths in config.py are set.
"""

import os
import sys

import cv2

import config
from modules.image_io import load_image
from modules.pipeline import STAGE_NAMES, run_stages

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INPUT_DIR = os.path.join(config.ROOT, "input")
STAGE_FILES = [
    "0_input.jpg",
    "1_illumination.jpg",
    "2_adaptive_tone.jpg",
    "3_detail_refine.jpg",
    "4_final.jpg",
]


def list_inputs(folder):
    if not os.path.isdir(folder):
        return []
    names = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXT:
            names.append(name)
    return names


def enhance_file(path, output_dir):
    image = load_image(path)
    stages = run_stages(image)
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for stage_name, filename, stage in zip(STAGE_NAMES, STAGE_FILES, stages):
        # One image keeps the historical flat names. A folder of images is
        # written as <stem>_<stage> so results do not overwrite each other.
        if len(list_inputs(INPUT_DIR)) == 1:
            out_name = filename
        else:
            out_name = f"{stem}_{filename}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, stage)
        written.append((stage_name, out_path))
    return written


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    names = list_inputs(folder)
    if not names:
        print(f"No images found in {folder}")
        print("Put a low-light photo in the input folder and run: python main.py")
        return 1

    output_dir = os.path.join(config.ROOT, config.OUTPUT_FOLDER)
    print(f"Enhancing {len(names)} image(s) from {folder}")
    for name in names:
        written = enhance_file(os.path.join(folder, name), output_dir)
        print(f"\n{name}")
        for stage_name, path in written:
            print(f"  {stage_name:<24} {path}")
    print("\nFinal image is 4_final.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
