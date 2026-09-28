import cv2
import csv
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def evaluate(original, enhanced):

    psnr = peak_signal_noise_ratio(original, enhanced)

    gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    ssim = structural_similarity(gray1, gray2)

    return psnr, ssim


def _technique_for(stage):
    """Look up the description written by the active pipeline."""
    try:
        from modules.pipeline import STAGE_TECHNIQUES
        return STAGE_TECHNIQUES.get(stage, stage)
    except Exception:
        return stage


def export_ablation_study(ablation_data, output_path="output/ablation_study.csv"):
    """
    Export ablation study results to a CSV file and print a Markdown table.

    Parameters
    ----------
    ablation_data : dict
        Keys   : stage names matching _TECHNIQUES above
        Values : (mean_psnr, mean_ssim) tuples
    output_path : str
        Path for the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    stages     = list(ablation_data.keys())
    rows       = []
    prev_psnr  = None

    for stage in stages:
        mean_psnr, mean_ssim = ablation_data[stage]
        technique = _technique_for(stage)
        delta     = (mean_psnr - prev_psnr) if prev_psnr is not None else 0.0
        prev_psnr = mean_psnr
        rows.append({
            "Stage"          : stage,
            "Technique Applied": technique,
            "Mean PSNR (dB)" : round(mean_psnr, 4),
            "Mean SSIM"      : round(mean_ssim, 4),
            "Delta PSNR Gain": round(delta, 4),
        })

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #
    fieldnames = ["Stage", "Technique Applied",
                  "Mean PSNR (dB)", "Mean SSIM", "Delta PSNR Gain"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------ #
    # Markdown table
    # ------------------------------------------------------------------ #
    c1, c2, c3, c4, c5 = 22, 47, 16, 12, 16

    sep  = f"| {'-'*c1} | {'-'*c2} | {'-'*c3} | {'-'*c4} | {'-'*c5} |"
    head = (f"| {'Stage':<{c1}} | {'Technique Applied':<{c2}} |"
            f" {'Mean PSNR (dB)':>{c3}} | {'Mean SSIM':>{c4}} |"
            f" {'Delta PSNR Gain':>{c5}} |")

    print("\n## Ablation Study\n")
    print(head)
    print(sep)

    for row in rows:
        delta_str = f"+{row['Delta PSNR Gain']:.4f}" if row["Delta PSNR Gain"] > 0 else f"{row['Delta PSNR Gain']:.4f}"
        print(
            f"| {row['Stage']:<{c1}} "
            f"| {row['Technique Applied']:<{c2}} "
            f"| {row['Mean PSNR (dB)']:>{c3}.4f} "
            f"| {row['Mean SSIM']:>{c4}.4f} "
            f"| {delta_str:>{c5}} |"
        )

    print(f"\nCSV saved to: {os.path.abspath(output_path)}")