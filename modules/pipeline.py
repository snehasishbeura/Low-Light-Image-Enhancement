"""
Low-light enhancement pipeline.

Stages, in order:

1. Illumination recovery (LIME max-RGB map + guided-filter smoothing)
2. Adaptive tone (crushed-exposure lift, or a mild midtone anchor)
3. Partial gray-world balance
4. Edge-preserving detail refine
5. Midtone saturation restore

Ground-truth images are never read here. Every decision uses the low-light
input only.
"""

import cv2
import numpy as np

import config
from modules.color_restore import restore_color
from modules.illumination import (
    initial_illumination,
    recover_reflectance,
    refine_illumination,
)
from modules.refine import refine_details
from modules.tone import apply_adaptive_tone, gray_world

# Names used by the ablation tables. Order matches ``run_stages``.
STAGE_NAMES = [
    "Raw Low-Light",
    "After Illumination",
    "After Adaptive Tone",
    "After Detail Refine",
    "After Color Restore",
]

STAGE_TECHNIQUES = {
    "Raw Low-Light": "No enhancement (baseline)",
    "After Illumination": (
        f"LIME illumination, gamma={config.ILLUM_GAMMA}, "
        f"guided radius={config.ILLUM_RADIUS}"
    ),
    "After Adaptive Tone": (
        "Crushed-exposure lift or tone gamma "
        f"{config.TONE_GAMMA} plus partial gray-world"
    ),
    "After Detail Refine": (
        "Guided-filter denoise with edge detail restored"
    ),
    "After Color Restore": f"Midtone saturation x{config.SATURATION}",
}


def _to_uint8(image):
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def run_stages(image):
    """
    Run every stage.

    Parameters
    ----------
    image : uint8 BGR

    Returns
    -------
    list of uint8 BGR images, one per ``STAGE_NAMES`` entry.
    The last image is the final enhancement.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    original = image
    linear = image.astype(np.float32) / 255.0

    illumination = refine_illumination(initial_illumination(linear))
    recovered = recover_reflectance(linear, illumination)
    toned, crushed = apply_adaptive_tone(recovered, initial_illumination(linear))
    balanced = gray_world(toned, config.GRAY_WORLD)
    refined = refine_details(balanced, crushed)
    final = restore_color(_to_uint8(refined))

    return [
        original,
        _to_uint8(recovered),
        _to_uint8(balanced),
        _to_uint8(refined),
        final,
    ]


def enhance(image):
    """Return only the final enhanced uint8 BGR image."""
    return run_stages(image)[-1]
