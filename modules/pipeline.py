"""
Low-light enhancement pipeline.

Stages, in order:

1. Illumination recovery (LIME max-RGB map + guided-filter smoothing)
2. Adaptive tone. Mixed frames use a paired shadow/highlight gamma instead
3. Partial gray-world balance
4. Edge-preserving detail refine
5. Midtone saturation restore
6. Non-local grain suppress on dark frames

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
from modules.refine import refine_details, suppress_amplified_grain
from modules.tone import apply_adaptive_tone, gray_world, is_crushed, is_mixed_light, paired_gamma_map

# Names used by the ablation tables. Order matches ``run_stages``.
STAGE_NAMES = [
    "Raw Low-Light",
    "After Illumination",
    "After Adaptive Tone",
    "After Detail Refine",
    "After Color Restore",
    "After Grain Suppress",
]

STAGE_TECHNIQUES = {
    "Raw Low-Light": "No enhancement (baseline)",
    "After Illumination": (
        f"LIME illumination, gamma={config.ILLUM_GAMMA}, "
        f"guided radius={config.ILLUM_RADIUS}"
    ),
    "After Adaptive Tone": (
        "Global tone, or paired shadow/highlight gamma on mixed light, "
        "plus partial gray-world"
    ),
    "After Detail Refine": (
        "Guided-filter denoise with edge detail restored"
    ),
    "After Color Restore": f"Midtone saturation x{config.SATURATION}",
    "After Grain Suppress": (
        "Non-local means on dark frames, strength from the illumination mean"
    ),
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

    raw_illumination = initial_illumination(linear)
    illumination = refine_illumination(raw_illumination)
    # One gamma for a uniformly dark frame. A shadow gamma and a highlight
    # gamma only when this frame itself is mixed. The global 0.82 is skipped
    # on that path because it lifts the bright regions again.
    if is_crushed(raw_illumination) or not is_mixed_light(illumination):
        recovered = recover_reflectance(linear, illumination)
        toned, crushed = apply_adaptive_tone(recovered, raw_illumination)
    else:
        gamma_map = paired_gamma_map(illumination)
        recovered = recover_reflectance(linear, illumination, gamma_map)
        toned = recovered
        crushed = False
    balanced = gray_world(toned, config.GRAY_WORLD)
    refined = refine_details(balanced, crushed)
    colored = restore_color(_to_uint8(refined))
    final = suppress_amplified_grain(colored, illumination)

    return [
        original,
        _to_uint8(recovered),
        _to_uint8(balanced),
        _to_uint8(refined),
        colored,
        final,
    ]


def enhance(image):
    """Return only the final enhanced uint8 BGR image."""
    return run_stages(image)[-1]
