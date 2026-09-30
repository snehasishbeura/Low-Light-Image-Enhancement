"""
Edge-preserving refine.

Smooth with a guided filter, then add back only the detail whose amplitude
is larger than a noise threshold. Edges stay. Fine grain does not.
A crushed exposure is smoothed more and its grain is not added back, because
that grain was amplified above the level of real edges.
"""

import cv2
import numpy as np

import config
from modules.filters import guided_filter_color


def refine_details(image, crushed):
    """``image`` is float BGR in [0, 1]."""
    if crushed:
        radius = config.CRUSH_REFINE_RADIUS
        eps = config.CRUSH_REFINE_EPS
        keep = config.CRUSH_DETAIL_KEEP
        tau = config.CRUSH_DETAIL_TAU
    else:
        radius = config.REFINE_RADIUS
        eps = config.REFINE_EPS
        keep = config.DETAIL_KEEP
        tau = config.DETAIL_TAU

    base = guided_filter_color(image, radius, eps)
    detail = image - base
    amplitude = np.mean(np.abs(detail), axis=2, keepdims=True)
    weight = np.clip((amplitude - tau) / (tau + 1e-6), 0.0, 1.0)
    return np.clip(base + detail * keep * weight, 0.0, 1.0)


def grain_strength(illumination):
    """
    Filter strength in the 0–255 units OpenCV's non-local means expects.

    A dark illumination mean means the division amplified noise, so the
    strength rises. At or above ``GRAIN_T_MEAN`` the frame is left unchanged.
    """
    tmean = float(np.mean(illumination))
    if tmean >= config.GRAIN_T_MEAN:
        return 0.0
    strength = config.GRAIN_H * (config.GRAIN_T_MEAN - tmean) / config.GRAIN_T_MEAN
    return float(np.clip(strength, 0.0, config.GRAIN_H))


def suppress_amplified_grain(image, illumination):
    """
    Remove amplified grain from a uint8 BGR image.

    ``illumination`` is the map that was divided out. It is only used to
    choose the strength. The filter itself never reads a ground-truth photo.
    """
    strength = grain_strength(illumination)
    if strength < 1.0:
        return image
    return cv2.fastNlMeansDenoisingColored(
        image, None, strength, strength + 2.0, 7, 21
    )
