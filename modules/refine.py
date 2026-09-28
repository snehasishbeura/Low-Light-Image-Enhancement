"""
Edge-preserving refine.

The previous pipeline ran a bilateral filter with a wide kernel
(d = 9, sigma = 75). That removed CLAHE noise, but it also wiped the texture
the ground-truth photo still has: book spines, fabric, facial detail.

This stage smooths with a guided filter, then adds back only the detail whose
amplitude is larger than a noise threshold. Edges stay. Fine grain does not.
Crushed exposures, which were amplified the most, use a stronger smooth.
"""

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
