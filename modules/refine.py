"""
Edge-preserving refine.

Smooth with a guided filter, then add back only the detail whose amplitude
is larger than a noise threshold. Edges stay. Fine grain does not.
A crushed exposure is smoothed more and its grain is not added back, because
that grain was amplified above the level of real edges.
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
