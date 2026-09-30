"""
Illumination-map enhancement (LIME).

Guo, Li, and Ling estimate a per-pixel illumination as the maximum of the
RGB channels, then smooth that map so it follows object structure instead of
sensor noise. Dividing the image by a gamma-compressed illumination lifts
dark regions more than regions that are already bright, which is what a
single global gamma cannot do.
"""

import numpy as np

import config
from modules.filters import guided_filter


def initial_illumination(image):
    """Max-RGB illumination, the LIME initial map. ``image`` is float BGR in [0, 1]."""
    return np.max(image, axis=2)


def refine_illumination(illumination):
    """Structure-aware smoothing of the initial illumination map."""
    refined = guided_filter(
        illumination,
        illumination,
        radius=config.ILLUM_RADIUS,
        eps=config.ILLUM_EPS,
    )
    return np.clip(refined, 1e-3, 1.0)


def recover_reflectance(image, illumination, gamma=None):
    """
    Recover a brighter image by dividing out a gamma-compressed illumination.

    ``gamma`` in (0, 1] controls the strength. Values closer to 1 lift shadows
    harder. It may be one number or a per-pixel map. The same value is applied
    to every channel, so colour ratios from the input are kept.
    """
    if gamma is None:
        gamma = config.ILLUM_GAMMA
    compressed = np.power(illumination, gamma)
    recovered = image / (compressed[..., None] + config.ILLUM_EPS_DIV)
    return np.clip(recovered, 0.0, 1.0)
