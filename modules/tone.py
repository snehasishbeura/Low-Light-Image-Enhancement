"""
Adaptive exposure after illumination recovery.

A fixed gamma under-lifts captures whose whole frame is crushed into the
bottom of the histogram (no highlight left to anchor the illumination map)
and can over-lift scenes that are genuinely dark. This stage reads only the
input illumination statistics — never the ground truth — and picks one of
three tones:

* uniformly crushed exposure: a stronger gain with a soft highlight knee
* ordinary low light that is still short of a normal midtone: a small,
  capped extra gain
* everything else: a gentle gamma on the recovered image
"""

import numpy as np

import config
from modules.filters import luminance


def _soft_gain(image, gain, knee):
    """Multiply, then compress values above ``knee`` so highlights do not clip hard."""
    scaled = image * gain
    over = np.maximum(scaled - knee, 0.0)
    compressed = np.minimum(scaled, knee) + (1.0 - knee) * (
        1.0 - np.exp(-over / (1.0 - knee))
    )
    return np.clip(compressed, 0.0, 1.0)


def is_crushed(illumination):
    """
    True when the capture has almost no bright pixels.

    A low mean, a low 99th percentile, and a small percentile ratio together
    mean the frame is uniformly underexposed rather than a dark scene that
    already contains a lamp or a window.
    """
    mean = float(illumination.mean())
    p50 = float(np.percentile(illumination, 50))
    p99 = float(np.percentile(illumination, 99))
    ratio = (p99 + 1e-6) / (p50 + 1e-4)
    return (
        mean < config.CRUSH_MEAN
        and p99 < config.CRUSH_P99
        and ratio < config.CRUSH_RATIO
    )


def apply_adaptive_tone(image, illumination):
    """Return the tone-mapped float image and whether the crushed path was used."""
    if is_crushed(illumination):
        toned = _soft_gain(image, config.CRUSH_GAIN, config.CRUSH_KNEE)
        return toned, True

    toned = np.power(np.clip(image, 0.0, 1.0), config.TONE_GAMMA)
    mean = float(illumination.mean())
    yy = luminance(toned)
    recovered_mean = float(yy.mean())
    p95 = float(np.percentile(yy, 95))
    needs_anchor = (
        mean > config.ANCHOR_TMEAN
        and config.ANCHOR_MEAN_LO < recovered_mean < config.ANCHOR_MEAN_HI
        and p95 < config.ANCHOR_P95
    )
    if needs_anchor:
        gain = float(np.clip(
            config.ANCHOR_TARGET / (p95 + 1e-3),
            1.0,
            config.ANCHOR_CAP,
        ))
        toned = _soft_gain(toned, gain, config.ANCHOR_KNEE)
    return toned, False


def gray_world(image, strength):
    """
    Partial gray-world white balance.

    Full gray-world fights scenes that are genuinely warm or cool. ``strength``
    between 0 and 1 only moves part of the way toward a neutral average.
    """
    if strength <= 0:
        return image
    channel_mean = image.reshape(-1, 3).mean(axis=0)
    gray = float(channel_mean.mean())
    scale = gray / (channel_mean + 1e-6)
    scale = 1.0 + strength * (scale - 1.0)
    return np.clip(image * scale, 0.0, 1.0)
