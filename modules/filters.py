"""Edge-aware filters used by the enhancement pipeline."""

import cv2
import numpy as np


def box_filter(image, radius):
    """Normalized box filter with reflected borders."""
    kernel = 2 * radius + 1
    return cv2.boxFilter(
        image, -1, (kernel, kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )


def guided_filter(guide, source, radius, eps):
    """
    He et al. guided filter for a single-channel guide and source.

    Both arrays are float32 in a comparable range (here, 0–1). A small
    ``eps`` keeps strong edges; a larger ``eps`` smooths more.
    """
    mean_guide = box_filter(guide, radius)
    mean_source = box_filter(source, radius)
    covariance = box_filter(guide * source, radius) - mean_guide * mean_source
    variance = box_filter(guide * guide, radius) - mean_guide * mean_guide
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return box_filter(a, radius) * guide + box_filter(b, radius)


def guided_filter_color(image, radius, eps):
    """Smooth each BGR channel, guided by its luminance."""
    luminance = (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    )
    smoothed = np.empty_like(image)
    for channel in range(3):
        smoothed[:, :, channel] = guided_filter(
            luminance, image[:, :, channel], radius, eps
        )
    return np.clip(smoothed, 0.0, 1.0)


def luminance(image):
    """Rec. 601 luminance of a float BGR image."""
    return (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    )
