"""Mild saturation restore on midtones. Hue is left untouched."""

import cv2
import numpy as np

import config


def restore_color(image):
    """
    ``image`` is uint8 BGR.

    Saturation is raised only where the value channel is a midtone, so shadow
    noise and clipped highlights are not painted more colourful.
    """
    factor = config.SATURATION
    if abs(factor - 1.0) < 1e-6:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2]
    midtone = np.clip((value - 15.0) / 70.0, 0.0, 1.0) * np.clip(
        (240.0 - value) / 40.0, 0.0, 1.0
    )
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + (factor - 1.0) * midtone), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
