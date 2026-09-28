"""Previous global gamma stage. Not used by the active pipeline."""

import cv2
import numpy as np

def gamma_correction(image, gamma=1.8):
    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)