"""Previous bilateral stage. Not used by the active pipeline.

d=9 and sigma=75 blurred texture (books, fabric, faces). Detail refine in
modules/refine.py replaces it.
"""

import cv2

def bilateral_filter(image):
    return cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )