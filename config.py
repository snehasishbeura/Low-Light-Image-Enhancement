"""Paths and pipeline parameters. Nothing here reads a ground-truth image."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return candidates[0]


# LOL eval15 (15 pairs). Filename in low/ matches filename in high/.
DATASET_LOW = _first_existing(
    os.path.join(ROOT, "data", "hf", "LOLdataset", "eval15", "low"),
    os.path.join(ROOT, "data", "LOLdataset", "eval15", "low"),
    r"C:\Users\notsnehasis\Downloads\LOLdataset\eval15\low",
)
DATASET_HIGH = _first_existing(
    os.path.join(ROOT, "data", "hf", "LOLdataset", "eval15", "high"),
    os.path.join(ROOT, "data", "LOLdataset", "eval15", "high"),
    r"C:\Users\notsnehasis\Downloads\LOLdataset\eval15\high",
)

# LOL-v2 Real captured test set. low00690.png pairs with normal00690.png.
LOLV2_REAL_LOW = _first_existing(
    os.path.join(ROOT, "data", "hf", "lol-v2-real", "Test", "Low"),
    os.path.join(ROOT, "data", "lol-v2-real", "Test", "Low"),
    r"C:\Users\notsnehasis\Downloads\LOL-v2\LOL-v2\Real_captured\Test\Low",
)
LOLV2_REAL_HIGH = _first_existing(
    os.path.join(ROOT, "data", "hf", "lol-v2-real", "Test", "Normal"),
    os.path.join(ROOT, "data", "lol-v2-real", "Test", "Normal"),
    r"C:\Users\notsnehasis\Downloads\LOL-v2\LOL-v2\Real_captured\Test\Normal",
)

OUTPUT_FOLDER = "output"

# --- 1. Illumination (LIME) -------------------------------------------------
ILLUM_GAMMA = 0.7          # exponent on the smoothed illumination map
ILLUM_RADIUS = 32          # guided-filter radius, in pixels
ILLUM_EPS = 1.5e-3         # guided-filter regularisation on the illumination
ILLUM_EPS_DIV = 1e-4       # floor used when dividing the image by illumination

# --- 2. Adaptive tone -------------------------------------------------------
TONE_GAMMA = 0.82          # additional gamma on ordinary (not crushed) images
CRUSH_MEAN = 0.045         # max-channel mean below this can be a crushed frame
CRUSH_P99 = 0.13           # and the 99th percentile is still dark
CRUSH_RATIO = 4.0          # and there is no bright anchor relative to the median
CRUSH_GAIN = 1.65
CRUSH_KNEE = 0.84
# Midtone anchor: only when the input already has some light, the recovery is
# still dim, and highlights are not already near white.
ANCHOR_TMEAN = 0.075
ANCHOR_MEAN_LO = 0.40
ANCHOR_MEAN_HI = 0.52
ANCHOR_P95 = 0.70
ANCHOR_TARGET = 0.70
ANCHOR_CAP = 1.10
ANCHOR_KNEE = 0.78
GRAY_WORLD = 0.10          # 0 = off, 1 = full gray-world

# --- 3. Edge-preserving refine ----------------------------------------------
REFINE_RADIUS = 4
REFINE_EPS = 0.006
DETAIL_KEEP = 0.25
DETAIL_TAU = 0.04
# Stronger smooth after the crushed-exposure gain, which amplifies noise more.
CRUSH_REFINE_RADIUS = 6
CRUSH_REFINE_EPS = 0.012
CRUSH_DETAIL_KEEP = 0.15
CRUSH_DETAIL_TAU = 0.05

# --- 4. Colour --------------------------------------------------------------
SATURATION = 1.04
