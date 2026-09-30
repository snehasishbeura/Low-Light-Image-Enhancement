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

# LOL-v2 Synthetic, from the Kaggle set ohmahler91/lolv1-and-lolv2.
# The bundled tree is datasets/LOLv2/Synthetic/{Train,Test}/{Low,Normal}.
# Scoring uses Test. Low and Normal share filenames (r00816405t.png).
LOLV2_SYN_LOW = _first_existing(
    os.path.join(ROOT, "datasets", "LOLv2", "Synthetic", "Test", "Low"),
    os.path.join(ROOT, "data", "hf", "lol-v2-synthetic", "Test", "Low"),
    os.path.join(ROOT, "data", "lol-v2-synthetic", "Test", "Low"),
    r"C:\Users\notsnehasis\Downloads\LOL-v2\LOL-v2\Synthetic\Test\Low",
)
LOLV2_SYN_HIGH = _first_existing(
    os.path.join(ROOT, "datasets", "LOLv2", "Synthetic", "Test", "Normal"),
    os.path.join(ROOT, "data", "hf", "lol-v2-synthetic", "Test", "Normal"),
    os.path.join(ROOT, "data", "lol-v2-synthetic", "Test", "Normal"),
    r"C:\Users\notsnehasis\Downloads\LOL-v2\LOL-v2\Synthetic\Test\Normal",
)

# UnLOL test split. Low/ is the dark input, High/ is the normal-light reference.
# Filenames match exactly (0103.jpeg). Captions in ts_caption.txt are scene
# descriptions and are not read by the enhancer or by PSNR/SSIM.
UNLOL_LOW = _first_existing(
    os.path.join(ROOT, "datasets", "UnLOL", "Test", "Low"),
    os.path.join(ROOT, "data", "UnLOL", "Test", "Low"),
    os.path.join(ROOT, "data", "UnLOL", "test", "Low"),
)
UNLOL_HIGH = _first_existing(
    os.path.join(ROOT, "datasets", "UnLOL", "Test", "High"),
    os.path.join(ROOT, "data", "UnLOL", "Test", "High"),
    os.path.join(ROOT, "data", "UnLOL", "test", "High"),
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

# Mixed light: a bright tail and a dark mass in the same frame.
# LOL's illumination p99 stays under about 0.38, so this path does not run there.
MIX_P99 = 0.55
MIX_P10 = 0.18
# Mean illumination above this means the frame is already partly lit.
# Dark scenes, including LOL and a lamp in a dark room, keep the single gamma.
MIX_MEAN = 0.30
# A large dark mass is underexposure. A small one is more often a dark object.
MIX_DARK = 0.10
MIX_DARK_MASS = 0.50
# Paired illumination exponents. Low T uses the shadow gamma, high T the highlight gamma.
PAIR_T_LO = 0.18
PAIR_T_HI = 0.55
GAMMA_SHADOW = 0.70            # same strength as ILLUM_GAMMA
GAMMA_SHADOW_LARGE = 0.92     # stronger lift when most of the frame is dark
GAMMA_HIGHLIGHT = 0.55        # already-bright areas are lifted less than 0.70

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
