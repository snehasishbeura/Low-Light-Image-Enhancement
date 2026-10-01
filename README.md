# Low-Light Image Enhancement

A classical enhancement method that lifts a low-light photo toward the brightness, local contrast, and sharpness of a normal exposure. It does not train a network. Every stage looks only at the low-light input. Ground truth is used afterwards, in `evaluate_lol.py`, to measure PSNR and SSIM.

## The problem

A low-light photo is not just a darker copy of a normal one.

- **Uneven darkness.** A corner, a face, or the far side of a room can sit near black while another part of the frame still has light. One brightness curve for the whole image leaves those regions dark.
- **Crushed exposure.** Some captures have almost no highlight left in the file. A mild lift never reaches a usable brightness.
- **Noise against detail.** The signal that is still there — text, fabric, edges — is buried in sensor noise. Smoothing the whole frame to hide that grain also wipes out the detail.
- **Dull colour.** Shadow regions lose saturation, so the brightened image looks gray even after the luminance is fixed.

## How this method addresses it

The method is built from LIME (Guo, Li, Ling), adaptive tone and gray-world colour in the AGCWD family (Huang, Cheng, Chiu), and an edge-aware guided filter. Parameters live in `config.py`.

```text
Low-light input
      │
      ▼
Illumination map          max(R, G, B), guided-filter smooth
      │                   one exponent on a dark frame
      │                   paired shadow and highlight exponents
      │                   only when the same frame is partly lit and partly dark
      ▼
Adaptive tone             extra gain only if the frame is uniformly crushed
      │                   otherwise a mild gamma, plus a capped midtone lift
      │                   mixed frames skip that second global gamma
      │                   partial gray-world balance
      ▼
Detail refine             guided filter, strong edges added back
      ▼
Colour restore            a little saturation on midtones only
      │
      ▼
Enhanced image
```

1. **Illumination map.** Per-pixel lighting is the maximum of R, G, and B. A guided filter smooths that map so it follows edges instead of texture. Dividing the image by the map raised to 0.7 brightens dark regions more than regions that are already lit. That is what removes patchy darkness without painting one gamma over the whole frame.
2. **Adaptive tone.** If the frame is uniformly crushed (very low mean, no bright anchor), a stronger soft gain lifts it. Otherwise a mild gamma (0.82) is enough, and a capped midtone anchor adds at most 10% when the recovery is still dim and the highlights are not already near white. A partial gray-world step (strength 0.10) pulls a colour cast back without forcing a gray scene.

   When the illumination map itself shows a bright tail, a dark mass, and a mean above 0.30, the frame is treated as mixed light. Those frames do not use one exponent. Shadows keep gamma 0.70, or 0.92 when more than half the map is dark, and already-bright areas use gamma 0.55. The extra global 0.82 is not applied on top. A dark scene, including every LOL eval15 frame, never takes this branch.
3. **Detail refine.** A second guided filter suppresses the grain that the lift amplified. Strong edges are added back, so book titles, fabric, and faces stay sharper than a wide blur would leave them. A uniformly crushed frame is the exception: the gain has already amplified the grain above the real edges, so that frame keeps the smoothed base and does not add the grain back.
4. **Colour restore.** Saturation is raised by 4% on midtones only, so shadows are not pushed into false colour and highlights are not oversaturated.

## Results

Higher PSNR is closer in pixel value. Higher SSIM is closer in structure. Scores below are this method on the paired test images.

| Dataset | Images | PSNR | SSIM |
|---|---:|---:|---:|
| LOL eval15 | 15 | **20.35 dB** | **0.8170** |
| LOL-v2 Real test | 100 | **18.63 dB** | **0.7787** |
| LOL-v2 Synthetic test | 100 | **19.88 dB** | **0.8282** |
| UnLOL test | 43 | **13.54 dB** | **0.6137** |

On LOL eval15 the mean is 20.35 dB / 0.8170, from 16.75 dB on the weakest frame to 26.49 dB on the strongest. The three uniformly crushed frames are the ones that moved: `23.png` 16.82 dB to 18.08 dB, `55.png` 15.78 dB to 16.75 dB, and `665.png` 18.23 dB to 19.06 dB, with SSIM rising by about 0.07 to 0.10 because the amplified grain is no longer kept. The other twelve frames are unchanged. LOL-v2 Real finishes at 18.63 dB / 0.7787 across 100 captured pairs. The synthetic test finishes at 19.88 dB / 0.8282. Illumination recovery does most of that work: the raw synthetic inputs sit at 11.22 dB / 0.4450, and the illumination stage alone reaches 20.23 dB / 0.8948. The best synthetic frame is `r191488c6t.png` at 31.08 dB. The weakest, `r01058910t.png` at 10.19 dB, stays soft because the darkness there is not a simple illumination scale.

UnLOL is a separate real-scene JPEG test (1280×1280). On its 43 pairs the method scores 13.54 dB / 0.6137. Twelve of those frames are mixed and partly lit, so they take the paired gamma. None of the 43 scores went down. The larger gains are `0502.jpeg` (+3.08 dB), `0604.jpeg` (+2.80 dB), `0703.jpeg` (+2.57 dB), `0103.jpeg` (+2.32 dB), and `1709.jpeg` (+1.85 dB). The best frame is still `1805.jpeg` (18.20 dB) and the weakest is `2903.jpeg` (9.49 dB). Both stay on the single-gamma path because the frame is dark overall. Raw inputs average 10.36 dB.

Comparisons (low-light | enhanced | ground truth):

- LOL: `results/LOL/comparisons/`
- LOL-v2 Real, three highest-PSNR frames: `results/LOLv2_Real/comparisons/`
- LOL-v2 Synthetic, best, median, and lowest PSNR: `results/LOLv2_Synthetic/comparisons/`
- UnLOL test, best, median, and lowest PSNR: `results/UnLOL/comparisons/`

Per-image scores: `results/LOL/metrics.csv`, `results/LOLv2_Real/metrics.csv`, `results/LOLv2_Synthetic/metrics.csv`, and `results/UnLOL/metrics.csv`.

## Stage screenshots

Input, then each stage, on `input/input.png`.

| Stage | Image |
|---|---|
| Input | ![input](screenshots/input.jpg) |
| Illumination | ![illumination](screenshots/illumination.jpg) |
| Adaptive tone | ![tone](screenshots/tone.jpg) |
| Detail refine | ![refine](screenshots/refine.jpg) |
| Final | ![final](screenshots/final.jpg) |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put low-light photos in `input/` and run:

```bash
python main.py
```

Stage images are written to `output/`.

## Datasets

The reference splits used for the scores above are included. Each one is a paired test set: the dark folder is the input, and the other folder is the normal-light reference. The enhancer never reads the reference.

```text
datasets/
├── LOL/eval15/                  15 pairs
│   ├── low/
│   └── high/
├── LOLv2/Real/Test/             100 captured pairs
│   ├── Low/
│   └── Normal/
├── LOLv2/Synthetic/Test/        100 synthetic pairs
│   ├── Low/
│   └── Normal/
└── UnLOL/Test/                  43 pairs, 1280×1280 JPEG
    ├── Low/
    └── High/
```

LOL uses the same filename in `low/` and `high/`, for example `1.png`. LOL-v2 Real pairs `low00690.png` with `normal00690.png`. Synthetic and UnLOL use the same filename in both folders, for example `r00816405t.png` and `0103.jpeg`.

The synthetic images are the test split from [LOLv1 & LOLv2 on Kaggle](https://www.kaggle.com/datasets/ohmahler91/lolv1-and-lolv2). The 900-pair training split is not included. UnLOL here is the test split only.

## Scoring against ground truth

`config.py` reads the bundled folders above first. If those are missing it falls back to `data/hf/...`, `data/...`, and the original Windows paths.

```bash
python evaluate_lol.py --dataset lol
python evaluate_lol.py --dataset lolv2
python evaluate_lol.py --dataset lolv2syn
python evaluate_lol.py --dataset unlol
python make_comparisons.py
python make_comparisons_v2.py
python make_comparisons_v2.py --dataset lolv2syn
python make_comparisons_v2.py --dataset unlol
python generate_final_report.py
```

## Layout

```text
config.py                 parameters and dataset paths
datasets/LOL              LOL eval15 pairs
datasets/LOLv2/Real       LOL-v2 Real captured test pairs
datasets/LOLv2/Synthetic  LOL-v2 Synthetic test pairs
datasets/UnLOL            UnLOL test pairs
main.py                   enhance images in input/
evaluate_lol.py           PSNR, SSIM, ablation on LOL, LOL-v2, or UnLOL
make_comparisons.py       LOL side-by-side figures
make_comparisons_v2.py    LOL-v2 and UnLOL comparison figures
modules/pipeline.py       stage order
modules/illumination.py   LIME map and division
modules/tone.py           crushed-exposure lift and midtone anchor
modules/refine.py         edge-preserving denoise
modules/color_restore.py  midtone saturation
modules/evaluation.py     PSNR and SSIM
```

## Limits

The method estimates illumination from the input. It cannot know whether a scene is supposed to be a bright room or a dark street, so a frame whose reference is deliberately dim can come out a little brighter than that photo. Very crushed frames still show some of the sensor noise that was hiding in the blacks; the refine stage suppresses that grain.

## Author

[Snehasish Beura](https://github.com/snehasishbeura)
