# FGDA: Frequency-Gated Directional Augmentation for Cross-Domain WS-VAD

Code for the paper "Frequency-Gated Directional Augmentation for Cross-Domain Weakly-Supervised Video Anomaly Detection".

## Overview

This repository contains the code for FGDA, a lightweight source-only augmentation for cross-domain weakly-supervised video anomaly detection (WS-VAD) on frozen CLIP features. The method injects frequency-gated directional perturbations into the lowest-scoring frames of source-normal features during MIL training.

## Repository structure

```
├── src/
│   ├── model.py                  # CLIPVAD dual-head model (VadCLIP architecture)
│   ├── freq_text_aug.py          # FGDA augmenter (shift/band/timedomain/noise modes)
│   ├── freq_text_options.py      # Variant table + CLI arguments + data-root helpers
│   ├── freq_text_prompts.py      # Class definitions + prompt templates (XD/UCF/union)
│   ├── freq_text_dataset.py      # Dataset loader with 16-frame stride convention
│   ├── freq_text_trainer.py      # Training loop (CLAS2 + CLASM + dispersion + aug)
│   ├── freq_text_test.py         # Evaluation (pooled AUC/AP + within-video AUC)
│   ├── freq_text_text.py         # CLIP text direction bank builder
│   ├── freq_text_events.py       # Event-prompt dictionary + per-frame response matrix
│   ├── probe_events.py           # Event-prompt / direction probe
│   ├── xd_train.py               # XD-Violence training entrypoint
│   ├── xd_option.py              # XD-Violence options
│   ├── union_train.py            # Multi-source (XD+UCF) training entrypoint
│   ├── build_union_list.py       # Merged XD+UCF training list builder
│   ├── cross_eval.py             # Cross-domain evaluation (source ckpt -> target test)
│   ├── analysis_crossdomain_transfer.py  # Direction transferability diagnostic
│   ├── analysis_spectral_bands.py         # Temporal band diagnostic (H2)
│   ├── analysis_zsad_floor.py             # Training-free ZSAD floor measurement
│   ├── analysis_official_metric.py        # Published-protocol trivial baselines
│   ├── stage0_extract_smoke.py   # CLIP feature extraction smoke test
│   ├── sht_extract.py            # ShanghaiTech CLIP feature extraction
│   ├── sht_build_gt.py           # ShanghaiTech GT construction
│   ├── clip/                     # Vendored CLIP (from VadCLIP), incl. the BPE vocab
│   └── utils/                    # Vendored utilities (from VadCLIP)
├── list/                         # Training/test lists + ground truth
│   ├── xd_CLIP_rgb.csv           # XD-Violence training list
│   ├── xd_CLIP_rgbtest.csv       # XD-Violence test list
│   ├── ucf_CLIP_rgb.csv          # UCF-Crime training list
│   ├── ucf_CLIP_rgbtest.csv      # UCF-Crime test list
│   ├── union_CLIP_rgb.csv        # XD+UCF merged training list (multi-source)
│   ├── gt.npy / gt_ucf.npy       # Frame-level ground truth
│   ├── gt_segment.npy / ...      # Segment-level GT
│   └── repair_list_paths.py      # Re-root the absolute feature paths stored in the CSVs
├── tests/
│   └── test_freq_text_aug.py     # 23 unit tests for the augmenter
├── LICENSE
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9+; `pip install -r requirements.txt` (PyTorch 1.13+ with CUDA recommended, numpy, scikit-learn, scipy, pandas, ftfy, regex, Pillow, opencv-python, tqdm). OpenAI CLIP is vendored in `src/clip/`.

## Data

Use the released VadCLIP CLIP ViT-B/16 frame features (stride 16, 512-d fp16, one feature per 16 frames): <https://github.com/nwpu-zxr/VadCLIP>. Lay them out as

```
<FGDA_DATA_ROOT>/
├── XDTrainClipFeatures/
├── XDTestClipFeatures/
├── UCFTrainClipFeatures/
├── UCFTestClipFeatures/
└── SHTClipFeatures/          # only needed for the ShanghaiTech target
```

The list CSVs were produced on the machine that built the feature banks and store absolute paths (`E:/dataset/...`). **Do not edit 55k rows by hand** — set `FGDA_DATA_ROOT` to the directory above (the one containing `XDTrainClipFeatures`) and the stored paths are re-rooted automatically at load time. If your directory names differ from the originals, `python list/repair_list_paths.py --roots <dir>` rewrites each list by locating the `.npy` basename (a `.bak` is kept); `--check-only` reports without writing.

Official VadCLIP checkpoints (needed for `--checkpoint official` and for the UCF->XD row): download `model_xd.pth` / `model_ucf.pth` from the VadCLIP repository and either set `FGDA_VADCLIP_XD` / `FGDA_VADCLIP_UCF` to their paths or drop them at `checkpoints/model_xd.pth` / `checkpoints/model_ucf.pth`.

ShanghaiTech (target only): extract features with `python src/sht_extract.py --split both`, then build the test list and frame GT with `python src/sht_build_gt.py`. Raw frames are read from `FGDA_SHT_ROOT` (default `E:\dataset\shanghaitech`) and outputs are written to `FGDA_SHT_FEAT_ROOT` (default `<FGDA_DATA_ROOT>/SHTClipFeatures`).

Environment overrides: `FGDA_DATA_ROOT`, `FGDA_SHT_ROOT`, `FGDA_SHT_FEAT_ROOT`, `FGDA_VADCLIP_XD`, `FGDA_VADCLIP_UCF`. With none set, behaviour is identical to the machine that produced the lists.

## Usage

### Train on XD-Violence

```bash
# E5 baseline (no augmentation)
python src/xd_train.py --variant E5 --seed 234

# E3: FGDA with random direction (the method)
python src/xd_train.py --variant E3 --seed 234 --save-threshold 0

# E1: FGDA with text direction (the ablated variant)
python src/xd_train.py --variant E1 --seed 234 --save-threshold 0

# N1: matched-magnitude noise control
python src/xd_train.py --variant N1 --seed 234 --save-threshold 0
```

Artefacts of a variant land in `runs/<dataset>_<variant>/`. The best model is written both under a self-describing filename (`XD_auc..._ap..._e{N}_...pth`) and as the stable alias `runs/<dataset>_<variant>/model_best.pth`; `latest_best.txt` also records the self-describing name.

### Cross-domain evaluation

```bash
# Evaluate an XD-trained checkpoint on UCF-Crime
python src/cross_eval.py --source xd --target ucf --checkpoint runs/xd_E3/model_best.pth

# Evaluate the official VadCLIP checkpoint on UCF-Crime
python src/cross_eval.py --source xd --target ucf --checkpoint official

# UCF-to-XD row (official UCF checkpoint)
python src/cross_eval.py --source ucf --target xd --checkpoint official
```

### Multi-source (union) training

```bash
# Build the merged XD+UCF training list
python src/build_union_list.py

# Train on the union list
python src/union_train.py --variant E3 --seed 234

# Evaluate union-trained model on held-out SHT
python src/cross_eval.py --source union --target sht --checkpoint runs/union_E3/model_best.pth
```

### Diagnostics

```bash
# Direction transferability probe
python src/analysis_crossdomain_transfer.py

# Spectral band diagnostic
python src/analysis_spectral_bands.py

# Training-free ZSAD floor
python src/analysis_zsad_floor.py

# Published-protocol trivial baselines
python src/analysis_official_metric.py
```

### Tests

```bash
python tests/test_freq_text_aug.py     # 23 tests, no pytest required
```

## Variants

| ID | Description |
|---|---|
| E5 | Baseline (no augmentation, VadCLIP as published) |
| E1 | FGDA with CLIP text class direction (ablated variant) |
| E3 | FGDA with fixed random direction (the method) |
| E4 | Gate frozen flat (degenerate) |
| E2 | Time-domain fallback (degenerate) |
| B1 | Multiplicative band gain (non-degenerate spectral form) |
| N1 | Plain i.i.d. noise control (RMS-matched, no structure) |

## Refinement sweep (paper Table 4): reproducing R1--R4

R1--R4 are the E3 method with one knob changed. There are no separate variant IDs; pass the override on top of `--variant E3` (CLI flags take precedence over the variant table):

| Paper row | Command |
|---|---|
| E3 default | `--variant E3` (rho=0.2, K=51, alpha in [0.1,0.5]) |
| R4 band mode | `--variant B1` (multiplicative band gain, identity-init gate) |
| R3 sparse high-freq | `--variant E3 --aug-ratio 0.09 --aug-block-len 25` |
| R1 coverage+ | `--variant E3 --aug-ratio 0.35 --aug-block-len 90` |
| R2 severity+ | `--variant E3 --aug-alpha-lo 0.3 --aug-alpha-hi 0.7` |

## Key findings

1. The cross-domain gain (+1.75+-0.74 pooled AUC on XD->UCF, mean +- s.d. over three runs) comes from the **injection structure** (frequency-gated, rank-1), not from semantic direction content.
2. CLIP text class directions, when used as perturbation directions, perform at baseline — the field's assumption that they carry transferable anomaly semantics does not survive a controlled attribution.
3. The hardest target (ShanghaiTech) exhibits a between-video style shift that dominates the pooled metric: a trivial magnitude cue achieves 0.86 within-video AUC but pools below chance.

## Citation

If you use this code, please cite the paper.

## License

MIT (see `LICENSE`).
