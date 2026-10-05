# RSI: Rank-1 Spectral Injection for Cross-Domain WS-VAD

Code for the paper "Rank-1 Spectral Injection for Cross-Domain Weakly-Supervised Video Anomaly Detection".

## Overview

This repository contains the code for rank-1 spectral injection (RSI), a lightweight source-only augmentation for cross-domain weakly-supervised video anomaly detection (WS-VAD) on frozen CLIP features. During MIL training the method forms a rank-1 update of the lowest-scoring frames of source-normal features in the frequency domain and injects it through a normalised, RMS-matched spectral field. A gate shapes the injected spectrum, but the paper's attribution study shows the gain comes from the injection scaffold rather than from the gate's shape or from semantic direction content.

## Repository structure

```
├── src/
│   ├── model.py                  # CLIPVAD dual-head model (VadCLIP architecture)
│   ├── freq_text_aug.py          # RSI augmenter (shift/band/timedomain/noise modes)
│   ├── freq_text_options.py      # Variant table + CLI arguments + data-root helpers
│   ├── freq_text_prompts.py      # Class definitions + prompt templates (XD/UCF/union)
│   ├── freq_text_dataset.py      # Dataset loader with 16-frame stride convention
│   ├── freq_text_trainer.py      # Training loop (CLAS2 + CLASM + dispersion + aug)
│   ├── freq_text_test.py         # Evaluation (pooled AUC/AP + within-video AUC)
│   ├── freq_text_text.py         # CLIP text direction bank builder
│   ├── freq_text_events.py       # Event-prompt dictionary + per-frame response matrix
│   ├── probe_events.py           # Event-prompt / direction probe
│   ├── xd_option.py              # XD-Violence options
│   ├── xd_train.py               # XD-Violence training entrypoint
│   ├── xd_test.py                # XD-Violence in-domain evaluation entrypoint
│   ├── ucf_option.py             # UCF-Crime options
│   ├── ucf_train.py              # UCF-Crime training entrypoint (UCF->XD row)
│   ├── ucf_test.py               # UCF-Crime in-domain evaluation entrypoint
│   ├── union_train.py            # Multi-source (XD+UCF) training entrypoint
│   ├── build_union_list.py       # Merged XD+UCF training list builder
│   ├── cross_eval.py             # Cross-domain evaluation (source ckpt -> target test)
│   ├── precondition_test.py      # Precondition-probe diagnostic
│   ├── analysis_crossdomain_transfer.py  # Direction transferability diagnostic
│   ├── analysis_spectral_bands.py         # Temporal band diagnostic (H2)
│   ├── analysis_zsad_floor.py             # Training-free ZSAD floor measurement
│   ├── analysis_official_metric.py        # Published-protocol trivial baselines
│   ├── analysis_gain_vs_difficulty.py     # Gain vs. difficulty stratification
│   ├── analysis_magnitude_orthogonal.py   # Magnitude-vs-direction orthogonality
│   ├── analysis_residual_direction.py     # Residual-direction attribution
│   ├── analysis_selective_routing.py     # Selective-routing diagnostic
│   ├── analysis_selective_trust.py        # Selective-trust diagnostic
│   ├── analysis_spread_confound.py        # Score-spread confound check
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
│   ├── test_freq_text_aug.py     # Unit tests for the augmenter
│   ├── test_training_protocol.py # Training-protocol regression tests
│   └── test_public_paths.py      # Entrypoint imports + path-helper API guard
├── run_ablation.py               # Crash-safe ablation driver (the R1--R4 sweep)
├── LICENSE
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9+; `pip install -r requirements.txt` (PyTorch 1.13+ with CUDA recommended, numpy, scikit-learn, scipy, pandas, ftfy, regex, Pillow, opencv-python, tqdm). OpenAI CLIP is vendored in `src/clip/`.

## Data

Use the released VadCLIP CLIP ViT-B/16 frame features (stride 16, 512-d fp16, one feature per 16 frames): <https://github.com/nwpu-zxr/VadCLIP>. Lay them out as

```
<data root>/
├── XDTrainClipFeatures/
├── XDTestClipFeatures/
├── UCFTrainClipFeatures/
├── UCFTestClipFeatures/
└── SHTClipFeatures/          # only needed for the ShanghaiTech target
```

All path resolution goes through `freq_text_options`: `data_root()`, `sht_root()`, `sht_feat_root()` and `official_checkpoints(source)`. Each honours the same precedence:

1. the `RSI_*` variable (current name)
2. the `FGDA_*` variable (pre-rename name, still honoured for existing setups)
3. a built-in default

| Variable | Controls | Default |
|---|---|---|
| `RSI_DATA_ROOT` | directory **containing** the feature banks above | `E:/dataset` |
| `RSI_SHT_ROOT` | raw ShanghaiTech frames | `E:\dataset\shanghaitech` |
| `RSI_SHT_FEAT_ROOT` | extracted ShanghaiTech features | `<data root>/SHTClipFeatures` |
| `RSI_VADCLIP_XD` / `RSI_VADCLIP_UCF` | released VadCLIP checkpoints | `<repo>/checkpoints/model_{xd,ucf}.pth`, then two development-machine paths |

An empty-string variable counts as unset, so you can blank a default in CI without deleting the export. `tests/test_public_paths.py` pins all of this.

The list CSVs were produced on the machine that built the feature banks and store absolute paths (`E:/dataset/...`). **Do not edit 55k rows by hand** — set `RSI_DATA_ROOT` to the directory above (the one containing `XDTrainClipFeatures`) and the stored paths are re-rooted automatically at load time. If your directory names differ from the originals, `python list/repair_list_paths.py --roots <dir>` rewrites each list by locating the `.npy` basename (a `.bak` is kept); `--check-only` reports without writing.

Official VadCLIP checkpoints are needed for `--checkpoint official`, which is also how the released UCF->XD row is produced. Download `model_xd.pth` / `model_ucf.pth` from the VadCLIP repository, then either set `RSI_VADCLIP_XD` / `RSI_VADCLIP_UCF`, or drop them at `checkpoints/model_xd.pth` / `checkpoints/model_ucf.pth` inside the clone.

ShanghaiTech (target only): extract features with `python src/sht_extract.py --split both`, then build the test list and frame GT with `python src/sht_build_gt.py`.

## Usage

### Train on XD-Violence

```bash
# E5 baseline (no augmentation)
python src/xd_train.py --variant E5 --seed 234

# E3: RSI with a fixed random rank-1 direction (the method)
python src/xd_train.py --variant E3 --seed 234 --save-threshold 0

# E1: RSI with the CLIP text class direction instead (ablated)
python src/xd_train.py --variant E1 --seed 234 --save-threshold 0

# E3a / E3b: the gate ablation of the paper's Table 5
python src/xd_train.py --variant E3a --seed 234 --save-threshold 0
python src/xd_train.py --variant E3b --seed 234 --save-threshold 0

# N1: matched-magnitude noise control
python src/xd_train.py --variant N1 --seed 234 --save-threshold 0
```

Artefacts of a variant land in `runs/<dataset>_<variant>/`. The best model is written both under a self-describing filename (`XD_auc..._ap..._e{N}_...pth`) and as the stable alias `runs/<dataset>_<variant>/model_best.pth`; `latest_best.txt` also records the self-describing name.

### Train on UCF-Crime (source side of the UCF->XD row)

```bash
python src/ucf_train.py --variant E5 --seed 234
python src/ucf_train.py --variant E3 --seed 234 --save-threshold 0
```

Note the metric asymmetry that the paper's protocol section states explicitly: on XD the
headline metric is AP, on UCF it is AUC. `--save-threshold` and the checkpoint naming follow
that choice, so a UCF checkpoint is written as `UCF_auc..._ap..._e{N}_...pth`.

### In-domain evaluation

```bash
python src/xd_test.py --variant E3      # loads runs/xd_E3/model_best.pth
python src/ucf_test.py --variant E3      # loads runs/ucf_E3/model_best.pth
```

Both wrappers resolve the checkpoint from `runs/<dataset>_<variant>/model_best.pth`; pass the
same `--variant` you trained with.

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

# Attribution diagnostics backing the paper's analysis section
python src/analysis_gain_vs_difficulty.py     # gain vs. difficulty stratification
python src/analysis_magnitude_orthogonal.py   # magnitude vs. direction orthogonality
python src/analysis_residual_direction.py     # residual-direction attribution
python src/analysis_selective_routing.py     # selective-routing diagnostic
python src/analysis_selective_trust.py        # selective-trust diagnostic
python src/analysis_spread_confound.py        # score-spread confound check
python src/precondition_test.py               # precondition-probe diagnostic
```

### Tests

```bash
python -m pytest tests/ -q      # 85 tests: augmenter units, training protocol, path API
```

## Variants

| ID | Description |
|---|---|
| E5 | Baseline (no augmentation, VadCLIP as published) |
| E3 | RSI with a fixed random rank-1 direction -- **the method** |
| E1 | RSI with the CLIP text class direction (ablated) |
| E3a | E3 with the designed gate frozen at initialisation |
| E3b | E3 with a frozen random spectrum (destroys the designed band shape) |
| N1 | Plain i.i.d. noise control (RMS-matched, no structure) |
| B1 | Multiplicative band gain (non-degenerate spectral form) |
| E4 | Gate frozen flat (degenerate) |
| E2 | Time-domain fallback (degenerate) |

## Refinement sweep (paper Table 4): reproducing R1--R4

R1--R4 are the E3 method with one knob changed. There are no separate variant IDs; pass the override on top of `--variant E3` (CLI flags take precedence over the variant table):

| Paper row | Command |
|---|---|
| E3 default | `--variant E3` (rho=0.2, K=51, alpha in [0.1,0.5]) |
| R4 band mode | `--variant B1` (multiplicative band gain, identity-init gate) |
| R3 sparse high-freq | `--variant E3 --aug-ratio 0.09 --aug-block-len 25` |
| R1 coverage+ | `--variant E3 --aug-ratio 0.35 --aug-block-len 90` |
| R2 severity+ | `--variant E3 --aug-alpha-lo 0.3 --aug-alpha-hi 0.7` |

`python run_ablation.py` drives the whole sweep crash-safely (it restarts only the runs that did not finish, so it is safe to re-invoke after an interruption).

## Key findings

1. The cross-domain gain (+1.75 +- 0.74 pooled AUC on XD->UCF, mean +- s.d. over three seeds) comes from the **injection scaffold** -- the normalised, RMS-matched rank-1 spectral field -- not from semantic direction content. The gate ablation narrows it further: freezing the designed gate (E3a) or replacing it with a frozen random spectrum (E3b) leaves results statistically indistinguishable from E3 (+0.64 +- 1.17 and +0.39 +- 0.55 respectively).
2. CLIP text class directions, when used as perturbation directions, perform at baseline; the field's assumption that they carry transferable anomaly semantics does not survive a controlled attribution.
3. The finding is not XD/UCF-specific: training on UCF and evaluating on XD moves AP2 by +1.66 over the released UCF baseline (66.14 vs 64.48).
4. The hardest target (ShanghaiTech) exhibits a between-video style shift that dominates the pooled metric: a trivial magnitude cue reaches 0.86 within-video AUC while pooling below chance.

## Citation

If you use this code, please cite the paper.

## License

MIT (see `LICENSE`).
