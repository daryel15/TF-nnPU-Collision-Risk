# TF-nnPU Level-1 Reproducibility Package

This package reproduces the processed-feature-level TF-nnPU experiment from:

**Robust Ego-Centric Collision Risk Estimation Under Positive–Unlabeled Supervision**

## Scope

Level 1 starts from the processed temporal dataset and reproduces:

1. data loading and scenario sorting;
2. causal prefix generation;
3. the original 80/20 random prefix split;
4. checkpoint-based TF-nnPU evaluation;
5. optional Stage-2 retraining from the canonical pretrained encoder.

The simulator, raw camera/LiDAR acquisition, YOLO detections, and physical-QCar raw data are intentionally outside Level 1.

## Canonical files

- `data/safe_col_mix.txt`
- `checkpoints/pretrained_encoder1.pt`
- `checkpoints/stage2_trained_model.pt`

SHA-256 hashes are stored in `expected_results.json`.

## Dataset columns

Each row contains:

`frame, distance, angle, sin(yaw), cos(yaw), relative-speed-magnitude, frame-label, sequence-id`

Expected counts:

- 981 sequences
- 15 frames per sequence
- 13,734 prefixes
- 10,987 training prefixes
- 2,747 validation prefixes
- 2,717 positive training prefixes
- 660 positive validation prefixes

## Model

Transformer temporal encoder:

- input dimension: 5
- `d_model`: 64
- heads: 4
- encoder layers: 2
- feed-forward dimension: 128
- dropout: 0.1
- maximum positional length: 500

Temporal attention pooling is followed by:

`LayerNorm(64) -> Linear(64,64) -> ReLU -> Dropout(0.1) -> Linear(64,1)`

## Stage-2 TF-nnPU settings

- positive prior: 0.10
- AdamW
- learning rate: `1e-4`
- weight decay: `1e-4`
- batch size: 32
- validation batch size: 64
- maximum epochs: 200
- early stopping metric: validation F1
- patience: 8
- minimum improvement: `1e-3`
- classification threshold: 0.5

## Exact checkpoint reproduction

Running `reproduce_level1.ipynb` with the included final Stage-2 checkpoint should produce approximately:

- Accuracy: 0.960684
- AUROC: 0.990302
- Precision: 0.938095
- Recall: 0.895455
- F1: 0.916279

Confusion matrix `[TN FP; FN TP]`:

```text
[[2048, 39],
 [  69, 591]]
```

Small floating-point differences can occur across PyTorch/CUDA platforms, but thresholded predictions should normally remain stable.

## Optional retraining

The notebook includes Stage-2 training code initialized from `pretrained_encoder1.pt`. The released final checkpoint is the canonical artifact for exact numerical reproduction of the reported trained model.

The historical research notebook did not fully isolate every random-number-generator state before Stage-2 head initialization. Therefore, a new from-scratch training run is expected to reproduce the methodology and comparable performance, not necessarily the exact byte-identical final checkpoint.

## Environment

The original experiments reported:

- Python 3.11.3
- PyTorch 2.5.1 + CUDA 12.1
- NVIDIA RTX A400
- Windows 10

CPU evaluation of the released checkpoint is also supported.
