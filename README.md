# TF-nnPU Collision-Risk Reproducibility Repository

This package consolidates the processed dataset, model checkpoints, executed research notebooks, fixed split files, corruption masks, and exported result tables used for the revised paper.

## Repository contents

### `data/`
- `safe_col_mix.txt`: processed 981-scenario dataset.
- Columns: `frame, distance, angle, sin(yaw), cos(yaw), speed, label, sequence_id`.

### `checkpoints/`
- `pretrained_encoder1.pt`: canonical Stage-1 Transformer encoder.
- `stage2_trained_model.pt`: canonical final TF-nnPU Stage-2 checkpoint.

### `notebooks/`
1. `01_main_tf_nnpu_reproduction.ipynb` — clean checkpoint reproduction.
2. `02_full_tf_nnpu_hyperparameter_sensitivity.ipynb` — full TF-nnPU research / sensitivity notebook.
3. `03_cost_sensitive_and_significance.ipynb` — TF-nnPU vs WBCE/Focal, symmetric label-noise analysis, bootstrap/permutation statistics.
4. `04_su_lstm_matched_selected.ipynb` — user-selected Su-LSTM baseline.
5. `05_tf_nnpu_vs_gat.ipynb` — matched GAT comparison.
6. `06_tf_nnpu_vs_cmpa.ipynb` — matched CMPA comparison.

### `pipeline_raw/`
- simulator camera/LiDAR acquisition notebook;
- camera/LiDAR feature-extraction notebook.

These are included for transparency but are not required to reproduce the Level-1 processed-feature experiments.

### `splits/`
- `main_prefix_split_seed42.npz`: exact prefix-level 80/20 split for the main t=2..15 experiment.
- `common_t4_t15_split_seed42.npz`: exact common t=4..15 prefix split used for the GAT/CMPA/Su-LSTM comparison.

The existing prefix-level split is intentionally preserved.

### `noise_masks/`
Exact deterministic corruption masks are exported for:
- common current-risk P->U corruption;
- common current-risk symmetric corruption;
- the selected Su-LSTM scenario-outcome P->U corruption;
- the cost-sensitive symmetric experiment.

### `results/`
CSV files containing the principal reported outputs.

## Main dataset statistics

- 981 scenarios
- 15 frames per scenario
- 14,715 rows
- 13,734 t=2..15 prefixes
- main split: 10,987 train / 2,747 validation prefixes
- common t=4..15 set: 11,772 prefixes
- common split: 9,417 train / 2,355 validation prefixes

## Canonical TF-nnPU checkpoint result at threshold 0.5

- Accuracy: 0.960684
- AUROC: 0.990302
- Precision: 0.938095
- Recall: 0.895455
- F1: 0.916279

## Environment reported by the original experiments

- Python 3.11.3
- PyTorch 2.5.1 + CUDA 12.1
- NVIDIA RTX A400
- Windows 10

See `MANIFEST_SHA256.txt` for file hashes.
