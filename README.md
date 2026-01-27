# SEDA-Former
A universal deep learning framework for empowering nanopore identification by reinforcing temporal signals
English Version｜Quick Usage Guide for the SEDA-Former Script
1. Environment

Python 3.8+ (recommended 3.9/3.10)

Key dependencies: numpy, pandas, pyarrow, scikit-learn, matplotlib, seaborn, torch

GPU is optional: the script automatically uses CUDA if available; otherwise it runs on CPU.

Note: reading parquet requires pyarrow (pip install pyarrow).

2. Data Requirements

Input is a parquet file specified by parquet_path.

Column convention in your script:

Column 1: label

Columns 4+: time-series signal features of length L

Each row corresponds to one sample (one event / one sequence segment).

3. End-to-end Pipeline (run top-to-bottom)

(A) Load data

pd.read_parquet(parquet_path)

(B) Label-wise cleaning (two stages)

Compute valid_length (number of non-zero entries) and keep samples within per-label quantiles (0.35–0.65).This data preprocessing step was applied only to the cholic acid conjugates dataset, because this dataset still contains unremoved outliers. No such preprocessing was performed on the other existing datasets, for which the corresponding setting is set to 0–1 by default.

Optionally remove extremes per label using a “max drop amplitude” score (controlled by x).

(C) Remove specified labels

remove_labels = [] (edit as needed)

(D) Train/Val/Test split

0.8 / 0.1 / 0.1 via two train_test_split calls with stratify

(E) Optional train downsampling

TRAIN_FRACTION = 1 uses the full training set

Set to 0.1 to keep 10% of the training set with per-class (stratified) sampling

(F) Standardization

Fit the scaler on training data only, then transform train/val/test

(G) Multi-channel construction (sliding-window std)

use_sliding_std = 1 enables multi-channel input

windows = [10, 20, 40] creates multiple sliding-std channels

Final tensor shape: [N, C, L], where C = 1 + len(windows) (raw + std channels)

(H) Training (TCN + window-wise attention + rejection)

Model: TCNWithAttention

TCN backbone: dilated Conv1d + ReLU + MaxPool per layer

Attention: unfold into windows and apply multi-head self-attention

Classifier: linear layer outputs logits

Training setup:

Optimizer: AdamW

Scheduler: OneCycleLR

Loss: CrossEntropyLoss (optionally re-weighted by dynamic class weights later)

(I) Validation threshold calibration (selective classification)

Build an Accuracy–Coverage curve on the validation set using max-softmax confidence.

With CAL_POLICY="target_coverage" and TARGET_VALUE=0.65, the script selects a threshold that meets the target coverage and minimizes risk.

The chosen threshold is written back to model.threshold, and per-class gate stats are saved to:

SAVE_JSON_PATH = "./val_gate_stats.json"

(J) Test evaluation + confusion matrix

Reports:

argmax accuracy (no rejection)

thresholded accuracy (accuracy on accepted samples only)

Not Predicted rate (rejected proportion)

Confusion matrices are computed on accepted samples only (samples with prediction -1 are excluded).

4. Common knobs to adjust

Data path: parquet_path

Removed labels: remove_labels

Downsampling ratio: TRAIN_FRACTION

Cleaning strength: the two x = 0.0 settings

Sliding-std: use_sliding_std, windows

Model/training:

num_layers, output_channels, attn_dim, num_heads, dropout

epochs=40, batch_size=64, learning rates

Rejection calibration:

CAL_POLICY (target_coverage / target_risk)

TARGET_VALUE

5. Outputs

Console logs per epoch: Train Acc, Val Acc(max), Val Acc(pred), Coverage

Plots: training curves, validation confidence histogram, Accuracy–Coverage curve, confusion matrices (absolute and normalized)

Gate statistics cache: val_gate_stats.json






#######################################################################
English Version: CA-Net Quick Start Guide
1. Overview

CA-Net is a baseline-aware histogram branch for nanopore event classification. The pipeline is:

raw (unstandardized) sequence → non-negative clipping (<=0 → 0) → baseline/bin-based histogram binning → z-score normalization on the histogram → StevieNet (1D ResNet) classifier.

Both forward() and predict() require baseline, because the per-sample bin step is defined as p_step = baseline / bin.

2. Requirements

Python 3.x

numpy, pandas, matplotlib

scikit-learn (data split, confusion matrix utilities)

PyTorch (model + training)

Parquet input requires a parquet engine supported by pandas (e.g., pyarrow).

3. Expected Parquet Schema

The script assumes:

Column 1: label (int class id)

Column 2: baseline (float)

Columns from the 4th onward: raw time-series features (length L)

The 3rd column can be a placeholder (the script uses columns[3:] for the sequence)

4. Key Configurations

parquet_path: path to your parquet dataset

TRAIN_FRACTION: fraction of the training set to use (e.g., 0.1 for 10%)

remove_labels = [8,12,14]: labels to be excluded

BIN: histogram dimension (e.g., 256/512/1024)

EPOCHS / BATCH / lr: training schedule

THRESH: rejection threshold (max softmax confidence < THRESH → prediction = -1)

5. How to Run

Set parquet_path to your dataset.

Run the script. It will:

clean samples per label (quantile filtering on valid length)

remove specified labels

stratify split into train/val/test (baseline is split consistently)

optionally downsample the training set via TRAIN_FRACTION

train CA-Net and print per epoch:

Train Acc

Val Acc(max) (argmax, no rejection)

Val Acc(pred) (with rejection)

Coverage (accepted proportion)

Inspect training curves: Train / Val(max) / Val(pred) / Coverage.

6. Rejection Threshold Calibration

Setting THRESH=0 disables rejection (Coverage ≈ 100%).

You can enable rejection by setting model.threshold = 0.8, etc.

An optional validation-based calibration module is included, supporting:

target_coverage: choose threshold to meet a target coverage

target_risk: choose threshold to meet a target risk (1-accuracy)

After calibration, model.threshold is updated and per-class gate statistics can be exported.

7. Test-Time Reporting

The test block reports:

per-class argmax accuracy (no rejection)

per-class threshold accuracy (accepted samples only)

per-class rejection rate

overall Accuracy-on-accepted / Coverage / Reject rate

optional confusion matrices on accepted samples only (absolute + normalized)

8. Important Notes

No StandardScaler on histogram input: this implementation strictly follows “raw + nonneg clipping + baseline binning + z-score”.

Baseline must be > 0: protected via clamp_min(1e-8).

NaN/Inf are mapped to 0 in the histogram input.

Changing BIN changes the StevieNet input length (B, 1, BIN).
