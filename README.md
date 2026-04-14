# SEDA-Former
A universal deep learning framework for empowering nanopore identification by reinforcing temporal signals｜Quick Usage Guide for the SEDA-Former Script

0. Environment

Python 3.8+ (recommended 3.9/3.10)

Key dependencies: numpy, pandas, pyarrow, scikit-learn, matplotlib, seaborn, torch

GPU is optional: the script automatically uses CUDA if available; otherwise it runs on CPU.

Note: reading parquet requires pyarrow (pip install pyarrow).

1. Data Cleaning 、Preprocessing and Prevention of data leakage

All data cleaning and preprocessing steps follow the same procedures as those adopted in the baseline comparison studies to ensure fair and reproducible evaluation.

To avoid information leakage, all normalization procedures are fitted using the training set only: the StandardScaler is fit on the training data and then applied (transform) separately to the validation and test sets. The sliding-window standard-deviation channels are also computed independently for the training, validation, and test sets after data splitting, so no information is shared across subsets, ensuring an unbiased evaluation.

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

Compute valid_length (number of non-zero entries) and keep samples within per-label quantiles .This data preprocessing step was applied only to the cholic acid conjugates dataset, because this dataset still contains unremoved outliers. No such preprocessing was performed on the other existing datasets, for which the corresponding setting is set to 0–1 by default.

Optionally remove extremes per label using a “max drop amplitude” score (controlled by x).

(C) Remove specified labels

remove_labels = [] (edit as needed)

(D) Train/Val/Test split

The dataset was randomly partitioned into training, validation, and test sets using an 8:1:1 ratio.

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

With CAL_POLICY="target_coverage" and TARGET_VALUE=0.8, the script selects a threshold that meets the target coverage and minimizes risk.

The chosen threshold is written back to model.threshold, and per-class gate stats are saved.

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


