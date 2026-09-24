# Stage-1 DPM U-Net deployment package

## Purpose

This model is a fast surrogate for WinProp Dominant Path Model (DPM) path-gain maps.
It predicts one 128 x 128 path-gain map for a 512 m x 512 m tile at 4 m resolution.

## Required normalized inputs

Each inference tile directory must contain:

- `building_height_norm.npy`
- `tx_position_height_norm.npy`
- `tx_distance_norm.npy`

All arrays must be finite `float32`-compatible 2-D arrays with the same shape and values
in the range `[0, 1]`. Use the included `normalization.json` and
`tx_feature_normalization.json`; do not recompute statistics from inference data.

The model does not require `path_gain_norm.npy` or `path_gain_valid_mask.npy` during
inference.

## Outputs

- `path_gain_pred_norm.npy`: prediction in the training standardized space.
- `path_gain_pred_db.npy`: inverse-standardized DPM path-gain prediction in dB.
- `path_gain_prediction.png`: visualization only; colors are not model inputs.
- `inference_metadata.json`: model, input, timing, range, and optional grid metadata.

## Training scope

- Frequency: 3.5 GHz
- Receiver height: 2 m
- Tile size: 512 m x 512 m
- Grid resolution: 4 m
- Training target: WinProp DPM output
- Input mode: building height + TX position/height + TX distance
- Parameters: 7,763,041

This checkpoint reproduces DPM; it is not validated as a direct predictor of real-world
RSRP. Other frequencies, receiver heights, antenna patterns, or propagation settings
require new labels and validation.

## Held-out test result

- MAE: 1.383 dB
- RMSE: 2.613 dB

## Integrity

- Checkpoint: `best.pt`
- SHA-256: `E44A4580AEFAAC6C0A2166A1824E4B140139EE23B0B87DCC7F78481FF4377C72`

## Canonical source

`D:\桌面\dac\code\dpm_unet_stage1`
