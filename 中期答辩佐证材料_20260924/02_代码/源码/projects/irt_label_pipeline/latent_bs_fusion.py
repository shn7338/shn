"""Reusable latent base-station feature construction for Stage2-B V4."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional


BASE_STAGE2B_CHANNELS = (
    "building_height_norm",
    "corrected_isotropic_map_norm",
    "sparse_signal_strength_norm",
    "sparse_measurement_mask",
)

DEFAULT_LATENT_FEATURES = (
    "location_probability",
    "location_confidence",
    "effective_power",
    "azimuth_sin",
    "azimuth_cos",
    "sector_gain_prior",
    "estimator_auxiliary_map",
)


def configured_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    fusion = config.get("latent_fusion", {})
    if not bool(fusion.get("enabled", False)):
        return ()
    names = tuple(str(name) for name in fusion.get("features", DEFAULT_LATENT_FEATURES))
    unknown = sorted(set(names) - set(DEFAULT_LATENT_FEATURES))
    if unknown:
        raise ValueError(f"unknown latent fusion features: {unknown}")
    if len(names) != len(set(names)):
        raise ValueError("latent fusion feature names must be unique")
    return names


def configured_stage2b_in_channels(config: dict[str, Any]) -> int:
    return len(BASE_STAGE2B_CHANNELS) + len(configured_feature_names(config))


def first_convolution_weight_key(state: dict[str, torch.Tensor]) -> str:
    candidates = (
        "input.block.0.weight",
        "input.block.0.block.0.weight",
    )
    for key in candidates:
        value = state.get(key)
        if value is not None and value.ndim == 4:
            return key
    matches = [
        key
        for key, value in state.items()
        if value.ndim == 4 and key.endswith("weight")
    ]
    if not matches:
        raise ValueError("model state contains no convolution weight")
    return matches[0]


def transplant_input_channels(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str, int, int]:
    """Copy a Stage2-B state and zero-initialize only new input planes."""

    key = first_convolution_weight_key(source_state)
    target_key = first_convolution_weight_key(target_state)
    if target_key != key:
        raise ValueError(f"first-convolution key changed: {key!r} -> {target_key!r}")
    source_channels = int(source_state[key].shape[1])
    target_channels = int(target_state[key].shape[1])
    if target_channels < source_channels:
        raise ValueError("target Stage2-B has fewer input channels than its source")
    transplanted: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        source_value = source_state[name]
        if name == key and source_channels != target_channels:
            value = torch.zeros_like(target_value)
            value[:, :source_channels].copy_(
                source_value.to(device=value.device, dtype=value.dtype)
            )
        else:
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"unexpected shape change for {name}: "
                    f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
                )
            value = source_value.to(device=target_value.device, dtype=target_value.dtype)
        transplanted[name] = value
    return transplanted, key, source_channels, target_channels


def build_latent_fusion_features(
    estimated: dict[str, torch.Tensor],
    fusion_config: dict[str, Any],
    *,
    estimator_signal_mean_db: float,
    estimator_signal_std_db: float,
    stage2b_signal_mean_db: float,
    stage2b_signal_std_db: float,
    estimator_power_mean_db: float,
    estimator_power_std_db: float,
    rows: int = 128,
    cols: int = 128,
    cell_size_m: float = 4.0,
    map_size_m: float = 512.0,
) -> torch.Tensor:
    """Convert estimator outputs into dense, normalized Stage2-B channels."""

    names = tuple(str(name) for name in fusion_config.get("features", DEFAULT_LATENT_FEATURES))
    if not names:
        batch = int(estimated["row_col_px"].shape[0])
        return estimated["row_col_px"].new_empty((batch, 0, rows, cols))
    logits = estimated["location_logits"].float()
    batch = int(logits.shape[0])
    if logits.shape[-2:] != (rows, cols):
        raise ValueError(f"unexpected estimator heatmap shape: {tuple(logits.shape)}")
    probabilities = torch.softmax(logits.flatten(1), dim=1).view(batch, 1, rows, cols)
    peak = probabilities.flatten(1).amax(dim=1).view(batch, 1, 1, 1)
    location_probability = probabilities / peak.clamp_min(1e-12)
    entropy = -torch.sum(
        probabilities.flatten(1) * torch.log(probabilities.flatten(1).clamp_min(1e-12)),
        dim=1,
    ) / math.log(rows * cols)
    location_confidence = (1.0 - entropy).view(batch, 1, 1, 1).expand(
        batch, 1, rows, cols
    )

    power_db = (
        estimated["power_norm"].float() * estimator_power_std_db
        + estimator_power_mean_db
    )
    power_min_db = float(fusion_config.get("power_min_db", 35.0))
    power_max_db = float(fusion_config.get("power_max_db", 70.0))
    power_center_db = (power_min_db + power_max_db) / 2.0
    power_half_range_db = (power_max_db - power_min_db) / 2.0
    effective_power = torch.clamp(
        (power_db - power_center_db) / power_half_range_db, -1.0, 1.0
    ).view(batch, 1, 1, 1).expand(batch, 1, rows, cols)

    direction = functional.normalize(estimated["direction_raw"].float(), dim=1)
    azimuth_sin = direction[:, 0].view(batch, 1, 1, 1).expand(batch, 1, rows, cols)
    azimuth_cos = direction[:, 1].view(batch, 1, 1, 1).expand(batch, 1, rows, cols)

    device = logits.device
    x_centers = (
        torch.arange(cols, device=device, dtype=torch.float32) + 0.5
    ) * cell_size_m
    y_centers = map_size_m - (
        torch.arange(rows, device=device, dtype=torch.float32) + 0.5
    ) * cell_size_m
    row_col = estimated["row_col_px"].float()
    tx_x_m = (row_col[:, 1] + 0.5) * cell_size_m
    tx_y_m = map_size_m - (row_col[:, 0] + 0.5) * cell_size_m
    delta_x = x_centers.view(1, 1, cols) - tx_x_m.view(batch, 1, 1)
    delta_y = y_centers.view(1, rows, 1) - tx_y_m.view(batch, 1, 1)
    radius = torch.sqrt(delta_x.square() + delta_y.square()).clamp_min(1e-4)
    unit_x = delta_x.expand(batch, rows, cols) / radius
    unit_y = delta_y.expand(batch, rows, cols) / radius
    dot = torch.clamp(
        unit_x * direction[:, 1].view(batch, 1, 1)
        + unit_y * direction[:, 0].view(batch, 1, 1),
        -1.0,
        1.0,
    )
    angular_offset_deg = torch.rad2deg(torch.acos(dot))
    horizontal_hpbw_deg = float(fusion_config.get("horizontal_hpbw_deg", 65.0))
    attenuation_cap_db = float(fusion_config.get("attenuation_cap_db", 30.0))
    attenuation_db = torch.clamp(
        12.0 * (angular_offset_deg / horizontal_hpbw_deg).square(),
        max=attenuation_cap_db,
    )
    sector_gain_prior = (1.0 - attenuation_db / attenuation_cap_db).unsqueeze(1)

    estimator_auxiliary_map_db = (
        estimated["map_norm"].float() * estimator_signal_std_db
        + estimator_signal_mean_db
    )
    estimator_auxiliary_map = (
        estimator_auxiliary_map_db - stage2b_signal_mean_db
    ) / stage2b_signal_std_db
    estimator_auxiliary_map = torch.clamp(estimator_auxiliary_map, -4.0, 4.0)

    available = {
        "location_probability": location_probability,
        "location_confidence": location_confidence,
        "effective_power": effective_power,
        "azimuth_sin": azimuth_sin,
        "azimuth_cos": azimuth_cos,
        "sector_gain_prior": sector_gain_prior,
        "estimator_auxiliary_map": estimator_auxiliary_map,
    }
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown latent fusion features: {unknown}")
    return torch.cat([available[name] for name in names], dim=1)

