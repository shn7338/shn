"""Masked losses and dB-domain metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "huber",
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = mask.bool()
    if not torch.any(valid):
        # Retain a gradient path so distributed/AMP training does not fail.
        return prediction.sum() * 0.0
    if kind == "mse":
        loss_map = F.mse_loss(prediction, target, reduction="none")
    elif kind == "huber":
        loss_map = F.smooth_l1_loss(prediction, target, beta=1.0, reduction="none")
    else:
        raise ValueError(f"Unknown loss: {kind}")
    if pixel_weight is None:
        return loss_map[valid].mean()
    weights = pixel_weight.to(loss_map.dtype)
    return (loss_map[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)


@dataclass
class DBMetrics:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    valid_count: int = 0

    def update(self, prediction_norm: torch.Tensor, target_norm: torch.Tensor, mask: torch.Tensor, gain_std_db: float) -> None:
        valid = mask.bool()
        if not torch.any(valid):
            return
        error_db = (prediction_norm[valid] - target_norm[valid]).float() * gain_std_db
        self.absolute_error_sum += float(error_db.abs().sum().item())
        self.squared_error_sum += float(error_db.square().sum().item())
        self.valid_count += int(error_db.numel())

    def compute(self) -> dict[str, float]:
        if self.valid_count == 0:
            return {"mae_db": float("nan"), "rmse_db": float("nan")}
        return {
            "mae_db": self.absolute_error_sum / self.valid_count,
            "rmse_db": (self.squared_error_sum / self.valid_count) ** 0.5,
        }
