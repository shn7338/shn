"""Multi-task latent base-station parameter estimator."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Up(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = functional.interpolate(
                inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat((skip, inputs), dim=1))


class BSParameterEstimator(nn.Module):
    """Estimate location, effective power and azimuth from sparse signal input.

    The dense decoder also predicts the full normalized signal map as an
    auxiliary task. This gives the latent parameter heads richer propagation
    gradients without replacing the existing three-stage final map pipeline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        scalar_hidden_channels: int = 256,
        softargmax_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        channels = int(base_channels)
        self.softargmax_temperature = float(softargmax_temperature)
        self.input = ConvBlock(in_channels, channels)
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16)
        self.up1 = Up(channels * 16, channels * 8, channels * 8)
        self.up2 = Up(channels * 8, channels * 4, channels * 4)
        self.up3 = Up(channels * 4, channels * 2, channels * 2)
        self.up4 = Up(channels * 2, channels, channels)
        self.spatial_head = nn.Conv2d(channels, 2, kernel_size=1)
        scalar_input = channels * 32
        self.scalar_head = nn.Sequential(
            nn.Linear(scalar_input, scalar_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(scalar_hidden_channels, scalar_hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Linear(scalar_hidden_channels // 2, 3),
        )
        row_grid, col_grid = torch.meshgrid(
            torch.arange(128, dtype=torch.float32),
            torch.arange(128, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("row_grid_px", row_grid.reshape(1, -1), persistent=False)
        self.register_buffer("col_grid_px", col_grid.reshape(1, -1), persistent=False)

    def spatial_softargmax(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(
            logits.flatten(1) / self.softargmax_temperature, dim=1
        )
        row = torch.sum(probabilities * self.row_grid_px, dim=1)
        col = torch.sum(probabilities * self.col_grid_px, dim=1)
        return torch.stack((row, col), dim=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        level1 = self.input(inputs)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        bottleneck = self.down4(level4)
        decoded = self.up1(bottleneck, level4)
        decoded = self.up2(decoded, level3)
        decoded = self.up3(decoded, level2)
        decoded = self.up4(decoded, level1)
        spatial = self.spatial_head(decoded)
        pooled = torch.cat(
            (
                functional.adaptive_avg_pool2d(bottleneck, 1).flatten(1),
                functional.adaptive_max_pool2d(bottleneck, 1).flatten(1),
            ),
            dim=1,
        )
        scalar = self.scalar_head(pooled)
        return {
            "location_logits": spatial[:, 0:1],
            "map_norm": spatial[:, 1:2],
            "row_col_px": self.spatial_softargmax(spatial[:, 0]),
            "power_norm": scalar[:, 0],
            "direction_raw": scalar[:, 1:3],
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

