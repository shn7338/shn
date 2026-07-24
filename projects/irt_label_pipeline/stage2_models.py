"""U-Net models for Stage2-A isotropic residual refinement."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = torch.nn.functional.interpolate(
                inputs,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat((skip, inputs), dim=1))


class ResidualUNet(nn.Module):
    """Two-channel U-Net predicting normalized IRT-minus-DPM residual."""

    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        channels = base_channels
        self.input = DoubleConv(in_channels, channels)
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16)
        self.up1 = Up(channels * 16, channels * 8, channels * 8)
        self.up2 = Up(channels * 8, channels * 4, channels * 4)
        self.up3 = Up(channels * 4, channels * 2, channels * 2)
        self.up4 = Up(channels * 2, channels, channels)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level1 = self.input(inputs)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        level5 = self.down4(level4)
        decoded = self.up1(level5, level4)
        decoded = self.up2(decoded, level3)
        decoded = self.up3(decoded, level2)
        decoded = self.up4(decoded, level1)
        return self.output(decoded)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
