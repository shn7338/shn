"""U-Net models for Stage2-A isotropic residual refinement.

``ResidualUNet`` is kept byte-for-byte compatible with the frozen Stage-1
DPM checkpoint. ``Geo2SigMapUNet`` follows the U-Net-Iso architecture in the
Geo2SigMap paper and its public implementation: four encoder/decoder levels,
two 3x3 Conv-BN-ReLU layers per block, max pooling, transposed convolution,
and skip concatenation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


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


class PaperDoubleConv(nn.Module):
    """Geo2SigMap convolution block: (3x3 Conv + BN + ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class PaperDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = PaperDoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class PaperUp(nn.Module):
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
        self.conv = PaperDoubleConv(out_channels + skip_channels, out_channels)

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


class Geo2SigMapUNet(nn.Module):
    """Paper-faithful U-Net backbone that predicts a normalized residual.

    The DPM baseline is added outside the module in dB space. This is
    equivalent to the ``pathloss=True`` residual connection in the authors'
    public implementation while keeping normalization explicit.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        gradient_checkpointing: bool = False,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        channels = base_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.input = PaperDoubleConv(in_channels, channels)
        self.down1 = PaperDown(channels, channels * 2)
        self.down2 = PaperDown(channels * 2, channels * 4)
        self.down3 = PaperDown(channels * 4, channels * 8)
        self.down4 = PaperDown(channels * 8, channels * 16)
        self.up1 = PaperUp(channels * 16, channels * 8, channels * 8)
        self.up2 = PaperUp(channels * 8, channels * 4, channels * 4)
        self.up3 = PaperUp(channels * 4, channels * 2, channels * 2)
        self.up4 = PaperUp(channels * 2, channels, channels)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _forward_block(
        self,
        module: nn.Module,
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        level1 = self._forward_block(self.input, inputs)
        level2 = self._forward_block(self.down1, level1)
        level3 = self._forward_block(self.down2, level2)
        level4 = self._forward_block(self.down3, level3)
        level5 = self._forward_block(self.down4, level4)
        decoded = self._forward_block(self.up1, level5, level4)
        decoded = self._forward_block(self.up2, decoded, level3)
        decoded = self._forward_block(self.up3, decoded, level2)
        decoded = self._forward_block(self.up4, decoded, level1)
        return self.output(decoded)


class ZeroResidualSpatialAdapter(nn.Module):
    """Project dense latent maps into one zero-initialized residual feature."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if inputs.shape[-2:] != size:
            inputs = functional.adaptive_avg_pool2d(inputs, size)
        return self.block(inputs)


class LatentAdapterGeo2SigMapUNet(nn.Module):
    """Geo2SigMap backbone with multiscale spatial adapters and scalar FiLM.

    The adapter projections and final FiLM layer are initialized to zero, so
    after loading a compatible backbone the module is exactly equivalent to
    that backbone before any V5 optimization.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        gradient_checkpointing: bool = False,
        spatial_channel_indices: tuple[int, ...] = (4, 9, 10),
        scalar_channel_indices: tuple[int, ...] = (5, 6, 7, 8),
        adapter_hidden_channels: int = 24,
        film_hidden_channels: int = 128,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if not spatial_channel_indices:
            raise ValueError("at least one spatial latent channel is required")
        if not scalar_channel_indices:
            raise ValueError("at least one scalar latent channel is required")
        if max((*spatial_channel_indices, *scalar_channel_indices)) >= in_channels:
            raise ValueError("latent adapter channel index exceeds model input")
        self.spatial_channel_indices = tuple(int(v) for v in spatial_channel_indices)
        self.scalar_channel_indices = tuple(int(v) for v in scalar_channel_indices)
        self.film_scale = float(film_scale)
        self.optimization_phase = "adapter"
        self.backbone = Geo2SigMapUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            gradient_checkpointing=gradient_checkpointing,
            zero_init_output=False,
        )
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        spatial_channels = len(self.spatial_channel_indices)
        self.encoder_adapters = nn.ModuleList(
            ZeroResidualSpatialAdapter(
                spatial_channels, out_channels, adapter_hidden_channels
            )
            for out_channels in channels
        )
        decoder_channels = (channels[3], channels[2], channels[1], channels[0])
        self.decoder_adapters = nn.ModuleList(
            ZeroResidualSpatialAdapter(
                spatial_channels, out_channels, adapter_hidden_channels
            )
            for out_channels in decoder_channels
        )
        self.film_channels = (channels[4], *decoder_channels)
        film_outputs = 2 * sum(self.film_channels)
        self.scalar_film = nn.Sequential(
            nn.Linear(len(self.scalar_channel_indices), film_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(film_hidden_channels, film_outputs),
        )
        nn.init.zeros_(self.scalar_film[-1].weight)
        nn.init.zeros_(self.scalar_film[-1].bias)
        self.set_optimization_phase("adapter")

    def adapter_parameters(self):
        yield from self.encoder_adapters.parameters()
        yield from self.decoder_adapters.parameters()
        yield from self.scalar_film.parameters()

    def late_backbone_parameters(self):
        for module in (self.backbone.up3, self.backbone.up4, self.backbone.output):
            yield from module.parameters()

    def set_optimization_phase(self, phase: str) -> None:
        if phase not in {"adapter", "decoder"}:
            raise ValueError(f"unsupported V5 optimization phase: {phase!r}")
        self.optimization_phase = phase
        self.backbone.requires_grad_(False)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(True)
        if phase == "decoder":
            for parameter in self.late_backbone_parameters():
                parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Preserve every frozen BatchNorm statistic. In decoder phase only
            # the two trainable late decoder blocks update their BN statistics.
            self.backbone.eval()
            if self.optimization_phase == "decoder":
                self.backbone.up3.train(True)
                self.backbone.up4.train(True)
                self.backbone.output.train(True)
        return self

    def _film_parameters(
        self,
        scalar_values: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        flat = self.scalar_film(scalar_values)
        batch = int(flat.shape[0])
        result: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for channels in self.film_channels:
            gamma = flat[:, offset : offset + channels]
            offset += channels
            beta = flat[:, offset : offset + channels]
            offset += channels
            result.append(
                (
                    gamma.view(batch, channels, 1, 1),
                    beta.view(batch, channels, 1, 1),
                )
            )
        return result

    def _apply_film(
        self,
        features: torch.Tensor,
        parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        gamma, beta = parameters
        return features * (1.0 + self.film_scale * gamma) + self.film_scale * beta

    def _forward_backbone_block(
        self,
        module: nn.Module,
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        # The backbone stays in eval mode while frozen so BatchNorm statistics
        # do not drift. Checkpointing must therefore follow the wrapper's
        # training state instead of the backbone module's ``training`` flag.
        if (
            self.backbone.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = inputs[:, self.spatial_channel_indices]
        scalar_maps = inputs[:, self.scalar_channel_indices]
        scalar_values = scalar_maps.mean(dim=(-2, -1))
        film = self._film_parameters(scalar_values)
        backbone = self.backbone

        level1 = self._forward_backbone_block(backbone.input, inputs)
        level1 = level1 + self.encoder_adapters[0](spatial, level1.shape[-2:])
        level2 = self._forward_backbone_block(backbone.down1, level1)
        level2 = level2 + self.encoder_adapters[1](spatial, level2.shape[-2:])
        level3 = self._forward_backbone_block(backbone.down2, level2)
        level3 = level3 + self.encoder_adapters[2](spatial, level3.shape[-2:])
        level4 = self._forward_backbone_block(backbone.down3, level3)
        level4 = level4 + self.encoder_adapters[3](spatial, level4.shape[-2:])
        level5 = self._forward_backbone_block(backbone.down4, level4)
        level5 = level5 + self.encoder_adapters[4](spatial, level5.shape[-2:])
        level5 = self._apply_film(level5, film[0])

        decoded = self._forward_backbone_block(backbone.up1, level5, level4)
        decoded = decoded + self.decoder_adapters[0](spatial, decoded.shape[-2:])
        decoded = self._apply_film(decoded, film[1])
        decoded = self._forward_backbone_block(backbone.up2, decoded, level3)
        decoded = decoded + self.decoder_adapters[1](spatial, decoded.shape[-2:])
        decoded = self._apply_film(decoded, film[2])
        decoded = self._forward_backbone_block(backbone.up3, decoded, level2)
        decoded = decoded + self.decoder_adapters[2](spatial, decoded.shape[-2:])
        decoded = self._apply_film(decoded, film[3])
        decoded = self._forward_backbone_block(backbone.up4, decoded, level1)
        decoded = decoded + self.decoder_adapters[3](spatial, decoded.shape[-2:])
        decoded = self._apply_film(decoded, film[4])
        return backbone.output(decoded)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
