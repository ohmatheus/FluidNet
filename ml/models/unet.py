from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn

NormType = Literal["none", "batch", "instance", "group"]
ActType = Literal["relu", "leaky_relu", "gelu", "silu"]
UpsampleType = Literal["nearest", "bilinear", "transpose"]
DownsampleType = Literal["stride", "avgpool", "maxpool"]
PaddingType = Literal["zeros", "reflect", "replicate", "circular"]
OutputActivationType = Literal["sigmoid_tanh", "linear_clamp"]


def _norm(norm: NormType, ch: int, groups: int, affine: bool = True) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(ch)
    if norm == "instance":
        return nn.InstanceNorm2d(ch, affine=affine)
    if norm == "group":
        g = min(groups, ch)
        while g > 1 and (ch % g) != 0:
            g -= 1
        return nn.GroupNorm(g, ch, affine=affine)


def _act(act: ActType) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)


class ConditioningEncoder(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.mlp(c.unsqueeze(-1)))  # (B,) → (B,1) → (B, cond_dim)


class FiLMLayer(nn.Module):
    def __init__(self, ch: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = cast("torch.Tensor", self.proj(cond_emb)).chunk(2, dim=-1)  # each (B, C)
        return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: NormType,
        act: ActType,
        groups: int,
        dropout: float,
        padding_mode: PaddingType = "zeros",
        film_cond_dim: int = 0,
    ) -> None:
        super().__init__()
        use_film = film_cond_dim > 0
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode, bias=(norm == "none"))
        self.n1 = _norm(norm, out_ch, groups, affine=not use_film)
        self.a1 = _act(act)

        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode, bias=(norm == "none"))
        self.n2 = _norm(norm, out_ch, groups, affine=not use_film)
        self.a2 = _act(act)

        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.film1 = FiLMLayer(out_ch, film_cond_dim) if use_film else None
        self.film2 = FiLMLayer(out_ch, film_cond_dim) if use_film else None

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.n1(self.c1(x))
        if self.film1 is not None and cond_emb is not None:
            x = self.film1(x, cond_emb)
        x = self.a1(x)
        x = self.drop(x)
        x = self.n2(self.c2(x))
        if self.film2 is not None and cond_emb is not None:
            x = self.film2(x, cond_emb)
        x = self.a2(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        *,
        norm: NormType,
        act: ActType,
        groups: int,
        dropout: float,
        padding_mode: PaddingType = "zeros",
        film_cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.b = ConvBlock(
            ch,
            ch,
            norm=norm,
            act=act,
            groups=groups,
            dropout=dropout,
            padding_mode=padding_mode,
            film_cond_dim=film_cond_dim,
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        return cast("torch.Tensor", x + self.b(x, cond_emb))


def _downsample(mode: DownsampleType, ch: int, padding_mode: PaddingType) -> nn.Module:
    if mode == "stride":
        return nn.Conv2d(ch, ch, 3, stride=2, padding=1, padding_mode=padding_mode)
    pool = nn.AvgPool2d(2) if mode == "avgpool" else nn.MaxPool2d(2)
    return nn.Sequential(pool, nn.Conv2d(ch, ch, 3, padding=1, padding_mode=padding_mode))


class Down(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: NormType,
        act: ActType,
        groups: int,
        dropout: float,
        use_residual: bool,
        downsample: DownsampleType = "stride",
        padding_mode: PaddingType = "zeros",
        film_cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.block = ConvBlock(
            in_ch,
            out_ch,
            norm=norm,
            act=act,
            groups=groups,
            dropout=dropout,
            padding_mode=padding_mode,
            film_cond_dim=film_cond_dim,
        )
        self.res = (
            ResBlock(
                out_ch,
                norm=norm,
                act=act,
                groups=groups,
                dropout=dropout,
                padding_mode=padding_mode,
                film_cond_dim=film_cond_dim,
            )
            if use_residual
            else nn.Identity()
        )
        self.down = _downsample(downsample, out_ch, padding_mode)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x, cond_emb)
        if isinstance(self.res, ResBlock):
            x = self.res(x, cond_emb)
        else:
            x = self.res(x)
        skip = x
        x = self.down(x)
        return x, skip


class Up(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        *,
        upsample: UpsampleType,
        norm: NormType,
        act: ActType,
        groups: int,
        dropout: float,
        use_residual: bool,
        padding_mode: PaddingType = "zeros",
        film_cond_dim: int = 0,
    ) -> None:
        super().__init__()

        self.up: nn.Module
        if upsample == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(
                scale_factor=2,
                mode=upsample,
                align_corners=False if upsample == "bilinear" else None,
            )

        self.block = ConvBlock(
            in_ch + skip_ch,
            out_ch,
            norm=norm,
            act=act,
            groups=groups,
            dropout=dropout,
            padding_mode=padding_mode,
            film_cond_dim=film_cond_dim,
        )
        self.res = (
            ResBlock(
                out_ch,
                norm=norm,
                act=act,
                groups=groups,
                dropout=dropout,
                padding_mode=padding_mode,
                film_cond_dim=film_cond_dim,
            )
            if use_residual
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh != 0 or dw != 0:
                skip = skip[..., dh // 2 : dh // 2 + x.shape[-2], dw // 2 : dw // 2 + x.shape[-1]]

        x = torch.cat([x, skip], dim=1)
        x = self.block(x, cond_emb)
        if isinstance(self.res, ResBlock):
            x = self.res(x, cond_emb)
        else:
            x = self.res(x)
        return x


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 7
    out_channels: int = 3

    base_channels: int = 32
    depth: int = 2

    norm: NormType = "group"
    act: ActType = "silu"
    group_norm_groups: int = 8
    dropout: float = 0.0

    upsample: UpsampleType = "nearest"
    downsample: DownsampleType = "stride"
    padding_mode: PaddingType = "zeros"
    use_residual: bool = False
    bottleneck_blocks: int = 1
    output_activation: OutputActivationType = "linear_clamp"

    use_film: bool = False
    film_cond_dim: int = 128


class UNet(nn.Module):
    """
    U-Net ONNX-friendly.
    """

    model_name: str = "Unet"

    def __init__(self, cfg: UNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()

        if self.cfg.depth < 1:
            raise ValueError("depth must be >= 1")

        film_cond_dim = self.cfg.film_cond_dim if self.cfg.use_film else 0

        if self.cfg.use_film:
            self.cond_encoder = ConditioningEncoder(self.cfg.film_cond_dim)

        self.stem = nn.Conv2d(
            self.cfg.in_channels, self.cfg.base_channels, 3, padding=1, padding_mode=self.cfg.padding_mode
        )

        downs: list[nn.Module] = []
        skip_chs: list[int] = []
        ch = self.cfg.base_channels
        for _ in range(self.cfg.depth):
            out_ch = ch * 2
            downs.append(
                Down(
                    ch,
                    out_ch,
                    norm=self.cfg.norm,
                    act=self.cfg.act,
                    groups=self.cfg.group_norm_groups,
                    dropout=self.cfg.dropout,
                    use_residual=self.cfg.use_residual,
                    downsample=self.cfg.downsample,
                    padding_mode=self.cfg.padding_mode,
                    film_cond_dim=film_cond_dim,
                )
            )
            skip_chs.append(out_ch)
            ch = out_ch
        self.downs = nn.ModuleList(downs)

        mids: list[nn.Module] = []
        for _ in range(max(0, self.cfg.bottleneck_blocks)):
            mids.append(
                ResBlock(
                    ch,
                    norm=self.cfg.norm,
                    act=self.cfg.act,
                    groups=self.cfg.group_norm_groups,
                    dropout=self.cfg.dropout,
                    padding_mode=self.cfg.padding_mode,
                    film_cond_dim=film_cond_dim,
                )
            )
        self.mid = nn.ModuleList(mids)

        ups: list[nn.Module] = []
        for i in reversed(range(self.cfg.depth)):
            skip_ch = skip_chs[i]
            out_ch = skip_ch // 2
            ups.append(
                Up(
                    ch,
                    skip_ch,
                    out_ch,
                    upsample=self.cfg.upsample,
                    norm=self.cfg.norm,
                    act=self.cfg.act,
                    groups=self.cfg.group_norm_groups,
                    dropout=self.cfg.dropout,
                    use_residual=self.cfg.use_residual,
                    padding_mode=self.cfg.padding_mode,
                    film_cond_dim=film_cond_dim,
                )
            )
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.head = nn.Conv2d(ch, self.cfg.out_channels, 1)

    def forward(self, x: torch.Tensor, cond_scalar: torch.Tensor | None = None) -> torch.Tensor:
        cond_emb: torch.Tensor | None = None
        if self.cfg.use_film and cond_scalar is not None:
            cond_emb = self.cond_encoder(cond_scalar)

        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, s = cast("Down", down)(x, cond_emb)
            skips.append(s)

        for block in self.mid:
            x = cast("ResBlock", block)(x, cond_emb)

        for up, skip in zip(self.ups, reversed(skips), strict=True):
            x = cast("Up", up)(x, skip, cond_emb)

        x = self.head(x)

        if self.cfg.output_activation == "sigmoid_tanh":
            density = torch.sigmoid(x[:, 0:1, :, :])
            velocity = torch.tanh(x[:, 1:3, :, :])
        else:
            density = torch.clamp(x[:, 0:1, :, :], 0.0, 1.0)
            velocity = torch.clamp(x[:, 1:3, :, :], -1.0, 1.0)

        return torch.cat([density, velocity], dim=1)


__all__ = ["UNet", "UNetConfig", "DownsampleType", "OutputActivationType", "PaddingType"]
