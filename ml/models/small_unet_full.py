from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn

NormType = Literal["none", "batch", "instance", "group"]
ActType = Literal["relu", "leaky_relu", "gelu", "silu"]
UpsampleType = Literal["nearest", "bilinear", "transpose"]


def _norm(norm: NormType, ch: int, groups: int) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(ch)
    if norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    if norm == "group":
        g = min(groups, ch)
        while g > 1 and (ch % g) != 0:
            g -= 1
        return nn.GroupNorm(g, ch)
    raise ValueError(f"unknown norm: {norm}")


def _act(act: ActType) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"unknown act: {act}")


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
    ) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=(norm == "none"))
        self.n1 = _norm(norm, out_ch, groups)
        self.a1 = _act(act)

        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=(norm == "none"))
        self.n2 = _norm(norm, out_ch, groups)
        self.a2 = _act(act)

        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.n1(self.c1(x)))
        x = self.drop(x)
        x = self.a2(self.n2(self.c2(x)))
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
    ) -> None:
        super().__init__()
        self.b = ConvBlock(ch, ch, norm=norm, act=act, groups=groups, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", x + self.b(x))


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
    ) -> None:
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, norm=norm, act=act, groups=groups, dropout=dropout)
        self.res = (
            ResBlock(out_ch, norm=norm, act=act, groups=groups, dropout=dropout) if use_residual else nn.Identity()
        )
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
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
    ) -> None:
        super().__init__()

        # Important for mypy: this attribute must accept both Upsample and ConvTranspose2d
        self.up: nn.Module
        if upsample == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(
                scale_factor=2,
                mode=upsample,
                align_corners=False if upsample == "bilinear" else None,
            )

        self.block = ConvBlock(in_ch + skip_ch, out_ch, norm=norm, act=act, groups=groups, dropout=dropout)
        self.res = (
            ResBlock(out_ch, norm=norm, act=act, groups=groups, dropout=dropout) if use_residual else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # If shapes don't match (odd sizes), crop skip to match x (center-ish).
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh != 0 or dw != 0:
                skip = skip[..., dh // 2 : dh // 2 + x.shape[-2], dw // 2 : dw // 2 + x.shape[-1]]

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.res(x)
        return x


@dataclass(frozen=True)
class SmallUNetFullConfig:
    in_channels: int = 7
    out_channels: int = 3

    base_channels: int = 32
    depth: int = 2

    norm: NormType = "group"
    act: ActType = "silu"
    group_norm_groups: int = 8
    dropout: float = 0.0

    upsample: UpsampleType = "nearest"
    use_residual: bool = False
    bottleneck_blocks: int = 1


class SmallUNetFull(nn.Module):
    """
    More configurable U-Net (still ONNX-friendly).
    """

    model_name: str = "SmallUnetFull"

    def __init__(self, cfg: SmallUNetFullConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SmallUNetFullConfig()

        if self.cfg.depth < 1:
            raise ValueError("depth must be >= 1")

        self.stem = nn.Conv2d(self.cfg.in_channels, self.cfg.base_channels, 3, padding=1)

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
                )
            )
        self.mid = nn.Sequential(*mids) if mids else nn.Identity()

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
                )
            )
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.head = nn.Conv2d(ch, self.cfg.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, s = cast("Down", down)(x)
            skips.append(s)

        x = self.mid(x)

        for up, skip in zip(self.ups, reversed(skips), strict=True):
            x = cast("Up", up)(x, skip)

        return cast("torch.Tensor", self.head(x))


__all__ = ["SmallUNetFull", "SmallUNetFullConfig"]
