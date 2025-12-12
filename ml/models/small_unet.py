from typing import cast

import torch
import torch.nn as nn


class SmallUNet(nn.Module):
    """
    https://github.com/milesial/Pytorch-UNet
    Simple baseline U-Net:
      - GELU everywhere
      - ConvTranspose2d for upsampling
      - tweakable via constructor args (in/out/base/depth)
    """

    def __init__(
        self,
        in_channels: int = 4, #(density, velx, vely, density-1)
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GELU(),
            )

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder: (block -> downsample) repeated
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.enc.append(block(ch, ch))
            self.down.append(nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1))
            ch *= 2

        self.mid = block(ch, ch)

        # Decoder: upsample -> concat skip -> block
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2))
            ch //= 2
            self.dec.append(block(ch * 2, ch))

        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for enc, down in zip(self.enc, self.down, strict=True):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.mid(x)

        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = up(x)

            # Handle odd shapes (shouldn't happen if H,W are divisible by 2**depth)
            if x.shape[-2:] != skip.shape[-2:]:
                x = x[..., : skip.shape[-2], : skip.shape[-1]]

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return cast("torch.Tensor", self.head(x))
