import torch


def normalize_grid(
    density: torch.Tensor,
    velx: torch.Tensor,
    vely: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Density: normalize to [0, 1]
    d_min = density.min()
    d_max = density.max()
    d_span = d_max - d_min
    if d_span > 0:
        density_norm = (density - d_min) / d_span
    else:
        density_norm = torch.zeros_like(density)

    # Velocity components: normalize to [-1, 1]
    # Find symmetric range around zero
    vx_abs_max = velx.abs().max()
    if vx_abs_max > 0:
        velx_norm = velx / vx_abs_max
    else:
        velx_norm = torch.zeros_like(velx)

    vy_abs_max = vely.abs().max()
    if vy_abs_max > 0:
        vely_norm = vely / vy_abs_max
    else:
        vely_norm = torch.zeros_like(vely)

    return density_norm, velx_norm, vely_norm


def denormalize_grid(
    density_norm: torch.Tensor,
    velx_norm: torch.Tensor,
    vely_norm: torch.Tensor,
    density_range: tuple[float, float] = (0.0, 1.0),
    velx_range: tuple[float, float] = (-1.0, 1.0),
    vely_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Density: [0, 1] -> [d_min, d_max]
    d_min, d_max = density_range
    density = density_norm * (d_max - d_min) + d_min

    # X-velocity: [-1, 1] -> [vx_min, vx_max]
    vx_min, vx_max = velx_range
    velx = velx_norm * (vx_max - vx_min) / 2.0 + (vx_max + vx_min) / 2.0

    # Y-velocity: [-1, 1] -> [vy_min, vy_max]
    vy_min, vy_max = vely_range
    vely = vely_norm * (vy_max - vy_min) / 2.0 + (vy_max + vy_min) / 2.0

    return density, velx, vely
