"""
physics-based constraints:
1. Divergence loss - enforces incompressibility (∇·v ~= 0) - https://arxiv.org/pdf/1806.02071
2. Gradient loss - preserves sharp density features - https://arxiv.org/pdf/1806.02071
^ those 2 helps preventing density 'spawning' durring rollout. Need to finetune weigths

3. Mass loss: Ignored. Assume the total fluid as equal mass at each frame, which is not our case since we have emission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spatial_gradients(
    field: torch.Tensor, dx: float = 1.0, dy: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial gradients using central differences.
    """
    if field.ndim == 3:
        field = field.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
        squeeze = True
    else:
        squeeze = False

    # ∂f/∂x ~= (f[i+1] - f[i-1]) / (2*dx)
    # Pad: left=1, right=1, top=1, bottom=1
    # Original shape: (B, C, H, W)
    # Padded shape: (B, C, H+2, W+2)
    field_padded = F.pad(field, (1, 1, 1, 1), mode="replicate")

    # X gradient (horizontal) - differences along width (last) dimension
    # Take slices [2:] and [:-2] which gives us (B, C, H+2, W)
    # But we want (B, C, H, W), so we need to also slice the height
    grad_x = (field_padded[:, :, 1:-1, 2:] - field_padded[:, :, 1:-1, :-2]) / (2.0 * dx)

    # differences along height dimension
    # (B, C, H, W+2)
    grad_y = (field_padded[:, :, 2:, 1:-1] - field_padded[:, :, :-2, 1:-1]) / (2.0 * dy)

    if squeeze:
        grad_x = grad_x.squeeze(1)
        grad_y = grad_y.squeeze(1)

    return grad_x, grad_y


def compute_divergence(velx: torch.Tensor, vely: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    Compute velocity divergence: ∇·v = ∂velx/∂x + ∂vely/∂y
    """
    grad_vx_x, _ = compute_spatial_gradients(velx, dx, dy)
    _, grad_vy_y = compute_spatial_gradients(vely, dx, dy)

    assert grad_vx_x.shape == grad_vy_y.shape, f"Shape mismatch: {grad_vx_x.shape} vs {grad_vy_y.shape}"

    return grad_vx_x + grad_vy_y


def divergence_loss(
    velx: torch.Tensor,
    vely: torch.Tensor,
    emitter_mask: torch.Tensor,
    collider_mask: torch.Tensor | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Divergence-free constraint: ∇·v ≈ 0 in fluid regions.

    Excludes emitter regions (where mass is injected) and optionally collider regions.
    """
    div = compute_divergence(velx, vely, dx, dy)

    # Create fluid region mask
    fluid_mask = (emitter_mask == 0.0).float()
    if collider_mask is not None:
        fluid_mask = fluid_mask * (collider_mask == 0.0).float()

    # Compute mean absolute divergence in fluid regions
    # Handle case where mask is all zeros
    num_fluid_cells = fluid_mask.sum()
    if num_fluid_cells < eps:
        return torch.tensor(0.0, device=velx.device, dtype=velx.dtype)

    masked_div = torch.abs(div) * fluid_mask
    loss = masked_div.sum() / (num_fluid_cells + eps)

    return loss


def gradient_loss(
    density_pred: torch.Tensor,
    density_target: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
) -> torch.Tensor:
    """
    Gradient/edge preservation loss for density field.
    Helps preserve sharp smoke edges by matching spatial gradients.
    """
    grad_pred_x, grad_pred_y = compute_spatial_gradients(density_pred, dx, dy)
    grad_target_x, grad_target_y = compute_spatial_gradients(density_target, dx, dy)

    loss_x = F.l1_loss(grad_pred_x, grad_target_x)
    loss_y = F.l1_loss(grad_pred_y, grad_target_y)

    return loss_x + loss_y


def emitter_spawn_loss(
    density_pred: torch.Tensor,
    density_current: torch.Tensor,
    emitter_mask: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize density spawning in non-emitter regions.
    """
    allowed_mask = (emitter_mask > 0.01) | (density_current > threshold)
    forbidden_density = density_pred * (~allowed_mask).float()
    return forbidden_density.mean()


class PhysicsAwareLoss(nn.Module):
    """
    Combines:
    - MSE reconstruction loss
    - Divergence-free constraint (∇·v ≈ 0)
    - Gradient preservation
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        divergence_weight: float = 0.1,
        gradient_weight: float = 0.1,
        emitter_weight: float = 0.1,
        grid_spacing: float = 1.0,
        enable_divergence: bool = True,
        enable_gradient: bool = True,
        enable_emitter: bool = True,
    ) -> None:
        super().__init__()

        self.mse_weight = mse_weight
        self.divergence_weight = divergence_weight
        self.gradient_weight = gradient_weight
        self.emitter_weight = emitter_weight
        self.grid_spacing = grid_spacing

        self.enable_divergence = enable_divergence
        self.enable_gradient = enable_gradient
        self.enable_emitter = enable_emitter

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Extract output channels
        density_pred = outputs[:, 0, :, :]  # (B, H, W)
        velx_pred = outputs[:, 1, :, :]
        vely_pred = outputs[:, 2, :, :]

        density_target = targets[:, 0, :, :]

        # Extract input channels
        emitter_mask = inputs[:, 4, :, :]  # (B, H, W)
        collider_mask = inputs[:, 5, :, :]

        loss_mse = self.mse_loss(outputs, targets)

        loss_dict = {"mse": loss_mse.item()}

        loss_div = torch.tensor(0.0, device=outputs.device)
        if self.enable_divergence and self.divergence_weight > 0:
            loss_div = divergence_loss(
                velx_pred,
                vely_pred,
                emitter_mask,
                collider_mask,
                dx=self.grid_spacing,
                dy=self.grid_spacing,
            )
            loss_dict["divergence"] = loss_div.item()

        loss_grad = torch.tensor(0.0, device=outputs.device)
        if self.enable_gradient and self.gradient_weight > 0:
            loss_grad = gradient_loss(
                density_pred,
                density_target,
                dx=self.grid_spacing,
                dy=self.grid_spacing,
            )
            loss_dict["gradient"] = loss_grad.item()

        loss_emitter = torch.tensor(0.0, device=outputs.device)
        if self.enable_emitter and self.emitter_weight > 0:
            density_pred_3d = outputs[:, 0:1, :, :]
            density_current = inputs[:, 0:1, :, :]
            emitter_mask_3d = inputs[:, 4:5, :, :]

            loss_emitter = emitter_spawn_loss(density_pred_3d, density_current, emitter_mask_3d)

            loss_dict["emitter"] = loss_emitter.item()

        total_loss = (
            self.mse_weight * loss_mse
            + self.divergence_weight * loss_div
            + self.gradient_weight * loss_grad
            + self.emitter_weight * loss_emitter
        )

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict
