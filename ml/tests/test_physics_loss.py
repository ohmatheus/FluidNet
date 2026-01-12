import pytest
import torch

from training.physics_loss import (
    PhysicsAwareLoss,
    compute_divergence,
    compute_spatial_gradients,
    divergence_loss,
    gradient_loss,
)


class TestGradientComputation:
    def test_constant_field_has_zero_gradient(self) -> None:
        field = torch.ones(2, 32, 32)
        grad_x, grad_y = compute_spatial_gradients(field, padding_mode="replicate")

        assert torch.allclose(grad_x, torch.zeros_like(grad_x), atol=1e-6)
        assert torch.allclose(grad_y, torch.zeros_like(grad_y), atol=1e-6)

    def test_linear_field_has_constant_gradient(self) -> None:
        # Linear ramp in x direction: field[y, x] = x
        x_coords = torch.arange(32).float().unsqueeze(0).expand(32, 32)
        field = x_coords.unsqueeze(0).expand(2, -1, -1)
        grad_x, grad_y = compute_spatial_gradients(field, dx=1.0, dy=1.0, padding_mode="replicate")

        # The gradient should be approximately 1.0 in x direction
        mean_grad_x = grad_x.mean().item()
        mean_grad_y = grad_y.mean().item()
        assert abs(mean_grad_x - 1.0) < 0.1, f"Expected grad_x ≈ 1.0, got {mean_grad_x}"
        assert abs(mean_grad_y) < 0.1, f"Expected grad_y ≈ 0.0, got {mean_grad_y}"

    def test_gradient_shape_preservation(self) -> None:
        field = torch.randn(4, 64, 64)
        grad_x, grad_y = compute_spatial_gradients(field)

        assert grad_x.shape == field.shape
        assert grad_y.shape == field.shape


class TestDivergence:
    def test_divergence_free_field(self) -> None:
        # Rotation field: vx = -y, vy = x (div = 0)
        H, W = 32, 32
        y_coords = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W).float().unsqueeze(0).expand(H, W)

        velx = -y_coords.unsqueeze(0).expand(2, -1, -1)
        vely = x_coords.unsqueeze(0).expand(2, -1, -1)

        div = compute_divergence(velx, vely, dx=1.0, dy=1.0, padding_mode="replicate")

        # Should be close to zero (numerical errors at boundaries)
        assert torch.abs(div).mean() < 0.1

    def test_divergence_loss_excludes_emitters(self) -> None:
        # Non-divergence-free field
        velx = torch.randn(2, 32, 32)
        vely = torch.randn(2, 32, 32)

        # All emitter (should return zero loss)
        emitter_mask = torch.ones(2, 32, 32)
        loss = divergence_loss(velx, vely, emitter_mask, padding_mode="replicate")

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_divergence_loss_non_zero_for_divergent_field(self) -> None:
        # Expansion field: vx = x, vy = y (div = 2)
        H, W = 32, 32
        x_coords = torch.arange(W).float().unsqueeze(0).expand(H, W)
        y_coords = torch.arange(H).float().unsqueeze(1).expand(H, W)

        velx = x_coords.unsqueeze(0).expand(2, -1, -1)
        vely = y_coords.unsqueeze(0).expand(2, -1, -1)

        emitter_mask = torch.zeros(2, 32, 32)
        loss = divergence_loss(velx, vely, emitter_mask, padding_mode="replicate")

        assert loss > 0.0


class TestGradientLoss:
    def test_identical_fields_have_zero_loss(self) -> None:
        density_pred = torch.randn(2, 64, 64)
        density_target = density_pred.clone()

        loss = gradient_loss(density_pred, density_target, padding_mode="replicate")

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_different_fields_have_nonzero_loss(self) -> None:
        density_pred = torch.randn(2, 64, 64)
        density_target = torch.randn(2, 64, 64)

        loss = gradient_loss(density_pred, density_target, padding_mode="replicate")

        assert loss > 0.0


class TestPhysicsAwareLoss:
    def test_forward_returns_loss_and_dict(self) -> None:
        loss_fn = PhysicsAwareLoss()

        outputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        inputs = torch.randn(2, 6, 64, 64)

        inputs[:, 4, :, :] = (inputs[:, 4, :, :] > 0).float()  # emitter mask
        inputs[:, 5, :, :] = (inputs[:, 5, :, :] > 0).float()  # collider mask

        loss, loss_dict = loss_fn(outputs, targets, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "mse" in loss_dict
        assert "divergence" in loss_dict
        assert "gradient" in loss_dict

    def test_disable_flags_work(self) -> None:
        loss_fn = PhysicsAwareLoss(
            enable_divergence=False,
            enable_gradient=False,
        )

        outputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        inputs = torch.randn(2, 6, 64, 64)

        inputs[:, 4, :, :] = (inputs[:, 4, :, :] > 0).float()
        inputs[:, 5, :, :] = (inputs[:, 5, :, :] > 0).float()

        loss, loss_dict = loss_fn(outputs, targets, inputs)

        assert "divergence" not in loss_dict
        assert "gradient" not in loss_dict
        assert "mse" in loss_dict

    def test_loss_components_are_finite(self) -> None:
        """All loss components should be finite (not NaN or Inf)."""
        loss_fn = PhysicsAwareLoss()

        outputs = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)
        inputs = torch.randn(2, 6, 64, 64)

        inputs[:, 4, :, :] = (inputs[:, 4, :, :] > 0).float()
        inputs[:, 5, :, :] = (inputs[:, 5, :, :] > 0).float()

        loss, loss_dict = loss_fn(outputs, targets, inputs)

        assert torch.isfinite(loss)
        for key, value in loss_dict.items():
            assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"
            assert not torch.isinf(torch.tensor(value)), f"{key} is Inf"


@pytest.mark.parametrize("padding_mode", ["zeros", "replicate", "reflect"])
def test_gradient_computation_with_padding_modes(padding_mode: str) -> None:
    field = torch.randn(2, 32, 32)
    grad_x, grad_y = compute_spatial_gradients(field, padding_mode=padding_mode)

    assert grad_x.shape == field.shape
    assert grad_y.shape == field.shape
    assert torch.isfinite(grad_x).all()
    assert torch.isfinite(grad_y).all()
