import torch

from training.physics_loss import compute_divergence


def compute_per_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse_density = torch.mean((pred[:, 0] - target[:, 0]) ** 2).item()
    mse_velx = torch.mean((pred[:, 1] - target[:, 1]) ** 2).item()
    mse_vely = torch.mean((pred[:, 2] - target[:, 2]) ** 2).item()

    return {"mse_density": mse_density, "mse_velx": mse_velx, "mse_vely": mse_vely}


def compute_divergence_norm(velx: torch.Tensor, vely: torch.Tensor, dx: float = 1.0, dy: float = 1.0, padding_mode: str = "zeros") -> float:
    """
    L2 norm of velocity divergence.
    Measures incompressibility violation (should be ~0 in free fluid).
    Target: < 0.1 for stable rollouts.
    """
    div = compute_divergence(velx, vely, dx, dy, padding_mode)
    norm = torch.sqrt(torch.mean(div**2)).item()
    return norm


def compute_kinetic_energy(velx: torch.Tensor, vely: torch.Tensor) -> float:
    """
    Total motion energy KE = 0.5 × Σ(vx² + vy²).
    Detects velocity explosion (should NOT grow exponentially).
    Target: Stable or slowly increasing trend.
    """
    ke = 0.5 * torch.sum(velx**2 + vely**2).item()
    return ke


def compute_collider_violation(density: torch.Tensor, collider_mask: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Average smoke density inside solid obstacles.
    Smoke should not penetrate colliders.
    Target: ~= 0 (< 0.01 acceptable).
    """
    num_collider_cells = collider_mask.sum()

    if num_collider_cells < eps:
        return 0.0

    density_in_collider = (density * collider_mask).sum()
    avg_violation = (density_in_collider / (num_collider_cells + eps)).item()

    return avg_violation


def compute_emitter_density_accuracy(
    density: torch.Tensor, emitter_mask: torch.Tensor, expected_injection: float = 0.8, eps: float = 1e-8
) -> float:
    """
    Error between predicted density at emitter vs expected injection value.
    Validates model maintains correct emission rate in rollouts.
    Target: < 0.1 error.
    """
    num_emitter_cells = emitter_mask.sum()

    if num_emitter_cells < eps:
        return 0.0

    avg_emitter_density = (density * emitter_mask).sum() / (num_emitter_cells + eps)
    error = torch.abs(avg_emitter_density - expected_injection).item()

    return error


class MetricsTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def reset(self) -> None:
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict: dict[str, float]) -> None:
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1

    def compute_averages(self) -> dict[str, float]:
        return {k: self.metrics[k] / self.counts[k] for k in self.metrics}
