import torch


def apply_horizontal_flip(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_flipped = torch.flip(x, dims=[2])
    y_flipped = torch.flip(y, dims=[2])

    x_flipped[1] = -x_flipped[1]
    y_flipped[1] = -y_flipped[1]

    return x_flipped, y_flipped


def apply_augmentation(x: torch.Tensor, y: torch.Tensor, flip_prob: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < flip_prob:
        return apply_horizontal_flip(x, y)
    return x, y
