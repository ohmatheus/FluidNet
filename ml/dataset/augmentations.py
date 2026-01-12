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


def apply_rollout_augmentation(
    x_0: torch.Tensor,
    y_seq: torch.Tensor,
    masks: torch.Tensor,
    flip_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < flip_prob:
        x_0 = torch.flip(x_0, dims=[2])  # (6, H, W) -> flip width
        y_seq = torch.flip(y_seq, dims=[3])  # (K, 3, H, W) -> flip width
        masks = torch.flip(masks, dims=[3])  # (K, 2, H, W) -> flip width

        # Negate vx (channel 1 in x_0, channel 1 in y_seq)
        x_0[1] = -x_0[1]
        y_seq[:, 1, :, :] = -y_seq[:, 1, :, :]

    return x_0, y_seq, masks
