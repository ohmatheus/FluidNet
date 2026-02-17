import random
from pathlib import Path
from typing import cast

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_splits(
    npz_dir: Path, split_ratios: tuple[float, float, float], seed: int
) -> tuple[list[int], list[int], list[int]]:
    seq_paths = sorted([p for p in npz_dir.iterdir() if p.name.startswith("seq_") and p.name.endswith(".npz")])
    n_seq = len(seq_paths)

    if n_seq < 1:
        raise ValueError(f"Need at least 1 sequence, found {n_seq}")

    indices = list(range(n_seq))
    random.Random(seed).shuffle(indices)

    ratio_sum = sum(split_ratios)
    if ratio_sum == 0:
        normalized_ratios = (1 / 3, 1 / 3, 1 / 3)
    else:
        normalized_ratios = cast("tuple[float, float, float]", tuple(r / ratio_sum for r in split_ratios))

    allocated = [int(r * n_seq) for r in normalized_ratios]

    leftover = n_seq - sum(allocated)
    if leftover > 0:
        fractional_parts = [(r * n_seq) % 1 for r in normalized_ratios]
        sorted_indices = sorted(range(3), key=lambda i: fractional_parts[i], reverse=True)
        for i in range(leftover):
            allocated[sorted_indices[i]] += 1

    n_train = allocated[0]
    n_val = allocated[1]
    n_test = allocated[2]

    assert n_train + n_val + n_test == n_seq

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx, val_idx, test_idx
