import torch
from torch.utils.data import DataLoader

from dataset.npz_sequence import FluidNPZSequenceDataset


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device)}")

        x = torch.randn(2048, 2048, device=device)
        y = x @ x
        print(f"CUDA test succeeded, tensor mean = {y.mean().item():.6f}")
    else:
        print("CUDA not available – check NVIDIA driver and PyTorch installation.")

    ds = FluidNPZSequenceDataset(npz_dir="../vdb-tools/numpy_output/", normalize=True, device="cuda")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for batch in loader:
        x, y = batch  # x: (B, 4, H, W), y: (B, 3, H, W)
        break


if __name__ == "__main__":
    main()
