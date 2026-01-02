from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray

from models.small_unet_full import SmallUNetFull, SmallUNetFullConfig

Backend = Literal["pytorch", "onnx"]


class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: str | Path, device: str) -> None:
        pass

    @abstractmethod
    def infer(self, input_data: NDArray[np.float32]) -> NDArray[np.float32]:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass


class PyTorchBackend(InferenceBackend):
    def __init__(self) -> None:
        self._device: str = "cpu"
        self._model: SmallUNetFull | None = None

    def load_model(self, model_path: str | Path, device: str) -> None:
        self._device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if "config" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'config' key at {model_path}")

        config = checkpoint["config"]
        required_keys = ["in_channels", "out_channels", "base_channels", "depth"]
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            raise KeyError(f"Checkpoint config missing: {missing_keys}")

        norm = config.get("norm", "group")
        act = config.get("act", "silu")
        group_norm_groups = config.get("group_norm_groups", 8)
        dropout = config.get("dropout", 0.0)
        upsample = config.get("upsample", "nearest")
        use_residual = config.get("use_residual", True)
        bottleneck_blocks = config.get("bottleneck_blocks", 1)

        self._model = SmallUNetFull(
            cfg=SmallUNetFullConfig(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                base_channels=config["base_channels"],
                depth=config["depth"],
                norm=norm,
                act=act,
                group_norm_groups=group_norm_groups,
                dropout=dropout,
                upsample=upsample,
                use_residual=use_residual,
                bottleneck_blocks=bottleneck_blocks,
            )
        )

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._model.to(self._device)

        print(f"Loaded PyTorch model from {model_path}")
        print(
            f"  Architecture: in_channels={config['in_channels']}, "
            f"out_channels={config['out_channels']}, "
            f"base_channels={config['base_channels']}, depth={config['depth']}"
        )

    def infer(self, input_data: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        # Convert numpy to torch, preserving device placement
        input_tensor = torch.from_numpy(input_data).float().to(self._device)

        with torch.no_grad():
            output_tensor = self._model(input_tensor)

        # Convert back to numpy and move to CPU
        return cast("NDArray[np.float32]", output_tensor.cpu().numpy())

    @property
    def device(self) -> str:
        return self._device


class ONNXBackend(InferenceBackend):
    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._device: str = "cpu"
        self._input_name: str = ""
        self._output_name: str = ""

    def load_model(self, model_path: str | Path, device: str) -> None:
        self._device = device

        # Map device string to ONNX Runtime provider
        available_providers = ort.get_available_providers()
        print(f"ONNX Available Providers: {available_providers}")

        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                print("Warning: CUDA requested but not available, falling back to CPU")

        session = ort.InferenceSession(str(model_path), sess_options=ort.SessionOptions(), providers=providers)
        self._session = session

        # Get input and output names
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

        print(f"Loaded ONNX model from {model_path}")
        print(f"  Using providers: {providers}")
        print(f"  Input: {self._input_name}, Output: {self._output_name}")

    def infer(self, input_data: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        # Ensure input is float32 and contiguous in memory
        input_data_contig = np.ascontiguousarray(input_data, dtype=np.float32)

        output = self._session.run([self._output_name], {self._input_name: input_data_contig})

        return cast("NDArray[np.float32]", output[0])  # ONNX returns a list of outputs

    @property
    def device(self) -> str:
        return self._device
