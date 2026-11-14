"""Image preprocessing shared by recording, training and inference (single source of truth)."""
import numpy as np
import torch

from .config import CROP_TOP_FRACTION


def crop_to_rgb(image) -> np.ndarray:
    """carla.Image (BGRA) -> uint8 HxWx3 RGB array with the top CROP_TOP_FRACTION (the sky) removed."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return bgra[int(image.height * CROP_TOP_FRACTION):, :, 2::-1].copy()  # drop sky rows, BGRA -> RGB


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB array -> float CxHxW tensor in [0, 1]."""
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
