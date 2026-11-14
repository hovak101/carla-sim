"""The steering network (single definition used by training and inference)."""
import torch
from torch import nn

from .config import INPUT_HEIGHT, INPUT_WIDTH
from .preprocessing import to_tensor


def get_device() -> str:
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class SteeringModel(nn.Module):
    """Camera image (3 x INPUT_HEIGHT x INPUT_WIDTH) -> steering in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

    @classmethod
    def create(cls, device: str) -> "SteeringModel":
        """Build the model on `device`. The dummy forward pass materializes the LazyLinear layer, so
        parameters exist before an optimizer is created or weights are loaded."""
        model = cls().to(device)
        model(torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device))
        return model

    @classmethod
    def load(cls, path: str, device: str) -> "SteeringModel":
        model = cls.create(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def predict(self, rgb) -> float:
        """Steering for one uint8 HxWx3 RGB image (as returned by preprocessing.crop_to_rgb)."""
        device = next(self.parameters()).device
        return self(to_tensor(rgb).unsqueeze(0).to(device)).item()
