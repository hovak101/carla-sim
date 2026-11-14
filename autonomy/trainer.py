"""Training / validation loop."""
import torch

from .config import MODEL_PATH


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device: str):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def run_epoch(self, loader, train: bool) -> float:
        """One pass over `loader`; returns the mean loss. Updates weights only when `train` is True."""
        self.model.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.model(x), y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total += loss.item() * len(x)
                n += len(x)
        return total / n

    def fit(self, train_loader, val_loader, epochs: int = 20, ckpt_path: str = MODEL_PATH) -> float:
        """Train for `epochs`, saving the best checkpoint (lowest validation loss). Returns the best val loss."""
        best = float("inf")
        for epoch in range(epochs):
            tr = self.run_epoch(train_loader, train=True)
            va = self.run_epoch(val_loader, train=False)
            print(f"epoch {epoch + 1:2d}  train {tr:.5f}  val {va:.5f}")
            if va < best:
                best = va
                self.model.save(ckpt_path)
        return best
