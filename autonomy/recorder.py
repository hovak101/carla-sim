"""Writes camera frames + the steering applied for the same sim frame to disk."""
import csv
import os
import shutil

from PIL import Image

from .config import DATA_DIR
from .preprocessing import crop_to_rgb

STALE_FRAMES = 20  # drop bookkeeping for frames that will never find a match


class DataRecorder:
    """Output: <data_dir>/images/<id>.png plus <data_dir>/steering.csv (id, image, frame, steering).
    Any previous recording in `data_dir` is wiped on construction."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        csv_path = os.path.join(data_dir, "steering.csv")

        shutil.rmtree(self.image_dir, ignore_errors=True)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        os.makedirs(self.image_dir, exist_ok=True)

        self.record_id = 0
        self._steer_by_frame = {}  # sim frame -> steer applied for that frame
        self._pending_images = {}  # sim frame -> image waiting for its steer value
        self._csv_file = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["id", "image", "frame", "steering"])

    def update(self, frame: int, steer: float, images: list) -> None:
        """Call once per tick while recording: `steer` was applied on sim frame `frame`, `images` are the
        camera frames received since the last call. Each image is saved once its steer value is known."""
        for image in images:
            self._pending_images[image.frame] = image
        self._steer_by_frame[frame] = steer

        for f in sorted(self._pending_images):
            if f in self._steer_by_frame:
                image = self._pending_images.pop(f)
                name = f"{self.record_id:06d}.png"
                Image.fromarray(crop_to_rgb(image)).save(os.path.join(self.image_dir, name))
                self._csv.writerow([self.record_id, name, f, self._steer_by_frame.pop(f)])
                self.record_id += 1
        self._csv_file.flush()

        for f in [f for f in self._steer_by_frame if f < frame - STALE_FRAMES]:
            del self._steer_by_frame[f]
        for f in [f for f in self._pending_images if f < frame - STALE_FRAMES]:
            del self._pending_images[f]

    def close(self) -> None:
        self._csv_file.close()
        print(f"Saved {self.record_id} frames to {self.data_dir}")
