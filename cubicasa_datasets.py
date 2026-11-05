import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from floortrans.loaders.house import House
from floorplan_loader import MATRIX_SIZE, compute_scaled_dims, resize_and_pad

log = logging.getLogger(__name__)

WINDOW_ICON_LABEL = 1  # icons_selected["Window"]
DOOR_ICON_LABEL = 2  # icons_selected["Door"]

MAX_LOAD_RETRIES = 5  # a small fraction of CubiCasa5K svgs hit edge cases in the vendored parser
# (e.g. floor plans with 256+ wall segments overflow its uint8 wall-id array); skip and retry
# rather than crash a multi-hour training run over one bad sample.


def _read_folder_list(data_dir: Path, split_file: str) -> list[str]:
    with open(data_dir / split_file) as f:
        return [line.strip() for line in f if line.strip()]


def _load_with_retries(load_fn, index: int, folders: list[str]):
    for _ in range(MAX_LOAD_RETRIES):
        try:
            return load_fn(index)
        except Exception as e:
            log.warning(f"Skipping unparseable sample {folders[index]!r} ({e!r}), retrying with another index")
            index = random.randrange(len(folders))
    raise RuntimeError(f"Failed to load a valid sample after {MAX_LOAD_RETRIES} retries")


class CubiCasaDetectionDataset(Dataset):
    """Door/window bounding boxes from CubiCasa5K, for training Faster R-CNN.

    Labels follow torchvision's detection convention (0 reserved for
    background): 1 == Window, 2 == Door.
    """

    def __init__(self, data_dir: str = "data/cubicasa5k", split: str = "train.txt",
                 size: int = MATRIX_SIZE, min_box_area: int = 16):
        self.data_dir = Path(data_dir)
        self.folders = _read_folder_list(self.data_dir, split)
        self.size = size
        self.min_box_area = min_box_area

    def __len__(self) -> int:
        return len(self.folders)

    def _boxes_for_label(self, icons: np.ndarray, label: int, scale: float,
                          pad_x: int, pad_y: int) -> list[list[float]]:
        mask = (icons == label).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, num_labels):  # label 0 is background
            if stats[i, cv2.CC_STAT_AREA] < self.min_box_area:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append([
                x * scale + pad_x, y * scale + pad_y,
                (x + bw) * scale + pad_x, (y + bh) * scale + pad_y,
            ])
        return boxes

    def __getitem__(self, index: int):
        return _load_with_retries(self._load, index, self.folders)

    def _load(self, index: int):
        folder = self.data_dir / self.folders[index].strip("/")
        image = cv2.imread(str(folder / "F1_scaled.png"))
        h, w = image.shape[:2]

        house = House(str(folder / "model.svg"), h, w)

        new_h, new_w, scale = compute_scaled_dims(h, w, self.size)
        pad_x = (self.size - new_w) // 2 if new_w < self.size else 0
        pad_y = (self.size - new_h) // 2 if new_h < self.size else 0

        boxes, labels = [], []
        for label in (WINDOW_ICON_LABEL, DOOR_ICON_LABEL):
            for box in self._boxes_for_label(house.icons, label, scale, pad_x, pad_y):
                boxes.append(box)
                labels.append(label)

        image = resize_and_pad(image, self.size)
        image_t = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return image_t, target
