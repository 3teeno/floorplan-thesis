import logging
import sys

import cv2
import ezdxf
import numpy as np
import torch

from floorplan_loader import compute_scaled_dims, crop_to_document, resize_and_pad
from floorplan_trace import _select_images, export_contours_dxf, trace_contours
from train_fasterrcnn import DEVICE, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_INPUT_SIZE = 512  # must match train_fasterrcnn.train()'s `size`
SCORE_THRESHOLD = 0.5
CLASS_NAMES = {1: "Window", 2: "Door"}  # torchvision detection convention: 0 is background
LAYER_COLORS = {"Door": 1, "Window": 5}  # red / blue in the DXF


def load_opening_model(checkpoint_path: str = "fasterrcnn_doors_windows.pt") -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> RGB float32 3xHxW in [0, 1] -- matches
    CubiCasaDetectionDataset's own normalization exactly (plain /255, no
    further scaling; torchvision's FasterRCNN applies ImageNet
    normalization internally, so the model was trained on this range).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def detect_openings(image: np.ndarray, model: torch.nn.Module, size: int = MODEL_INPUT_SIZE,
                     score_threshold: float = SCORE_THRESHOLD) -> list[dict]:
    """Door/window bounding boxes in `image`'s own pixel coordinates, via the
    CubiCasa5K-trained Faster R-CNN (train_fasterrcnn.py) -- a real trained
    detector, not a heuristic, so it catches door/window icons trace_contours
    has no way to recognize as such (it only sees generic wall/furniture
    shapes, with no semantic classification at all).

    Boxes come back out of the model in the resized+padded (size x size)
    frame the model was trained on; compute_scaled_dims/resize_and_pad's own
    scale+pad geometry is inverted here to map them back to `image`'s
    original pixel coordinates, the same geometry CubiCasaDetectionDataset
    applies forwards when building training labels.
    """
    h, w = image.shape[:2]
    new_h, new_w, scale = compute_scaled_dims(h, w, size)
    pad_x = (size - new_w) // 2 if new_w < size else 0
    pad_y = (size - new_h) // 2 if new_h < size else 0

    resized = resize_and_pad(image, size=size)
    prediction = model([_to_tensor(resized).to(DEVICE)])[0]

    detections = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "class": CLASS_NAMES[int(label)],
            "score": float(score),
            "box": ((x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale),
        })
    return detections


def export_openings_dxf(dxf_path: str, detections: list[dict], height: int) -> None:
    """Add each detected door/window as a closed rectangle on its own layer to
    an existing DXF (e.g. one export_contours_dxf already wrote the traced
    wall geometry to) -- same "open, add, resave" pattern as
    add_text_labels_to_dxf in floorplan_trace.py.

    Coordinates are flipped (`height - y`) for the same reason as every other
    export here: image space has y growing downward, DXF/CAD space has y
    growing upward.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    existing_layers = {layer.dxf.name for layer in doc.layers}

    for det in detections:
        layer = det["class"]
        if layer not in existing_layers:
            doc.layers.add(layer, color=LAYER_COLORS.get(layer, 7))
            existing_layers.add(layer)
        x1, y1, x2, y2 = det["box"]
        points = [(x1, height - y1), (x2, height - y1), (x2, height - y2), (x1, height - y2)]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    doc.saveas(dxf_path)
