"""Crop the plan and trace it -- nothing else, for checking the tracing on its own.

Per image:
  1. crop    -- crop_to_document (photo/scan background) + isolate_plan (marketing layout around it)
  2. trace   -- every line as a SINGLE line: ink = anything clearly darker than the paper (so light-
                grey lines are kept, unlike the main pipeline's Otsu), thinned to centrelines by
                extract_wall_skeleton (usual text mask, no OCR); then the diagonal stubs thinning
                leaves at thick wall ends and corners are pruned (floorplan_trace.trace_single_lines)

Writes to FP_TRACE_OUT (default output/crop_trace/), per image:
  <name>.dxf          -- layer "Walls": the single-line trace
  <name>_trace.png    -- cropped plan | single lines over it | single lines alone

Usage:
  python floorplan_crop_trace.py IMG1.jpg [IMG2.jpg ...]
  python floorplan_crop_trace.py                            -> every image
"""
import logging
import os
import sys
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from floorplan_loader import crop_to_document, list_floorplan_images
from floorplan_pipeline import isolate_plan
from floorplan_trace import trace_single_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT = Path(os.environ.get("FP_TRACE_OUT", "output/crop_trace"))


def write_dxf(path: Path, lines: list[np.ndarray], height: int) -> None:
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.add("Walls", color=7)
    for line in lines:
        msp.add_lwpolyline([(float(x), height - float(y)) for x, y in line], close=False,
                           dxfattribs={"layer": "Walls"})
    doc.saveas(str(path))


def write_png(path: Path, image: np.ndarray, lines: list[np.ndarray]) -> None:
    """Three panels: the cropped plan, the single lines over it, the single lines alone."""
    pts = [p.astype(np.int32) for p in lines]
    over = cv2.addWeighted(image, 0.35, np.full_like(image, 255), 0.65, 0)
    cv2.polylines(over, pts, False, (255, 0, 0), 2, cv2.LINE_AA)
    alone = np.full_like(image, 255)
    cv2.polylines(alone, pts, False, (0, 0, 0), 2, cv2.LINE_AA)
    gap = np.full((image.shape[0], 12, 3), 200, np.uint8)
    cv2.imwrite(str(path), np.hstack([image, gap, over, gap, alone]))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(sys.argv[1:])
    for path in [p for p in list_floorplan_images() if not wanted or p.name in wanted]:
        try:
            image = isolate_plan(crop_to_document(cv2.imread(str(path))))   # 1. crop
            lines = trace_single_lines(image)                                     # 2. trace
            h = image.shape[0]
            write_dxf(OUT / f"{path.stem}.dxf", lines, h)
            write_png(OUT / f"{path.stem}_trace.png", image, lines)
            log.info(f"{path.name}: cropped to {image.shape[1]}x{h}, {len(lines)} single lines")
        except Exception:
            log.exception(f"{path.name}: FAILED")
