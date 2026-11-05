import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MATRIX_SIZE = 1024
SEAM_BLUR_WIDTH = 10  # px on each side of the content/padding seam, per paper Algorithm 1

MIN_DOCUMENT_AREA_FRACTION = 0.10  # ignore bright regions smaller than this — not a real document, just noise


def list_floorplan_images(images_dir: str = "output/floorplan_images") -> list[Path]:
    """Return every floor plan image path in the given directory."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} does not exist")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_floorplan_images(images_dir: str = "output/floorplan_images"):
    """Yield (path, image) pairs for every floor plan image.

    Images are loaded as BGR numpy arrays via OpenCV — the format expected by
    ultralytics/YOLO and most cv2-based vectorization steps (contour/line
    detection, ezdxf export, etc.).
    """
    for path in list_floorplan_images(images_dir):
        image = cv2.imread(str(path))
        if image is None:
            log.warning(f"Could not read {path}, skipping")
            continue
        yield path, image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 arbitrary corner points into (top-left, top-right, bottom-right,
    bottom-left) order, required for a correct perspective transform."""
    rect = np.zeros((4, 2), dtype=np.float32)
    total = pts.sum(axis=1)
    rect[0] = pts[np.argmin(total)]  # smallest x+y -> top-left
    rect[2] = pts[np.argmax(total)]  # largest x+y -> bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # smallest x-y -> top-right
    rect[3] = pts[np.argmax(diff)]  # largest x-y -> bottom-left
    return rect


def crop_to_document(image: np.ndarray, min_area_fraction: float = MIN_DOCUMENT_AREA_FRACTION) -> np.ndarray:
    """Crop (and de-skew) a photo down to just the floor plan, discarding
    whatever background surrounds it — a desk, a laptop bezel, a hand.

    Floor plans (screens, printouts, scans) are reliably a bright region
    against a darker surrounding scene, so this finds the largest bright
    contour, and if it looks like a photographed rectangle (approximates to
    4 corners), perspective-warps just that region flat — the same technique
    document-scanner apps use. Falls back to a plain axis-aligned crop of the
    bounding box if the region isn't a clean quadrilateral (e.g. a curled
    photo, glare, an irregular crop), and leaves the image untouched entirely
    if nothing large enough looks like a document — an already-clean, already
    -cropped floor plan image should pass through this as a no-op.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) / (width * height) < min_area_fraction:
        return image

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(largest)
        return image[y:y + h, x:x + w]

    tl, tr, br, bl = _order_corners(approx.reshape(4, 2).astype(np.float32))
    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                   dtype=np.float32)

    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(image, transform, (max_width, max_height))


def _blur_seam(canvas: np.ndarray, position: int, axis: int, width: int) -> None:
    """Blur a thin strip centered on `position` in-place, to hide the content/padding seam."""
    lo, hi = max(0, position - width), min(canvas.shape[axis], position + width)
    ksize = (hi - lo) | 1  # GaussianBlur requires an odd kernel size
    if ksize < 3:
        return
    if axis == 0:
        canvas[lo:hi] = cv2.GaussianBlur(canvas[lo:hi], (ksize, ksize), sigmaX=0)
    else:
        canvas[:, lo:hi] = cv2.GaussianBlur(canvas[:, lo:hi], (ksize, ksize), sigmaX=0)


def compute_scaled_dims(h: int, w: int, size: int = MATRIX_SIZE) -> tuple[int, int, float]:
    """Geometry shared by resize_and_pad/resize_mask: longer side -> size, aspect kept."""
    scale = size / max(h, w)
    return round(h * scale), round(w * scale), scale


def resize_and_pad(image: np.ndarray, size: int = MATRIX_SIZE,
                    seam_blur_width: int = SEAM_BLUR_WIDTH) -> np.ndarray:
    """Scale the longer side to `size` (aspect ratio preserved), pad the shorter side
    with the average edge color, and blur only the seam between content and padding.

    This is Algorithm 1 from Surikov et al. 2020 — it avoids the hard border
    artifacts a plain resize-to-square + full-image blur would feed into a
    segmentation model.
    """
    h, w = image.shape[:2]
    new_h, new_w, scale = compute_scaled_dims(h, w, size)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)

    if new_h == size and new_w == size:
        return scaled

    canvas = np.empty((size, size, image.shape[2]), dtype=image.dtype)
    if new_h < size:
        pad = (size - new_h) // 2
        canvas[:pad] = scaled[0].mean(axis=0).astype(image.dtype)
        canvas[pad:pad + new_h] = scaled
        canvas[pad + new_h:] = scaled[-1].mean(axis=0).astype(image.dtype)
        _blur_seam(canvas, pad, axis=0, width=seam_blur_width)
        _blur_seam(canvas, pad + new_h, axis=0, width=seam_blur_width)
    else:
        pad = (size - new_w) // 2
        canvas[:, :pad] = scaled[:, 0].mean(axis=0).astype(image.dtype)
        canvas[:, pad:pad + new_w] = scaled
        canvas[:, pad + new_w:] = scaled[:, -1].mean(axis=0).astype(image.dtype)
        _blur_seam(canvas, pad, axis=1, width=seam_blur_width)
        _blur_seam(canvas, pad + new_w, axis=1, width=seam_blur_width)
    return canvas


