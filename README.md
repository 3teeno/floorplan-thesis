# Floor-plan vectorisation pipeline

Turns a raster floor-plan image from a property listing into a layered CAD drawing (DXF):
single-line walls, text read from the plan, detected doors and windows, and rooms with their
areas, calibrated from the listing's total living area. Built for a master's thesis on Finnish
Oikotie.fi listings.

## Pipeline

For each image, `floorplan_pipeline.py`:

1. crops the document from its background and isolates the drawing from the marketing layout
2. reads the plan's text (PaddleOCR, with Tesseract as a second pass)
3. traces every stroke as a single straight centreline (`floorplan_trace.trace_single_lines`)
4. detects doors and windows with a Faster R-CNN trained on CubiCasa5K
5. segments rooms from the wall pixels, names them from the room codes on the plan
   (MH, OH, KT, ...), and divides the listed floor area among them
6. writes one DXF (layers Trace, Text, Door, Window, Area, Info, Legend) plus per-image and
   per-room CSV statistics

## Files

| File | Purpose |
| --- | --- |
| `floorplan_pipeline.py` | Main pipeline: runs all steps per image, room segmentation, area calibration, CSV output |
| `floorplan_trace.py` | OCR, text masking, binarisation, single-line tracing, DXF writers |
| `floorplan_loader.py` | Image loading, document cropping, resizing |
| `floorplan_detect_openings.py` | Door/window detection with the trained Faster R-CNN |
| `floorplan_crop_trace.py` | Crop + trace only, with a preview image, for checking the tracing |
| `floorplan_scraper.py` | Scraper for Oikotie.fi for-sale listings (respects robots.txt) |
| `train_fasterrcnn.py`, `cubicasa_datasets.py`, `download_cubicasa5k.py`, `floortrans/` | Detector training on CubiCasa5K (`floortrans/` is the CubiCasa5K reference loader) |
| `evaluation/compare.py` | Room-count and room-area evaluation, before vs after |
| `evaluation/make_review.py` | Per-plan review CSV and input/output images grouped by outcome |
| `scripts/` | Batch runs and the daily scraping job |

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install tesseract            # macOS; second-pass OCR
```

The trained detector `fasterrcnn_doors_windows.pt` is not in the repository (165 MB). Download it
from the repository's Releases page into the project folder, or retrain it:

```bash
.venv/bin/python download_cubicasa5k.py      # 5.5 GB, CC BY-NC 4.0
.venv/bin/python train_fasterrcnn.py         # about 18 h on CPU
```

## Usage

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PYTHONPATH=.
.venv/bin/python floorplan_scraper.py --region both --limit 120 --only-with-floorplan
.venv/bin/python floorplan_pipeline.py <image name in output/floorplan_images/>   # one image
scripts/run_all.sh                                                                  # all images
scripts/run_evaluation.sh && .venv/bin/python evaluation/compare.py output/final_500/images.txt output/final_500
```

Output goes to `output/vectorized_lines_full/` (DXF) and `output/run_stats.csv`,
`output/run_rooms.csv`. Environment variables: `FP_OUT_DIR`, `FP_STATS_CSV`, `FP_ROOMS_CSV`,
`FP_IMAGE_LIST`, `MAX_IMAGES_PER_RUN`, `MAX_IMAGE_PX`; `FP_ROOM_WALLS=contours` and
`FP_TEXT_MASK_MODE=box` select the earlier behaviour for comparison.

## Data

Scraped listings and images are not included: the listings contain addresses and the images
belong to the listing sites. CubiCasa5K is licensed CC BY-NC 4.0 (non-commercial use).
