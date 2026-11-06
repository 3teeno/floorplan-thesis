"""Full pipeline: tracing + text OCR + door/window detection + room areas, in ONE run per
image, writing a single DXF with every layer (walls/furniture, OCR'd text, doors, windows,
rooms) plus a color-key legend at the bottom of the drawing.

Steps, over the same cropped image:
  1. read_text_regions          -- OCR the plan's text (floorplan_trace.py)
  2. trace_contours              -- classical contour tracing, text masked out
                                     (only feeds segment_rooms with FP_ROOM_WALLS=contours;
                                     the default floods between the image's dark pixels)
  3. trace_single_lines          -- the Trace layer's actual drawn geometry: every stroke,
                                     thick walls included, thinned to ONE centreline; ink is
                                     anything clearly darker than the paper, so light-grey
                                     lines (door swings, fixtures, windows) are kept too
  4. add_text_labels_to_dxf      -- OCR'd labels as TEXT entities
  5. detect_openings             -- Faster R-CNN door/window detector (CubiCasa5K)
  6. export_openings_dxf         -- doors/windows as colored rectangles
  7. lookup_listing_metadata /
     strip_known_address /
     add_metadata_to_dxf         -- scraped ground-truth rooms/address/build
                                     year (output/listings.csv), NOT OCR,
                                     written as a title-block header above the
                                     drawing -- Oikotie images only
  8. segment_rooms /
     add_room_areas_to_dxf       -- the listing's scraped size_m2 distributed
                                     across the traced room polygons (flood-
                                     fill of the open space, doors blocked so
                                     it can't leak between rooms) instead of
                                     printed as one flat total -- uses
                                     trace_contours' own polygons (step 2), NOT
                                     the skeleton lines drawn in step 3; those
                                     are a cosmetic-only alternate view of the
                                     same wall ink, not a replacement for it
  9. add_legend                  -- color-key legend for whichever layers this
                                     particular plan actually used

Usage:
  python floorplan_pipeline.py               -> every image in output/floorplan_images/
  python floorplan_pipeline.py IMG_8666.JPG  -> just that one image

Every run also APPENDS (never overwrites -- resuming a run or rerunning a single image doesn't
lose earlier rows) to two CSVs for downstream reporting/analysis, not just the per-image log line:
  output/run_stats.csv  -- one row per image: ground-truth listing fields (size_m2/rooms/
                            room_configuration/build_year from listings.csv, not OCR), success/
                            failure + error message, detection counts (wall lines/text labels/
                            doors/windows), room-segmentation yield (total/shown/named/gap-closed/
                            recovered), the calibration scale actually used, and processing time.
  output/run_rooms.csv  -- one row per detected room across every image: its text (or "Other"/
                            blank), area in both px^2 and calibrated m^2, whether it was shown
                            (>= AREA_MIN_DISPLAY_M2), recovered (erosion fallback), or gap_closed
                            (Pass 3) -- for room-level analysis the per-image CSV only summarizes.
"""
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pandas as pd

from floorplan_detect_openings import detect_openings, export_openings_dxf, load_opening_model
from floorplan_loader import crop_to_document
from floorplan_trace import (KNOWN_LABELS, _FLOORPLAN_LABEL_RE, _select_images, add_text_labels_to_dxf,
                              _binarize_ink, _geometry_text_mask, detect_text_mask, export_contours_dxf,
                              read_text_regions, trace_contours, trace_single_lines)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("ezdxf").setLevel(logging.WARNING)  # its own INFO chatter (dictionary/header noise) drowns ours
log = logging.getLogger(__name__)

OUT_DIR = Path(os.environ.get("FP_OUT_DIR", "output/vectorized_lines_full"))

# One row per image / one row per detected room -- written incrementally (flushed after every
# image, not buffered until the run ends) so a multi-hour unattended batch leaves usable partial
# data if it's ever interrupted, and so progress is inspectable while it's still running. Meant
# for exactly this: reporting pipeline results (success rate, room-detection yield, calibration
# accuracy) as thesis-ready data, not just a human-readable log line.
# FP_OUT_DIR/FP_STATS_CSV/FP_ROOMS_CSV let a one-off run (e.g. a dedicated review batch) write to
# its own isolated folder/CSVs instead of the main backlog's -- same resume/dedup logic, just
# scoped to that run's own tracking files, so it can't skip-as-already-done against the main run's
# history or vice versa.
STATS_CSV = Path(os.environ.get("FP_STATS_CSV", "output/run_stats.csv"))
ROOMS_CSV = Path(os.environ.get("FP_ROOMS_CSV", "output/run_rooms.csv"))
STATS_FIELDS = [
    "image_name", "listing_id", "has_metadata", "listed_size_m2", "listed_rooms",
    "listed_room_configuration", "listed_build_year", "status", "error_message",
    "width_px", "height_px", "num_wall_lines", "num_text_labels", "num_doors", "num_windows",
    "num_rooms_total", "num_rooms_shown", "num_rooms_named", "num_rooms_gap_closed",
    "num_rooms_recovered", "total_area_px2", "calibration_scale_m2_per_px2", "processing_time_sec",
]
ROOM_FIELDS = ["image_name", "listing_id", "room_text", "area_m2", "area_px2", "shown", "recovered", "gap_closed"]

TEXT_LAYER_COLOR = 3  # green -- OCR'd labels, visually distinct from the plain wall/furniture trace
LEGEND_LAYER = "Legend"

INFO_LAYER = "Info"
INFO_LAYER_COLOR = 6  # magenta -- distinct from every other layer's color

# (layer name, description, color) -- must match how each layer is actually created
# elsewhere: Trace gets no explicit color from export_contours_dxf (ezdxf default = 7,
# "ByLayer" white/black); Door/Window colors come from floorplan_detect_openings.LAYER_COLORS;
# Text is pre-created with TEXT_LAYER_COLOR below before add_text_labels_to_dxf runs.
LEGEND_ENTRIES = [
    ("Trace", "Traced walls and furniture", 7),
    ("Text", "Text read from the plan (room labels, dimensions)", TEXT_LAYER_COLOR),
    ("Door", "Detected door", 1),
    ("Window", "Detected window", 5),
    ("Info", "Listing details (from the listing, not read from the image)", INFO_LAYER_COLOR),
    ("Area", "Room area (calibrated from listing floor area), one colour per room", 2),
]

LISTINGS_CSV = Path("output/listings.csv")
_OIKOTIE_NAME_RE = re.compile(r"^oikotie_(\d+)_floorplan")
_listings_df: pd.DataFrame | None = None


def _get_listings() -> pd.DataFrame:
    """Lazily load + cache the scraper's ground-truth listing table
    (output/listings.csv), keyed by listing_id. This is the authoritative
    size_m2/rooms/address -- scraped straight from the Oikotie listing page --
    as opposed to a size figure OCR'd off the plan image itself, which may be
    missing, illegible, mirrored, or (as with the C32/C37/C42 building in
    oikotie_17380049) describe a *different* unit on the same sheet than the
    one this listing_id actually refers to.
    """
    global _listings_df
    if _listings_df is None:
        _listings_df = (pd.read_csv(LISTINGS_CSV, dtype={"listing_id": str}).set_index("listing_id")
                         if LISTINGS_CSV.exists() else pd.DataFrame())
    return _listings_df


def lookup_listing_metadata(image_name: str) -> dict | None:
    """Match `image_name` (e.g. "oikotie_24662660_floorplan_0.jpg") back to its
    scraped listing row. Returns None for anything that isn't a scraped
    Oikotie image, or whose listing_id isn't in listings.csv."""
    m = _OIKOTIE_NAME_RE.match(image_name)
    if not m:
        return None
    listings = _get_listings()
    if listings.empty or m.group(1) not in listings.index:
        return None
    row = listings.loc[m.group(1)]
    return {k: (None if pd.isna(row.get(k)) else row[k])
            for k in ("size_m2", "rooms", "room_configuration", "address", "build_year")}


def strip_known_address(labels: list[dict], address: str | None) -> list[dict]:
    """Drop any OCR'd label that's substantially a fragment of THIS listing's
    own scraped street address. A street name (unlike a HUOM note or a firm
    name) has no lexical marker that would let _is_marketing_caption catch it
    generically -- but we already know this exact listing's real address, so
    check against that directly instead."""
    if not address:
        return labels
    addr_core = re.sub(r"[^A-ZÄÖÅ0-9]", "", str(address).upper())
    if not addr_core:
        return labels
    kept = []
    for lbl in labels:
        core = re.sub(r"[^A-ZÄÖÅ0-9]", "", lbl["text"].upper())
        if len(core) >= 4 and core in addr_core:
            continue
        kept.append(lbl)
    return kept


def _fmt_num(v) -> str:
    """96.5 -> "96.5", 4.0 -> "4" -- pandas reads these numeric CSV columns as
    float even when every real value is a whole number."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def add_metadata_to_dxf(dxf_path: str, meta: dict, height: int) -> None:
    """Write the listing's scraped rooms/address/build-year as a title-block-
    style header ABOVE the drawing (image y < 0, i.e. DXF y > height after the
    standard flip) -- distinct in both position and color from anything the
    OCR pass read off the plan itself, so the two are never mistaken for one
    another. Deliberately does NOT include size_m2 as a flat total figure --
    that's instead distributed across the traced room polygons themselves (see
    segment_rooms / add_room_areas_to_dxf), which is the whole point of doing
    this from geometry rather than just printing the listing's one number.
    Silently does nothing if `meta` carries no other usable field."""
    lines = []
    room_bits = [b for b in (meta.get("rooms"), meta.get("room_configuration")) if b is not None]
    if room_bits:
        lines.append(" / ".join(_fmt_num(b) if isinstance(b, float) else str(b) for b in room_bits))
    if meta.get("address") is not None:
        lines.append(str(meta["address"]))
    if meta.get("build_year") is not None:
        lines.append(f"built {_fmt_num(meta['build_year'])}")
    if not lines:
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    if INFO_LAYER not in {lyr.dxf.name for lyr in doc.layers}:
        doc.layers.add(INFO_LAYER, color=INFO_LAYER_COLOR)

    text_h = max(height * 0.018, 16.0)
    row_h = text_h * 1.6
    top = height + row_h * (len(lines) + 0.5)  # topmost line's y; each next line sits row_h below it

    for i, line in enumerate(lines):
        t = msp.add_text(line, dxfattribs={"layer": INFO_LAYER, "color": INFO_LAYER_COLOR, "height": text_h})
        t.set_placement((0, top - row_h * i))
    doc.saveas(dxf_path)


AREA_LAYER = "Area"
AREA_LAYER_COLOR = 2  # yellow -- distinct from Info's magenta and Text's green
ROOM_MIN_AREA_FRAC = 0.006   # discard an enclosed-space component smaller than this fraction of the image
                             # (anti-aliasing pinholes, a slice cut off between two close-together fixtures)
ROOM_NESTED_MAX_FRAC = 0.5    # a thin-outlined "room" under this share of the room around it is furniture
NESTED_MIN_OPEN_SHARE = 0.5   # ...when at least this share of the band just outside it is open floor
NESTED_HOST_SHARE = 0.8       # ...belonging (at least this share) to one single surrounding room
ROOM_MIN_WIDTH_FRAC = 0.015  # a candidate room narrower on average than this fraction of max(w,h) is a sliver
ROOM_LABEL_MAX_DIST_FRAC = 0.20  # match a room to an OCR'd label only within this fraction of max(w,h)
ROOM_OUTLINE_EPSILON_PX = 3.0    # RDP simplification tolerance for each room's drawn boundary
ROOM_RECOVERY_MIN_SOLIDITY = 0.70  # a recovered leak-fragment below this (area / its own convex-hull
                                   # area) is a thin bent margin remnant, not a real room's blocky core
AREA_MIN_DISPLAY_M2 = 2.0  # don't draw an outline/label for a room calibrated smaller than this --
                           # still counted in the calibration sum (it's real floor area), just not
                           # drawn (a closet-sized room clutters the plan more than it informs)

# One color per room, cycled if there are more rooms than colors -- ACI (AutoCAD Color Index)
# codes chosen to stay visually distinct from every other layer's fixed color (Trace=7, Text=3,
# Door=1, Window=5, Info=6) and from each other (spread across the ACI hue wheel, avoiding the
# near-white/near-black ends). Set per-ENTITY, not just on the shared "Area" layer, so each
# room's outline+label pair stands out from its neighbors, not just from other layers.
ROOM_COLOR_PALETTE = [30, 50, 70, 110, 130, 150, 190, 200, 230, 8]


ROOM_WALL_STROKE_FRAC = 0.002  # rasterised wall thickness, as a fraction of the image's own larger dimension

# A dimension/scale annotation ("ar=485mm", "2400mm") reads as alphanumeric as a real room code,
# but should never BE allowed to claim one -- a real room code in this dataset's own convention
# never contains "=" and never ends with a bare length-unit suffix straight after a number (a
# trailing digit it DOES sometimes carry, e.g. "AH1"/"AH2" for two similar rooms, names a
# duplicate, not a measurement). Checked before any label is allowed into room-matching at all --
# without this, a misread scale annotation can win a match a real room's own label should have
# gotten (confirmed on oikotie_24716483: "ar=485mm" claimed an 11.8 m^2 "room" carved from the
# exact same merged-exterior blob the real open-plan OH needed below, via the ordinary erosion
# recovery above -- blocking Pass 3's own later, correct match for OH until this was filtered out).
ROOM_DIMENSION_LABEL_RE = re.compile(r"=|\d\s*(MM|CM|M2|M²)\b", re.IGNORECASE)


def _is_non_room_label(text: str) -> bool:
    """True for a dimension/scale annotation or a floor marker ("1.KRS") -- never a real room
    code, so never eligible to claim a room's area, but still fine to OCR/draw as its own Text-
    layer annotation elsewhere (this doesn't touch floorplan_trace.py's own keep/reject decision).
    Reuses floorplan_trace's own _FLOORPLAN_LABEL_RE for the floor-marker half, NOT its
    _is_floorplan_label wrapper -- that wrapper also matches anything in KNOWN_LABELS, which is
    exactly the set of real room codes this check must NOT exclude (confirmed necessary: on
    oikotie_24721956, "1.KRS" -- a floor marker, not a room -- claimed a 102k px^2 "room" via Pass
    3 before this was added)."""
    if ROOM_DIMENSION_LABEL_RE.search(text) or _FLOORPLAN_LABEL_RE.search(text):
        return True
    t = text.strip().upper()
    if t in KNOWN_LABELS:
        return False
    if "+" in t:  # a listing configuration copied onto the plan ("4h+kt+s"), not one room
        return True
    # A room code may carry ONE trailing digit for a duplicate ("MH2", "AH1"); any other digit
    # makes it a dimension, door/window code or area ("09V", "15.5", "16x4", "F26B/V", "060").
    return any(c.isdigit() for c in re.sub(r"(?<=[A-ZÄÖÅ])\d$", "", t))


LIVING_LABEL_RE = re.compile(r"^(MH|OH|OH/K|H|AH|HO|TH)\d?$|^(MAKUU|OLO|TAKKA|TYÖ)?HUONE\d?$")
OUTDOOR_LABEL_RE = re.compile(r"PARV|TERASSI|LASITETTU|PIHA|BALKONG")


def _label_rank(text: str) -> int:
    """Which label names a room when several fall inside it: living-room words first, then other
    known room codes, then anything else."""
    t = text.strip().upper()
    return 0 if LIVING_LABEL_RE.match(t) else 1 if t in KNOWN_LABELS else 2


def counts_in_living_area(room: dict) -> bool:
    """A balcony or terrace is not part of a listing's size_m2, so it must not take a share of it."""
    return not (room.get("text") and OUTDOOR_LABEL_RE.search(room["text"].upper()))


def _is_subsequence(short: str, long: str) -> bool:
    """True if every character of `short`, in order, appears somewhere in `long` (not necessarily
    contiguous) -- the signature of an OCR INSERTION error (a short real read like "TK" coming
    out as "TEKN" by inserting extra letters, not reading something else entirely)."""
    it = iter(long)
    return all(c in it for c in short)


OCR_DUPLICATE_PROXIMITY_FRAC = 0.08  # two DIFFERENT KNOWN_LABELS reads this close together, where one's
                                     # text is an insertion away from the other's, are treated as the
                                     # same misread room code printed twice, not two real rooms


def _is_likely_ocr_duplicate(lab: dict, others: list[dict], w: int, h: int) -> bool:
    """True if some OTHER label (any match status -- this must catch a bogus orphan sitting near an
    already-matched real label, not just two orphans) has different text that's a subsequence
    match with this one's AND sits within OCR_DUPLICATE_PROXIMITY_FRAC.

    Checked against plain proximity alone first and rejected: scanning this project's own 15-image
    sample, legitimate DIFFERENT real rooms sit just as close together as the one confirmed bogus
    pair this was built for (oikotie_24721956's "TEKN" misread 112px/0.058 from the real "TK" --
    well inside the range of genuine nearby pairs measured elsewhere, e.g. "KHH"/"OH" at 0.057,
    "KPH"/"S" at 0.051, "OH"/"K" at 0.057) -- proximity alone would veto as many good matches as
    bad ones. The subsequence requirement is what actually discriminates: none of those genuine
    nearby pairs share one, but "TK" is literally spelled out, in order, inside "TEKN". When both
    the suspect orphan and its apparent source are themselves still orphans (as in the confirmed
    case), both get vetoed, since there's no independent way to tell which, if either, is real --
    left unresolved rather than guessed at, consistent with this function's neighbors throughout
    segment_rooms.
    """
    max_dist = OCR_DUPLICATE_PROXIMITY_FRAC * max(w, h)
    for other in others:
        if other["text"] == lab["text"]:
            continue
        if len(min(lab["text"], other["text"], key=len)) < 2:
            continue  # a 1-char code trivially "matches" almost anything -- too weak a signal alone
        if not (_is_subsequence(lab["text"], other["text"]) or _is_subsequence(other["text"], lab["text"])):
            continue
        if np.hypot(lab["x"] - other["x"], lab["y"] - other["y"]) <= max_dist:
            return True
    return False


ORPHAN_CLOSE_K_START = 4         # smallest candidate bridge half-width to try, in px
ORPHAN_CLOSE_K_MAX_FRAC = 0.20   # give up past this fraction of max(w, h) -- a gap this wide isn't a
                                 # tracing gap or a missing door, it's genuinely open to the exterior
ORPHAN_CLOSE_K_MULT = 1.3        # grow the candidate kernel by this factor each failed attempt
ORPHAN_CLOSE_WIN_FACTOR = 4      # the closing is applied only within +/- this many multiples of k of
                                 # the label (padded for correct kernel context at the window's own
                                 # edge), never the whole image -- what keeps one label's own fix from
                                 # reaching across the plan and bridging some OTHER, unrelated gap
ORPHAN_CLOSE_WIN_MIN_PX = 40     # floor on that window radius, for a very small starting k
ORPHAN_CLOSE_OVERLAP_FLOOR_PX = 30   # tolerate at most this much incidental overlap with area
ORPHAN_CLOSE_OVERLAP_FRAC = 0.01     # already claimed elsewhere (contour/approx noise at the seam),
                                     # or 1% of the new area, whichever is larger -- anything beyond
                                     # that means the "gap" is real wall between two rooms that must
                                     # stay separate, not a tracing gap or a missing door


def _flood_component(patched: np.ndarray, lx: int, ly: int, w: int, h: int):
    """Area, boolean mask, and border-touching flag for the connected OPEN (value-0) region of
    `patched` reachable from (lx, ly) -- via cv2.floodFill instead of a full-image
    connectedComponentsWithStats pass, so cost scales with the filled region's own size, not the
    whole image. Matters here specifically: Pass 3 retries this up to ~15 times per orphan label,
    and connectedComponentsWithStats over a full ~1900x1080 source photo repeated that often is
    what pushed a real test image (oikotie_24721956, 48 labels) well past a two-minute timeout --
    flood-filling just the region actually being asked about is the fix, not a smaller k_max."""
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(patched, mask, (lx, ly), 0, loDiff=0, upDiff=0, flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    sub = mask[1:-1, 1:-1] > 0
    area = int(sub.sum())
    touches = bool(sub[0, :].any() or sub[-1, :].any() or sub[:, 0].any() or sub[:, -1].any())
    return area, sub, touches


DOOR_GAP_MIN_RUN_PX = 4       # a candidate wall-ink run shorter than this, along a floating door's own
                              # implied wall line, is noise (a stray OCR/furniture speck) -- never a
                              # real wall, never a valid bridge endpoint
DOOR_GAP_MAX_SEARCH_PX = 80   # give up past this distance from the door -- the nearest substantial
                              # ink this far away isn't this door's own wall gap
DOOR_GAP_AREA_TOL_FRAC = 0.05  # reject a candidate bridge if it moves the TOTAL counted-room area
                               # (every room, not just the one component the bridge touches) by more
                               # than this fraction -- see _close_door_gaps for why this one check
                               # catches both ways a floating-door bridge can go wrong


def _ink_runs(line: np.ndarray) -> list[tuple[int, int]]:
    """Every contiguous run of non-zero pixels in a 1-D ink scanline, as (start, end) inclusive.
    Used by _close_door_gaps to tell a real wall from a tiny irrelevant ink fragment along a door's
    own implied wall line -- see that function's docstring for why stopping at the first non-zero
    pixel (not a whole run) was tried first and found insufficient."""
    ink_at = np.nonzero(line)[0]
    if len(ink_at) == 0:
        return []
    runs = []
    start = prev = int(ink_at[0])
    for y in ink_at[1:]:
        y = int(y)
        if y != prev + 1:
            runs.append((start, prev))
            start = y
        prev = y
    runs.append((start, prev))
    return runs


def _close_door_gaps(ink: np.ndarray, doors: list[dict], w: int, h: int, stroke: int) -> np.ndarray:
    """Close a wall-tracing gap anchored to a Faster R-CNN-detected DOOR, run on the base `ink`
    BEFORE segment_rooms' own main connected-components pass. Unlike _close_orphan_labels (Pass 3,
    which only ever recovers an orphan fragment AFTER the main pass has already run), this can
    split an already-wrongly-merged room back into two, because it edits the ink everything else
    downstream (including Pass 3) is computed from.

    The anchor: a door whose own two perpendicular sides (relative to its wall's orientation, read
    off the detection box's own aspect ratio -- taller than wide means a vertical wall) land in the
    SAME open-space connected component. A real door sits IN a wall, so its two sides should
    normally be two different rooms (or a room and the exterior); landing in the same component
    means there's a real gap somewhere along that door's own implied wall line, independent of the
    door opening itself (already blocked solid, same as segment_rooms' main pass). "Floating door",
    for short.

    Scans outward from the door along its own wall line (the column it sits in, for a vertical
    wall, else the row) in both directions and bridges to the nearest SUBSTANTIAL ink run -- not
    the nearest single ink pixel. An earlier version stopped at the first non-zero pixel, which can
    be a tiny irrelevant fragment (a stray OCR/furniture speck) rather than the real wall, producing
    a bridge far short of the actual gap and leaving the room merged (confirmed on a real test
    image, oikotie_24680506: a 6px bridge where the true gap was ~35px, no split at all). Fixed by
    extracting every contiguous ink run along that line (_ink_runs) and requiring
    >= DOOR_GAP_MIN_RUN_PX before a run counts as real wall.

    A floating door is necessary but NOT sufficient evidence a gap exists right there. On a real
    40-image batch, blindly bridging every floating door's nearest run produced two different kinds
    of wrong result alongside the genuine fixes:
      - A bad MERGE: bridging across what was actually a real wall between two legitimately
        separate rooms (oikotie_22132240, 6 rooms -> 5, a real 2471 px^2 room swallowed into its
        neighbour).
      - A fabricated room: bridging through open/exterior space carved a brand-new enclosed "room"
        out of area that was never part of any counted room before -- not a split of anything, pure
        invention (oikotie_24694301: 8 floating doors fired on one over-eager open-plan image, total
        counted room area inflated 13789 -> 60528 px^2; oikotie_24704745: a new 56203 px^2 "room"
        appeared with none of the original 7 rooms shrinking to source it from).
    Both failure modes share one tell a genuine split doesn't have: the TOTAL area of every counted
    room changes. A real split only partitions existing area (conserved, up to the new bridge
    stroke's own few pixels); a bad merge loses area, a fabricated room gains it. So each candidate
    bridge is applied one at a time, cumulatively, and kept only if the total counted-room area
    stays within DOOR_GAP_AREA_TOL_FRAC of what it was just before that one bridge -- discarded
    otherwise. Confirmed on the same batch: this gate alone, with no other change, rejected exactly
    the bad merge (1 candidate, 0 kept) and both fabricated-room images (24694301: 8 candidates, 3
    kept, net room count unchanged; 24704745: 1 candidate, 0 kept) while leaving every genuine split
    untouched, including the original oikotie_24680506 case (3 candidates, 3 kept, the wrongly-
    merged 106642 px^2 room correctly split back into three: 76890 + 15031 + 14574).
    """
    open_space = (ink == 0).astype(np.uint8)
    n, lbl, _, _ = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    candidates = []
    for d in doors:
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        bw, bh = x1 - x0, y1 - y0
        vertical_wall = bh > bw
        pad = 3
        if vertical_wall:
            p1, p2 = (x0 - pad, (y0 + y1) // 2), (x1 + pad, (y0 + y1) // 2)
        else:
            p1, p2 = ((x0 + x1) // 2, y0 - pad), ((x0 + x1) // 2, y1 + pad)
        c1 = lbl[p1[1], p1[0]] if 0 <= p1[1] < h and 0 <= p1[0] < w else -1
        c2 = lbl[p2[1], p2[0]] if 0 <= p2[1] < h and 0 <= p2[0] < w else -1
        if c1 != c2 or c1 == 0:
            continue  # not floating -- the door's two sides already land in different components
        mid = (x0 + x1) // 2 if vertical_wall else (y0 + y1) // 2
        line = ink[:, mid] if vertical_wall else ink[mid, :]
        runs = [r for r in _ink_runs(line) if r[1] - r[0] + 1 >= DOOR_GAP_MIN_RUN_PX]
        lo, hi = (y0, y1) if vertical_wall else (x0, x1)
        above = [r for r in runs if r[1] < lo]
        below = [r for r in runs if r[0] > hi]
        if above:
            a_end = above[-1][1]
            if 2 <= lo - a_end <= DOOR_GAP_MAX_SEARCH_PX:
                candidates.append((mid, a_end, lo, vertical_wall))
        if below:
            b_start = below[0][0]
            if 2 <= b_start - hi <= DOOR_GAP_MAX_SEARCH_PX:
                candidates.append((mid, hi, b_start, vertical_wall))
    if not candidates:
        return ink

    def _counted_room_area(raster: np.ndarray) -> int:
        space = (raster == 0).astype(np.uint8)
        cn, clbl, cstats, _ = cv2.connectedComponentsWithStats(space, connectivity=4)
        border = (set(clbl[0, :].tolist()) | set(clbl[-1, :].tolist())
                  | set(clbl[:, 0].tolist()) | set(clbl[:, -1].tolist()))
        min_area = ROOM_MIN_AREA_FRAC * h * w
        return sum(int(cstats[i, cv2.CC_STAT_AREA]) for i in range(1, cn)
                   if i not in border and cstats[i, cv2.CC_STAT_AREA] >= min_area)

    cur = ink.copy()
    base_total = _counted_room_area(cur)
    for mid, a, b, vertical_wall in candidates:
        trial = cur.copy()
        if vertical_wall:
            cv2.line(trial, (mid, a), (mid, b), 255, thickness=stroke)
        else:
            cv2.line(trial, (a, mid), (b, mid), 255, thickness=stroke)
        new_total = _counted_room_area(trial)
        ok = (new_total > 0) if base_total == 0 else (abs(new_total - base_total) <= DOOR_GAP_AREA_TOL_FRAC * base_total)
        if ok:
            cur = trial
            if new_total:
                base_total = new_total
    return cur


def _close_orphan_labels(ink: np.ndarray, labels: list[dict], rooms: list[dict], used_labels: set,
                          w: int, h: int) -> list[dict]:
    """Pass 3, for whichever labels passes 1-2 still couldn't place: bridge a SMALL, LOCAL gap
    around just that one label, enough to enclose it, measure the resulting area, then throw the
    bridge away -- it only ever exists inside this function's own scratch copy of `ink`; the real
    wall trace/DXF is never touched. (The user's own idea: "if we just close it temporarily to
    measure the distance and then remove the added closing".)

    The two earlier, REJECTED attempts at gap-bridging (segment_rooms' own docstring above) both
    failed the same way: a single kernel size applied globally can't tell "a real gap that should
    be bridged" from "a real room/notch that should stay open", because the right answer differs in
    different parts of the SAME image. This version never answers that question globally at all:
      - It only tries to close a gap AROUND an orphan label's own position -- i.e. only where we
        already have strong independent evidence (an OCR'd room code) that a real room exists,
        never a blind sweep over the whole wall network. "Strong evidence" is deliberately narrow:
        only an EXACT match (case-insensitive) to KNOWN_LABELS, floorplan_trace's own curated
        vocabulary of real room codes -- not just any alphanumeric OCR read. Confirmed necessary on
        a second real test image (oikotie_24721956, 48 labels): several reads that pass every
        other filter in this codebase are still OCR noise/fragments ("ERSSI", "RD", "HΛ", "KHHH")
        that are NOT real codes, and each one independently found and claimed its own large,
        plausible-looking enclosure once allowed into this search at all -- removing the earlier
        multi-label merge (below) stopped them gluing together, but did nothing to stop any ONE
        of them winning alone.
      - It searches UPWARD from the smallest possible kernel and stops at the first one that
        encloses the label -- the minimum width that does the job, not one fixed guess.
      - The closing itself is applied only within a small window around that one label (padded by
        the kernel radius for correct edge behaviour at the window boundary, then pasted back into
        an otherwise-untouched copy of `ink`) -- so even a large kernel, for a label that genuinely
        needs one, can't reach far enough to bridge some unrelated gap elsewhere in the same plan.
      - Before accepting a candidate, it checks the newly-enclosed area doesn't overlap (more than
        a small tolerance) any pixel already claimed by an earlier match or an earlier orphan in
        this same pass -- rejects outright if it does, since that means this "gap" was actually a
        stretch of real wall between two rooms that must stay separate. Confirmed necessary: on
        oikotie_24716483, OH's own true gap needed k=140 -- big enough that a naive version of this
        reached far enough to also swallow a separate, already-claimed label's area 500+px away.
    Each accepted enclosure is named by its own anchor label ONLY, never by every label that
    happens to land inside it. An earlier version combined co-located orphans into one name
    (e.g. "ET / RK", for a real case on oikotie_24716483 where two codes genuinely share one open
    area with no partition between them) -- dropped after a second real test image
    (oikotie_24721956, 48 labels, several fragmentary/noisy OCR reads) showed it gluing UNRELATED
    reads into nonsense names instead ("HH / HΛ / KHHH / TEKN", "AULA / OH / RT", and even capped
    at 2 labels, "PARVEKE / o sa" / "ERSSI / TKH" -- a real code glued to an OCR fragment). A
    second orphan that lands in the same enclosure as the one just accepted is simply left
    unresolved -- its own later attempt will find the same area already in `claimed` and get
    rejected by the overlap check above, so nothing is drawn twice, but ET/RK-style genuine
    co-labeling is no longer attempted at all. Conservative on purpose: one missed nice-to-have is
    a smaller cost than a second real image's worth of glued-nonsense room names.

    Deliberately conservative otherwise too: most orphans on real test images (SK/PY, KPH, ULP,
    ILP, KT on oikotie_24716483) never close even at ORPHAN_CLOSE_K_MAX_FRAC -- genuinely wide
    open to the true exterior, not a bridgeable gap at any reasonable width -- and are correctly
    left unmatched, exactly as before this pass existed.

    An orphan whose text looks like an OCR misread of another nearby label (_is_likely_ocr_
    duplicate -- see its own docstring) is skipped outright, before any closing is attempted at
    all. Confirmed necessary on a THIRD real test image (oikotie_24721956): "TEKN", a real
    KNOWN_LABELS code, is ALSO what PaddleOCR misread a real "TK" stairwell label as, 112px away
    -- not caught by the KNOWN_LABELS-only restriction above (both reads are individually
    legitimate codes), and not safely catchable by proximity alone either (this project's own
    15-image sample has genuine different-room pairs just as close, e.g. "KHH"/"OH" at a tighter
    fraction of image size than this bogus pair) -- it took the much more specific "one text is
    the other with extra letters inserted" signature to tell the two apart without also rejecting
    real nearby rooms elsewhere.
    """
    k_max = ORPHAN_CLOSE_K_MAX_FRAC * max(w, h)
    claimed = np.zeros((h, w), dtype=np.uint8)
    for r in rooms:
        # A room with a real OCR match, or a PRIMARY (not erosion-recovered) room even without
        # one, is a confirmed, distinct interior space -- protect it. An unlabeled RECOVERED
        # fragment is uncertain scrap from the same merged-exterior blob an orphan label here may
        # legitimately need to reclaim (confirmed necessary: without this exception, two leftover
        # unlabeled recovered fragments on oikotie_24716483 -- scraps of the exact blob OH needed --
        # blocked OH's own later, correct match below).
        if r.get("text") is not None or not r.get("recovered"):
            cv2.fillPoly(claimed, [r["polygon"].astype(np.int32)], 1)

    # Indexed, not text-keyed: this dataset can print the same code on several different physical
    # rooms (e.g. "AH" x3 on one test image, each its own closet) -- resolving one must not skip
    # the others. Restricted to an EXACT KNOWN_LABELS match -- see this function's own docstring
    # for why a looser "looks alphanumeric" test isn't enough.
    orphans = [(li, lab) for li, lab in enumerate(labels)
               if li not in used_labels and lab["text"].upper() in KNOWN_LABELS]
    resolved: set = set()
    new_rooms = []
    for li, lab in orphans:
        if li in resolved:
            continue
        lx, ly = int(round(lab["x"])), int(round(lab["y"]))
        if not (0 <= ly < h and 0 <= lx < w) or ink[ly, lx] != 0:
            continue  # label sits on wall ink itself (e.g. a caption printed on the wall line)
        if _is_likely_ocr_duplicate(lab, labels, w, h):
            continue  # e.g. a real "TK" misread as "TEKN" nearby -- see that helper's own docstring

        k = ORPHAN_CLOSE_K_START
        while k <= k_max:
            win = max(ORPHAN_CLOSE_WIN_MIN_PX, int(k * ORPHAN_CLOSE_WIN_FACTOR))
            x0, y0 = max(0, lx - win), max(0, ly - win)
            x1, y1 = min(w, lx + win), min(h, ly + win)
            # MORPH_RECT, not MORPH_ELLIPSE: OpenCV computes a rectangular dilate/erode with a
            # fast, kernel-size-independent separable (running min/max) filter; an elliptical one
            # does not decompose that way, so cv2.morphologyEx with a large elliptical kernel got
            # slower as k grew -- confirmed on oikotie_24721956 (large 1912x1082 photo, 48 labels):
            # 534s for one call to segment_rooms with MORPH_ELLIPSE, 2.2s after this one-line
            # switch to MORPH_RECT (same image, same labels, otherwise identical code). A squarer
            # bridge shape is a fine trade for that -- this is a scratch closing used only to
            # measure area, never drawn into the real wall trace/DXF.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2 + 1,) * 2)
            patched = ink.copy()
            patched[y0:y1, x0:x1] = cv2.morphologyEx(ink[y0:y1, x0:x1], cv2.MORPH_CLOSE, kernel)

            area, comp_mask, touches_border = _flood_component(patched, lx, ly, w, h)
            if area > 0 and not touches_border:
                overlap = int(np.sum(comp_mask & (claimed > 0)))
                if overlap > max(ORPHAN_CLOSE_OVERLAP_FLOOR_PX, ORPHAN_CLOSE_OVERLAP_FRAC * area):
                    break  # real wall between two rooms that must stay separate -- not a gap
                ys, xs = np.nonzero(comp_mask)
                cy_, cx_ = float(ys.mean()), float(xs.mean())
                contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                outline = max(contours, key=cv2.contourArea)
                outline = cv2.approxPolyDP(outline, ROOM_OUTLINE_EPSILON_PX, closed=True).reshape(-1, 2)
                new_rooms.append({"cx": cx_, "cy": cy_, "area_px2": float(area), "polygon": outline,
                                   "gap_closed": True, "text": lab["text"]})
                claimed |= comp_mask.astype(np.uint8)
                resolved.add(li)

                # Drop any unlabeled RECOVERED fragment this new enclosure has now substantially
                # subsumed (half or more of its own area) -- it was uncertain scrap of the exact
                # same merged blob to begin with (that's why it was never excluded from `claimed`
                # above), so keeping both would double-count the same floor area twice.
                still = []
                for r in rooms:
                    if r.get("recovered") and r.get("text") is None and r["area_px2"] > 0:
                        old_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(old_mask, [r["polygon"].astype(np.int32)], 1)
                        shared = int(np.sum((old_mask > 0) & comp_mask))
                        if shared / r["area_px2"] >= 0.5:
                            continue  # superseded -- don't keep both
                    still.append(r)
                rooms[:] = still
                break
            k = max(k + 1, int(round(k * ORPHAN_CLOSE_K_MULT)))
    return new_rooms


def segment_rooms(polygons: list[np.ndarray], openings: list[dict], shape: tuple[int, int],
                   labels: list[dict], wall_ink: np.ndarray | None = None) -> list[dict]:
    """Interior room regions, recovered from the SAME traced wall/furniture
    polygons already in the DXF (not a fresh re-detection) by rasterising them
    and taking the connected components of everything left open.

    TRIED AND REJECTED (2026-10-07): calibrating against one independently
    measured "building interior footprint" instead of summing these rooms --
    the idea being that the sum is only as accurate as the LEAST cleanly
    segmented room, while a single robust total would make every room an
    independent estimate instead. Measuring that total turned out to be the
    problem: a convex hull of the wall ink came out 2.3x the honest room sum
    on a real test building (a notch-heavy, non-convex shape makes the hull
    balloon far past "bounded, small" error), and morphological closing at
    increasing kernel sizes never found a stable middle ground either --
    7k px^2 at a 5px kernel (still fragmented), 54k at 30px, 184k at 60px,
    with no plateau anywhere near the true ~84k, because the real wall-tracing
    gap and this building's genuine concave notches are a similar size, so no
    single kernel bridges one without also swallowing the other. Reverted;
    the sum below, despite depending on individual segmentation quality, is
    the more accurate of the two in practice. A real fix would need to close
    the SPECIFIC wall gap (detect two dangling, roughly collinear wall-polygon
    endpoints and bridge just those two points), not a
    blanket geometric measure that can't tell a missing wall from a real
    concave notch.

    Rasterised as a STROKED outline (cv2.polylines, ROOM_WALL_STROKE_FRAC of
    the image's own scale thick), not cv2.fillPoly. A traced "wall" polygon
    here is, in general, only the OUTER edge of the wall's own ring shape --
    findContours (RETR_LIST, no hierarchy) never pairs it with the matching
    inner edge that would let anything punch the room's interior out as a
    hole -- so filling it solid fills the entire room behind that wall too,
    not just the wall's own thickness. Every other place in this codebase
    only ever strokes these polygons (export_contours_dxf writes them as an
    unfilled LWPOLYLINE, every debug render in this project used
    cv2.polylines) which is exactly why this was never visible before.

    A door OR a window is a genuine gap in the wall ink by construction -- flood-filling the open space would otherwise walk
    straight through every doorway and merge every room into one component,
    AND (easy to miss: a window sits in an *exterior* wall, but that gap still
    connects that room's interior straight through to the true outside) walk
    out through every window and merge every room into the exterior background
    that surrounds the whole building -- which is exactly the component this
    function discards as "not a room" one paragraph down, so an unblocked
    window silently erases every interior room in the entire plan, not just
    the one room it belongs to. Each Faster R-CNN-detected door/window box
    (already computed for the DXF's Door/Window layers) is rasterised as solid
    ink here too, specifically to block both.

    Beyond doors/windows, the wall network can still have a genuine TRACING
    gap -- a real wall whose ink dipped below the binarisation threshold
    somewhere along its run -- which lets the room behind it
    flood straight out and merge into the exterior component this function
    would otherwise just discard as "not a room". Recovered instead, by
    eroding that component (a real crop margin is thin, a leaked room's bulk
    is not) and keeping only what's left that is both blocky (not a thin bent
    margin remnant) and no longer touching the image border (unlike a
    corner of margin, which stays adjacent to it even after eroding) -- see
    the inline comments below for the two rejected approaches (a convex-hull
    seal, a convex-hull filter) and why. Recovered fragments get their own
    "Other"-labelled entry (or a real OCR label, if one happens to be close
    enough) with their own eroded boundary -- smaller than the true room by
    construction, a safer bias than either rejected approach's overestimate.

    The component touching the image's own border is otherwise discarded as
    "outside the building" (or blank crop margin), not a room -- everything
    else above ROOM_MIN_AREA_FRAC is a room candidate, matched to an already-
    OCR'd text label in two passes: first whichever label's own position falls
    literally INSIDE that room's traced boundary (point-in-polygon -- the
    strong signal, especially for a recovered room, whose centroid is the
    erosion-shrunk fragment's, not the true room's), then, for whatever's
    still unmatched, whichever label sits closest to the room's centroid
    (within ROOM_LABEL_MAX_DIST_FRAC of the image's own scale) -- or left
    unnamed if neither test finds one (a room with no printed code on the
    source plan has nothing to match -- this doesn't invent one).

    TRIED AND REJECTED (2026-10-07, follow-up #9's own TODO): closing the
    SPECIFIC wall-tracing gap, instead of recovering from it after the fact,
    by bridging two dangling wall-polygon endpoints that flank a real gap.
    Two independent ways to find that pair, both unsafe on real data:
      - Treat each wall polygon as a thin ribbon and bridge its two short-end
        "caps" to another polygon's nearest cap if close and roughly
        collinear: found ZERO usable bridges on either test image, because
        this dataset's actual traced polygons are coarse wall-network blobs
        (aspect ratio ~2, not a clean elongated single-wall ribbon >=3), not
        the tidy per-segment ribbons the method assumes.
      - Skeletonize the rasterised wall ink (skimage.morphology.skeletonize)
        and bridge pairs of degree-1 skeleton endpoints that are close and
        roughly collinear with their own local skeleton direction: found
        real candidates this time, but far too liberal -- 11 bridges fired
        on one test image and merged OH/ET/RK/ULP/SK-PY (five DIFFERENT real
        rooms) into one 369k px^2 blob, because any two wall endpoints that
        happen to point toward each other within the angle tolerance pass
        the test even when they don't flank the same physical gap.
    Same root failure as follow-up #8/#9: a geometric rule that's loose
    enough to find the real gap is also loose enough to bridge across real
    walls between rooms that must stay separate, and this is the third
    independently-designed heuristic in this project to hit exactly that
    wall.

    REFINED INTO A WORKING VERSION (follow-up #11): the failure above was
    always a blanket, GLOBAL rule trying to answer "where are the real gaps"
    with no anchor. Pass 3 below (_close_orphan_labels) never asks that
    globally at all -- it only tries to close a gap AROUND an already-OCR'd
    orphan label's own position (strong independent evidence a room really
    exists there), searches for the MINIMUM kernel that encloses it, applies
    that kernel only in a small window around that one label, and rejects
    the result outright if it would overlap any area already claimed
    elsewhere. See its own docstring for the full account, including why
    each of those pieces turned out to be necessary (not just conservative
    caution) on the same two real test images this docstring's own rejected
    attempts were measured against.

    A SECOND, independent anchor (follow-up #12, Pass 0 below, _close_door_gaps): a Faster
    R-CNN-detected DOOR whose own two sides land in the same open-space component is equally
    strong independent evidence of a real gap, right there, along that door's own wall -- and
    unlike Pass 3, this one runs BEFORE the main pass, so it can split an already-wrongly-merged
    room back into two, not just recover an orphan fragment after the fact. Gated the same way in
    spirit (only ever acts on strong local evidence, never a blanket sweep), but the gate itself
    had to be different: a floating door turned out to be necessary but not sufficient evidence
    on its own (see that function's docstring for the real bad-merge and fabricated-room cases
    this produced before the gate was added).

    Dimension/scale annotations ("ar=485mm") are stripped from `labels`
    before any of the three passes below runs -- see ROOM_DIMENSION_LABEL_RE
    -- because one of those was found actively blocking Pass 3's own correct
    match (it had already claimed, as a fake "room", a fragment of the exact
    area an orphaned real label needed).

    Returns [{"text": str | None, "area_px2": float, "cx": int, "cy": int,
    "polygon": Nx2 int array}, ...] in image-pixel space, one entry per
    recovered room; "polygon" is that room's own traced boundary (its open-
    space connected component's contour, simplified), for drawing an outline
    that makes it visually unambiguous which physical room an area label
    belongs to -- independent of whether a text match was found for it.
    """
    labels = [lab for lab in labels if not _is_non_room_label(lab["text"])]
    h, w = shape
    stroke = max(2, round(ROOM_WALL_STROKE_FRAC * max(w, h)))
    if wall_ink is not None:
        # Direct mode (floorplan_rooms_direct.py): walls straight from the image's own ink pixels,
        # no tracing/simplification/axis-snapping in between.
        ink = wall_ink.copy()
    else:
        ink = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            cv2.polylines(ink, [poly.astype(np.int32)], True, 255, thickness=stroke)

    for d in openings:
        if d["class"] not in ("Door", "Window"):
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in d["box"])
        cv2.rectangle(ink, (x0, y0), (x1, y1), 255, thickness=-1)

    # Pass 0: door-anchored wall-tracing gap closing -- see _close_door_gaps' own docstring. Runs
    # before the main pass below so a genuine fix here can SPLIT an already-wrongly-merged room,
    # not just recover an orphan fragment after the fact (that's Pass 3, _close_orphan_labels).
    ink = _close_door_gaps(ink, [d for d in openings if d["class"] == "Door"], w, h, stroke)

    open_space = (ink == 0).astype(np.uint8)
    n, lbl, stats, centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if n <= 1:
        return []
    exterior = set(lbl[0, :].tolist()) | set(lbl[-1, :].tolist()) | set(lbl[:, 0].tolist()) | set(lbl[:, -1].tolist())
    exterior.discard(0)

    min_area = ROOM_MIN_AREA_FRAC * h * w
    max_dist = ROOM_LABEL_MAX_DIST_FRAC * max(w, h)
    room_ids = [i for i in range(1, n) if i not in exterior and stats[i, cv2.CC_STAT_AREA] >= min_area]
    rooms = []
    for i in room_ids:
        contours, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)
        # A sliver (mean width = 2 x area / perimeter) is the gap between two close parallel lines --
        # a window's double line, a wall drawn as two strokes -- not a room.
        if 2 * stats[i, cv2.CC_STAT_AREA] / max(cv2.arcLength(outline, True), 1) < ROOM_MIN_WIDTH_FRAC * max(w, h):
            continue
        outline = cv2.approxPolyDP(outline, ROOM_OUTLINE_EPSILON_PX, closed=True).reshape(-1, 2)
        rooms.append({"cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
                      "area_px2": float(stats[i, cv2.CC_STAT_AREA]), "text": None, "polygon": outline,
                      "cc": i})

    # A closed piece of furniture (a rug, a table, a kitchen island) drawn inside a room encloses its
    # own little area, which comes out as a separate "room". Such an outline is a THIN line, so just
    # beyond it lies the surrounding room; a real room is behind thick walls (or other rooms). Test
    # the band just outside each candidate: if it is mostly open floor of one single larger room,
    # give the candidate's area to that room instead of counting it as a room.
    band_px = stroke + 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)
    by_id = {r["cc"]: r for r in rooms}
    merged = set()
    for r in sorted(rooms, key=lambda r: r["area_px2"]):
        mask = (lbl == r["cc"]).astype(np.uint8)
        band = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
        around = lbl[band & (open_space > 0)]
        if around.size < NESTED_MIN_OPEN_SHARE * band.sum():
            continue  # walled in, not a thin outline
        ids, counts = np.unique(around, return_counts=True)
        host_id = ids[np.argmax(counts)]
        host = by_id.get(host_id)
        if (host is not None and host_id not in merged and counts.max() >= NESTED_HOST_SHARE * around.size
                and r["area_px2"] < ROOM_NESTED_MAX_FRAC * host["area_px2"]):
            host["area_px2"] += r["area_px2"]
            merged.add(r["cc"])
    rooms = [r for r in rooms if r["cc"] not in merged]
    for r in rooms:
        r.pop("hull", None)

    # Recover floor area that leaked into an exterior-touching component through a real wall-
    # tracing gap. Two earlier attempts, both rejected:
    #   - A convex-hull-based filter (keep only the part of the exterior-touching component inside
    #     the building's rough hull footprint): on a non-convex building the hull is generously
    #     oversized at every real concave notch, so it counted a lot of genuine outside-the-building
    #     space as "recoverable" (turned one missing ~20 m² room into two implausible ~40 m² and
    #     ~12 m² "Other" entries, while shrinking every real room to compensate).
    #   - Erosion alone (the true crop margin is thin, a leaked ROOM's bulk is not, so eroding by
    #     ~2x the expected margin thickness should eat the margin ring while the room mostly
    #     survives): closer, but the margin ring is NOT uniformly thin around an irregular
    #     building's silhouette (it's much wider at a concave notch than elsewhere), so a chunk of
    #     genuine margin outlived the erosion too, on this image getting mislabeled as a fake 12.9 m²
    #     "AH" room.
    # Two more filters on top of erosion, neither sufficient alone:
    #   - Solidity (contour area / its own convex-hull area): a thin, bent margin/notch remnant
    #     scores low (measured 0.44/0.64 on two such remnants here) vs a real room's blocky core.
    #     But NOT sufficient alone -- a rectangular corner of pure margin, bounded by the crop's own
    #     straight edge on one side and the building's straight wall on the other, can be just as
    #     solid as a real room (measured 0.96 on exactly such a corner here, higher than the genuine
    #     leaked room's own 0.79).
    #   - Still touching the image border after this same erosion: the true leaked room, once
    #     detached from the thin margin ring by erosion, should no longer reach the crop edge at
    #     all -- but a margin corner is by definition adjacent to that edge and mostly still touches
    #     it even after erosion. This is what actually catches the high-solidity margin corner
    #     solidity missed.
    # Both together, on this image: keeps only the one genuine leaked room (solidity 0.79, detached
    # from the border); rejects the high-solidity margin corner (0.96, still touching the border)
    # and the two low-solidity margin remnants.
    if exterior:
        margin_px = max(3, round(ISOLATE_MARGIN_FRAC * max(w, h)))
        erode_px = margin_px * 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        exterior_mask = np.isin(lbl, list(exterior)).astype(np.uint8)
        eroded = cv2.erode(exterior_mask, kernel)
        rn, rlbl, rstats, rcentroids = cv2.connectedComponentsWithStats(eroded, connectivity=4)
        still_touches_border = (set(rlbl[0, :].tolist()) | set(rlbl[-1, :].tolist())
                                 | set(rlbl[:, 0].tolist()) | set(rlbl[:, -1].tolist()))
        for j in range(1, rn):
            if rstats[j, cv2.CC_STAT_AREA] < min_area or j in still_touches_border:
                continue
            contours, _ = cv2.findContours((rlbl == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = max(contours, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(outline))
            solidity = rstats[j, cv2.CC_STAT_AREA] / hull_area if hull_area > 0 else 0
            if solidity < ROOM_RECOVERY_MIN_SOLIDITY:
                continue
            outline = cv2.approxPolyDP(outline, ROOM_OUTLINE_EPSILON_PX, closed=True).reshape(-1, 2)
            rooms.append({"cx": float(rcentroids[j][0]), "cy": float(rcentroids[j][1]),
                          "area_px2": float(rstats[j, cv2.CC_STAT_AREA]), "text": None, "polygon": outline,
                          "recovered": True})

    # Pass 1: point-in-polygon containment. A label's own (x, y) sitting literally inside a
    # room's traced boundary is a much stronger signal than "nearest centroid" -- it matters most
    # for a RECOVERED room, whose centroid is the erosion-shrunk fragment's own centroid, not the
    # true room's, so a label placed off-center (common -- OCR reads wherever the text happens to
    # sit on the plan, not the room's geometric middle) can end up centroid-closer to some other,
    # wrong room while still falling inside its own room's actual boundary. Globally-sorted-by-
    # distance, same reason as pass 2 below: two rooms could otherwise both contain the same label
    # (only possible for overlapping/adjacent polygons) and grab it independently.
    contained = sorted(((float(np.hypot(lab["x"] - r["cx"], lab["y"] - r["cy"])), ri, li)
                        for ri, r in enumerate(rooms) for li, lab in enumerate(labels)
                        if cv2.pointPolygonTest(r["polygon"].astype(np.float32),
                                                 (float(lab["x"]), float(lab["y"])), False) >= 0),
                       key=lambda p: (_label_rank(labels[p[2]]["text"]), p[0]))
    used_rooms, used_labels = set(), set()
    for dist, ri, li in contained:
        if ri in used_rooms or li in used_labels:
            continue
        rooms[ri]["text"] = labels[li]["text"]
        used_rooms.add(ri)
        used_labels.add(li)

    # Pass 2: nearest-centroid fallback, for whatever pass 1 didn't resolve (a label outside every
    # room's own boundary -- e.g. printed in a hallway/doorway between the rooms it names). Not
    # "whichever label is nearest" independently for each room -- otherwise two different rooms
    # can both claim the same nearby label (e.g. both grab "WC") while a room that's genuinely a
    # bit further from its own label goes unmatched. Assign the globally closest (room, label)
    # pair first, then the next-closest whose room and label are both still free.
    pairs = sorted(((float(np.hypot(lab["x"] - r["cx"], lab["y"] - r["cy"])), ri, li)
                    for ri, r in enumerate(rooms) for li, lab in enumerate(labels)
                    if ri not in used_rooms and li not in used_labels),
                   key=lambda p: p[0])
    for dist, ri, li in pairs:
        if dist > max_dist:
            break
        if ri in used_rooms or li in used_labels:
            continue
        rooms[ri]["text"] = labels[li]["text"]
        used_rooms.add(ri)
        used_labels.add(li)

    # Pass 3: label-anchored gap closing, for whatever's STILL unmatched after both passes above --
    # an orphan label whose own room never became its own component at all (merged straight into
    # the exterior-touching blob through a gap too wide for the erosion-based recovery above to
    # separate back out). See _close_orphan_labels' own docstring for how this stays safe where
    # this function's own two earlier, REJECTED gap-bridging attempts (above) weren't.
    rooms.extend(_close_orphan_labels(ink, labels, rooms, used_labels, w, h))

    for r in rooms:
        r["cx"], r["cy"] = int(round(r["cx"])), int(round(r["cy"]))
    return rooms


# Which walls room segmentation floods between:
#   "ink"      -- the image's own dark pixels (Otsu, letters erased, 3x3 closing). Thick walls are
#                 solid, corners stay closed, light lines (furniture, notes) are ignored.
#   "contours" -- trace_contours' outlines, stroked. A thick wall's inside (between its outer and
#                 inner outline) is then an enclosed empty area and gets counted as a room.
#   "lines"    -- the drawn single-line trace. Straightening moves line ends a few px, which opens
#                 gaps at corners that rooms leak through.
ROOM_WALLS = os.environ.get("FP_ROOM_WALLS", "ink")
PLAN_OUTLINE_MIN_FRAC = 0.02  # "ink": outer boundaries of drawn shapes at least this share of the image are walls


def room_wall_ink(image: np.ndarray, labels: list[dict], wall_lines: list[np.ndarray]) -> np.ndarray | None:
    """Wall mask for segment_rooms per ROOM_WALLS (None = segment_rooms strokes the contours itself)."""
    h, w = image.shape[:2]
    if ROOM_WALLS == "lines":
        return lines_to_wall_ink(wall_lines, (h, w))
    if ROOM_WALLS == "ink":
        text_mask = _geometry_text_mask(image, labels)
        ink = cv2.morphologyEx(_binarize_ink(image, text_mask, "otsu"), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # The building's outer edge is often only a light window line, which the dark-pixel walls
        # leave out, so a room behind it leaks outside and is thrown away. Draw the outer boundary
        # of everything drawn (light lines included) as a wall too.
        light = cv2.morphologyEx(_binarize_ink(image, text_mask, "paper"), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        outer, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stroke = max(2, round(ROOM_WALL_STROKE_FRAC * max(w, h)))
        big = [c for c in outer if cv2.contourArea(c) >= PLAN_OUTLINE_MIN_FRAC * h * w]
        cv2.drawContours(ink, big, -1, 255, thickness=stroke)
        return ink
    return None


def lines_to_wall_ink(lines: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """The single-line trace rasterised as walls for segment_rooms (same stroke width it uses)."""
    h, w = shape
    ink = np.zeros((h, w), dtype=np.uint8)
    stroke = max(2, round(ROOM_WALL_STROKE_FRAC * max(w, h)))
    cv2.polylines(ink, [l.astype(np.int32) for l in lines], False, 255, thickness=stroke)
    return ink


def add_room_areas_to_dxf(dxf_path: str, rooms: list[dict], size_m2: float, height: int) -> int:
    """Calibrate segment_rooms' pixel areas against the listing's own scraped
    total floor area (size_m2 / sum of every recovered room's pixel area =
    m2-per-px2), then draw each room's own traced boundary plus its real-world
    area as TEXT at the centroid, both on a new "Area" layer.

    The boundary outline (not just a floating number) is what makes it
    unambiguous which physical room a label belongs to -- CAD viewers don't
    support DXF hover tooltips, so this is the in-file alternative: select or
    look at the outlined region, not a text entity, to see what it covers.
    Drawn as its own LWPOLYLINE distinct from the wall trace on the Trace
    layer, so toggling the Area layer off leaves the original trace untouched.
    Each room's outline+label pair additionally gets its own color, cycling
    through ROOM_COLOR_PALETTE, as a second, faster way to tell neighbouring
    rooms apart at a glance (the shared layer color is still set, as a
    fallback for anything that ignores per-entity color overrides).

    This is a real assumption, not a measurement: it takes the segmented
    rooms' combined pixel area as standing for the listing's whole living
    area, so it inherits whatever the segmentation over- or under-counts
    (a corridor that failed to separate from a room, a balcony the listing
    excludes but the trace includes). Treat the printed per-room figures as
    a plausible breakdown of the known total, not an independently verified
    room-by-room survey. (Calibrating instead against an independently
    measured "whole building footprint" was tried and reverted -- see
    segment_rooms' docstring -- because measuring that total reliably turned
    out to be the harder, less accurate problem on real test data.)

    A room calibrated below AREA_MIN_DISPLAY_M2 is skipped -- not drawn, no
    outline, no label -- but still included in the calibration sum above, since
    it's real floor area and dropping it from the sum would just inflate every
    other room to compensate (the exact mistake a missing/leaked room already
    makes without this).
    """
    total_px2 = sum(r["area_px2"] for r in rooms if counts_in_living_area(r))
    if total_px2 <= 0:
        return 0
    scale = size_m2 / total_px2  # m^2 per px^2 (balconies/terraces left out: not in size_m2)

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    if AREA_LAYER not in {lyr.dxf.name for lyr in doc.layers}:
        doc.layers.add(AREA_LAYER, color=AREA_LAYER_COLOR)

    text_h = max(height * 0.014, 12.0)
    drawn = 0
    for i, r in enumerate(rooms):
        area_m2 = r["area_px2"] * scale
        if area_m2 < AREA_MIN_DISPLAY_M2:
            continue  # still counted in `scale`'s calibration above -- just not drawn
        color = ROOM_COLOR_PALETTE[i % len(ROOM_COLOR_PALETTE)]
        is_other = r["text"] is None and r.get("recovered")
        if r["text"] is not None:
            text = f"{r['text']} {area_m2:.1f} m²"
        elif is_other:
            text = f"Other {area_m2:.1f} m²"  # reclaimed from a wall-tracing gap, not a normally segmented room
        else:
            text = f"{area_m2:.1f} m²"

        if not is_other:
            # An "Other" entry's shape is an eroded, filtered leftover -- not a real traced room
            # boundary like every other entry here -- so draw the number, not a misleading outline.
            outline = [(float(x), height - float(y)) for x, y in r["polygon"]]
            msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": AREA_LAYER, "color": color})

        t = msp.add_text(text, dxfattribs={"layer": AREA_LAYER, "color": color, "height": text_h})
        t.set_placement((r["cx"], height - r["cy"]), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        drawn += 1
    doc.saveas(dxf_path)
    return drawn


def add_legend(dxf_path: str, width: int, height: int) -> None:
    """Add a color-key legend below the drawing, listing only the layers this
    DXF actually uses -- a plan with no detected windows doesn't get a
    "Window" row it has nothing to show for.

    Everything here is written in the same image-pixel coordinate space as
    every other export in this pipeline (DXF y already runs 0..height after
    the same height-y flip), so "below the drawing" is simply y < 0. Legend
    text/swatch size is scaled to this image's own resolution (1.2% of its
    larger dimension) so it stays legibly sized whether the source was a
    ~600px scrape or a ~3000px photo, rather than using one fixed pixel size
    across wildly different source resolutions.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    used_layers = {e.dxf.layer for e in msp}
    entries = [e for e in LEGEND_ENTRIES if e[0] in used_layers]
    if not entries:
        return
    if LEGEND_LAYER not in {lyr.dxf.name for lyr in doc.layers}:
        doc.layers.add(LEGEND_LAYER)

    unit = max(width, height) * 0.012
    text_h = max(unit, 14.0)
    swatch_w, swatch_h = text_h * 2.2, text_h * 0.9
    row_h = text_h * 1.8
    margin = text_h * 2.5

    title = msp.add_text("LEGEND", dxfattribs={"layer": LEGEND_LAYER, "height": text_h * 1.2, "color": 7})
    title.set_placement((0, -margin))

    for i, (layer, desc, color) in enumerate(entries):
        y = -margin - row_h * (i + 1)
        # SOLID (always rendered filled, unlike a LINE's lineweight which most
        # viewers hide on screen by default) -- vertex order must be
        # bottom-left, bottom-right, top-left, top-right to avoid a bowtie.
        msp.add_solid([(0, y), (swatch_w, y), (0, y + swatch_h), (swatch_w, y + swatch_h)],
                      dxfattribs={"layer": LEGEND_LAYER, "color": color})
        label = msp.add_text(f"{layer} - {desc}",
                              dxfattribs={"layer": LEGEND_LAYER, "height": text_h, "color": 7})
        label.set_placement((swatch_w + text_h, y))

    doc.saveas(dxf_path)


ISOLATE_DILATE_FRAC = 0.04  # bridge gaps up to this fraction of max(w, h) when grouping ink into one
                            # connected blob -- NOT a flat pixel count (see ISOLATE_DILATE_FLOOR_PX):
                            # confirmed necessary on a real listing (oikotie_24682800, 723x1023) whose
                            # own wall lines are thin/light enough that the OLD flat 8px dilation left
                            # its single connected floor plan shattered into 25 separate components (the
                            # largest only 5.6% of the image), so the relative-size rule below discarded
                            # nearly the whole plan as if each fragment were a separate marketing element.
                            # 0.04 (~40px on that image) reassembles it into a handful of components
                            # whose largest alone is already ~35% of the image -- the real floor plan.
ISOLATE_DILATE_FLOOR_PX = 8  # floor for a small source image, same as the old flat value
ISOLATE_MIN_AREA_FRAC = 0.05    # reject everything if even the largest component is smaller than this
ISOLATE_RELATIVE_AREA_FRAC = 0.30  # also keep any OTHER component at least this fraction of the largest's size
ISOLATE_MARGIN_FRAC = 0.03   # pad the isolated bbox by this fraction of its own size
ISOLATE_PROTECT_MIN_AREA_FRAC = 0.002  # an excluded component at least this big is worth a targeted OCR
                                       # check before being discarded as junk (below this, not worth the
                                       # call -- a stray dot/watermark fragment, never a real small room)
ISOLATE_PROTECT_CROP_PAD_FRAC = 0.15  # pad a protect-check crop by this fraction of its own size, so a
                                      # room code printed just outside the component's own ink isn't missed


def isolate_plan(image: np.ndarray) -> np.ndarray:
    """Crop a scraped Oikotie image down to just the floor-plan drawing(s),
    discarding the marketing composite around them (title block, apartment-unit
    table, scale bar, agency credit, address). Must run BEFORE OCR/tracing --
    otherwise that surrounding text gets read and masked right alongside the
    real plan, which is what motivated this in the first place.

    This finds "the plan" as the connected clusters of non-text ink after a small
    dilation -- gaps of ISOLATE_DILATE_FRAC of the image's own scale (a door opening, the space between
    two wall runs) get bridged into one blob, but the much larger gap to a
    genuinely separate sheet element (a scale bar, a corner watermark) does
    not. A density-blur approach was tried first and
    measured looser: on a sheet whose scale-bar numerals are too widely
    spaced to group into a text row (so they survive the text mask as stray
    ink), its wider blur window bridges the plan to that stray ink and returns
    close to the whole, uncropped sheet.

    Critically, this does NOT just keep the single largest component: a
    multi-storey sheet draws each floor as its own separate wall network, with
    real whitespace between them, so a second floor is exactly as "disconnected
    from the largest blob" as a scale bar is. The two are told apart by size,
    not connectivity -- floor plans are all roughly comparable in extent to one
    another (measured on this dataset: a real second floor is 59-81% the size
    of the first), while a marketing element is much smaller (<22%). So every
    component within ISOLATE_RELATIVE_AREA_FRAC of the single largest one is
    kept and their bounding boxes unioned, instead of keeping only the biggest.

    That size rule has its own gap, found on a real listing: a small ANCILLARY
    structure (a storage shed, a garage) connected to the main house only by a
    thin property line is real floor area, but is nowhere near 30% of the main
    house's own size -- the exact same rule that correctly excludes a small
    marketing element excludes a small real room too (confirmed: a "VAR."
    storage room, counted in the listing's own size_m2, was lost entirely).
    Size alone can't tell these apart; a real room-code label can. Before
    discarding a component below ISOLATE_RELATIVE_AREA_FRAC (but still bigger
    than ISOLATE_PROTECT_MIN_AREA_FRAC -- not worth the check for a speck),
    this runs a TARGETED OCR pass on just that component's own small crop
    (padded by ISOLATE_PROTECT_CROP_PAD_FRAC) -- cheap even though it runs on
    every image, since the common case (no excluded components, or ones that
    fail this check) pays for nothing beyond that one small crop. A
    KNOWN_LABELS match there overrides the size rule and keeps it.

    The reverse check (vet anything kept BY the size rule, drop it if it has no KNOWN_LABELS match)
    was also tried, to catch a decorative/informational diagram -- a site map, a legend -- large
    enough to clear ISOLATE_RELATIVE_AREA_FRAC on its own. REVERTED: on real data it rejected
    genuine floor-plan sections more often than it caught junk, whenever a section's own room text
    didn't happen to exactly match the curated KNOWN_LABELS set (confirmed on two different real
    listings: "makuutila", a legitimate Finnish term for a sleeping area, isn't in it). See
    segment_rooms' own docstring for this project's established pattern of blanket/global
    heuristics failing this same way; the full account of what was tried and why it was reverted is
    at this function's own git history / project memory, not repeated here.

    Falls back to the whole image if nothing found, or if the found region
    already covers effectively the whole frame (nothing worth cropping).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[detect_text_mask(image)] = 0

    dilate_px = max(ISOLATE_DILATE_FLOOR_PX, int(ISOLATE_DILATE_FRAC * max(w, h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    dilated = cv2.dilate(binary, kernel)
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        return image

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = areas.max()
    if largest_area < ISOLATE_MIN_AREA_FRAC * h * w:
        return image
    keep = {1 + i for i, a in enumerate(areas) if a >= ISOLATE_RELATIVE_AREA_FRAC * largest_area}

    def has_known_label(i: int) -> bool:
        x, y, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = int(ISOLATE_PROTECT_CROP_PAD_FRAC * max(cw, ch))
        x0c, y0c = max(0, x - pad), max(0, y - pad)
        x1c, y1c = min(w, x + cw + pad), min(h, y + ch + pad)
        crop = image[y0c:y1c, x0c:x1c]
        if crop.size == 0:
            return False
        return any(lab["text"].upper() in KNOWN_LABELS for lab in read_text_regions(crop))

    # A component too small for the relative-size rule might still be a real (if small) room, not
    # marketing junk -- see this function's own docstring. Only worth checking (and only costs
    # anything) for the rare component that's both excluded AND not tiny-speck-small.
    protect_min = ISOLATE_PROTECT_MIN_AREA_FRAC * h * w
    for i in range(1, n):
        if i in keep or stats[i, cv2.CC_STAT_AREA] < protect_min:
            continue
        if has_known_label(i):
            keep.add(i)

    # TRIED AND REVERTED (2026-10-08): vetting every size-qualified component the same way --
    # require a KNOWN_LABELS match or drop it -- to catch a decorative/informational diagram large
    # enough to clear ISOLATE_RELATIVE_AREA_FRAC on its own (found on oikotie_24455117: a small
    # site/location map with building-block letters and a street name, no real room). Shipped once,
    # then reverted after it was shown to cause WORSE harm than it prevented: a direct per-image
    # comparison across a 226-image overlap (old vs vetted pipeline, same images) found 54 images
    # with FEWER named rooms than before, several losing 3-5 at once -- confirmed on two of them
    # (oikotie_24679139, oikotie_24654958) that the discarded component was a REAL part of the
    # floor plan whose text just didn't happen to exactly match KNOWN_LABELS (e.g. "makuutila", a
    # legitimate Finnish term for a sleeping area, and "SK/PY", a real but uncurated code) -- the
    # curated whitelist is too narrow to reliably tell "real room, different naming convention"
    # from "decorative junk" this way, and that gap bites often enough (the kept-by-size path
    # covers every multi-section plan, not just the rare site-map case) that it's a net loss, not a
    # net gain, on real data. A site map slipping through has bounded, known downstream harm (its
    # ink is never traced as a NAMED room -- nothing in its text matches a real code either, so
    # Pass 1-3 simply never name anything there); a real floor-plan section being discarded loses
    # square footage and room counts outright, with no downstream recovery possible. Between those
    # two failure modes, the user's own explicit priority (accuracy -- real content must not be
    # lost) picks the first. Not reattempted with a looser text criterion either: the obvious loosen
    # ("any OCR'd text at all", not specifically a KNOWN_LABELS match) was checked against the
    # original site-map case and found to still accept it (its own street-name text alone clears
    # that bar), so it wouldn't have actually re-fixed what this was built for -- the whack-a-mole
    # pattern this project has hit with every other global/blanket heuristic in this file.

    mask = np.isin(labels_img, list(keep)) & (binary > 0)  # bbox from the ORIGINAL ink, not the dilated blobs
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    if (x1 - x0) >= 0.97 * w and (y1 - y0) >= 0.97 * h:
        return image  # already effectively the whole frame -- nothing to crop
    mx, my = (x1 - x0) * ISOLATE_MARGIN_FRAC, (y1 - y0) * ISOLATE_MARGIN_FRAC
    return image[int(max(0, y0 - my)):int(min(h, y1 + my)), int(max(0, x0 - mx)):int(min(w, x1 + mx))]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_opening_model()

    stats_is_new = not STATS_CSV.exists()
    rooms_is_new = not ROOMS_CSV.exists()

    # Resume support: a long run across thousands of images is restarted periodically on purpose
    # (see MAX_IMAGES_PER_RUN below -- bounds memory growth by giving each chunk a fresh process),
    # so a restart must pick up where the last one left off, not redo already-completed images.
    # Only "ok" rows count as done -- a "failed" row is retried, in case the failure was transient
    # (e.g. memory pressure from a previous chunk) rather than something wrong with that image
    # specifically.
    already_done = set()
    if STATS_CSV.exists():
        with open(STATS_CSV, newline="") as f:
            already_done = {row["image_name"] for row in csv.DictReader(f) if row.get("status") == "ok"}

    # Bounds how many NEW images one process handles before exiting cleanly -- confirmed necessary
    # on a real overnight run: a single process left running across the whole ~3800-image dataset
    # grew to ~4GB resident on an 8GB machine, which pushed the system into heavy swapping (vm_stat
    # showed hundreds of millions of page-outs) and slowed per-image time from ~13s to 45-120s.
    # Unset/0 = no limit (the old, single-process-for-everything behavior).
    max_images = int(os.environ.get("MAX_IMAGES_PER_RUN", "0")) or None

    stats_f = open(STATS_CSV, "a", newline="")
    rooms_f = open(ROOMS_CSV, "a", newline="")
    stats_w = csv.DictWriter(stats_f, fieldnames=STATS_FIELDS)
    rooms_w = csv.DictWriter(rooms_f, fieldnames=ROOM_FIELDS)
    if stats_is_new:
        stats_w.writeheader()
    if rooms_is_new:
        rooms_w.writeheader()

    target = sys.argv[1] if len(sys.argv) > 1 else None
    # Temporary triage filter: some Oikotie images are genuinely huge (confirmed: one real
    # 7308x5167 / 37.8-megapixel photo took 340s and +1.3GB just for read_text_regions/OCR alone,
    # with detect_openings and the rest adding more on top) -- real-data correlation across the
    # first ~450 processed images (width_px*height_px vs processing_time_sec in run_stats.csv)
    # confirmed size predicts slowness directionally, if noisily (OCR label count and detected-
    # opening count matter too, so even some small images run long): under ~1 megapixel, >1/3 of
    # images finish under 15s; above ~2 megapixels, well under 2% do, with means in the
    # 50-366s range and worst cases over 1400s. MAX_IMAGE_PX, when set, skips (not fails) anything
    # over that pixel count -- logged with its own "skipped_large" status (not "ok", so a later
    # run with the cap lifted or removed will still pick these up) so the easy majority can be
    # powered through now instead of blocking on the slow tail.
    max_image_px = int(os.environ.get("MAX_IMAGE_PX", "0")) or None

    n_ok = n_failed = n_skipped = n_skipped_large = 0
    # FP_IMAGE_LIST: a text file of image names (one per line) to restrict a run to, e.g. a fixed
    # evaluation set; images not in it are skipped without being counted.
    image_list = os.environ.get("FP_IMAGE_LIST")
    wanted = set(Path(image_list).read_text().split()) if image_list else None
    for path, image in _select_images(target):
        if wanted is not None and path.name not in wanted:
            continue
        if path.name in already_done:
            n_skipped += 1
            continue
        if max_image_px and image.shape[0] * image.shape[1] > max_image_px:
            n_skipped_large += 1
            stats_w.writerow({
                "image_name": path.name, "listing_id": _OIKOTIE_NAME_RE.match(path.name).group(1)
                if _OIKOTIE_NAME_RE.match(path.name) else "",
                "has_metadata": "", "listed_size_m2": "", "listed_rooms": "",
                "listed_room_configuration": "", "listed_build_year": "",
                "status": "skipped_large", "error_message": f"{image.shape[1]}x{image.shape[0]} "
                f"= {image.shape[0]*image.shape[1]} px > MAX_IMAGE_PX={max_image_px}",
                "width_px": image.shape[1], "height_px": image.shape[0],
                "num_wall_lines": "", "num_text_labels": "", "num_doors": "", "num_windows": "",
                "num_rooms_total": "", "num_rooms_shown": "", "num_rooms_named": "",
                "num_rooms_gap_closed": "", "num_rooms_recovered": "", "total_area_px2": "",
                "calibration_scale_m2_per_px2": "", "processing_time_sec": 0,
            })
            stats_f.flush()
            continue
        if max_images and (n_ok + n_failed) >= max_images:
            log.info(f"Reached MAX_IMAGES_PER_RUN={max_images} new images this invocation -- "
                     f"stopping cleanly (rerun the same command to resume with the rest).")
            break
        # A single bad/corrupt/edge-case image must not take down an unattended run across
        # thousands of images -- log it and move on, instead of the whole process dying partway
        # through a run meant to finish unattended (confirmed necessary: this loop body had no
        # error handling at all before a 3700+-image overnight run made that a real risk, not a
        # theoretical one).
        t0 = time.time()
        m = _OIKOTIE_NAME_RE.match(path.name)
        listing_id = m.group(1) if m else ""
        try:
            image = crop_to_document(image)  # discard any background around the floor plan (desk, bezel, hand)
            image = isolate_plan(image)      # discard the marketing composite around it -- BEFORE any OCR/tracing
            height, width = image.shape[:2]
            dxf_path = OUT_DIR / f"{path.stem}.dxf"
            meta = lookup_listing_metadata(path.name)                # scraped size_m2/rooms/address, not OCR

            labels = read_text_regions(image)                       # OCR the plan's text
            if meta:
                # A generic street name/number has no lexical marker distinguishing it from a real
                # room label (unlike a HUOM note or a company name) -- but we already know THIS
                # listing's own address, so strip any OCR read that's substantially a fragment of it.
                labels = strip_known_address(labels, meta.get("address"))
            polygons = trace_contours(image, text_regions=labels)   # trace walls/furniture, text masked out
            # Drawn as single lines: every stroke thinned to one centreline, thick walls included, with
            # light-grey lines kept (see trace_single_lines). Feeds segment_rooms too when
            # ROOM_WALLS == "lines"; otherwise rooms still come from `polygons` above.
            wall_lines = trace_single_lines(image, text_regions=labels)
            export_contours_dxf(wall_lines, str(dxf_path), height=height, closed=False)

            doc = ezdxf.readfile(str(dxf_path))
            if "Text" not in {lyr.dxf.name for lyr in doc.layers}:
                doc.layers.add("Text", color=TEXT_LAYER_COLOR)      # pre-create so add_text_labels_to_dxf reuses it
            doc.saveas(str(dxf_path))

            add_text_labels_to_dxf(str(dxf_path), labels, height=height)

            detections = detect_openings(image, model)               # Faster R-CNN door/window detection
            export_openings_dxf(str(dxf_path), detections, height=height)

            rooms, rooms_drawn = [], 0
            if meta and meta.get("size_m2") is not None:
                # The listing's one size_m2 figure, distributed across the traced room polygons
                # instead of printed as a flat total -- doors block the flood-fill so it can't
                # leak from one room into the next through a doorway gap.
                rooms = segment_rooms(polygons, detections, (height, width), labels,
                                      wall_ink=room_wall_ink(image, labels, wall_lines))
                rooms_drawn = add_room_areas_to_dxf(str(dxf_path), rooms, meta["size_m2"], height=height)

            if meta:
                add_metadata_to_dxf(str(dxf_path), meta, height=height)

            add_legend(str(dxf_path), width=width, height=height)     # must run last -- it reads back which layers exist

            n_doors = sum(1 for d in detections if d["class"] == "Door")
            n_windows = sum(1 for d in detections if d["class"] == "Window")
            room_note = (f", {rooms_drawn}/{len(rooms)} rooms shown (>= {AREA_MIN_DISPLAY_M2:g} m², "
                         f"of {_fmt_num(meta['size_m2'])} m² listed)") if rooms else ""
            log.info(f"{path.name}: {len(wall_lines)} wall lines, "
                     f"{len(labels)} text labels, {n_doors} doors, {n_windows} windows{room_note} -> {dxf_path.name}")

            total_px2 = sum(r["area_px2"] for r in rooms if counts_in_living_area(r))
            scale = meta["size_m2"] / total_px2 if (meta and rooms and total_px2 > 0) else None
            stats_w.writerow({
                "image_name": path.name, "listing_id": listing_id, "has_metadata": bool(meta),
                "listed_size_m2": meta.get("size_m2") if meta else None,
                "listed_rooms": meta.get("rooms") if meta else None,
                "listed_room_configuration": meta.get("room_configuration") if meta else None,
                "listed_build_year": meta.get("build_year") if meta else None,
                "status": "ok", "error_message": "",
                "width_px": width, "height_px": height,
                "num_wall_lines": len(wall_lines), "num_text_labels": len(labels),
                "num_doors": n_doors, "num_windows": n_windows,
                "num_rooms_total": len(rooms), "num_rooms_shown": rooms_drawn,
                "num_rooms_named": sum(1 for r in rooms if r.get("text")),
                "num_rooms_gap_closed": sum(1 for r in rooms if r.get("gap_closed")),
                "num_rooms_recovered": sum(1 for r in rooms if r.get("recovered")),
                "total_area_px2": total_px2, "calibration_scale_m2_per_px2": scale,
                "processing_time_sec": round(time.time() - t0, 2),
            })
            for r in rooms:
                area_m2 = r["area_px2"] * scale if scale else None
                rooms_w.writerow({
                    "image_name": path.name, "listing_id": listing_id,
                    "room_text": r.get("text") or ("Other" if r.get("recovered") else ""),
                    "area_m2": round(area_m2, 2) if area_m2 is not None else None,
                    "area_px2": r["area_px2"], "shown": bool(area_m2 and area_m2 >= AREA_MIN_DISPLAY_M2),
                    "recovered": bool(r.get("recovered")), "gap_closed": bool(r.get("gap_closed")),
                })
            stats_f.flush()
            rooms_f.flush()
            n_ok += 1
        except Exception as exc:
            n_failed += 1
            log.exception(f"{path.name}: FAILED, skipping")
            stats_w.writerow({
                "image_name": path.name, "listing_id": listing_id, "has_metadata": "",
                "listed_size_m2": "", "listed_rooms": "", "listed_room_configuration": "",
                "listed_build_year": "", "status": "failed", "error_message": f"{type(exc).__name__}: {exc}",
                "width_px": "", "height_px": "", "num_wall_lines": "", "num_text_labels": "",
                "num_doors": "", "num_windows": "", "num_rooms_total": "", "num_rooms_shown": "",
                "num_rooms_named": "", "num_rooms_gap_closed": "", "num_rooms_recovered": "",
                "total_area_px2": "", "calibration_scale_m2_per_px2": "",
                "processing_time_sec": round(time.time() - t0, 2),
            })
            stats_f.flush()

    stats_f.close()
    rooms_f.close()
    log.info(f"DONE (this invocation): {n_ok} succeeded, {n_failed} failed, {n_skipped} already-done "
             f"skipped, {n_skipped_large} too-large skipped (MAX_IMAGE_PX={max_image_px}), "
             f"{n_ok + n_failed} newly processed -> {STATS_CSV}, {ROOMS_CSV}")
