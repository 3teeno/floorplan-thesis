import logging
import os
import re
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from skimage.morphology import skeletonize

try:
    import pytesseract  # free, local OCR; needs the `tesseract` binary on PATH
except ImportError:
    pytesseract = None

from floorplan_loader import crop_to_document, list_floorplan_images, load_floorplan_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LAYER_NAME = "Trace"


# No real-world scale/DPI is known anywhere in this pipeline (images are arbitrary-
# resolution screenshots), so these tolerances are in pixels, not mm — same ballpark
# as GAP_TOL_PX above, tune to taste.

# guards against _connected_chains transitively linking many unrelated fragments (A-B-C-D close pairwise,
# but A and D far apart) into one bogus "chain" spanning most of the drawing
# an algebraic circle fit can find a huge, spurious radius that looks locally straight for scattered/noisy
# points; a real small arc's radius should roughly match the visual size of its own point cluster
# span much more than that are picking up contamination points near the arc (the door leaf line, hinge
# marks) that get pulled in by proximity but aren't actually part of the curve, inflating the fitted span
# into a near-full circle

# Real OCR (pytesseract) turned out unreliable here: it recognizes *characters*, and this
# dataset's text renders mirrored/backwards (see the earlier mirrored-room-label discussion),
# which tesseract mostly fails to read at all. This heuristic instead recognizes the *shape*
# of text — small, similarly-sized glyph blobs lining up in rows — which doesn't care whether
# the characters are readable, mirrored, or in any particular orientation.
TEXT_GLYPH_MIN_HEIGHT_PX = 6
TEXT_GLYPH_MAX_HEIGHT_PX = 40
TEXT_GLYPH_MAX_WIDTH_PX = 40
TEXT_ROW_Y_TOLERANCE_PX = 10  # glyphs in the same text row have centroids within this many px of the row average
TEXT_ROW_GAP_PX = 25  # max horizontal gap between glyphs to still count as the same word/line
TEXT_MIN_ROW_GLYPHS = 2  # need at least this many aligned small glyphs to call it text, not a stray symbol —
# 2 catches short 2-letter room labels ("WC", "MH") that 3 was missing, at the cost of occasionally
# masking a real pair of small marks (e.g. two nearby hinge dots) that happen to align similarly
TEXT_ORPHAN_ADOPT_GAP_RATIO = 1.2  # after rows are built, adopt a leftover lone glyph-sized blob into
# an accepted row if it shares that row's baseline and sits within this multiple of the row height of
# its edge — catches split floor numbers ("KRS" + a detached "1") that have no row-mate of their own


# Contour-based tracing: walls in these drawings are solid filled bars, not thin outlined
# strokes, so thresholding + cv2.findContours traces each wall's actual boundary as ONE
# closed polygon directly; a traced contour is connected by construction. This is the right tool for rectilinear architectural drawings; a general
# bitmap-to-vector tracer (Potrace and similar) instead fits smooth splines through
# contours, which is right for logos/photos but would round off every 90 deg wall corner.
CONTOUR_MIN_AREA_PX = 12.0  # drop contours smaller than this — noise specks, not real shapes
CONTOUR_APPROX_EPSILON_PX = 2.0  # RDP simplification tolerance for cv2.approxPolyDP
CONTOUR_AXIS_SNAP_TOLERANCE_DEG = 5.0  # same idea as AXIS_SNAP_TOLERANCE_DEG, applied per polygon edge

INK_MODE = os.environ.get("FP_INK_MODE", "otsu")  # "otsu" (default) or "paper" (see _binarize_ink)
INK_PAPER_OFFSET = 40  # "paper" mode: a pixel this much darker than the paper level counts as ink
TEXT_MASK_MODE = os.environ.get("FP_TEXT_MASK_MODE", "strokes")  # "strokes" or "box" (old: blank whole box)
TEXT_MASK_INSIDE_FRAC = 0.9  # "strokes": erase an ink component with at least this share inside text boxes


def _binarize_ink(image: np.ndarray, text_mask: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Wall/furniture ink as a 0/255 mask (255 = ink), text already erased -- the shared first
    step of trace_contours and extract_wall_skeleton. Plain OTSU.

    TRIED AND REVERTED (2026-10-08): recovering a faint/antialiased wall run OTSU's single global
    threshold misses (confirmed on a real listing, oikotie_24721956: a stretch of exterior wall
    measured ~100-230, light gray, against an OTSU cutoff of 147, so it never registered as ink,
    merging several rooms into one) by adding cv2.adaptiveThreshold's own extra ink on top,
    filtered down to components shaped like a wall run (elongated AND long in absolute terms, not
    just elongated -- a short hatching stroke is just as "elongated" as a real wall). Genuinely
    recovered the missing wall, confirmed by direct pixel inspection and by visual inspection of
    the resulting render -- but also confirmed, on this project's own cleanest, most-validated
    test image (oikotie_17380049), to spuriously bridge two previously-separate real wall blobs
    into one via trace_contours' raw cv2.findContours connectivity, merging two real rooms into
    one 69k px^2 "ET". The shape/length filter on each ACCEPTED extra component isn't enough to
    prevent this: a single short bridge that individually passes the filter can still touch two
    separate base blobs and permanently fuse them at the findContours level, before any filtering
    of the RESULT is even possible -- confirmed this persisted after tightening the filter twice.
    extract_wall_skeleton's own rendering looked clean throughout this because it has an
    additional, separate safety net (branch-length pruning after skeletonizing) that
    trace_contours/segment_rooms' own path does not share -- validating only through that
    rendering masked the real regression for a full iteration.
    Reverted to plain OTSU; the missing-wall-run case is an open, unresolved limitation again (see
    segment_rooms' own docstring's prior TRIED AND REJECTED entries for the same class of
    problem). A real fix would need to verify a candidate extra component's effect on actual ROOM
    separation (e.g. by checking it doesn't merge two otherwise-separate base ink components) as
    part of accepting it, not just its own shape -- not attempted, a materially bigger change than
    a threshold adjustment.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (mode or INK_MODE) == "paper":
        # Anything clearly darker than the paper is ink -- keeps the light-grey lines (door swings,
        # fixtures, windows) that Otsu drops when a plan also has solid black walls.
        paper = np.percentile(gray, 90)
        binary = np.where(gray < paper - INK_PAPER_OFFSET, 255, 0).astype(np.uint8)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if TEXT_MASK_MODE == "box":
        binary[text_mask] = 0
        return binary
    # Erase only ink that lies (almost) entirely inside the text boxes -- the letters themselves.
    # A stroke that continues outside a box is a wall or line the text was printed on or next to,
    # and stays whole; blanking the whole box cut such walls and let rooms merge.
    n, cc = cv2.connectedComponents(binary, connectivity=8)
    inside = np.bincount(cc[text_mask], minlength=n)
    total = np.bincount(cc.ravel(), minlength=n)
    erase = inside >= TEXT_MASK_INSIDE_FRAC * np.maximum(total, 1)
    erase[0] = False
    binary[erase[cc]] = 0
    return binary


def add_text_labels_to_dxf(dxf_path: str, labels: list[dict], height: int,
                           layer: str = "Text", label_height: float = 12.0) -> None:
    """Add each OCR'd label (from read_text_regions) to an existing DXF as a real
    TEXT entity at its position, all on one `layer` — the plan keeps its text as
    text, separate from the traced wall/shape geometry.

    `labels`: [{"text", "x", "y"}], coords in the same image space as the geometry.
    y is flipped (`height - y`) like every other export here (image y grows down,
    CAD y grows up); `height` is the pixel height of that same image.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    if layer not in {lyr.dxf.name for lyr in doc.layers}:
        doc.layers.add(layer)

    for lbl in labels:
        if not lbl.get("text"):
            continue
        t = msp.add_text(lbl["text"], dxfattribs={"layer": layer, "height": label_height})
        t.set_placement((float(lbl["x"]), height - float(lbl["y"])))

    doc.saveas(dxf_path)


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    """True if axis-aligned boxes (x0, y0, x1, y1) intersect at all."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def detect_text_regions(image: np.ndarray) -> list[dict]:
    """Bounding boxes of text — rows of small, similarly-sized glyph blobs —
    found by *shape*, not OCR, so it works regardless of orientation, mirroring,
    or whether the text is real language. Each region:
    ``{"bbox": (x0, y0, x1, y1), "cx": int, "cy": int}``.

    A row needs TEXT_MIN_ROW_GLYPHS aligned neighbours; a lone glyph that has no
    row-mates is picked up only by the orphan-adoption pass below, when it sits
    right next to an accepted row (a floor number beside "KRS").
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    glyphs = []
    for i in range(1, num_labels):  # label 0 is background
        x, y, w, h, _ = stats[i]
        if TEXT_GLYPH_MIN_HEIGHT_PX <= h <= TEXT_GLYPH_MAX_HEIGHT_PX and w <= TEXT_GLYPH_MAX_WIDTH_PX:
            glyphs.append((x, y, w, h, centroids[i][1]))
    glyphs.sort(key=lambda g: g[4])  # by centroid y

    regions = []
    used = [False] * len(glyphs)
    for i, (_, _, _, _, cy) in enumerate(glyphs):
        if used[i]:
            continue
        row = [i]
        row_cy_sum = cy  # running average, not a fixed anchor — per-letter baseline drift within a
        # single word (different letters sit at slightly different visual centers) can otherwise
        # exceed the tolerance by the 3rd/4th glyph even though each is close to its neighbor
        for j in range(i + 1, len(glyphs)):
            if used[j]:
                continue
            xj, _, wj, _, cyj = glyphs[j]
            if abs(cyj - row_cy_sum / len(row)) > TEXT_ROW_Y_TOLERANCE_PX:
                continue
            close = any(xj - (glyphs[k][0] + glyphs[k][2]) <= TEXT_ROW_GAP_PX and
                        (glyphs[k][0] - (xj + wj)) <= TEXT_ROW_GAP_PX for k in row)
            if close:
                row.append(j)
                row_cy_sum += cyj

        if len(row) >= TEXT_MIN_ROW_GLYPHS:
            x0 = min(glyphs[k][0] for k in row)
            y0 = min(glyphs[k][1] for k in row)
            x1 = max(glyphs[k][0] + glyphs[k][2] for k in row)
            y1 = max(glyphs[k][1] + glyphs[k][3] for k in row)
            regions.append({"bbox": (int(x0), int(y0), int(x1), int(y1)),
                            "cx": int((x0 + x1) // 2), "cy": int((y0 + y1) // 2)})
            for k in row:
                used[k] = True

    # A lone floor number ("1" next to "KRS"), a single detached digit or letter,
    # never forms a row of its own, so the loop above drops it and it gets traced
    # as geometry. Adopt any leftover blob that overlaps an accepted row's
    # baseline and sits within ~one row-height of its edge. Looser height cap
    # than the row-forming glyph filter (a bare digit runs a touch taller than a
    # lowercase letter); adjacency is tested against each row's ORIGINAL extent
    # so adoptions can't snowball along the plan.
    orig = [r["bbox"] for r in regions]
    tall_cap = int(TEXT_GLYPH_MAX_HEIGHT_PX * 1.6)
    for i in range(1, num_labels):
        cx0, cy0, cw, ch, _ = stats[i]
        if not (TEXT_GLYPH_MIN_HEIGHT_PX <= ch <= tall_cap and cw <= TEXT_GLYPH_MAX_WIDTH_PX):
            continue
        cx1, cy1 = cx0 + cw, cy0 + ch
        for ridx, (rx0, ry0, rx1, ry1) in enumerate(orig):
            rh = ry1 - ry0
            if cy0 > ry1 or cy1 < ry0 or ch > 1.6 * rh:  # not this baseline, or too tall for it
                continue
            if cx0 >= rx0 - 1 and cx1 <= rx1 + 1:
                break  # already inside this row's box
            if min(abs(cx0 - rx1), abs(cx1 - rx0)) <= TEXT_ORPHAN_ADOPT_GAP_RATIO * rh:
                bx0, by0, bx1, by1 = regions[ridx]["bbox"]
                nb = (int(min(bx0, cx0)), int(min(by0, cy0)),
                      int(max(bx1, cx1)), int(max(by1, cy1)))
                regions[ridx]["bbox"] = nb
                regions[ridx]["cx"], regions[ridx]["cy"] = (nb[0] + nb[2]) // 2, (nb[1] + nb[3]) // 2
                break

    return regions


def detect_text_mask(image: np.ndarray, regions: list[dict] | None = None) -> np.ndarray:
    """Boolean mask (True = text) — text-region boxes rasterised, for callers
    that only need to *exclude* text pixels from geometry tracing.

    `regions` defaults to the shape heuristic (detect_text_regions). Pass an
    explicit list (each needs a "bbox") to rasterise those instead — e.g. the
    OCR output of read_text_regions(), which catches isolated numbers and
    rotated dimension strings the shape heuristic drops (a lone digit blob has
    no row-mates; a multi-digit string is one over-wide component).
    read_text_regions() keeps the text as text instead."""
    if regions is None:
        regions = detect_text_regions(image)
    mask = np.zeros(image.shape[:2], dtype=bool)
    pad = 2
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    return mask


def _geometry_text_mask(image: np.ndarray, text_regions: list[dict] | None) -> np.ndarray:
    """Text-pixel mask for the geometry passes (trace_contours / extract_wall_skeleton). The shape heuristic alone misses isolated numbers and
    rotated dimension strings, so those get traced as polygons; unioning in the
    OCR boxes from read_text_regions() — which do catch numbers — plugs that
    gap, while the heuristic still covers mirrored/gibberish text OCR rejects.

    FP_NO_TEXT_MASK=1 turns the mask off (experiment: text is traced as geometry)."""
    if os.environ.get("FP_NO_TEXT_MASK") == "1":
        return np.zeros(image.shape[:2], dtype=bool)
    mask = detect_text_mask(image)
    if text_regions:
        mask |= detect_text_mask(image, regions=text_regions)
    return mask


# Finnish floor-plan label abbreviations, uppercased -- an OCR result matching one
# of these is trusted even when it's short and would otherwise fail the noise filter.
KNOWN_LABELS = {
    "MH", "MH1", "MH2", "MH3", "OH", "OH/K", "K", "KT", "KK", "RT", "ET", "AULA",
    "KPH", "KH", "PH", "WC", "S", "SAUNA", "VH", "KHH", "VAR", "AK", "PARVEKE",
    "TERASSI", "LASITETTU PARVEKE", "TEKN", "TK", "LEV",
    "H", "H1", "H2", "H3", "AH", "HO", "TH",  # H = huone (room), AH = asuinhuone, TH = tyohuone
    # the same rooms spelled out, as some (often older) plans print them
    "HUONE", "MAKUUHUONE", "OLOHUONE", "TAKKAHUONE", "TYÖHUONE", "KEITTIÖ", "KYLPYHUONE",
    "PESUHUONE", "VAATEHUONE", "ETEINEN", "VARASTO", "KODINHOITOHUONE",
}
OCR_UPSCALE = 3          # small labels OCR far better enlarged
OCR_MIN_ALNUM = 2        # drop results with fewer real characters than this (unless KNOWN_LABELS)
OCR_MIN_SCORE = 0.5      # PaddleOCR per-line confidence floor

_paddle_ocr = None


def _get_paddle():
    """Lazily build one PaddleOCR reader and reuse it. First call downloads a few
    small models to ~/.paddlex (needs network once, then cached). Set env
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True to skip the connectivity probe."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang="en", use_textline_orientation=True)
    return _paddle_ocr


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("-_/\\|.,")


# Words that flag a marketing/title-block caption rather than a room label or
# dimension -- a real estate agency name, a legal entity suffix, an address
# marker. Checked as whole words (case-insensitive), not substrings, so it
# can't misfire on a room code that happens to contain these letters.
_MARKETING_WORDS = {
    "OY", "OYJ", "LKV", "AB", "LTD", "KY", "ARKKITEHDIT", "ARKKITEHTITOIMISTO",
    "KIINTEISTONVALITYS", "KIINTEISTOVALITYS", "ASUNNOT", "REALTY", "ASUMISOIKEUS",
}
# Long, unambiguous prefixes of common drawing-metadata/firm words -- checked as a plain
# substring (safe here, unlike _MARKETING_WORDS' short tokens, since nothing this long is
# a real room code) specifically because a crop edge can truncate the OCR'd string mid-word
# ("Mittakaava" -> "Mittaka", "Asumisoikeus" -> "Asumisoike") before it reaches a whole word.
_MARKETING_PREFIXES = ("MITTAKA", "ASUMISOIKE", "ARKKITEHT", "KIINTEIST")
_POSTAL_CODE_RE = re.compile(r"\b\d{5}\b")  # a Finnish postal code -- never appears in a room/floor/scale label


def _is_marketing_caption(text: str) -> bool:
    """True for a title-block/marketing caption -- a HUOM(.) note, an address
    line, or a firm's name/credit -- as opposed to a genuine room label or
    dimension. These slip past a plain size/alnum filter because PaddleOCR
    reads a wrapped multi-line note or address as several separately short
    lines, each individually well within a room label's normal length; a
    length cap alone can't catch them; content-based rules can."""
    upper = text.upper()
    if upper.startswith("HUOM"):  # Finnish drawing-note marker ("HUOM." = "NOTE") -- never a room label
        return True
    if _POSTAL_CODE_RE.search(text):  # an address line's zip code
        return True
    if "&" in text:  # "Helin & Co", "X & Y Arkkitehdit" -- a firm name joiner, never a room label
        return True
    if any(p in upper for p in _MARKETING_PREFIXES):
        return True
    words = set(re.findall(r"[A-ZÄÖÅ]+", upper))
    return bool(words & _MARKETING_WORDS)


def _keep_label(text: str) -> bool:
    alnum = sum(c.isalnum() for c in text)
    if not text or alnum < len(text) / 2:
        return False
    if _is_marketing_caption(text):
        return False
    return (text.upper() in KNOWN_LABELS
            or alnum >= 3
            or (alnum == OCR_MIN_ALNUM
                and (text.isdigit() or (text.isalpha() and text.isupper()))))


def _core(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalnum())


# A floor marker ("1. KRS" / "2KRS" / "KRS 3"), a drawing scale ("1:100"), or an
# area ("84,0 m2" / "84 M2"). Used to whitelist the noisy cropped-tesseract
# second-chance pass down to only high-value, unambiguous reads.
_FLOORPLAN_LABEL_RE = re.compile(
    r"\d{1,2}\s*\.?\s*KRS|KRS\s*\d{1,2}|KERROS|\d{1,3}\s*[:/]\s*\d{2,4}|\d{1,4}([.,]\d+)?\s*M[²2]\b",
    re.IGNORECASE,
)


def _is_floorplan_label(text: str) -> bool:
    return bool(text) and (text.upper() in KNOWN_LABELS
                           or _FLOORPLAN_LABEL_RE.search(text) is not None)


def _dedup_labels(labels: list[dict]) -> list[dict]:
    """Drop near-duplicate labels. The whole-image PaddleOCR pass and the
    cropped tesseract second-chance pass frequently read the same physical
    label twice — a few dozen to ~200 px apart, since their boxes don't line up
    exactly (or one has "KRS" where the other has the fuller "1.KRS"). Two
    labels collapse when one's alphanumeric core contains the other's and their
    boxes either overlap or their centres are within ~4 box-heights; the longer
    read wins. A room code that genuinely recurs (two "MH" bedrooms) sits much
    further apart than that and is kept."""
    kept: list[dict] = []
    for lab in sorted(labels, key=lambda l: -len(_core(l["text"]))):
        lc = _core(lab["text"])
        dup = False
        for k in kept:
            kc = _core(k["text"])
            if kc.find(lc) < 0 and lc.find(kc) < 0:
                continue
            h = max(lab["bbox"][3] - lab["bbox"][1], k["bbox"][3] - k["bbox"][1])
            if (_boxes_overlap(lab["bbox"], k["bbox"])
                    or np.hypot(lab["x"] - k["x"], lab["y"] - k["y"]) <= max(4 * h, 80)):
                dup = True
                break
        if not dup:
            kept.append(lab)
    return kept


def _read_paddle(image: np.ndarray) -> list[dict]:
    """PaddleOCR finds and reads text lines itself -- one pass over the whole
    image, its own boxes + orientation handling (best for rotated plan labels)."""
    out = []
    for res in _get_paddle().predict(image):
        d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
        texts = d.get("rec_texts", []) or []
        polys = d.get("rec_polys", d.get("dt_polys", [])) or []
        scores = d.get("rec_scores", [1.0] * len(texts)) or []
        for text, poly, score in zip(texts, polys, scores):
            if score < OCR_MIN_SCORE:
                continue
            text = _clean(text)
            if not _keep_label(text):
                continue
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            out.append({"text": text, "bbox": (int(x0), int(y0), int(x1), int(y1)),
                        "x": int((x0 + x1) / 2), "y": int((y0 + y1) / 2)})
    return out


def _read_tesseract(image: np.ndarray, regions: list[dict], try_mirrored: bool) -> list[dict]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed (and `tesseract` must be on PATH)")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    labels = []
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        pad = max(3, (y1 - y0) // 3)
        crop = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants = [crop] + ([cv2.flip(crop, 1), cv2.flip(crop, 0)] if try_mirrored else [])
        best = max((pytesseract.image_to_string(v, config="--psm 7").strip() for v in variants),
                   key=lambda s: sum(c.isalnum() for c in s))
        text = _clean(best)
        if _keep_label(text):
            labels.append({"text": text, "bbox": r["bbox"], "x": r["cx"], "y": r["cy"]})
    return labels


def read_text_regions(image: np.ndarray, regions: list[dict] | None = None,
                      engine: str = "auto", try_mirrored: bool = True) -> list[dict]:
    """Read the plan's text and return it *as text*, positioned:
    ``[{"text": str, "bbox": (x0,y0,x1,y1), "x": cx, "y": cy}]`` in `image`
    pixel coordinates. Meant to run alongside trace_contours so labels become
    TEXT entities in the DXF instead of being polygonised or discarded.

    engine: "paddle" (accurate, handles rotation, downloads models once),
    "tesseract" (lighter, per-region, weaker on small/rotated labels), or
    "auto" (paddle if importable, else tesseract). All local and free.

    The paddle path is hybrid: the whole-image PaddleOCR pass, then a fast
    cropped tesseract pass over the shape-detected text rows PaddleOCR missed
    (upscaled per box, so it copes with the small size), so a stray label like
    a "1.KRS" floor number is recovered as text instead of being left as
    polygons. Cropped PaddleOCR would be more accurate but is ~15 s per box.
    """
    if engine == "auto":
        try:
            import paddleocr  # noqa: F401
            engine = "paddle"
        except ImportError:
            engine = "tesseract"

    if engine == "paddle":
        labels = _read_paddle(image)
        missed = [r for r in detect_text_regions(image)
                  if not any(_boxes_overlap(r["bbox"], l["bbox"]) for l in labels)]
        if missed and pytesseract is not None:
            # Cropped tesseract on a "missed row" is unreliable: fixtures the
            # shape heuristic mis-grouped become hallucinated words, and mirrored
            # plan text (e.g. F1_original) reads as uppercase gibberish that a
            # plain is-uppercase test can't tell from a real label. So keep a
            # second-chance read only when it matches a known abbreviation or an
            # unambiguous pattern — a floor marker ("1.KRS") or a scale ("1:100")
            # — which is the high-value info worth recovering anyway.
            extra = [e for e in _read_tesseract(image, missed, try_mirrored=False)
                     if _is_floorplan_label(e["text"])]
            labels = _dedup_labels(labels + extra)
        return labels
    if regions is None:
        regions = detect_text_regions(image)
    return _read_tesseract(image, regions, try_mirrored)


def snap_polygon_to_axes(polygon: np.ndarray, tolerance_deg: float = CONTOUR_AXIS_SNAP_TOLERANCE_DEG,
                          closed: bool = True) -> np.ndarray:
    """Force each near-horizontal/near-vertical edge of a polygon/polyline to be
    *exactly* horizontal/vertical.

    A vertex is shared by two edges; for a genuine rectilinear shape those two
    edges are perpendicular (one wants to fix the vertex's y, the other its
    x), so fixing them in a single left-to-right pass doesn't fight itself.
    Genuinely diagonal edges are left untouched, same as the line-based version.

    closed=False treats `polygon` as an open chain (e.g. one wall-skeleton
    branch) instead of a closed ring -- no edge is checked between the last
    point and the first, since none exists.
    """
    pts = polygon.astype(float).copy()
    n = len(pts)
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        angle = np.degrees(np.arctan2(pts[j][1] - pts[i][1], pts[j][0] - pts[i][0])) % 180
        if angle <= tolerance_deg or angle >= 180 - tolerance_deg:
            avg_y = (pts[i][1] + pts[j][1]) / 2
            pts[i][1] = avg_y
            pts[j][1] = avg_y
        elif abs(angle - 90) <= tolerance_deg:
            avg_x = (pts[i][0] + pts[j][0]) / 2
            pts[i][0] = avg_x
            pts[j][0] = avg_x
    return pts


def trace_contours(image: np.ndarray, text_regions: list[dict] | None = None) -> list[np.ndarray]:
    """Every solid shape in the image, traced as a closed polygon directly from
    its filled boundary — walls, furniture, hatching — instead of independently
    detected line segments. Connected by construction (no gaps to close), and
    axis-snapped per edge afterward so real 90 deg corners land exactly on 90.

    Runs at the image's native resolution. Text is masked out first so glyph strokes aren't traced as shapes — pass
    `text_regions` (from read_text_regions) so OCR'd numbers/dimensions the
    shape heuristic misses are erased here instead of polygonised.

    Binarization is plain OTSU (see _binarize_ink's own docstring -- an adaptive-threshold
    recovery pass for faint wall ink was tried here and reverted).
    """
    text_mask = _geometry_text_mask(image, text_regions)
    binary = _binarize_ink(image, text_mask)

    raw_contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in raw_contours:
        if cv2.contourArea(contour) < CONTOUR_MIN_AREA_PX:
            continue
        simplified = cv2.approxPolyDP(contour, CONTOUR_APPROX_EPSILON_PX, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        polygons.append(snap_polygon_to_axes(simplified))
    return polygons


WALL_SKELETON_EPSILON_PX = 2.0  # RDP simplification tolerance per branch, same scale as CONTOUR_APPROX_EPSILON_PX
WALL_SKELETON_MIN_LEN_FRACTION = 0.02  # drop a skeleton branch shorter than this fraction of max(w, h) --
WALL_SKELETON_MIN_LEN_FLOOR = 10.0     # a stray junction spur, not a real wall run (floor for small sources)


def extract_wall_skeleton(image: np.ndarray, text_regions: list[dict] | None = None,
                          ink_mode: str | None = None, min_len: float | None = None) -> list[np.ndarray]:
    """Every wall/furniture ink blob reduced to its own 1px CENTERLINE, as open
    polylines -- unlike trace_contours, which traces each blob's own full outer
    boundary (both long edges of a wall's real thickness, plus its two end
    caps), so a straight wall renders as two parallel lines with a closing cap
    at each end, not the single line an architectural plan would normally draw.

    Same binarization/text-masking as trace_contours, but skimage.morphology.
    skeletonize (medial-axis thinning) first, before vectorizing -- this is
    NOT a per-polygon heuristic (e.g. "collapse anything elongated to its long
    axis"): a real wall network is a branching, often bent shape (a traced
    exterior-wall blob on one real test image had 74 vertices in its own
    outline), not a simple rectangle, and skeletonizing finds the true
    centerline of that whole shape, following every bend and junction,
    instead of flattening it to one straight segment.

    The skeleton is then walked into vector polylines: pixels with exactly 1
    neighbor (a branch endpoint) or 3+ (a junction) are "nodes"; every simple
    path of skeleton pixels between two nodes, with no other node in between,
    becomes one open polyline (cv2.approxPolyDP, NOT closed -- an open wall
    segment has no reason to loop back to its own start). A junction with
    several walls meeting at it becomes several polylines sharing that one
    point, not one shape -- which is what lets this stay simple: no attempt
    to reconstruct "rooms" or even trace order here, just the raw line graph.

    Two noise sources specific to this representation, neither caught by
    trace_contours' own CONTOUR_MIN_AREA_PX (that filters by blob AREA,
    after tracing full outlines -- not applicable here, before anything is
    traced) or by a whole-polygon minimum-size filter (tuned for a whole
    wall-NETWORK's own bounding box, not one skeleton BRANCH between two
    junctions, which is typically far smaller -- reusing it here dropped
    339 of 352 branches on a real test image, almost the entire wall
    network, not just noise):
      - A stray ink speck (anti-aliasing, a dimension tick mark) skeletonizes
        into its own tiny disconnected fragment -- dropped by removing any
        binary connected component smaller than CONTOUR_MIN_AREA_PX BEFORE
        skeletonizing, the same area floor trace_contours applies after.
      - A blobby wall corner/junction, once thinned, often sprouts a short
        spurious "spur" branch (a skeletonization artifact, not a real wall)
        -- dropped by WALL_SKELETON_MIN_LEN_FRACTION/FLOOR, measuring each
        branch's own pixel-path length (not its simplified vertex count, so
        a long but gently-curved branch isn't penalized for being long).
      Verified together on oikotie_17380049 (549x466): 352 raw branches -> 57
      after both filters, each one matching a real wall run on visual
      inspection, no stray specks or spurs left.

    Cosmetic/visual ONLY, by design: this is meant for how the Trace layer
    is DRAWN, not as a replacement for trace_contours' own polygons, which
    segment_rooms and every other area/room computation in floorplan_pipeline.py
    still use unchanged -- swapping the underlying wall representation for
    those would need its own, separate re-validation this doesn't attempt.
    A room's own drawn outline (the "Area" layer) is therefore still
    positioned relative to the original wall polygon's edge, not this
    centerline -- a small, usually imperceptible offset (about half the
    traced wall's own pixel thickness), not a bug.

    Verified on IMG_8666.JPG (a real, furniture-dense CAD-style photo, not
    just a simple test case): furniture/fixture icons (a toilet, a sink, a
    stair run's hatching, kitchen cabinets) came through recognizable and
    undistorted -- they were already thin-stroked in the source image, so
    skeletonizing barely changes them; only the thick, filled wall ink was
    meaningfully thinned.

    Binarization is plain OTSU (see _binarize_ink's own docstring -- an adaptive-threshold
    recovery pass for faint wall ink was tried here and reverted).
    """
    text_mask = _geometry_text_mask(image, text_regions)
    binary = _binarize_ink(image, text_mask, ink_mode)

    # Drop ink specks (anti-aliasing, a stray dimension tick) BEFORE skeletonizing -- same area
    # floor trace_contours applies to its own traced blobs, applied here one step earlier since
    # nothing is "a blob" yet to filter by area once it's already a 1px-wide skeleton.
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n_cc):
        if cc_stats[i, cv2.CC_STAT_AREA] >= CONTOUR_MIN_AREA_PX:
            clean[cc_labels == i] = 255

    skel = skeletonize(clean > 0)
    h, w = skel.shape
    if min_len is None:
        min_len = max(WALL_SKELETON_MIN_LEN_FLOOR, WALL_SKELETON_MIN_LEN_FRACTION * max(w, h))
    pixels = set(zip(*np.nonzero(skel)[::-1]))  # (x, y)

    def neighbours(p):
        """Skeleton neighbours of p, with a diagonal link dropped whenever the two pixels are
        also joined through a shared side neighbour. Without this, every step of a staircase
        (a line at an angle) has 3 neighbours, reads as a junction, and the line is chopped into
        1-2 px pieces that can never be straightened."""
        x, y = p
        out = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (x + dx, y + dy) in pixels]
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            q = (x + dx, y + dy)
            if q in pixels and (x + dx, y) not in pixels and (x, y + dy) not in pixels:
                out.append(q)
        return out

    adj = {p: neighbours(p) for p in pixels}
    node_pts = {p for p, n in adj.items() if len(n) != 2}  # line ends (1), junctions (3+), isolated (0)

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while cur not in node_pts and cur != start:
            nexts = [q for q in adj[cur] if q != prev]
            if not nexts:
                break
            prev, cur = cur, nexts[0]
            path.append(cur)
        return path

    visited_edges = set()
    polylines = []

    def add(path):
        for i in range(len(path) - 1):
            visited_edges.add(frozenset([path[i], path[i + 1]]))
        if len(path) <= 3 and path[0] in node_pts and path[-1] in node_pts and len(adj[path[0]]) >= 3 \
                and len(adj[path[-1]]) >= 3:
            return  # a link inside one junction (a junction is often 2-3 touching pixels), not a line
        pts = np.array(path, dtype=np.float32)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if length < min_len:
            return  # a short junction-thinning spur, not a real wall run
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2), WALL_SKELETON_EPSILON_PX, closed=False).reshape(-1, 2)
        if len(simplified) >= 2:
            polylines.append(snap_polygon_to_axes(simplified, closed=False))

    for start in node_pts:
        for nxt in adj[start]:
            if frozenset([start, nxt]) not in visited_edges:
                add(walk(start, nxt))
    # closed loops with no node on them at all (a room outline drawn as one ring)
    for p in pixels:
        for nxt in adj[p]:
            if frozenset([p, nxt]) not in visited_edges:
                add(walk(p, nxt))
    return polylines


SPUR_FACTOR = 2.5      # a dead-end branch up to this many half-thicknesses long is a thinning stub
SPUR_MIN_PX = 6.0      # ...and never shorter than this, so anti-aliasing whiskers on thin lines go too
SPECK_FRACTION = 0.02  # a line with BOTH ends free, shorter than this fraction of the longer side, is a speck
JOIN_PX = 3.0          # two line ends closer than this are the same junction
STRAIGHTEN_EPS_PX = 3.0  # RDP tolerance when straightening a re-joined line
STRAIGHTEN_SNAP_DEG = 8.0  # ...then edges within this of horizontal/vertical are snapped onto the axis


def _join_chains(lines: list[np.ndarray], tol: float = JOIN_PX) -> list[np.ndarray]:
    """Re-join line pieces end to end wherever exactly two ends meet (a plain bend or a split left
    by thinning, not a real junction), so one drawn line becomes one polyline again."""
    lines = [np.asarray(l, dtype=float) for l in lines if len(l) >= 2]
    grid: dict = {}
    for i, l in enumerate(lines):
        for side, p in ((0, l[0]), (1, l[-1])):
            grid.setdefault((int(p[0] // tol), int(p[1] // tol)), []).append((i, side))

    def end(i, side):
        return lines[i][-1] if side == 1 else lines[i][0]

    def others(i, side):
        p = end(i, side)
        gx, gy = int(p[0] // tol), int(p[1] // tol)
        return [(j, s) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j, s in grid.get((gx + dx, gy + dy), ())
                if (j, s) != (i, side) and np.hypot(*(end(j, s) - p)) <= tol]

    used = [False] * len(lines)
    out = []
    for i in range(len(lines)):
        if used[i]:
            continue
        used[i] = True
        chain = lines[i]
        for tail in (True, False):
            cur = (i, 1 if tail else 0)
            while True:
                o = others(*cur)
                if len(o) != 1 or used[o[0][0]]:
                    break
                j, s = o[0]
                used[j] = True
                seg = lines[j] if s == 0 else lines[j][::-1]  # oriented from the shared point outward
                chain = np.vstack([chain, seg[1:]]) if tail else np.vstack([seg[::-1][:-1], chain])
                cur = (j, 1 - s)
        out.append(chain)
    return out


def trace_single_lines(image: np.ndarray, text_regions: list[dict] | None = None,
                       ink_mode: str = "paper") -> list[np.ndarray]:
    """Every drawn stroke as ONE line, thick walls included -- the drawn Trace layer.

    Differs from extract_wall_skeleton's defaults in two ways:
      - ink = anything clearly darker than the paper ("paper" mode of _binarize_ink), so light-grey
        lines (door swings, fixtures, windows) are kept; Otsu drops them whenever a plan also has
        solid black walls (24-44% of drawn pixels on the plans checked).
      - no blanket length filter (it also cut short wall pieces between two junctions, leaving
        gaps where walls meet). Instead only dead-end stubs are pruned: thinning a thick wall
        leaves short diagonal branches from the centreline out to the wall's corners, each with one
        free end and about the wall's half-thickness long (distance transform at its junction end).
        Dead-end branches up to max(SPUR_FACTOR half-thicknesses, SPUR_MIN_PX) are dropped until
        none are left; isolated lines shorter than SPECK_FRACTION of the longer side are specks.
    Finally the pieces thinning split each line into are re-joined (_join_chains) and every whole
    line is straightened (RDP, STRAIGHTEN_EPS_PX) and snapped to horizontal/vertical.

    ink_mode="otsu" keeps only the dark strokes -- the walls -- which is what room segmentation
    should flood between; the light lines (furniture, notes) would only cut rooms into slivers.
    """
    lines = [l for l in extract_wall_skeleton(image, text_regions, ink_mode=ink_mode, min_len=0.0) if len(l) >= 2]
    ink = _binarize_ink(image, _geometry_text_mask(image, text_regions), ink_mode)
    half = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = half.shape
    speck = SPECK_FRACTION * max(h, w)

    def half_at(pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        return float(half[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)].max())

    while lines:
        ends = np.array([pt for line in lines for pt in (line[0], line[-1])], dtype=float)
        owner = np.repeat(np.arange(len(lines)), 2)

        def free(i, pt):  # no OTHER line has an end within JOIN_PX of this one
            return not np.any((np.hypot(*(ends - pt).T) <= JOIN_PX) & (owner != i))

        kept = []
        for i, line in enumerate(lines):
            fa, fb = free(i, line[0]), free(i, line[-1])
            length = float(np.sum(np.hypot(*np.diff(line, axis=0).T)))
            if fa and fb and length < speck:
                continue
            if fa != fb and length <= max(SPUR_FACTOR * half_at(line[-1] if fa else line[0]), SPUR_MIN_PX):
                continue
            kept.append(line)
        if len(kept) == len(lines):
            break
        lines = kept
    # Thinning splits a drawn line at every tiny junction, so it arrives as many 1-2 px pieces that
    # can't be straightened on their own. Re-join them, then straighten each whole line.
    out = []
    for chain in _join_chains(lines):
        simplified = cv2.approxPolyDP(chain.astype(np.float32).reshape(-1, 1, 2), STRAIGHTEN_EPS_PX,
                                      closed=False).reshape(-1, 2)
        if len(simplified) >= 2:
            out.append(snap_polygon_to_axes(simplified, STRAIGHTEN_SNAP_DEG, closed=False))
    return out


def export_contours_dxf(polygons: list[np.ndarray], out_path: str, height: int, closed: bool = True) -> None:
    """Write each traced polygon/polyline as a DXF LWPOLYLINE on a single layer.

    Coordinates are flipped (`height - y`) as in every export here:
    image space has y growing downward, DXF/CAD space has y growing upward.

    closed=False (for extract_wall_skeleton's open branches, not
    trace_contours' closed-ring polygons) skips the DXF's own closing
    segment back to the first point -- with close=True on an open chain,
    ezdxf draws a spurious extra edge from the last vertex back to the
    first, which is wrong for a wall branch that doesn't loop.
    """
    doc = ezdxf.new()
    doc.layers.add(LAYER_NAME)
    msp = doc.modelspace()

    for polygon in polygons:
        points = [(float(x), height - float(y)) for x, y in polygon]
        msp.add_lwpolyline(points, close=closed, dxfattribs={"layer": LAYER_NAME})

    doc.saveas(out_path)


def _select_images(filename: str | None):
    """All images in output/floorplan_images/, or just one by exact filename
    (e.g. "IMG_8666.JPG") if given.
    """
    if filename is None:
        yield from load_floorplan_images()
        return
    matches = [p for p in list_floorplan_images() if p.name == filename]
    if not matches:
        available = ", ".join(p.name for p in list_floorplan_images())
        raise SystemExit(f"no image named {filename!r} in output/floorplan_images/ — available: {available}")
    image = cv2.imread(str(matches[0]))
    if image is None:
        raise SystemExit(f"found {matches[0]} but could not read it as an image")
    yield matches[0], image
