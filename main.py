"""
DP Vision API for Thane TestFit Studio

Professional DP workflow:
1. Receive aligned/cropped DP image from the one-file HTML frontend.
2. Segment filled DP colour regions using OpenCV.
3. Run OCR inside/near polygons when pytesseract is available.
4. Classify patches using colour + OCR + layer/legend dictionary.
5. Return clean CAD-style polygons with label, layer, confidence and reason.

This is intentionally a backend. Do not move this logic back into the browser for
official approval-quality DP tracing.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    import pytesseract
except Exception:  # pragma: no cover - optional system dependency
    pytesseract = None


APP_VERSION = "0.2.0"
app = FastAPI(title="DP Vision API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your app/domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class DPClass:
    key: str
    label: str
    layer: str
    category: str
    color: Tuple[int, int, int, int]
    hsv_ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]
    min_area_px: int = 180


DP_CLASSES: Tuple[DPClass, ...] = (
    DPClass(
        key="yellow_residential",
        label="Residential Zone",
        layer="DP_ZONE_RESIDENTIAL",
        category="yellow_residential_zone",
        color=(250, 204, 21, 220),
        hsv_ranges=(((18, 35, 90), (42, 255, 255)), ((42, 18, 130), (64, 180, 255))),
    ),
    DPClass(
        key="green_open_space",
        label="Green Open Space Reservation - Verify Exact Type",
        layer="DP_GREEN_OPEN_SPACE_REVIEW",
        category="green_open_space_or_rg",
        color=(34, 197, 94, 220),
        hsv_ranges=(((38, 25, 70), (92, 255, 255)),),
    ),
    DPClass(
        key="blue_water",
        label="Nallah / Drain / Water Body",
        layer="DP_WATER_NALLAH",
        category="blue_water_nallah_or_drain",
        color=(14, 165, 233, 220),
        hsv_ranges=(((88, 25, 55), (140, 255, 255)),),
    ),
    DPClass(
        key="magenta_reservation",
        label="Public Purpose / Reservation - Verify",
        layer="DP_RESERVATION_REVIEW",
        category="magenta_reservation_or_public_purpose",
        color=(217, 70, 239, 210),
        hsv_ranges=(((135, 25, 55), (178, 255, 255)),),
    ),
    DPClass(
        key="orange_amenity",
        label="Public / Semi-public / Amenity Reservation - Verify",
        layer="DP_PUBLIC_PURPOSE_AMENITY_REVIEW",
        category="orange_public_semipublic_amenity",
        color=(249, 115, 22, 210),
        hsv_ranges=(((5, 30, 80), (24, 255, 255)),),
    ),
    DPClass(
        key="red_road",
        label="Road Widening / Proposed DP Road - Verify",
        layer="DP_ROAD_WIDENING_AFFECTED_AREA",
        category="road_widening_affected_area",
        color=(239, 68, 68, 210),
        hsv_ranges=(((0, 40, 70), (8, 255, 255)), ((172, 40, 70), (179, 255, 255))),
        min_area_px=120,
    ),
)


TEXT_RULES: Tuple[Tuple[re.Pattern[str], str, str, str, str], ...] = (
    (re.compile(r"\b(residential|r\s*zone|r-?zone|r1|r\s*1)\b", re.I), "Residential Zone", "DP_ZONE_RESIDENTIAL", "yellow_residential_zone", "text says residential/R zone"),
    (re.compile(r"\b(play\s*ground|playground|pg)\b", re.I), "Playground Reservation", "DP_RESERVATION_PLAYGROUND", "green_open_space_or_rg", "text says playground/PG"),
    (re.compile(r"\b(garden|gdn|\bg\b)\b", re.I), "Garden Reservation", "DP_RESERVATION_GARDEN", "green_open_space_or_rg", "text says garden/G"),
    (re.compile(r"\b(park|open\s*space|ros|recreation|rg)\b", re.I), "Park / Open Space Reservation", "DP_RESERVATION_OPEN_SPACE", "green_open_space_or_rg", "text says park/open space/RG/ROS"),
    (re.compile(r"\b(lake|talav|pond|water\s*body)\b", re.I), "Lake / Water Body", "DP_WATER_BODY_LAKE", "blue_water_nallah_or_drain", "text says lake/talav/water body"),
    (re.compile(r"\b(nallah|nala|drain|storm\s*water|sw[dm])\b", re.I), "Nallah / Drain / Water Course", "DP_WATER_NALLAH", "blue_water_nallah_or_drain", "text says nallah/drain"),
    (re.compile(r"\b(road\s*widening|widening|dp\s*road|proposed\s*road)\b", re.I), "Road Widening Affected Area", "DP_ROAD_WIDENING_AFFECTED_AREA", "road_widening_affected_area", "text says road widening/DP road"),
    (re.compile(r"\b(school|primary\s*school|secondary\s*school)\b", re.I), "School Reservation", "DP_RESERVATION_SCHOOL", "magenta_reservation_or_public_purpose", "text says school"),
    (re.compile(r"\b(hospital|dispensary|health)\b", re.I), "Health / Hospital Reservation", "DP_RESERVATION_HEALTH", "magenta_reservation_or_public_purpose", "text says hospital/health"),
    (re.compile(r"\b(amenity|public\s*purpose|public\s*semi|semi\s*public|psp)\b", re.I), "Public / Semi-public / Amenity Reservation", "DP_PUBLIC_SEMI_PUBLIC_REVIEW", "magenta_reservation_or_public_purpose", "text says amenity/public-semi-public"),
)


def _decode_data_url(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise HTTPException(status_code=400, detail="image_data_url is missing or invalid")
    header, b64 = data_url.split(",", 1)
    try:
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image_data_url: {exc}") from exc
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s/()._-]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _ocr_crop(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 16) -> str:
    if pytesseract is None:
        return ""
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
    except Exception:
        return ""
    return _clean_text(text)


def _classify_from_text(base: DPClass, text: str) -> Tuple[str, str, str, int, str]:
    cleaned = _clean_text(text)
    for pattern, label, layer, category, reason in TEXT_RULES:
        if pattern.search(cleaned):
            return label, layer, category, 94, f"{reason}; OCR: {cleaned[:120]}"
    if cleaned:
        return base.label, base.layer, base.category, 78, f"{base.key} colour + OCR unread/ambiguous: {cleaned[:120]}"
    return base.label, base.layer, base.category, 64, f"{base.key} colour region; no readable OCR text inside/near polygon"


def _site_influence_mask(shape: Tuple[int, int], site_polygon: List[List[float]], buffer_px: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(site_polygon) >= 3:
        pts = np.array(site_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    else:
        mask[:] = 255
    if buffer_px > 0:
        k = max(3, int(buffer_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _mask_for_class(hsv: np.ndarray, cls: DPClass, influence: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cls.hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    mask = cv2.bitwise_and(mask, influence)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _contour_to_polygon(contour: np.ndarray) -> List[List[float]]:
    peri = cv2.arcLength(contour, True)
    epsilon = max(1.5, 0.006 * peri)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        rect = cv2.boxPoints(cv2.minAreaRect(contour))
        approx = np.int32(rect)
    return [[float(x), float(y)] for x, y in approx]


def _line_red_coverage(mask: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Approximate how much of a candidate line is actually red ink."""
    x1, y1 = p1
    x2, y2 = p2
    length = int(max(2, np.hypot(x2 - x1, y2 - y1)))
    xs = np.linspace(x1, x2, length).astype(np.int32)
    ys = np.linspace(y1, y2, length).astype(np.int32)
    h, w = mask.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    vals = mask[ys, xs] > 0
    return float(vals.mean()) if vals.size else 0.0


def _line_dash_score(mask: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Higher score means the sampled line looks intermittent/dashed rather than one solid fill."""
    x1, y1 = p1
    x2, y2 = p2
    length = int(max(2, np.hypot(x2 - x1, y2 - y1)))
    xs = np.linspace(x1, x2, length).astype(np.int32)
    ys = np.linspace(y1, y2, length).astype(np.int32)
    h, w = mask.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    vals = (mask[ys, xs] > 0).astype(np.int8)
    if vals.size < 8:
        return 0.0
    transitions = int(np.sum(np.abs(np.diff(vals))))
    return float(min(1.0, transitions / max(4, vals.size / 12)))


def _remove_large_red_fills(red_mask: np.ndarray) -> np.ndarray:
    """Keep thin/dashed red linework, remove large filled red shapes that are more likely road/patch fills."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    cleaned = np.zeros_like(red_mask)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        bbox_area = max(1, int(w) * int(h))
        fill_ratio = float(area) / bbox_area
        long_thin = min(w, h) <= max(10, 0.22 * max(w, h))
        small_dash = area < 1200 and max(w, h) < 180
        # TILR dashes/lines are usually thin or small. Big compact red blobs are not.
        if long_thin or small_dash or fill_ratio < 0.35:
            cleaned[labels == idx] = 255
    return cleaned


def _dedupe_segments(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Avoid returning the same Hough segment many times."""
    kept: List[Dict[str, Any]] = []
    for feat in sorted(features, key=lambda f: (f.get("confidence", 0), f.get("lengthPx", 0)), reverse=True):
        pts = feat.get("geometry") or []
        if len(pts) < 2:
            continue
        x1, y1 = pts[0]
        x2, y2 = pts[-1]
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        ang = np.arctan2(y2 - y1, x2 - x1)
        duplicate = False
        for other in kept:
            opt = other.get("geometry") or []
            ox1, oy1 = opt[0]
            ox2, oy2 = opt[-1]
            omx, omy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5
            oang = np.arctan2(oy2 - oy1, ox2 - ox1)
            angle_delta = abs(((ang - oang + np.pi / 2) % np.pi) - np.pi / 2)
            midpoint_dist = float(np.hypot(mx - omx, my - omy))
            if angle_delta < np.deg2rad(5) and midpoint_dist < 10:
                duplicate = True
                break
        if not duplicate:
            kept.append(feat)
    return kept


def _extract_tilr_survey_lines(
    img_bgr: np.ndarray,
    hsv: np.ndarray,
    influence: np.ndarray,
    options: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect red dashed TILR / survey-number parcel linework as a separate line layer.

    These are returned as line vectors, never as zone polygons. They are useful as a
    survey-reference overlay and should remain separate from DP zone patches and road widening.
    """
    if options.get("return_tilr_survey_lines", True) is False:
        return []

    # DP scans vary from pure red to faded pink/red. Keep this broad, but still red-family.
    red1 = cv2.inRange(hsv, np.array((0, 35, 55)), np.array((12, 255, 255)))
    red2 = cv2.inRange(hsv, np.array((165, 35, 55)), np.array((179, 255, 255)))
    pink_red = cv2.inRange(hsv, np.array((145, 25, 70)), np.array((179, 210, 255)))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), pink_red)
    red_mask = cv2.bitwise_and(red_mask, influence)

    # Remove big fills, preserve thin dashed strokes.
    red_mask = _remove_large_red_fills(red_mask)

    # Slightly connect dash fragments so Hough can see the intended line.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    detect_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    detect_mask = cv2.morphologyEx(detect_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    h, w = red_mask.shape[:2]
    min_len = int(options.get("tilr_min_line_px", max(28, min(w, h) * 0.025)))
    max_gap = int(options.get("tilr_max_dash_gap_px", 24))
    hough = cv2.HoughLinesP(
        detect_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=int(options.get("tilr_hough_threshold", 24)),
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if hough is None:
        return []

    features: List[Dict[str, Any]] = []
    for idx, line in enumerate(hough.reshape(-1, 4)):
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        coverage = _line_red_coverage(red_mask, (x1, y1), (x2, y2))
        dash_score = _line_dash_score(red_mask, (x1, y1), (x2, y2))

        # Reject extremely weak false positives and mostly-solid red boundaries from filled areas.
        if coverage < 0.16:
            continue
        if coverage > 0.96 and dash_score < 0.08:
            continue

        conf = int(max(52, min(92, 55 + coverage * 30 + dash_score * 18 + min(10, length / 80))))
        geometry = [[float(x1), float(y1)], [float(x2), float(y2)]]
        features.append(
            {
                "id": f"tilr_survey_line_{idx + 1}",
                "type": "line",
                "label": "TILR / Survey Number Line",
                "layer": "DP_TILR_SURVEY_LINES",
                "category": "tilr_survey_number_lines",
                "geometry": geometry,
                "path": geometry,
                "confidence": conf,
                "reason": f"Red dashed/parcel-style linework detected; coverage={coverage:.2f}, dash_score={dash_score:.2f}",
                "color": [220, 38, 38, 235],
                "dashArray": [5, 4],
                "lengthPx": round(length, 2),
                "reviewStatus": "cloud_detected_unverified",
                "needsManualConfirmation": True,
                "source": "opencv_red_dashed_tilr_line_detection",
                "traceMethod": "cloud_cv_tilr_survey_line_hough",
            }
        )

    features = _dedupe_segments(features)
    features = sorted(features, key=lambda f: (float(f.get("confidence", 0)), float(f.get("lengthPx", 0))), reverse=True)
    return features[: int(options.get("tilr_max_lines", 60))]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "DP Vision API",
        "version": APP_VERSION,
        "opencv": cv2.__version__,
        "ocr": "pytesseract" if pytesseract is not None else "disabled",
    }


@app.post("/trace")
def trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    img_bgr = _decode_data_url(str(payload.get("image_data_url", "")))
    h, w = img_bgr.shape[:2]
    options = payload.get("options") or {}
    site_polygon = payload.get("site_polygon_px") or []
    buffer_px = int(options.get("site_buffer_px", 140))
    min_area_ratio = float(options.get("min_polygon_area_ratio", 0.0012))
    influence = _site_influence_mask((h, w), site_polygon, buffer_px)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    min_area = max(100, int(w * h * min_area_ratio))
    patches: List[Dict[str, Any]] = []
    vectors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for cls in DP_CLASSES:
        mask = _mask_for_class(hsv, cls, influence)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ci, contour in enumerate(contours):
            area_px = cv2.contourArea(contour)
            if area_px < max(cls.min_area_px, min_area):
                continue
            bbox = cv2.boundingRect(contour)
            polygon = _contour_to_polygon(contour)
            if len(polygon) < 3:
                continue
            ocr_inside = _ocr_crop(img_bgr, bbox, pad=18)
            label, layer, category, confidence, reason = _classify_from_text(cls, ocr_inside)

            patch = {
                "id": f"{cls.key}_{ci + 1}",
                "type": "polygon",
                "label": label,
                "layer": layer,
                "category": category,
                "geometry": polygon,
                "confidence": confidence,
                "reason": reason,
                "color": list(cls.color),
                "ocrText": ocr_inside,
                "areaPx": round(float(area_px), 2),
                "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "width": int(bbox[2]), "height": int(bbox[3])},
                "reviewStatus": "needs_architect_review" if confidence < 88 or "Verify" in label else "cloud_detected_unverified",
                "needsManualConfirmation": confidence < 88 or "Verify" in label,
                "source": "opencv_colour_segmentation_plus_ocr",
            }
            patches.append(patch)
            vectors.append(dict(patch, path=polygon))

    tilr_vectors = _extract_tilr_survey_lines(img_bgr, hsv, influence, options)
    if tilr_vectors:
        vectors.extend(tilr_vectors)
        warnings.append(f"Detected {len(tilr_vectors)} TILR/survey-number red dashed line candidate(s) on separate layer DP_TILR_SURVEY_LINES. Verify against official TILR/CAD survey records.")

    if not vectors:
        warnings.append("No stable filled DP colour polygons or TILR survey lines were detected in the submitted crop. Check alignment, crop area, scan quality, or use DXF/native PDF vector source.")

    # Keep useful architectural features first and avoid flooding the frontend.
    vectors = sorted(vectors, key=lambda v: (float(v.get("confidence", 0)), float(v.get("areaPx", 0)), float(v.get("lengthPx", 0))), reverse=True)[:120]
    vector_ids = {v["id"] for v in vectors}
    patches = [p for p in patches if p["id"] in vector_ids]

    return {
        "service": "DP Vision API",
        "version": APP_VERSION,
        "extractionMode": "cloud_opencv_polygon_segmentation_ocr_tilr_v2",
        "imageSize": {"width": w, "height": h},
        "patches": patches,
        "vectors": vectors,
        "findings": [
            {
                "id": f"finding_{i + 1}",
                "title": v["label"],
                "category": v["category"],
                "layer": v["layer"],
                "status": v.get("reviewStatus", "cloud_detected_unverified"),
                "confidence": v["confidence"],
                "note": v["reason"],
                "vectorId": v["id"],
            }
            for i, v in enumerate(vectors)
        ],
        "warnings": warnings,
        "meta": {
            "sitePolygonPx": site_polygon,
            "siteBufferPx": buffer_px,
            "minAreaPx": min_area,
            "ocrEnabled": pytesseract is not None,
            "tilrSurveyLinesEnabled": options.get("return_tilr_survey_lines", True) is not False,
            "tilrLayer": "DP_TILR_SURVEY_LINES",
            "professionalWarning": "Architect must verify all returned DP features against the official DP sheet/legend before lock/export.",
        },
    }
