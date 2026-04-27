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


APP_VERSION = "0.1.0"
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

    if not vectors:
        warnings.append("No stable filled DP colour polygons were detected in the submitted crop. Check alignment, crop area, scan quality, or use DXF/native PDF vector source.")

    # Keep useful architectural features first and avoid flooding the frontend.
    vectors = sorted(vectors, key=lambda v: (float(v.get("confidence", 0)), float(v.get("areaPx", 0))), reverse=True)[:80]
    vector_ids = {v["id"] for v in vectors}
    patches = [p for p in patches if p["id"] in vector_ids]

    return {
        "service": "DP Vision API",
        "version": APP_VERSION,
        "extractionMode": "cloud_opencv_polygon_segmentation_ocr_v1",
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
            "professionalWarning": "Architect must verify all returned DP features against the official DP sheet/legend before lock/export.",
        },
    }
