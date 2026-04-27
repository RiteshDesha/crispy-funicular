"""
DP Vision API for Thane TestFit Studio
Version 0.4.0 Recovery Build

Goal of this build:
- Never let TILR/survey-line detection break normal DP zone polygon output.
- Return zone patches and TILR survey lines as separate feature families.
- Add debug counts so the frontend can tell whether the API returned zero features,
  zero patches, or only raw TILR candidates.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None

APP_VERSION = "0.4.0"
app = FastAPI(title="DP Vision API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    min_area_px: int = 80


DP_CLASSES: Tuple[DPClass, ...] = (
    DPClass("yellow_residential", "Residential Zone", "DP_ZONE_RESIDENTIAL", "yellow_residential_zone", (250, 204, 21, 220), (((16, 25, 80), (48, 255, 255)), ((48, 14, 120), (70, 190, 255)))),
    DPClass("green_open_space", "Green Open Space Reservation - Verify Exact Type", "DP_GREEN_OPEN_SPACE_REVIEW", "green_open_space_or_rg", (34, 197, 94, 220), (((34, 18, 55), (96, 255, 255)),)),
    DPClass("blue_water", "Nallah / Drain / Water Body", "DP_WATER_NALLAH", "blue_water_nallah_or_drain", (14, 165, 233, 220), (((84, 18, 45), (142, 255, 255)),)),
    DPClass("magenta_reservation", "Public Purpose / Reservation - Verify", "DP_RESERVATION_REVIEW", "magenta_reservation_or_public_purpose", (217, 70, 239, 210), (((132, 18, 45), (179, 255, 255)),)),
    DPClass("orange_amenity", "Public / Semi-public / Amenity Reservation - Verify", "DP_PUBLIC_PURPOSE_AMENITY_REVIEW", "orange_public_semipublic_amenity", (249, 115, 22, 210), (((4, 20, 60), (26, 255, 255)),)),
    DPClass("red_road", "Road Widening / Proposed DP Road - Verify", "DP_ROAD_WIDENING_AFFECTED_AREA", "road_widening_affected_area", (239, 68, 68, 210), (((0, 28, 55), (10, 255, 255)), ((168, 28, 55), (179, 255, 255))), 60),
)

TEXT_RULES: Tuple[Tuple[re.Pattern[str], str, str, str, str], ...] = (
    (re.compile(r"\b(residential|r\s*zone|r-?zone|r1|r\s*1)\b", re.I), "Residential Zone", "DP_ZONE_RESIDENTIAL", "yellow_residential_zone", "OCR says residential/R zone"),
    (re.compile(r"\b(play\s*ground|playground|pg)\b", re.I), "Playground Reservation", "DP_RESERVATION_PLAYGROUND", "green_open_space_or_rg", "OCR says playground/PG"),
    (re.compile(r"\b(garden|gdn|\bg\b)\b", re.I), "Garden Reservation", "DP_RESERVATION_GARDEN", "green_open_space_or_rg", "OCR says garden/G"),
    (re.compile(r"\b(park|open\s*space|ros|recreation|rg)\b", re.I), "Park / Open Space Reservation", "DP_RESERVATION_OPEN_SPACE", "green_open_space_or_rg", "OCR says park/open space/RG/ROS"),
    (re.compile(r"\b(lake|talav|pond|water\s*body)\b", re.I), "Lake / Water Body", "DP_WATER_BODY_LAKE", "blue_water_nallah_or_drain", "OCR says lake/talav/water body"),
    (re.compile(r"\b(nallah|nala|drain|storm\s*water|sw[dm])\b", re.I), "Nallah / Drain / Water Course", "DP_WATER_NALLAH", "blue_water_nallah_or_drain", "OCR says nallah/drain"),
    (re.compile(r"\b(road\s*widening|widening|dp\s*road|proposed\s*road)\b", re.I), "Road Widening Affected Area", "DP_ROAD_WIDENING_AFFECTED_AREA", "road_widening_affected_area", "OCR says road widening/DP road"),
    (re.compile(r"\b(school|primary\s*school|secondary\s*school)\b", re.I), "School Reservation", "DP_RESERVATION_SCHOOL", "magenta_reservation_or_public_purpose", "OCR says school"),
    (re.compile(r"\b(hospital|dispensary|health)\b", re.I), "Health / Hospital Reservation", "DP_RESERVATION_HEALTH", "magenta_reservation_or_public_purpose", "OCR says hospital/health"),
    (re.compile(r"\b(amenity|public\s*purpose|public\s*semi|semi\s*public|psp)\b", re.I), "Public / Semi-public / Amenity Reservation", "DP_PUBLIC_SEMI_PUBLIC_REVIEW", "magenta_reservation_or_public_purpose", "OCR says amenity/public-semi-public"),
)


def _decode_data_url(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise HTTPException(status_code=400, detail="image_data_url is missing or invalid")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image_data_url: {exc}") from exc
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _clean_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s/()._-]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()[:500]


def _ocr_crop(img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 20) -> str:
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
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    try:
        return _clean_text(pytesseract.image_to_string(gray, config="--psm 6"))
    except Exception:
        return ""


def _classify_from_text(base: DPClass, text: str) -> Tuple[str, str, str, int, str]:
    cleaned = _clean_text(text)
    for pattern, label, layer, category, reason in TEXT_RULES:
        if pattern.search(cleaned):
            return label, layer, category, 94, f"{reason}; OCR: {cleaned[:120]}"
    if cleaned:
        return base.label, base.layer, base.category, 78, f"{base.key} colour + ambiguous OCR: {cleaned[:120]}"
    return base.label, base.layer, base.category, 66, f"{base.key} colour region; no readable OCR label"


def _site_influence_mask(shape: Tuple[int, int], site_polygon: List[List[float]], buffer_px: int, force_full: bool = False) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if force_full or len(site_polygon) < 3:
        mask[:] = 255
    else:
        pts = np.array(site_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    if buffer_px > 0 and not force_full:
        k = max(3, min(801, int(buffer_px) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _mask_for_class(hsv: np.ndarray, cls: DPClass, influence: np.ndarray) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cls.hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    mask = cv2.bitwise_and(mask, influence)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    return mask


def _contour_to_polygon(contour: np.ndarray) -> List[List[float]]:
    peri = cv2.arcLength(contour, True)
    epsilon = max(1.2, 0.005 * peri)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        rect = cv2.boxPoints(cv2.minAreaRect(contour))
        approx = np.int32(rect)
    return [[float(x), float(y)] for x, y in approx]


def _segment_zone_patches(img_bgr: np.ndarray, hsv: np.ndarray, influence: np.ndarray, min_area: int, tag: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"pass": tag, "minAreaPx": min_area, "classes": {}}
    for cls in DP_CLASSES:
        mask = _mask_for_class(hsv, cls, influence)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = 0
        for ci, contour in enumerate(contours):
            area_px = float(cv2.contourArea(contour))
            if area_px < max(cls.min_area_px, min_area):
                continue
            bbox = cv2.boundingRect(contour)
            polygon = _contour_to_polygon(contour)
            if len(polygon) < 3:
                continue
            ocr_inside = _ocr_crop(img_bgr, bbox, pad=18)
            label, layer, category, confidence, reason = _classify_from_text(cls, ocr_inside)
            patch = {
                "id": f"{tag}_{cls.key}_{ci + 1}",
                "type": "polygon",
                "label": label,
                "layer": layer,
                "category": category,
                "geometry": polygon,
                "path": polygon,
                "confidence": confidence,
                "reason": reason,
                "color": list(cls.color),
                "ocrText": ocr_inside,
                "areaPx": round(area_px, 2),
                "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "width": int(bbox[2]), "height": int(bbox[3])},
                "reviewStatus": "needs_architect_review" if confidence < 88 or "Verify" in label else "cloud_detected_unverified",
                "needsManualConfirmation": confidence < 88 or "Verify" in label,
                "source": "opencv_colour_segmentation_plus_optional_ocr",
            }
            patches.append(patch)
            kept += 1
        debug["classes"][cls.key] = {"contours": len(contours), "kept": kept, "maskPixels": int(cv2.countNonZero(mask))}
    return patches, debug


def _line_red_coverage(mask: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
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


def _dash_score(mask: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
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


def _dedupe_lines(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for f in sorted(features, key=lambda item: (item.get("confidence", 0), item.get("lengthPx", 0)), reverse=True):
        p = f["geometry"]
        mx = (p[0][0] + p[1][0]) / 2
        my = (p[0][1] + p[1][1]) / 2
        ang = np.arctan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
        duplicate = False
        for g in kept:
            gp = g["geometry"]
            gmx = (gp[0][0] + gp[1][0]) / 2
            gmy = (gp[0][1] + gp[1][1]) / 2
            gang = np.arctan2(gp[1][1] - gp[0][1], gp[1][0] - gp[0][0])
            if np.hypot(mx - gmx, my - gmy) < 16 and abs(np.sin(ang - gang)) < 0.12:
                duplicate = True
                break
        if not duplicate:
            kept.append(f)
    return kept


def _extract_tilr_survey_lines(img_bgr: np.ndarray, hsv: np.ndarray, influence: np.ndarray, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    if options.get("return_tilr_survey_lines", True) is False:
        return []
    red1 = cv2.inRange(hsv, np.array((0, 18, 45)), np.array((13, 255, 255)))
    red2 = cv2.inRange(hsv, np.array((160, 18, 45)), np.array((179, 255, 255)))
    pink = cv2.inRange(hsv, np.array((140, 14, 55)), np.array((179, 210, 255)))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), pink)
    red_mask = cv2.bitwise_and(red_mask, influence)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    profile = str(options.get("tilr_detection_profile", "aggressive_top_overlay"))
    min_len = int(options.get("tilr_min_line_px", 12 if profile.startswith("aggressive") else 24))
    max_gap = int(options.get("tilr_max_dash_gap_px", 72 if profile.startswith("aggressive") else 36))
    threshold = int(options.get("tilr_hough_threshold", 8 if profile.startswith("aggressive") else 18))
    max_lines = int(options.get("tilr_max_lines", 220))

    candidate_masks = [red_mask]
    for size in [(9, 3), (19, 3), (39, 3), (3, 9), (3, 19), (3, 39), (11, 11)]:
        candidate_masks.append(cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, size), iterations=1))

    sample_mask = cv2.dilate(red_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    features: List[Dict[str, Any]] = []

    def add_segment(x1: int, y1: int, x2: int, y2: int, method: str, idx_hint: int) -> None:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            return
        raw_cov = _line_red_coverage(red_mask, (x1, y1), (x2, y2))
        near_cov = _line_red_coverage(sample_mask, (x1, y1), (x2, y2))
        dash = _dash_score(red_mask, (x1, y1), (x2, y2))
        if near_cov < 0.08 and raw_cov < 0.025:
            return
        conf = int(max(46, min(94, 48 + near_cov * 30 + raw_cov * 20 + dash * 18 + min(12, length / 110))))
        geom = [[float(x1), float(y1)], [float(x2), float(y2)]]
        features.append({
            "id": f"tilr_survey_line_{method}_{idx_hint}_{len(features)+1}",
            "type": "line",
            "label": "TILR / Survey Number Line",
            "layer": "DP_TILR_SURVEY_LINES",
            "category": "tilr_survey_number_lines",
            "geometry": geom,
            "path": geom,
            "confidence": conf,
            "reason": f"Red dashed/TILR line candidate via {method}; raw={raw_cov:.2f}, near={near_cov:.2f}, dash={dash:.2f}",
            "color": [239, 68, 68, 255],
            "dashArray": [8, 5],
            "lengthPx": round(length, 2),
            "reviewStatus": "cloud_detected_unverified",
            "needsManualConfirmation": True,
            "source": "opencv_red_dashed_tilr_detection_recovery_v4",
            "traceMethod": f"cloud_cv_tilr_{method}_v4",
            "meaning": "Red dashed survey-number / TILR land-record reference line. Separate from zone patches and road widening.",
        })

    for vi, mask in enumerate(candidate_masks):
        hough = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=threshold, minLineLength=min_len, maxLineGap=max_gap)
        if hough is None:
            continue
        for li, line in enumerate(hough.reshape(-1, 4)):
            x1, y1, x2, y2 = [int(v) for v in line]
            add_segment(x1, y1, x2, y2, "hough", vi * 1000 + li)

    return _dedupe_lines(features)[:max_lines]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "DP Vision API",
        "version": APP_VERSION,
        "opencv": cv2.__version__,
        "ocr": "pytesseract" if pytesseract is not None else "disabled",
        "recoveryBuild": True,
    }


@app.post("/trace")
def trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    img_bgr = _decode_data_url(str(payload.get("image_data_url", "")))
    h, w = img_bgr.shape[:2]
    options = payload.get("options") or {}
    site_polygon = payload.get("site_polygon_px") or []
    buffer_px = int(options.get("site_buffer_px", 360))
    min_area_ratio = float(options.get("min_polygon_area_ratio", 0.00035))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    min_area = max(45, int(w * h * min_area_ratio))
    influence = _site_influence_mask((h, w), site_polygon, buffer_px)
    patches, debug_primary = _segment_zone_patches(img_bgr, hsv, influence, min_area, "site")

    warnings: List[str] = []
    debug_passes = [debug_primary]

    if not patches:
        # Recovery pass: old DP scans can be misaligned/cropped. Do not return empty immediately.
        # Search the submitted crop with a much lower area threshold, but tag results for review.
        full_influence = _site_influence_mask((h, w), site_polygon, 0, force_full=True)
        recovery_min = max(30, int(w * h * 0.00012))
        patches, debug_recovery = _segment_zone_patches(img_bgr, hsv, full_influence, recovery_min, "recovery")
        debug_passes.append(debug_recovery)
        if patches:
            warnings.append("No zone patches survived the site-buffer pass, so the API used a recovery scan over the submitted crop. Verify alignment before locking.")

    vectors: List[Dict[str, Any]] = [dict(p, path=p.get("geometry", [])) for p in patches]
    tilr_vectors = _extract_tilr_survey_lines(img_bgr, hsv, influence, options)
    if tilr_vectors:
        vectors.extend(tilr_vectors)
        warnings.append(f"Detected {len(tilr_vectors)} TILR/survey-number red dashed line candidate(s) on separate layer DP_TILR_SURVEY_LINES. Verify against official TILR/CAD survey records.")

    if not vectors:
        warnings.append("API ran but detected zero DP zone patches and zero TILR lines. Check if the submitted crop actually contains coloured DP data, not a blank/misaligned area.")

    # Protect zones from being pushed out by many TILR lines.
    zone_vectors = [v for v in vectors if v.get("type") == "polygon"]
    line_vectors = [v for v in vectors if v.get("type") != "polygon"]
    zone_vectors = sorted(zone_vectors, key=lambda v: (float(v.get("confidence", 0)), float(v.get("areaPx", 0))), reverse=True)[:100]
    line_vectors = sorted(line_vectors, key=lambda v: (float(v.get("confidence", 0)), float(v.get("lengthPx", 0))), reverse=True)[:220]
    vectors = zone_vectors + line_vectors
    vector_ids = {v["id"] for v in zone_vectors}
    patches = [p for p in patches if p["id"] in vector_ids]

    return {
        "service": "DP Vision API",
        "version": APP_VERSION,
        "extractionMode": "cloud_opencv_zone_polygons_plus_separate_tilr_recovery_v4",
        "imageSize": {"width": w, "height": h},
        "patches": patches,
        "vectors": vectors,
        "tilrSurveyLines": line_vectors,
        "findings": [
            {
                "id": f"finding_{i+1}",
                "title": v.get("label", "DP Feature"),
                "category": v.get("category", "cloud_dp_feature"),
                "layer": v.get("layer", "DP_CLOUD_TRACE"),
                "status": v.get("reviewStatus", "cloud_detected_unverified"),
                "confidence": v.get("confidence", 70),
                "note": v.get("reason", ""),
                "vectorId": v.get("id", f"vector_{i+1}"),
            }
            for i, v in enumerate(vectors)
        ],
        "warnings": warnings,
        "debugCounts": {
            "patches": len(patches),
            "vectors": len(vectors),
            "zoneVectors": len(zone_vectors),
            "tilrVectors": len(line_vectors),
            "passes": debug_passes,
        },
        "meta": {
            "sitePolygonPx": site_polygon,
            "siteBufferPx": buffer_px,
            "minAreaPx": min_area,
            "ocrEnabled": pytesseract is not None,
            "recoveryBuild": True,
            "professionalWarning": "Architect must verify all returned DP features against the official DP sheet/legend before lock/export.",
        },
    }
