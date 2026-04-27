"""
DP Vision Tracer API for Thane TestFit Studio.

This service converts the cropped, aligned DP map image sent by the browser into
clean, draftsman-style CAD polylines. The browser remains the viewer/alignment
surface; Python handles CV, geometry cleanup, and semantic layer assignment.
"""
from __future__ import annotations

import base64
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

Point = Tuple[float, float]
Segment = Tuple[float, float, float, float]

app = FastAPI(title="DP Vision Tracer API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TraceRequest(BaseModel):
    image_data_url: str = Field(..., description="PNG/JPEG data URL of the crop prepared by the browser")
    site_polygon_px: List[List[float]] = Field(default_factory=list, description="Site polygon in crop pixel coordinates")
    crop: Dict[str, float] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class LayerSpec:
    category: str
    label: str
    layer: str
    color: Tuple[int, int, int, int]
    kind: str


LAYER_SPECS: Dict[str, LayerSpec] = {
    "red_road_or_widening": LayerSpec("red_road_or_widening", "DP road / road widening line", "DP_ROAD_WIDENING", (239, 68, 68, 235), "line"),
    "pale_red_survey_or_alignment_line": LayerSpec("pale_red_survey_or_alignment_line", "Faint red survey / alignment line", "DP_FAINT_RED_SURVEY_LINE", (248, 113, 113, 225), "line"),
    "pink_survey_or_reservation_line": LayerSpec("pink_survey_or_reservation_line", "Pink survey / reservation linework", "DP_PINK_SURVEY_LINEWORK", (244, 114, 182, 225), "line"),
    "magenta_reservation_or_public_purpose": LayerSpec("magenta_reservation_or_public_purpose", "Reservation / public purpose patch", "DP_RESERVATION", (217, 70, 239, 215), "polygon"),
    "yellow_orange_amenity_zone": LayerSpec("yellow_orange_amenity_zone", "Amenity / mixed DP marking", "DP_AMENITY_MARKING", (245, 158, 11, 220), "polygon"),
    "green_open_space_or_rg": LayerSpec("green_open_space_or_rg", "Green / RG / open space marking", "DP_OPEN_SPACE", (34, 197, 94, 215), "polygon"),
    "blue_water_nallah_or_drain": LayerSpec("blue_water_nallah_or_drain", "Water body / nallah / drain marking", "DP_WATER_NALLAH", (14, 165, 233, 225), "line"),
    "dark_boundary_or_cad_line": LayerSpec("dark_boundary_or_cad_line", "Dark boundary / base map linework", "DP_BASE_LINEWORK", (30, 41, 59, 230), "line"),
    "grey_dotted_boundary_or_survey_line": LayerSpec("grey_dotted_boundary_or_survey_line", "Grey dotted survey / boundary line", "DP_DOTTED_SURVEY_BOUNDARY", (100, 116, 139, 230), "line"),
}


def decode_data_url(data_url: str) -> np.ndarray:
    match = re.match(r"^data:image/[^;]+;base64,(.*)$", data_url, flags=re.I | re.S)
    if not match:
        raise ValueError("Expected image data URL")
    raw = base64.b64decode(match.group(1))
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def close_ring(path: List[Point]) -> List[Point]:
    if not path:
        return path
    if math.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]) > 1.5:
        return path + [path[0]]
    return path


def polygon_area(path: List[Point]) -> float:
    if len(path) < 3:
        return 0.0
    return abs(cv2.contourArea(np.array(path, dtype=np.float32)))


def rdp(path: List[Point], eps: float) -> List[Point]:
    if len(path) <= 2:
        return path
    arr = np.array(path, dtype=np.float32).reshape((-1, 1, 2))
    out = cv2.approxPolyDP(arr, eps, False).reshape((-1, 2))
    return [(float(x), float(y)) for x, y in out]


def make_site_mask(shape: Tuple[int, int], site_polygon_px: List[List[float]], buffer_px: int = 70) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if len(site_polygon_px) >= 3:
        pts = np.array(site_polygon_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    else:
        mask[:] = 255
    return mask


def color_masks(img_bgr: np.ndarray, site_mask: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    # HSV hue ranges are OpenCV units: 0..179.
    red_strong = ((h <= 8) | (h >= 170)) & (s > 55) & (v > 95)
    red_faint = ((h <= 12) | (h >= 168)) & (s > 18) & (s <= 85) & (v > 115)
    pink = (h >= 145) & (h <= 176) & (s > 24) & (v > 115)
    magenta_fill = (h >= 138) & (h <= 176) & (s > 32) & (v > 130)
    yellow = (h >= 12) & (h <= 38) & (s > 30) & (v > 130)
    green = (h >= 42) & (h <= 95) & (s > 22) & (v > 105)
    blue = (h >= 88) & (h <= 132) & (s > 25) & (v > 95)

    # Grey / black dotted map lines: low saturation, not white, and darker than paper texture.
    grey_dots = (s < 42) & (v > 60) & (v < 190)
    dark = (gray < 145) & (s < 75)

    raw = {
        "red_road_or_widening": red_strong.astype(np.uint8) * 255,
        "pale_red_survey_or_alignment_line": red_faint.astype(np.uint8) * 255,
        "pink_survey_or_reservation_line": pink.astype(np.uint8) * 255,
        "magenta_reservation_or_public_purpose": magenta_fill.astype(np.uint8) * 255,
        "yellow_orange_amenity_zone": yellow.astype(np.uint8) * 255,
        "green_open_space_or_rg": green.astype(np.uint8) * 255,
        "blue_water_nallah_or_drain": blue.astype(np.uint8) * 255,
        "grey_dotted_boundary_or_survey_line": grey_dots.astype(np.uint8) * 255,
        "dark_boundary_or_cad_line": dark.astype(np.uint8) * 255,
    }

    masks: Dict[str, np.ndarray] = {}
    for key, mask in raw.items():
        mask = cv2.bitwise_and(mask, site_mask)
        if LAYER_SPECS[key].kind == "polygon":
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        else:
            mask = cv2.medianBlur(mask, 3)
            # Close broken/dotted DP linework without ballooning all line positions.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            if key == "grey_dotted_boundary_or_survey_line":
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        masks[key] = mask
    return masks


def dominant_site_angles(site_polygon_px: List[List[float]]) -> List[float]:
    angles: List[float] = []
    if len(site_polygon_px) >= 2:
        pts = [(float(p[0]), float(p[1])) for p in site_polygon_px]
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 20:
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                # Normalize orientation to 0..pi.
                if ang < 0:
                    ang += math.pi
                angles.append(ang)
    # Add orthogonal axes too, because many DP sheets have vertical/horizontal parcel lines.
    angles.extend([0.0, math.pi / 2])
    return angles


def angle_delta(a: float, b: float) -> float:
    d = abs((a - b + math.pi / 2) % math.pi - math.pi / 2)
    return d


def snap_angle(angle: float, doms: List[float], tol_deg: float = 8.0) -> float:
    if not doms:
        return angle
    best = min(doms, key=lambda x: angle_delta(angle, x))
    if angle_delta(angle, best) <= math.radians(tol_deg):
        return best
    return angle


def segment_from_hough_line(line: np.ndarray) -> Segment:
    x1, y1, x2, y2 = line[0]
    return float(x1), float(y1), float(x2), float(y2)


def line_angle(seg: Segment) -> float:
    x1, y1, x2, y2 = seg
    ang = math.atan2(y2 - y1, x2 - x1)
    if ang < 0:
        ang += math.pi
    return ang


def fit_segment_to_angle(seg: Segment, angle: float) -> Segment:
    x1, y1, x2, y2 = seg
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = math.cos(angle), math.sin(angle)
    return cx - ux * length / 2, cy - uy * length / 2, cx + ux * length / 2, cy + uy * length / 2


def merge_collinear_segments(segments: List[Segment], doms: List[float], max_perp_gap: float = 10.0, max_inline_gap: float = 55.0) -> List[Segment]:
    buckets: Dict[Tuple[int, int], List[Tuple[float, float, float, Point]]] = {}
    for seg in segments:
        x1, y1, x2, y2 = seg
        if math.hypot(x2 - x1, y2 - y1) < 14:
            continue
        ang = snap_angle(line_angle(seg), doms, tol_deg=9)
        seg = fit_segment_to_angle(seg, ang)
        ux, uy = math.cos(ang), math.sin(ang)
        nx, ny = -uy, ux
        cx, cy = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
        offset = cx * nx + cy * ny
        angle_bucket = round(ang / math.radians(3))
        offset_bucket = round(offset / max_perp_gap)
        t1 = seg[0] * ux + seg[1] * uy
        t2 = seg[2] * ux + seg[3] * uy
        buckets.setdefault((angle_bucket, offset_bucket), []).append((min(t1, t2), max(t1, t2), offset, (ux, uy)))

    out: List[Segment] = []
    for _, items in buckets.items():
        if not items:
            continue
        items.sort(key=lambda x: x[0])
        cur_a, cur_b, offset_sum, u = items[0]
        count = 1
        for a, b, off, u2 in items[1:]:
            if a <= cur_b + max_inline_gap:
                cur_b = max(cur_b, b)
                offset_sum += off
                count += 1
            else:
                off_avg = offset_sum / max(1, count)
                ux, uy = u
                nx, ny = -uy, ux
                out.append((ux * cur_a + nx * off_avg, uy * cur_a + ny * off_avg, ux * cur_b + nx * off_avg, uy * cur_b + ny * off_avg))
                cur_a, cur_b, offset_sum, u, count = a, b, off, u2, 1
        off_avg = offset_sum / max(1, count)
        ux, uy = u
        nx, ny = -uy, ux
        out.append((ux * cur_a + nx * off_avg, uy * cur_a + ny * off_avg, ux * cur_b + nx * off_avg, uy * cur_b + ny * off_avg))
    return out


def extract_line_vectors(mask: np.ndarray, category: str, doms: List[float], min_line_len: int) -> List[Dict[str, Any]]:
    h, w = mask.shape
    # V18.5.3 cloud-stability fix: remove isolated text/speckle components before Hough.
    # This prevents a Render/Railway worker from freezing on huge grey/dark masks.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    min_area = 10 if category in ("grey_dotted_boundary_or_survey_line", "dark_boundary_or_cad_line") else 16
    max_area = max(2500, int(w * h * 0.08))
    kept = 0
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 1.8 or area > 45 or category == "grey_dotted_boundary_or_survey_line":
            clean[labels == i] = 255
            kept += 1
        if kept > 3500:
            break
    if int(clean.sum()) > 0:
        mask = clean
    edges = cv2.Canny(mask, 40, 120)
    # Use the mask itself as well for very faint lines that don't create good Canny edges.
    lines = cv2.HoughLinesP(
        cv2.bitwise_or(edges, mask),
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(min_line_len * 0.35)),
        minLineLength=max(18, min_line_len),
        maxLineGap=max(10, int(min_line_len * 0.8)),
    )
    if lines is None:
        return []
    raw_segments = [segment_from_hough_line(l) for l in lines]
    merged = merge_collinear_segments(raw_segments, doms)
    spec = LAYER_SPECS[category]
    vectors: List[Dict[str, Any]] = []
    for i, seg in enumerate(merged):
        x1, y1, x2, y2 = seg
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_line_len * 0.75:
            continue
        vectors.append({
            "id": f"cv_{category}_{i}",
            "type": "line",
            "category": category,
            "label": spec.label,
            "layer": spec.layer,
            "color": list(spec.color),
            "confidence": int(min(93, 54 + length / max(w, h) * 100)),
            "source": "python_cv_draftsman_backend",
            "traceMethod": "hough_collinear_merge_angle_snap",
            "path": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
        })
    return vectors


def extract_polygon_vectors(mask: np.ndarray, category: str, min_area: float) -> List[Dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spec = LAYER_SPECS[category]
    vectors: List[Dict[str, Any]] = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:18]
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(2.8, min(18.0, peri * 0.012))
        approx = cv2.approxPolyDP(cnt, eps, True).reshape((-1, 2))
        if len(approx) < 3:
            continue
        path = [(float(x), float(y)) for x, y in approx]
        # Simplify once more but do not destroy irregular reservation boundaries.
        path = close_ring(rdp(path, max(2.2, eps * 0.55)))
        if len(path) < 4 or polygon_area(path) < min_area:
            continue
        vectors.append({
            "id": f"cv_{category}_{i}",
            "type": "polygon",
            "category": category,
            "label": spec.label,
            "layer": spec.layer,
            "color": list(spec.color),
            "confidence": int(min(94, 62 + area / (mask.shape[0] * mask.shape[1]) * 220)),
            "source": "python_cv_draftsman_backend",
            "traceMethod": "color_mask_contour_polygonize_simplify",
            "path": [[round(x, 2), round(y, 2)] for x, y in path],
        })
    return vectors


def build_findings(vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for i, v in enumerate(vectors, start=1):
        kind = "closed zone polygon" if v.get("type") == "polygon" else "drafted line segment"
        findings.append({
            "id": f"finding_{i}",
            "title": v.get("label") or v.get("category") or "DP feature",
            "category": v.get("category"),
            "layer": v.get("layer"),
            "status": "detected_near_plot",
            "confidence": v.get("confidence", 70),
            "note": f"Python CV extracted this as a {kind}. Review and lock before approval use.",
            "vectorId": v.get("id"),
        })
    return findings


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "DP Vision Tracer API", "version": "1.1.0"}


@app.get("/")
def root() -> Dict[str, str]:
    return health()


@app.post("/trace")
def trace(req: TraceRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    img = decode_data_url(req.image_data_url)
    h, w = img.shape[:2]
    if w < 40 or h < 40:
        raise ValueError("Crop is too small for DP tracing")

    # Server-side safety: even if the HTML sends a large crop, downscale before OpenCV.
    max_dim = int(req.options.get("server_max_dimension", 1450))
    scale_back = 1.0
    if max(h, w) > max_dim:
        scale_back = max(h, w) / max_dim
        img = cv2.resize(img, (int(w / scale_back), int(h / scale_back)), interpolation=cv2.INTER_AREA)
        req.site_polygon_px = [[float(x) / scale_back, float(y) / scale_back] for x, y in req.site_polygon_px]
        h, w = img.shape[:2]

    site_mask = make_site_mask((h, w), req.site_polygon_px, buffer_px=int(req.options.get("site_buffer_px", 80)))
    masks = color_masks(img, site_mask)
    doms = dominant_site_angles(req.site_polygon_px)

    min_dim = min(h, w)
    min_line_len = max(24, int(min_dim * float(req.options.get("min_line_ratio", 0.055))))
    min_poly_area = max(220.0, float(w * h) * float(req.options.get("min_polygon_area_ratio", 0.0022)))
    time_budget = float(req.options.get("server_time_budget_sec", 22))

    vectors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Priority order. Avoid running dark/grey masks first because they are the slowest/noisiest.
    ordered_categories = [
        "magenta_reservation_or_public_purpose",
        "yellow_orange_amenity_zone",
        "green_open_space_or_rg",
        "red_road_or_widening",
        "pale_red_survey_or_alignment_line",
        "pink_survey_or_reservation_line",
        "blue_water_nallah_or_drain",
        "grey_dotted_boundary_or_survey_line",
        "dark_boundary_or_cad_line",
    ]

    for category in ordered_categories:
        if time.perf_counter() - started > time_budget:
            warnings.append("Cloud API reached safe time budget, returned partial DP features instead of hanging.")
            break
        spec = LAYER_SPECS[category]
        mask = masks.get(category)
        if mask is None or int(mask.sum()) == 0:
            continue
        coverage = int(np.count_nonzero(mask)) / float(w * h)
        # Massive masks usually mean paper texture/text bleed. Skip slow Hough on them.
        if spec.kind == "line" and coverage > 0.22:
            warnings.append(f"Skipped noisy layer {spec.layer}; mask coverage was too high for stable drafting.")
            continue
        if spec.kind == "polygon":
            vectors.extend(extract_polygon_vectors(mask, category, min_poly_area))
        else:
            vectors.extend(extract_line_vectors(mask, category, doms, min_line_len))

    # De-duplicate tiny duplicate vectors by approximate bbox and layer.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for v in vectors:
        pts = v.get("path") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        key = (v.get("layer"), round(min(xs) / 8), round(min(ys) / 8), round(max(xs) / 8), round(max(ys) / 8), v.get("type"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)

    # Keep the response useful and lightweight for deck.gl.
    max_vectors = int(req.options.get("max_response_vectors", 80))
    deduped = sorted(deduped, key=lambda v: (0 if v.get("type") == "polygon" else 1, -float(v.get("confidence", 0))))[:max_vectors]

    if scale_back != 1.0:
        for v in deduped:
            pts = v.get("path") or []
            v["path"] = [[round(float(x) * scale_back, 2), round(float(y) * scale_back, 2)] for x, y in pts]
        warnings.append(f"Server downscaled crop by {scale_back:.2f}x for stable cloud processing, then scaled vectors back.")

    if not deduped:
        warnings.append("Python CV backend did not find stable DP features in the crop. Check alignment, crop clarity, or use a cleaner DP source.")

    return {
        "status": "ok",
        "extractionMode": "cloud_cv_draftsman_trace_v1_1",
        "imageSize": {"width": w, "height": h},
        "crop": req.crop,
        "vectors": deduped,
        "findings": build_findings(deduped),
        "warnings": warnings,
        "averageConfidence": int(round(np.mean([v.get("confidence", 0) for v in deduped]))) if deduped else None,
        "processingSec": round(time.perf_counter() - started, 2),
    }
