# DP Vision API Backend

This is the hosted backend for the updated Thane TestFit Studio DP workflow.

The HTML file remains one standalone frontend. Official DP polygon segmentation is done here, not in the browser.

## What it does

- Receives the aligned/cropped DP image from the HTML app.
- Uses OpenCV to segment filled DP colour regions.
- Uses OCR through Tesseract when available.
- Classifies polygons with colour plus OCR dictionary matching.
- Detects red dashed TILR / survey-number lines as a separate `DP_TILR_SURVEY_LINES` line layer.
- Returns clean CAD-style `patches` / `vectors` with zone polygons and separate TILR/survey line vectors:
  - label
  - layer
  - geometry in crop pixel coordinates
  - confidence
  - reason
  - review status

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then put this in the HTML DP Vision API URL field:

```text
http://localhost:8000
```

## Render deployment

Render start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Build command:

```bash
pip install -r requirements.txt
```

Health URL:

```text
/health
```

Trace endpoint:

```text
POST /trace
```

## Important

This backend is a real segmentation/OCR starter, but every DP sheet still needs architect verification. The frontend review panel exists because OCR and old scanned DP sheets will never be 100 percent automatic.
