# DP Vision API Backend v0.4.0 Recovery Build

This backend fixes the issue where adding aggressive TILR detection could leave the frontend with zero DP features.

## What changed

- Zone patches and TILR lines are detected as separate feature families.
- TILR lines cannot push zone patches out of the returned result.
- If the normal site-buffer scan finds no zone patches, the API runs a recovery scan over the submitted crop.
- The API returns `debugCounts` so the HTML app can show whether the backend found patches, vectors, or only TILR lines.

## Render settings

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Health check:

```text
/health
```

Expected version:

```text
0.4.0
```
