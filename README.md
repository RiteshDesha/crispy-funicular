# DP Vision Cloud API v1.1.0

This is the hosted backend for the one-file Thane TestFit Studio HTML app.

## What changed in v1.1.0

- Added `/` health route as well as `/health`.
- Added server-side image downscaling so large DP crops do not hang the API.
- Added safe time budget so the API returns partial results instead of freezing.
- Added noisy-mask skipping for huge dark/grey paper texture regions.
- Kept CORS open for one-file HTML usage.

## Deploy on Render

1. Upload this folder to GitHub.
2. Create a Render Web Service from that repo.
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

5. Copy the Render base URL, for example:

```text
https://dp-vision-tracer-api.onrender.com
```

6. Paste only the base URL into the HTML app. Do not paste `/trace` at the end.
7. Click **Test** first, then **Analyze DP Map**.

The HTML app now uses Cloud Vision Pro only by default. It will not silently fall back to the old browser trace unless Browser Basic Trace is manually selected.
