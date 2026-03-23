# Changelog

## v0.2.0 - 2026-03-23

- Replaced the queue-based worker with a simpler HTTP service in `service.py`
- Switched to the official `glmocr[selfhosted]` PDF/layout pipeline
- Added startup, throughput, and cost metrics
- Added DATEV benchmark tooling for real-world PDF measurements
- Preloaded both GLM-OCR and PP-DocLayoutV3 in the Docker image
- Updated the repo docs for RunPod Load Balancing deployment
- Strengthened the default OCR prompt for better fidelity on complex pages
- Cleaned the RunPod metadata and entrypoint shims

## v0.2.1 - 2026-03-23

- Added a dedicated `/ocr/single` endpoint for fast single-page OCR
- Kept `/glmocr/parse` as the explicit full document OCR path
- Imported the stronger Markdown-preserving fallback prompt from the local OCR workbench idea
- Added server-side PDF page rendering for single-page OCR fallback
- Added smoke tests using the public receipt example from the official vLLM GLM-OCR recipe
