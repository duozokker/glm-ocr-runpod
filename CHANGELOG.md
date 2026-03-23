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
