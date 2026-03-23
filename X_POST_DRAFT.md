GLM-OCR is now running fully self-hosted on RunPod with the official PDF/layout pipeline.

What changed:
- dropped queue-worker complexity
- switched to `glmocr[selfhosted]`
- split the API into `single OCR` and `full document OCR`
- added health, metrics, and cost tracking
- preloaded GLM-OCR + PP-DocLayoutV3 into the image
- benchmarked against large DATEV-style PDFs

Goal:
EU-hosted OCR without depending on external MaaS OCR APIs.

Repo:
https://github.com/duozokker/glm-ocr-runpod

Release:
v0.2.0
