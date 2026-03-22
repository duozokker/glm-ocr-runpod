FROM runpod/worker-v1-vllm:stable-cuda12.1.0

# git is required for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install transformers from main branch (required by GLM-OCR, not yet on PyPI)
RUN pip uninstall -y transformers || true \
    && pip install -U git+https://github.com/huggingface/transformers.git

# Bake model weights into the image to eliminate cold start downloads (~2 GB)
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

# The worker-v1-vllm image has its own handler — configure via env vars
ENV MODEL_NAME="zai-org/GLM-OCR"
ENV GPU_MEMORY_UTILIZATION="0.95"
ENV MAX_MODEL_LEN="4096"
ENV DISABLE_LOG_STATS="true"
