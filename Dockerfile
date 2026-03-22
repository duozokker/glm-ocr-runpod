FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl build-essential && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

# Install vLLM nightly (pip wheels built for CUDA 12.1, compatible with 12.4)
RUN pip3 install --break-system-packages \
    vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install transformers from main branch (required by GLM-OCR, needs >=5.0.0)
RUN pip3 install --break-system-packages \
    git+https://github.com/huggingface/transformers.git

# Install RunPod SDK and requests for handler
RUN pip3 install --break-system-packages runpod requests

# Bake model weights into the image (~2 GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
