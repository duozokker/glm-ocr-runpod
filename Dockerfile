FROM vllm/vllm-openai:nightly

# git is required for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install bleeding-edge transformers (v5+) required by GLM-OCR
RUN pip uninstall -y transformers || true \
    && pip install -U git+https://github.com/huggingface/transformers.git

# Bake model weights into the image to eliminate cold start downloads (~2 GB)
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

EXPOSE 8080

CMD [ \
    "vllm", "serve", "zai-org/GLM-OCR", \
    "--port", "8080", \
    "--allowed-local-media-path", "/tmp", \
    "--gpu-memory-utilization", "0.95", \
    "--speculative-config", "{\"method\": \"mtp\", \"num_speculative_tokens\": 1}" \
]
