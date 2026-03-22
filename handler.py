"""
RunPod Serverless handler for GLM-OCR via vLLM.

Runpod Hub expects a handler.py in the repository root (or .runpod/).
This file is the canonical entrypoint used by the Docker image.
"""

import os
import subprocess
import sys
import time

import requests
import runpod

VLLM_HOST = "http://localhost:8000"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "")
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "180"))


def start_vllm():
    """Start the local vLLM server and wait for health checks to pass."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--port",
        "8000",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 1}',
    ]

    if MAX_MODEL_LEN:
        cmd.extend(["--max-model-len", MAX_MODEL_LEN])

    print(f"Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    for i in range(STARTUP_TIMEOUT):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            response = requests.get(f"{VLLM_HOST}/health", timeout=2)
            if response.status_code == 200:
                print(f"vLLM ready after {i + 1}s")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    raise RuntimeError(f"vLLM failed to start within {STARTUP_TIMEOUT}s")


def handler(job):
    """Process a single RunPod job."""
    job_input = job["input"]

    if "openai_route" in job_input:
        route = job_input["openai_route"]
        body = job_input.get("openai_input", {})
    elif "messages" in job_input:
        route = "/v1/chat/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    elif "prompt" in job_input:
        route = "/v1/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    else:
        return {"error": "Invalid input: expected 'messages', 'prompt', or 'openai_route'"}

    try:
        response = requests.post(f"{VLLM_HOST}{route}", json=body, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def main():
    start_vllm()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
