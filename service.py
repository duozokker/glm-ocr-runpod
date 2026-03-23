import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from flask import Flask, Response, jsonify, request
from glmocr import GlmOcr


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
VLLM_HOST = os.getenv("VLLM_HOST", "http://127.0.0.1:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "glm-ocr")
GPU_COST_PER_SEC = float(os.getenv("GPU_COST_PER_SEC", "0.00016"))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "900"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))
GLMOCR_CONFIG_PATH = os.getenv("GLMOCR_CONFIG_PATH", "/app/glmocr.config.yaml")
GLMOCR_LAYOUT_DEVICE = os.getenv("GLMOCR_LAYOUT_DEVICE", "cpu")
HEALTH_POLL_INTERVAL = float(os.getenv("HEALTH_POLL_INTERVAL", "1.0"))


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RequestSample:
    document: str
    elapsed_seconds: float
    pages: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.boot_started_at = time.time()
        self.vllm_ready_at: float | None = None
        self.pipeline_ready_at: float | None = None
        self.total_requests = 0
        self.total_documents = 0
        self.total_pages = 0
        self.total_request_seconds = 0.0
        self.total_cost_usd = 0.0
        self.last_samples: deque[RequestSample] = deque(maxlen=100)

    def mark_vllm_ready(self) -> None:
        with self.lock:
            self.vllm_ready_at = time.time()

    def mark_pipeline_ready(self) -> None:
        with self.lock:
            self.pipeline_ready_at = time.time()

    def record_request(self, documents: list[dict[str, Any]], elapsed_seconds: float) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_request_seconds += elapsed_seconds
            for document in documents:
                sample = RequestSample(
                    document=document["document"],
                    elapsed_seconds=float(document["elapsed_seconds"]),
                    pages=int(document["pages"]),
                    cost_usd=float(document["estimated_cost_usd"]),
                )
                self.total_documents += 1
                self.total_pages += sample.pages
                self.total_cost_usd += sample.cost_usd
                self.last_samples.append(sample)

    def snapshot(self, stage: str, ready: bool) -> dict[str, Any]:
        with self.lock:
            avg_doc_seconds = self.total_request_seconds / self.total_documents if self.total_documents else None
            avg_page_seconds = self.total_request_seconds / self.total_pages if self.total_pages else None
            return {
                "stage": stage,
                "ready": ready,
                "uptime_seconds": round(time.time() - self.boot_started_at, 3),
                "boot": {
                    "started_at": self.boot_started_at,
                    "vllm_ready_after_seconds": None
                    if self.vllm_ready_at is None
                    else round(self.vllm_ready_at - self.boot_started_at, 3),
                    "pipeline_ready_after_seconds": None
                    if self.pipeline_ready_at is None
                    else round(self.pipeline_ready_at - self.boot_started_at, 3),
                },
                "totals": {
                    "requests": self.total_requests,
                    "documents": self.total_documents,
                    "pages": self.total_pages,
                    "request_seconds": round(self.total_request_seconds, 3),
                    "estimated_cost_usd": round(self.total_cost_usd, 6),
                    "avg_seconds_per_document": None if avg_doc_seconds is None else round(avg_doc_seconds, 3),
                    "avg_seconds_per_page": None if avg_page_seconds is None else round(avg_page_seconds, 3),
                    "estimated_cost_per_1000_documents_usd": None
                    if self.total_documents == 0
                    else round((self.total_cost_usd / self.total_documents) * 1000, 4),
                    "estimated_cost_per_1000_pages_usd": None
                    if self.total_pages == 0
                    else round((self.total_cost_usd / self.total_pages) * 1000, 4),
                },
                "recent_samples": [
                    {
                        "document": sample.document,
                        "elapsed_seconds": round(sample.elapsed_seconds, 3),
                        "pages": sample.pages,
                        "cost_usd": round(sample.cost_usd, 6),
                        "timestamp": sample.timestamp,
                    }
                    for sample in self.last_samples
                ],
            }


class ServiceState:
    def __init__(self) -> None:
        self.stage = "booting"
        self.ready = False
        self.vllm_process: subprocess.Popen[str] | None = None
        self.parser: GlmOcr | None = None
        self.metrics = Metrics()
        self.error: str | None = None
        self.lock = threading.Lock()

    def set_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error
            self.stage = "failed"
            self.ready = False

    def mark_ready(self) -> None:
        with self.lock:
            self.stage = "ready"
            self.ready = True
            self.error = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = self.metrics.snapshot(self.stage, self.ready)
            if self.error:
                data["error"] = self.error
            return data


state = ServiceState()
app = Flask(__name__)


def build_vllm_command() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--port",
        "8080",
        "--gpu-memory-utilization",
        os.getenv("GPU_MEMORY_UTILIZATION", "0.95"),
        "--max-model-len",
        os.getenv("MAX_MODEL_LEN", "16384"),
        "--allowed-local-media-path",
        "/",
    ]

    if env_flag("TRUST_REMOTE_CODE", True):
        cmd.append("--trust-remote-code")

    speculative = os.getenv(
        "SPECULATIVE_CONFIG",
        '{"method":"mtp","num_speculative_tokens":1}',
    )
    if speculative:
        cmd.extend(["--speculative-config", speculative])

    max_num_seqs = os.getenv("MAX_NUM_SEQS")
    if max_num_seqs:
        cmd.extend(["--max-num-seqs", max_num_seqs])

    limit_mm_per_prompt = os.getenv("LIMIT_MM_PER_PROMPT")
    if limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])

    extra_args = os.getenv("VLLM_EXTRA_ARGS")
    if extra_args:
        cmd.extend(extra_args.split())

    return cmd


def wait_for_vllm() -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        process = state.vllm_process
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            response = requests.get(f"{VLLM_HOST}/health", timeout=2)
            if response.status_code == 200:
                state.metrics.mark_vllm_ready()
                return
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    raise TimeoutError(f"vLLM did not become healthy within {STARTUP_TIMEOUT}s")


def startup_worker() -> None:
    try:
        state.set_stage("starting_vllm")
        cmd = build_vllm_command()
        print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
        state.vllm_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
        wait_for_vllm()

        state.set_stage("starting_glmocr")
        state.parser = GlmOcr(
            config_path=GLMOCR_CONFIG_PATH,
            mode="selfhosted",
            layout_device=GLMOCR_LAYOUT_DEVICE,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        state.metrics.mark_pipeline_ready()
        state.mark_ready()
        print("GLM-OCR service is ready", flush=True)
    except Exception as exc:
        state.set_error(str(exc))
        print(f"Startup failed: {exc}", flush=True)


def page_count_from_result(result: Any) -> int:
    json_result = getattr(result, "json_result", None)
    if isinstance(json_result, list):
        return len(json_result)
    return 1


def build_document_response(document: str, result: Any, elapsed_seconds: float) -> dict[str, Any]:
    pages = max(page_count_from_result(result), 1)
    cost = elapsed_seconds * GPU_COST_PER_SEC
    return {
        "document": document,
        "pages": pages,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "pages_per_second": round(pages / elapsed_seconds, 4) if elapsed_seconds > 0 else None,
        "estimated_cost_usd": round(cost, 6),
        "markdown_result": getattr(result, "markdown_result", ""),
        "json_result": getattr(result, "json_result", None),
        "original_images": getattr(result, "original_images", [document]),
    }


def save_result_if_requested(result: Any, output_dir: str | None, save_layout_visualization: bool) -> None:
    if not output_dir:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.save(
        output_dir=output_dir,
        save_layout_visualization=save_layout_visualization,
    )


def proxy_openai_request(route: str) -> Response:
    url = f"{VLLM_HOST}{route}"
    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            data=request.get_data(),
            params=request.args,
            timeout=REQUEST_TIMEOUT,
        )
        excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        headers = [(name, value) for name, value in response.raw.headers.items() if name.lower() not in excluded_headers]
        return Response(response.content, response.status_code, headers)
    except requests.RequestException as exc:
        return jsonify({"error": str(exc)}), 502


@app.get("/health")
@app.get("/ready")
def health() -> tuple[Response, int] | Response:
    snapshot = state.snapshot()
    if snapshot["ready"]:
        return jsonify(snapshot)
    return jsonify(snapshot), 503


@app.get("/metrics")
def metrics() -> Response:
    return jsonify(state.snapshot())


@app.get("/openai/v1/models")
def openai_models() -> Response:
    return proxy_openai_request("/v1/models")


@app.post("/openai/v1/chat/completions")
def openai_chat_completions() -> Response:
    return proxy_openai_request("/v1/chat/completions")


@app.post("/glmocr/parse")
def glmocr_parse() -> tuple[Response, int] | Response:
    if not state.ready or state.parser is None:
        return jsonify(state.snapshot()), 503

    payload = request.get_json(silent=True) or {}
    documents = payload.get("documents") or payload.get("images")
    if not documents:
        single = payload.get("document") or payload.get("image")
        documents = [single] if single else []
    documents = [document for document in documents if document]

    if not documents:
        return jsonify({"error": "Expected 'document' or 'documents' in request body."}), 400

    save_layout_visualization = bool(payload.get("save_layout_visualization", False))
    output_dir = payload.get("output_dir")
    include_results = bool(payload.get("include_results", True))

    request_started = time.perf_counter()
    parsed_documents: list[dict[str, Any]] = []

    try:
        for document in documents:
            document_started = time.perf_counter()
            result = state.parser.parse(
                document,
                save_layout_visualization=save_layout_visualization,
            )
            document_elapsed = time.perf_counter() - document_started
            if output_dir:
                save_result_if_requested(result, output_dir, save_layout_visualization)
            parsed = build_document_response(document, result, document_elapsed)
            if not include_results:
                parsed.pop("markdown_result", None)
                parsed.pop("json_result", None)
            parsed_documents.append(parsed)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    request_elapsed = time.perf_counter() - request_started
    state.metrics.record_request(parsed_documents, request_elapsed)

    total_pages = sum(document["pages"] for document in parsed_documents)
    total_cost = sum(document["estimated_cost_usd"] for document in parsed_documents)
    return jsonify(
        {
            "documents": parsed_documents,
            "summary": {
                "documents": len(parsed_documents),
                "pages": total_pages,
                "elapsed_seconds": round(request_elapsed, 3),
                "pages_per_second": round(total_pages / request_elapsed, 4) if request_elapsed > 0 else None,
                "estimated_cost_usd": round(total_cost, 6),
                "estimated_cost_per_1000_documents_usd": round((total_cost / len(parsed_documents)) * 1000, 4),
                "estimated_cost_per_1000_pages_usd": round((total_cost / total_pages) * 1000, 4)
                if total_pages > 0
                else None,
            },
        }
    )


@app.post("/warmup")
def warmup() -> tuple[Response, int] | Response:
    if not state.ready:
        return jsonify(state.snapshot()), 503

    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9YlR5X0AAAAASUVORK5CYII="
    )
    with tempfile.TemporaryDirectory(prefix="glmocr_warmup_") as output_dir:
        result = state.parser.parse(tiny_png, save_layout_visualization=False)
        save_result_if_requested(result, output_dir, False)
    return jsonify({"status": "ok"})


def shutdown() -> None:
    parser = state.parser
    if parser is not None:
        try:
            parser.close()
        except Exception:
            pass

    process = state.vllm_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    threading.Thread(target=startup_worker, daemon=True).start()
    try:
        app.run(host=APP_HOST, port=APP_PORT)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
