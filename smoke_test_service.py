#!/usr/bin/env python3

import argparse
import json
import os
import time

import requests


DEFAULT_SINGLE_IMAGE = (
    "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/"
    "wpf272043/keepme/image/receipt.png"
)


def wait_until_ready(base_url: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    last_payload = {}
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
            last_payload = response.json()
            if response.status_code == 200 and last_payload.get("ready"):
                return last_payload
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Service at {base_url} did not become ready within {timeout}s. Last payload: {last_payload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the GLM-OCR service")
    parser.add_argument("--base-url", default=os.getenv("GLMOCR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--single-image", default=DEFAULT_SINGLE_IMAGE)
    parser.add_argument("--document")
    args = parser.parse_args()

    print(json.dumps({"health": wait_until_ready(args.base_url, args.timeout)}, indent=2))

    single = requests.post(
        f"{args.base_url.rstrip('/')}/ocr/single",
        json={"image": args.single_image},
        timeout=args.timeout,
    )
    single.raise_for_status()
    print(json.dumps({"single": single.json()}, indent=2))

    if args.document:
        document = requests.post(
            f"{args.base_url.rstrip('/')}/glmocr/parse",
            json={"document": args.document, "include_results": False},
            timeout=args.timeout,
        )
        document.raise_for_status()
        print(json.dumps({"document": document.json()}, indent=2))


if __name__ == "__main__":
    main()
