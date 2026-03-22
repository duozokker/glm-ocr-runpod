#!/usr/bin/env python3
"""
Test script for GLM-OCR on RunPod Serverless.

Supports:
  - Remote images (URL)
  - Local images (auto-converted to base64)
  - All GLM-OCR prompt types: text, formula, table, JSON schema

Usage:
  python test_endpoint.py --image https://example.com/document.png
  python test_endpoint.py --image ./invoice.jpg --prompt "table"
  python test_endpoint.py --image ./id_card.png --prompt '{"name": "", "dob": ""}'

Environment (.env or exported):
  RUNPOD_API_KEY      - Your RunPod API key
  RUNPOD_ENDPOINT_ID  - Your serverless endpoint ID
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional if env vars are exported directly

try:
    import requests
except ImportError:
    print("Missing dependency: pip install -r requirements.txt")
    sys.exit(1)


PROMPT_SHORTCUTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


def image_to_base64_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    suffix = p.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(suffix, "image/png")

    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def resolve_prompt(raw: str) -> str:
    return PROMPT_SHORTCUTS.get(raw.lower(), raw)


def call_endpoint(
    api_key: str,
    endpoint_id: str,
    image: str,
    prompt: str,
    timeout: int = 120,
) -> dict:
    """Call RunPod serverless endpoint via OpenAI-compatible proxy."""

    if image.startswith(("http://", "https://", "data:")):
        image_url = image
    else:
        image_url = image_to_base64_url(image)

    payload = {
        "model": "zai-org/GLM-OCR",
        "max_tokens": 16384,
        "temperature": 0.01,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    # Use RunPod's OpenAI-compatible proxy endpoint
    url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Sending request to {endpoint_id}...")
    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()

    print(f"\nTotal:     {elapsed:.2f}s")
    print(f"{'='*60}")

    # Standard OpenAI response format
    choices = result.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        print(content)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def main():
    parser = argparse.ArgumentParser(description="Test GLM-OCR on RunPod Serverless")
    parser.add_argument(
        "--image", required=True, help="Image path (local file or URL)"
    )
    parser.add_argument(
        "--prompt",
        default="text",
        help='Prompt: "text", "formula", "table", or custom string/JSON schema',
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RUNPOD_API_KEY"),
        help="RunPod API key (or set RUNPOD_API_KEY env var)",
    )
    parser.add_argument(
        "--endpoint-id",
        default=os.getenv("RUNPOD_ENDPOINT_ID"),
        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key or RUNPOD_API_KEY required")
        sys.exit(1)
    if not args.endpoint_id:
        print("Error: --endpoint-id or RUNPOD_ENDPOINT_ID required")
        sys.exit(1)

    prompt = resolve_prompt(args.prompt)
    print(f"Image:  {args.image}")
    print(f"Prompt: {prompt}")
    print()

    call_endpoint(args.api_key, args.endpoint_id, args.image, prompt, args.timeout)


if __name__ == "__main__":
    main()
