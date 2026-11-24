from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


MAX_REQUEST_BODY_BYTES = 1_048_576
MAX_PROMPT_CHARS = 60_000


class PayloadTooLarge(ValueError):
    """The request body or prompt exceeds the configured input limit."""


def read_json_body(
    headers: Mapping[str, str],
    read_body: Callable[[int], bytes],
) -> dict[str, Any]:
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        raise ValueError("Content-Length header is required")

    try:
        length = int(raw_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("Content-Length must be an integer") from exc

    if length <= 0:
        raise ValueError("Content-Length must be positive")
    if length > MAX_REQUEST_BODY_BYTES:
        raise PayloadTooLarge("request body exceeds the 1 MiB limit")

    raw = read_body(length)
    if len(raw) != length:
        raise ValueError("incomplete request body")

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("request body must contain valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required and must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PayloadTooLarge("prompt exceeds the 60000-character limit")

    return payload
