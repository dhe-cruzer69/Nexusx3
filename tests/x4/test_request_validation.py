from __future__ import annotations

from io import BytesIO

import pytest

from x4.request_validation import (
    MAX_PROMPT_CHARS,
    MAX_REQUEST_BODY_BYTES,
    PayloadTooLarge,
    read_json_body,
)


def parse(raw: bytes) -> dict[str, object]:
    return read_json_body({"Content-Length": str(len(raw))}, BytesIO(raw).read)


def test_valid_chat_request_is_parsed() -> None:
    assert parse(b'{"prompt":"hello","estimated_budget":0.01}') == {
        "prompt": "hello",
        "estimated_budget": 0.01,
    }


@pytest.mark.parametrize("value", [None, "", "abc", "1.5", "1e2"])
def test_rejects_missing_or_invalid_content_length(value: str | None) -> None:
    headers = {} if value is None else {"Content-Length": value}
    with pytest.raises(ValueError, match="Content-Length"):
        read_json_body(headers, lambda size: b"")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_rejects_non_positive_content_length(value: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        read_json_body({"Content-Length": value}, lambda size: b"")


def test_rejects_oversized_body_before_reading() -> None:
    def should_not_read(size: int) -> bytes:
        pytest.fail("oversized body must be rejected before reading")

    with pytest.raises(PayloadTooLarge):
        read_json_body(
            {"Content-Length": str(MAX_REQUEST_BODY_BYTES + 1)},
            should_not_read,
        )


def test_rejects_incomplete_body() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        read_json_body({"Content-Length": "10"}, lambda size: b"{}")


def test_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse(b'{"prompt":')


def test_rejects_non_object_json() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        parse(b'["hello"]')


@pytest.mark.parametrize("payload", [{}, {"prompt": None}, {"prompt": ""}, {"prompt": "  "}])
def test_rejects_missing_or_empty_prompt(payload: dict[str, object]) -> None:
    import json

    with pytest.raises(ValueError, match="prompt"):
        parse(json.dumps(payload).encode())


def test_rejects_oversized_prompt() -> None:
    import json

    raw = json.dumps({"prompt": "x" * (MAX_PROMPT_CHARS + 1)}).encode()
    with pytest.raises(PayloadTooLarge, match="prompt"):
        parse(raw)
