import json
from typing import Any, Union

import httpx

from tests.unit.vcr import CustomVCR


def _parse_sse(content: str) -> list[Union[dict[str, Any], str]]:
    """Parse SSE content into a list of parsed JSON events or the literal '[DONE]' sentinel."""
    events: list[dict[str, Any] | str] = []
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"Unexpected SSE line: {block!r}"
        payload = block[len("data: ") :]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


class TestVercelChatStreamRouter:
    async def test_basic_chat_completion(
        self,
        httpx_client: httpx.AsyncClient,
        anthropic_api_key: str,  # noqa: ARG002
        custom_vcr: CustomVCR,
    ) -> None:
        """Test that a simple message is sent to Anthropic and returns a streamed response."""
        params = {
            "provider_type": "builtin",
            "provider": "ANTHROPIC",
            "model_name": "claude-haiku-4-5-20251001",
        }
        body = {
            "trigger": "submit-message",
            "id": "test-session-1",
            "messages": [
                {
                    "id": "test-msg-1",
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "What is the capital of France? Answer in one word.",
                        }
                    ],
                }
            ],
        }
        with custom_vcr.use_cassette():
            response = await httpx_client.post("/vercel_chat_stream", params=params, json=body)

        assert response.status_code == 200
        events = _parse_sse(response.text)

        # Verify overall event sequence structure
        types = [e["type"] if isinstance(e, dict) else e for e in events]
        assert types[0] == "start"
        assert types[1] == "start-step"
        assert "text-start" in types
        assert "text-delta" in types
        assert "text-end" in types
        assert types[-3] == "finish-step"
        assert types[-2] == "finish"
        assert types[-1] == "[DONE]"

        # Verify the finish event signals a clean stop
        finish_event = next(e for e in events if isinstance(e, dict) and e.get("type") == "finish")
        assert finish_event["finishReason"] == "stop"

        # Verify text-start, text-delta, text-end all share the same stream ID
        text_events = [
            e
            for e in events
            if isinstance(e, dict) and e.get("type") in ("text-start", "text-delta", "text-end")
        ]
        stream_ids = {e["id"] for e in text_events}
        assert len(stream_ids) == 1, "text-start, text-delta, text-end should share one stream ID"

        # Verify the full response text contains the expected answer
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text-delta"]
        assert text_deltas, "Expected at least one text-delta event"
        full_text = "".join(e["delta"] for e in text_deltas)
        assert isinstance(full_text, str)
        assert "Paris" in full_text
