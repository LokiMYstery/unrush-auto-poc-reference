"""Minimal Volcengine V3 unidirectional streaming TTS client.

This module intentionally keeps Volcengine concepts explicit.  ``resource_id``
and ``speaker`` are caller-selected provider parameters, while the new-console
API key is read by the caller from a secret source and used only in the
WebSocket handshake.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException


VOLCENGINE_V3_UNIDIRECTIONAL_ENDPOINT = (
    "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"
)

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0x1
MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x9
MESSAGE_TYPE_AUDIO_ONLY_RESPONSE = 0xB
MESSAGE_TYPE_ERROR = 0xF

EVENT_SESSION_FINISHED = 152
EVENT_TTS_RESPONSE = 352
SUCCESS_STATUS_CODE = 20_000_000


class VolcengineProtocolError(RuntimeError):
    """Raised when Volcengine rejects a request or returns an invalid frame."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class VolcengineTimeoutError(VolcengineProtocolError):
    """Raised when a Volcengine request exceeds its bounded lifetime."""


class VolcengineConnectionError(VolcengineProtocolError):
    """Raised when the upstream WebSocket cannot be established or maintained."""


def load_api_key() -> str:
    """Load exactly one new-console API key source without logging its value."""

    environment_value = os.getenv("VOLCENGINE_TTS_API_KEY", "").strip()
    file_value = os.getenv("VOLCENGINE_TTS_API_KEY_FILE", "").strip()
    if environment_value and file_value:
        raise ValueError("Configure only one Volcengine API key source")
    if file_value:
        try:
            environment_value = Path(file_value).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError("Unable to read Volcengine API key file") from exc
    if not environment_value:
        raise ValueError("Volcengine API key is not configured")
    return environment_value


@dataclass(frozen=True)
class VolcengineServerFrame:
    message_type: int
    event: int | None
    session_id: str | None
    payload: bytes
    error_code: int | None = None

    def json_payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VolcengineProtocolError("Volcengine returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise VolcengineProtocolError("Volcengine returned a non-object JSON payload")
        return value


def new_console_headers(
    *,
    api_key: str,
    resource_id: str,
    request_id: str | None = None,
) -> dict[str, str]:
    """Build only the new-console authentication headers."""

    api_key = api_key.strip()
    resource_id = resource_id.strip()
    if not api_key:
        raise ValueError("Volcengine API key is required")
    if not resource_id:
        raise ValueError("Volcengine resource ID is required")
    return {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id or str(uuid4()),
    }


def build_send_text_frame(
    *,
    text: str,
    speaker: str,
    uid: str,
    sample_rate: int = 24_000,
) -> bytes:
    """Encode the V3 SendText full-client request frame."""

    text = text.strip()
    speaker = speaker.strip()
    uid = uid.strip()
    if not text:
        raise ValueError("text is required")
    if not speaker:
        raise ValueError("speaker is required")
    if not uid:
        raise ValueError("uid is required")
    if sample_rate != 24_000:
        raise ValueError("Phase E smoke test requires 24 kHz PCM")

    payload = json.dumps(
        {
            "user": {"uid": uid},
            "req_params": {
                "text": text,
                "speaker": speaker,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": sample_rate,
                },
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    # v1 protocol, 4-byte header; full-client request with no event number;
    # JSON serialization; no compression.
    header = bytes((0x11, MESSAGE_TYPE_FULL_CLIENT_REQUEST << 4, 0x10, 0x00))
    return header + struct.pack(">I", len(payload)) + payload


def parse_server_frame(data: bytes) -> VolcengineServerFrame:
    """Parse the V3 server frame shapes used by unidirectional TTS."""

    if len(data) < 4:
        raise VolcengineProtocolError("Volcengine frame is shorter than its header")
    version = data[0] >> 4
    header_size = (data[0] & 0x0F) * 4
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    if version != 1 or header_size < 4 or len(data) < header_size:
        raise VolcengineProtocolError("Volcengine frame header is invalid")

    cursor = header_size
    if message_type == MESSAGE_TYPE_ERROR:
        error_code, cursor = _read_u32(data, cursor, "error code")
        payload, cursor = _read_sized_bytes(data, cursor, "error payload")
        _require_consumed(data, cursor)
        payload = _decompress(payload, compression)
        return VolcengineServerFrame(
            message_type=message_type,
            event=None,
            session_id=None,
            payload=payload,
            error_code=error_code,
        )

    if flags != 0x4:
        raise VolcengineProtocolError(f"Unsupported Volcengine frame flags: {flags}")

    event, cursor = _read_u32(data, cursor, "event")
    session_bytes, cursor = _read_sized_bytes(data, cursor, "session ID")
    payload, cursor = _read_sized_bytes(data, cursor, "payload")
    _require_consumed(data, cursor)
    try:
        session_id = session_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VolcengineProtocolError("Volcengine session ID is invalid") from exc
    return VolcengineServerFrame(
        message_type=message_type,
        event=event,
        session_id=session_id,
        payload=_decompress(payload, compression),
    )


async def stream_pcm(
    *,
    api_key: str,
    resource_id: str,
    speaker: str,
    text: str,
    uid: str,
    endpoint: str = VOLCENGINE_V3_UNIDIRECTIONAL_ENDPOINT,
    timeout_seconds: float = 30.0,
    request_id: str | None = None,
    handshake_observer: Any | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM chunks from one Volcengine V3 synthesis request."""

    headers = new_console_headers(
        api_key=api_key,
        resource_id=resource_id,
        request_id=request_id,
    )
    request = build_send_text_frame(text=text, speaker=speaker, uid=uid)
    try:
        async with asyncio.timeout(timeout_seconds):
            async with connect(
                endpoint,
                additional_headers=headers,
                compression=None,
                proxy=None,
                open_timeout=min(timeout_seconds, 10.0),
                close_timeout=3.0,
                max_size=None,
            ) as websocket:
                if handshake_observer is not None:
                    handshake_observer(
                        headers["X-Api-Request-Id"],
                        websocket.response.headers.get("X-Tt-Logid"),
                    )
                await websocket.send(request)
                async for message in websocket:
                    if not isinstance(message, bytes):
                        raise VolcengineProtocolError("Volcengine returned a text WebSocket message")
                    frame = parse_server_frame(message)
                    if frame.message_type == MESSAGE_TYPE_ERROR:
                        raise VolcengineProtocolError(
                            _error_message(frame.payload),
                            code=frame.error_code,
                        )
                    if frame.event == EVENT_TTS_RESPONSE:
                        if frame.message_type != MESSAGE_TYPE_AUDIO_ONLY_RESPONSE:
                            raise VolcengineProtocolError("TTSResponse was not an audio frame")
                        if frame.payload:
                            yield frame.payload
                        continue
                    if frame.event == EVENT_SESSION_FINISHED:
                        if frame.message_type != MESSAGE_TYPE_FULL_SERVER_RESPONSE:
                            raise VolcengineProtocolError("SessionFinished had an invalid frame type")
                        result = frame.json_payload()
                        status_code = result.get("status_code")
                        normalized_status_code = (
                            int(status_code)
                            if isinstance(status_code, str) and status_code.isdigit()
                            else status_code
                        )
                        # The live V3 service may return an empty JSON object on
                        # successful SessionFinished.  Error outcomes use the
                        # dedicated error frame; validate status_code only when
                        # the provider includes it.
                        if normalized_status_code is not None and normalized_status_code != SUCCESS_STATUS_CODE:
                            raise VolcengineProtocolError(
                                str(result.get("message") or "Volcengine synthesis failed"),
                                code=normalized_status_code if isinstance(normalized_status_code, int) else None,
                            )
                        return
                raise VolcengineProtocolError("Volcengine closed before SessionFinished")
    except TimeoutError as exc:
        raise VolcengineTimeoutError("Volcengine synthesis timed out") from exc
    except (OSError, WebSocketException) as exc:
        raise VolcengineConnectionError(f"Volcengine WebSocket failed: {type(exc).__name__}") from exc


def _read_u32(data: bytes, cursor: int, label: str) -> tuple[int, int]:
    if len(data) < cursor + 4:
        raise VolcengineProtocolError(f"Volcengine frame is missing {label}")
    return struct.unpack_from(">I", data, cursor)[0], cursor + 4


def _read_sized_bytes(data: bytes, cursor: int, label: str) -> tuple[bytes, int]:
    size, cursor = _read_u32(data, cursor, f"{label} size")
    end = cursor + size
    if end > len(data):
        raise VolcengineProtocolError(f"Volcengine frame has a truncated {label}")
    return data[cursor:end], end


def _require_consumed(data: bytes, cursor: int) -> None:
    if cursor != len(data):
        raise VolcengineProtocolError("Volcengine frame has unexpected trailing bytes")


def _decompress(payload: bytes, compression: int) -> bytes:
    if compression == 0:
        return payload
    if compression == 1:
        try:
            return gzip.decompress(payload)
        except OSError as exc:
            raise VolcengineProtocolError("Volcengine returned invalid gzip data") from exc
    raise VolcengineProtocolError(f"Unsupported Volcengine compression: {compression}")


def _error_message(payload: bytes) -> str:
    try:
        decoded = json.loads(payload.decode("utf-8"))
        if isinstance(decoded, dict):
            return str(decoded.get("message") or decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return payload.decode("utf-8", errors="replace") or "Volcengine provider error"
