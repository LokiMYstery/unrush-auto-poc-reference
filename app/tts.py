from __future__ import annotations

import os
import struct
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

from app.config import VOICE_DIR, tts_enabled
from app.provider.volcengine_v3 import load_api_key, stream_pcm
from app import store


WELCOME_TEXT = {
    "zh": "欢迎回来，我先放一段稳定的声音陪你进入状态。",
    "en": "Welcome back. I’ll start with something steady while you settle in.",
}
WELCOME_VOICE_IDS: dict[str, str | None] = {"zh": None, "en": None}
SYNTHESIS_COUNT = 0
DEFAULT_VOICES = {
    # Original poc_copy voice/catalog.json; Chinese remains on its original public voice.
    "zh": ("zh_female_jitangnv_uranus_bigtts", "seed-tts-2.0"),
    "en": ("", ""),
}


def voice_language(run_id: str | None, phase: str) -> str:
    if phase.startswith("WELCOME_"):
        return "en" if phase == "WELCOME_EN" else "zh"
    run = store.get_run(run_id) if run_id else None
    context = store.loads(run.get("context_json"), {}) if run else {}
    return "en" if str(context.get("language") or "zh").lower().startswith("en") else "zh"


def voice_speaker(run_id: str | None, phase: str) -> str:
    language = voice_language(run_id, phase)
    # Language-specific configuration prevents a new English voice replacing Chinese.
    return os.getenv(f"VOLCENGINE_TTS_VOICE_ID_{language.upper()}") or \
        os.getenv("VOLCENGINE_TTS_VOICE_ID") or os.getenv("VOLCENGINE_TTS_SPEAKER") or DEFAULT_VOICES[language][0]


def voice_resource(run_id: str | None, phase: str) -> str:
    language = voice_language(run_id, phase)
    return os.getenv(f"VOLCENGINE_TTS_RESOURCE_ID_{language.upper()}") or \
        os.getenv("VOLCENGINE_TTS_RESOURCE_ID") or DEFAULT_VOICES[language][1]


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 24_000) -> bytes:
    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


async def synthesize_voice(run_id: str | None, phase: str, text: str) -> str | None:
    global SYNTHESIS_COUNT
    text = " ".join((text or "").split()).strip()
    if not text or not tts_enabled():
        return None
    voice_id = f"voice_{uuid.uuid4().hex}"
    path = VOICE_DIR / f"{voice_id}.wav"
    try:
        pcm_chunks: list[bytes] = []
        async for chunk in stream_pcm(
            api_key=load_api_key(),
            resource_id=voice_resource(run_id, phase),
            speaker=voice_speaker(run_id, phase),
            text=text,
            uid=os.getenv("VOLCENGINE_TTS_UID", "unrush-auto-poc"),
            timeout_seconds=float(os.getenv("VOLCENGINE_TTS_TIMEOUT_SECONDS", "12")),
        ):
            pcm_chunks.append(chunk)
        wav = pcm_to_wav_bytes(b"".join(pcm_chunks))
        path.write_bytes(wav)
        store.record_voice(voice_id=voice_id, run_id=run_id, phase=phase, path=str(path), content_type="audio/wav")
        SYNTHESIS_COUNT += 1
        return voice_id
    except Exception:
        return None


def tiny_test_wav(path: Path) -> None:
    frames = b"".join(struct.pack("<h", 0) for _ in range(240))
    path.write_bytes(pcm_to_wav_bytes(frames))


async def prebuild_welcome_voices() -> None:
    for language, text in WELCOME_TEXT.items():
        WELCOME_VOICE_IDS[language] = await synthesize_voice(None, f"WELCOME_{language.upper()}", text)


def welcome_voice_id(language: str) -> str | None:
    return WELCOME_VOICE_IDS.get("en" if language.lower().startswith("en") else "zh")


def synthesis_count() -> int:
    return SYNTHESIS_COUNT
