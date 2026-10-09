from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from app.catalog import MusicProfile


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = "ok"


async def validate_playback_unit(profile: MusicProfile) -> ValidationResult:
    if not profile.playback_unit_id:
        return ValidationResult(False, "missing_playback_unit_id")
    base_url = os.getenv("AUTO_RESOURCE_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return ValidationResult(True, "local_catalog_only")
    timeout = float(os.getenv("AUTO_RESOURCE_TIMEOUT_SECONDS", "3"))
    headers = _auth_headers()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_reason = "metadata_missing_preset_id"
            for service_song_id in _service_song_ids(profile):
                metadata_response = await client.get(f"{base_url}/api/scenes/song/meta", params={"songId": service_song_id}, headers=headers)
                if metadata_response.status_code in {401, 403}:
                    return ValidationResult(True, "local_catalog_only_resource_auth_required")
                if metadata_response.status_code >= 400:
                    last_reason = f"metadata_{metadata_response.status_code}"
                    continue
                metadata_payload = metadata_response.json()
                if not _business_ok(metadata_payload):
                    last_reason = "metadata_business_error"
                    continue
                preset_id = _find_value(metadata_payload, "presetId")
                if not preset_id:
                    last_reason = "metadata_missing_preset_id"
                    continue
                preset_response = await client.get(f"{base_url}/audio-resources/api/v1/presets/{preset_id}", headers=headers)
                if preset_response.status_code in {401, 403}:
                    return ValidationResult(True, "local_catalog_only_resource_auth_required")
                if preset_response.status_code >= 400:
                    last_reason = f"ar03_{preset_response.status_code}"
                    continue
                preset_payload = preset_response.json()
                if not _business_ok(preset_payload):
                    last_reason = "ar03_business_error"
                    continue
                if not _has_nonempty_tracks(preset_payload):
                    last_reason = "ar03_tracks_empty"
                    continue
                return ValidationResult(True, "public_resources_ok")
            return ValidationResult(False, last_reason)
    except Exception as exc:
        return ValidationResult(False, f"resource_validation_error:{type(exc).__name__}")


def _service_song_ids(profile: MusicProfile) -> list[str]:
    ids: list[str] = []
    primary = profile.service_song_id()
    if primary:
        ids.append(primary)
    for binding in profile.resource_bindings.get("bindings", []):
        if isinstance(binding, dict) and isinstance(binding.get("serviceSongId"), str) and binding["serviceSongId"] not in ids:
            ids.append(binding["serviceSongId"])
    if profile.song_id not in ids:
        ids.append(profile.song_id)
    return ids


def _business_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return True
    for key in ("code", "statusCode"):
        if key in value and value.get(key) not in {0, "0", 200, "200"}:
            return False
    success = value.get("success")
    if success is False:
        return False
    return True


def _has_nonempty_tracks(value: object) -> bool:
    tracks = _find_value(value, "tracks")
    if isinstance(tracks, list) and tracks:
        return True
    audio_resources = _find_value(value, "audioResources")
    if isinstance(audio_resources, list) and audio_resources:
        return True
    resources = _find_value(value, "resources")
    if isinstance(resources, list) and resources:
        return True
    return False


def _find_value(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_value(child, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_value(child, key)
            if found is not None:
                return found
    return None


def _auth_headers() -> dict[str, str]:
    headers = {
        "Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "Accept-Language": os.getenv("AUTO_RESOURCE_LANGUAGE", "zh-CN"),
        "X-Region": os.getenv("AUTO_RESOURCE_REGION", "CN"),
    }
    token = os.getenv("AUTO_RESOURCE_AUTH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
