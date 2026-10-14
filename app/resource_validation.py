from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from app.catalog import MusicProfile
from app.resource_auth import auth_headers, refresh_resource_token


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = "ok"


async def validate_playback_unit(profile: MusicProfile, origin_scene_id: str | None = None) -> ValidationResult:
    if not profile.playback_unit_id:
        return ValidationResult(False, "missing_playback_unit_id")
    base_url = os.getenv("AUTO_RESOURCE_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return ValidationResult(True, "local_catalog_only")
    timeout = float(os.getenv("AUTO_RESOURCE_TIMEOUT_SECONDS", "3"))
    service_song_id = profile.service_song_id(origin_scene_id)
    if not service_song_id:
        return ValidationResult(False, "missing_service_song_id")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_reason = "metadata_missing_preset_id"
            metadata_response = await _get_resource(client, base_url, f"{base_url}/api/scenes/song/meta", params={"songId": service_song_id})
            if metadata_response.status_code in {401, 403}:
                return ValidationResult(False, "resource_auth_required")
            if metadata_response.status_code >= 400:
                return ValidationResult(False, f"metadata_{metadata_response.status_code}")
            metadata_payload = metadata_response.json()
            if not _business_ok(metadata_payload):
                return ValidationResult(False, "metadata_business_error")
            preset_id = _find_value(metadata_payload, "presetId")
            if not preset_id:
                return ValidationResult(False, last_reason)
            preset_response = await _get_resource(client, base_url, f"{base_url}/audio-resources/api/v1/presets/{preset_id}")
            if preset_response.status_code in {401, 403}:
                return ValidationResult(False, "resource_auth_required")
            if preset_response.status_code >= 400:
                return ValidationResult(False, f"ar03_{preset_response.status_code}")
            preset_payload = preset_response.json()
            if not _business_ok(preset_payload):
                return ValidationResult(False, "ar03_business_error")
            if not _has_nonempty_tracks(preset_payload):
                return ValidationResult(False, "ar03_tracks_empty")
            return ValidationResult(True, "public_resources_ok")
    except Exception as exc:
        return ValidationResult(False, f"resource_validation_error:{type(exc).__name__}")


async def _get_resource(client: httpx.AsyncClient, base_url: str, url: str, **kwargs: object) -> httpx.Response:
    headers, access_token = auth_headers(base_url)
    response = await client.get(url, headers=headers, **kwargs)
    if response.status_code in {401, 403} and await refresh_resource_token(base_url, rejected_access_token=access_token):
        retry_headers, _ = auth_headers(base_url)
        response = await client.get(url, headers=retry_headers, **kwargs)
    return response


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
