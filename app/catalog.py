from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DESCRIPTION_EMBEDDINGS_PATH, MUSIC_PROFILE_PATH, PLAYBACK_UNIT_CATALOG_PATH
from app.music_profile_v2_2 import (
    MusicDescriptionRecallService,
    MusicToolError,
    get_music_description_recall_service,
    get_music_profile_store,
)


PRIORITIES = ("high", "middle", "low")


@dataclass(frozen=True)
class MusicProfile:
    playback_unit_id: str
    song_id: str
    unit_kind: str
    track_order: int
    enabled: bool
    names: dict[str, str]
    summary: str
    tags: dict[str, list[dict[str, str]]]
    axes: dict[str, float | None]
    song_inner_id: str
    resource_bindings: dict[str, Any]
    parent_names: list[str] = field(default_factory=list)

    @property
    def is_song_unit(self) -> bool:
        return self.unit_kind in {"song", "touchpad_track"}

    def service_song_id(self, origin_scene_id: str | None = None) -> str:
        bindings = self.resource_bindings.get("bindings")
        default_scene = self.resource_bindings.get("defaultBindingSceneId")
        if isinstance(bindings, list):
            if origin_scene_id:
                for binding in bindings:
                    if isinstance(binding, dict) and binding.get("sceneId") == origin_scene_id and isinstance(binding.get("serviceSongId"), str):
                        return binding["serviceSongId"]
            for binding in bindings:
                if isinstance(binding, dict) and binding.get("sceneId") == default_scene and isinstance(binding.get("serviceSongId"), str):
                    return binding["serviceSongId"]
            for binding in bindings:
                if isinstance(binding, dict) and isinstance(binding.get("serviceSongId"), str):
                    return binding["serviceSongId"]
        return self.song_id


class Catalog:
    def __init__(
        self,
        profiles: list[MusicProfile],
        raw_version: str,
        playback_catalog_version: str,
    ) -> None:
        self.profiles = [p for p in profiles if p.enabled and p.is_song_unit]
        self.by_playback_unit_id = {p.playback_unit_id: p for p in self.profiles}
        self.by_song_id: dict[str, list[MusicProfile]] = {}
        self.by_service_song_id: dict[str, list[MusicProfile]] = {}
        for profile in self.profiles:
            self.by_song_id.setdefault(profile.song_id, []).append(profile)
            for binding in profile.resource_bindings.get("bindings", []):
                if isinstance(binding, dict) and isinstance(binding.get("serviceSongId"), str):
                    self.by_service_song_id.setdefault(binding["serviceSongId"], []).append(profile)
        self.raw_version = raw_version
        self.playback_catalog_version = playback_catalog_version

    def exists(self, playback_unit_id: str | None) -> bool:
        return bool(playback_unit_id and playback_unit_id in self.by_playback_unit_id)

    def profile(self, playback_unit_id: str) -> MusicProfile | None:
        return self.by_playback_unit_id.get(playback_unit_id)

    def candidate_ids_from_payload(self, payload: dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for item in payload.get("songCandidates") or []:
            if isinstance(item, dict):
                self._add_identifier(
                    ids,
                    item.get("playbackUnitId"),
                    item.get("songId") or item.get("carrierUuid") or item.get("carrierUUID"),
                )
        for scene in payload.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for playback_id in scene.get("playbackUnitIds") or []:
                self._add_identifier(ids, playback_id, None)
            for song_id in scene.get("songIds") or []:
                self._add_identifier(ids, None, song_id)
            for carrier_uuid in scene.get("carrierUuids") or scene.get("carrierUUIDs") or []:
                self._add_identifier(ids, None, carrier_uuid)
        for playback_id in payload.get("catalog", {}).get("playbackUnitIds", []) if isinstance(payload.get("catalog"), dict) else []:
            self._add_identifier(ids, playback_id, None)
        return {item for item in ids if item in self.by_playback_unit_id}

    def _add_identifier(self, ids: set[str], playback_unit_id: Any, song_id: Any) -> None:
        if isinstance(playback_unit_id, str) and playback_unit_id in self.by_playback_unit_id:
            ids.add(playback_unit_id)
        if isinstance(song_id, str):
            for profile in self.by_song_id.get(song_id, []) + self.by_service_song_id.get(song_id, []):
                ids.add(profile.playback_unit_id)

    def choose_fast(self, payload: dict[str, Any]) -> MusicProfile | None:
        _, ordered = self.fast_choice(payload)
        return ordered[0] if ordered else None

    def ordered_fast_candidates(self, payload: dict[str, Any]) -> list[MusicProfile]:
        _, ordered = self.fast_choice(payload)
        return ordered

    def fast_choice(self, payload: dict[str, Any]) -> tuple[str | None, list[MusicProfile]]:
        selected_scene_id = payload.get("selectedSceneId")
        scenes = [scene for scene in payload.get("scenes") or [] if isinstance(scene, dict)]
        if selected_scene_id:
            scenes = sorted(scenes, key=lambda s: 0 if s.get("sceneId") == selected_scene_id else 1)
        ordered: list[MusicProfile] = []
        seen: set[str] = set()
        for priority in PRIORITIES:
            group = [scene for scene in scenes if (scene.get("priority") or "middle") == priority]
            if not any(scene.get("preordered") is True for scene in group):
                random.shuffle(group)
            for scene in group:
                scene_profiles: list[MusicProfile] = []
                for playback_id in scene.get("playbackUnitIds") or []:
                    profile = self.profile(str(playback_id))
                    if profile:
                        scene_profiles.append(profile)
                for song_id in scene.get("songIds") or []:
                    candidates = self.default_units_for_song(str(song_id))
                    if candidates:
                        scene_profiles.extend(candidates)
                for carrier_uuid in scene.get("carrierUuids") or scene.get("carrierUUIDs") or []:
                    scene_profiles.extend(self.default_units_for_song(str(carrier_uuid)))
                for profile in scene_profiles:
                    if profile.playback_unit_id not in seen:
                        ordered.append(profile)
                        seen.add(profile.playback_unit_id)
                if ordered:
                    scene_id = scene.get("sceneId")
                    return str(scene_id) if scene_id is not None else selected_scene_id, ordered
        song_candidates = [item for item in payload.get("songCandidates") or [] if isinstance(item, dict)]
        for priority in PRIORITIES:
            group = [item for item in song_candidates if (item.get("priority") or "middle") == priority]
            random.shuffle(group)
            for item in group:
                profile = self.profile(str(item.get("playbackUnitId") or ""))
                if profile:
                    scene_id = item.get("sceneId")
                    return str(scene_id) if scene_id is not None else selected_scene_id, [profile]
                lookup_id = str(item.get("songId") or item.get("carrierUuid") or item.get("carrierUUID") or "")
                song_profiles = self.default_units_for_song(lookup_id)
                if song_profiles:
                    scene_id = item.get("sceneId")
                    return str(scene_id) if scene_id is not None else selected_scene_id, song_profiles
        return selected_scene_id, [self.profiles[0]] if self.profiles else []

    def default_units_for_song(self, song_id: str) -> list[MusicProfile]:
        candidates = self.by_song_id.get(song_id, []) + self.by_service_song_id.get(song_id, [])
        if not candidates:
            return []
        song_units = [profile for profile in candidates if profile.unit_kind == "song"]
        if song_units:
            return [sorted(song_units, key=lambda item: item.track_order)[0]]
        return [sorted(candidates, key=lambda item: (abs(item.track_order - 4), item.track_order, item.playback_unit_id))[0]]

    def same_song_candidates(self, current_playback_unit_id: str | None, allowed: set[str]) -> list[MusicProfile]:
        current = self.profile(current_playback_unit_id or "")
        if current is None:
            return []
        profiles: list[MusicProfile] = [current]
        profiles.extend(
            profile for profile in self.by_song_id.get(current.song_id, [])
            if profile.playback_unit_id != current.playback_unit_id
            and (not allowed or profile.playback_unit_id in allowed)
        )
        return profiles

    def search_same_song(self, args: dict[str, Any], allowed: set[str], current_id: str) -> dict[str, Any]:
        profiles = self.same_song_candidates(current_id, allowed)
        if not profiles:
            return {"toolStatus": "empty", "candidates": [], "scope": "current_carrier_song"}
        scoped_ids = [profile.playback_unit_id for profile in profiles]
        if args.get("nameQuery") or args.get("tagPreferences") or args.get("axisTargets"):
            result = get_music_profile_store().search(
                name_query=args.get("nameQuery"), tag_preferences=args.get("tagPreferences"),
                axis_targets=args.get("axisTargets"), top_k=len(scoped_ids), language=str(args.get("language") or "zh"),
                include_playback_unit_ids=scoped_ids,
            )
        else:
            result = {"toolStatus": "ok", "candidates": [{"playbackUnitId": item.playback_unit_id} for item in profiles]}
        result.update({"scope": "current_carrier_song", "carrierSongId": profiles[0].song_id,
                       "internalTrackCount": len(scoped_ids), "completeInternalScope": True,
                       "supportsInternalAdjustment": profiles[0].unit_kind == "touchpad_track"})
        return result

    def search_profiles(self, args: dict[str, Any], allowed: set[str], current_playback_unit_id: str | None = None) -> dict[str, Any]:
        exclude = [current_playback_unit_id] if current_playback_unit_id else []
        return get_music_profile_store().search(
            name_query=args.get("nameQuery"),
            tag_preferences=args.get("tagPreferences"),
            axis_targets=args.get("axisTargets"),
            candidate_kinds=args.get("candidateKinds"),
            exclude_playback_unit_ids=exclude + [item for item in args.get("excludePlaybackUnitIds", []) if isinstance(item, str)],
            top_k=max(1, min(int(args.get("topK") or args.get("limit") or 5), 8)),
            language=str(args.get("language") or "zh"),
        )

    def description_recall(self, args: dict[str, Any], allowed: set[str], current_playback_unit_id: str | None = None) -> dict[str, Any]:
        query = str(args.get("queryText") or "").strip()
        language = str(args.get("textLanguage") or "en")
        if not query or language != "en" or any(ord(char) > 127 for char in query):
            raise MusicToolError(
                "INVALID_DESCRIPTION_QUERY",
                "music_description_recall requires an explicit English queryText from the model; the server does not translate or rewrite it.",
                rejected=True,
            )
        exclude = [current_playback_unit_id] if current_playback_unit_id else []
        return description_recall_service().recall(
            query,
            top_k=max(1, min(int(args.get("topK") or args.get("limit") or 5), 8)),
            candidate_kinds=args.get("candidateKinds"),
            exclude_playback_unit_ids=exclude + [item for item in args.get("excludePlaybackUnitIds", []) if isinstance(item, str)],
            text_language="en",
        )

    def profiles_from_tool_result(self, result: dict[str, Any], allowed: set[str]) -> list[MusicProfile]:
        return self._profiles_from_donor_candidates(result, allowed)

    def _profiles_from_donor_candidates(self, result: dict[str, Any], allowed: set[str]) -> list[MusicProfile]:
        profiles: list[MusicProfile] = []
        for candidate in result.get("candidates", []) if isinstance(result, dict) else []:
            playback_id = candidate.get("playbackUnitId") if isinstance(candidate, dict) else None
            if not isinstance(playback_id, str):
                continue
            if allowed and playback_id not in allowed:
                continue
            profile = self.profile(playback_id)
            if profile:
                profiles.append(profile)
        return profiles


def _profile_from_raw(raw: dict[str, Any], binding: dict[str, Any] | None = None) -> MusicProfile:
    agent_profile = raw.get("agentProfile") if isinstance(raw.get("agentProfile"), dict) else {}
    song_inner_id = str(raw.get("songId") or "")
    return MusicProfile(
        playback_unit_id=str(raw.get("playbackUnitId") or ""),
        song_id=song_inner_id,
        unit_kind=str(raw.get("unitKind") or ""),
        track_order=int(raw.get("trackOrder") or 0),
        enabled=bool(raw.get("enabled", True)),
        names=raw.get("names") if isinstance(raw.get("names"), dict) else {},
        summary=str(agent_profile.get("summary") or ""),
        tags=agent_profile.get("tags") if isinstance(agent_profile.get("tags"), dict) else {},
        axes=agent_profile.get("perceptualAxes") if isinstance(agent_profile.get("perceptualAxes"), dict) else {},
        song_inner_id=song_inner_id,
        resource_bindings=binding or {},
        parent_names=[name for name in raw.get("parentNames", []) if isinstance(name, str) and name],
    )


def _load_playback_catalog(path: Path) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("records") if isinstance(raw, dict) else []
    bindings = raw.get("resourceBindingsBySongInnerId") if isinstance(raw, dict) else {}
    by_playback: dict[str, dict[str, Any]] = {}
    if isinstance(records, list):
        by_playback = {
            str(record.get("playbackUnitId")): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("playbackUnitId"), str)
        }
    return str(raw.get("version") if isinstance(raw, dict) else "unknown"), by_playback, bindings if isinstance(bindings, dict) else {}


class EndpointTextEncoder:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def encode(self, text: str) -> list[float]:
        body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(os.getenv("AUTO_DESCRIPTION_EMBEDDING_TIMEOUT_SECONDS", "5"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        embedding = payload.get("embedding") if isinstance(payload, dict) else None
        if not isinstance(embedding, list):
            raise MusicToolError("TEXT_ENCODER_UNAVAILABLE", "Embedding endpoint did not return embedding[].", retryable=True)
        return [float(value) for value in embedding]


@lru_cache(maxsize=1)
def description_recall_service() -> MusicDescriptionRecallService:
    endpoint = os.getenv("AUTO_DESCRIPTION_EMBEDDING_ENDPOINT", "").strip()
    if endpoint:
        return MusicDescriptionRecallService.from_npz(
            store=get_music_profile_store(),
            path=DESCRIPTION_EMBEDDINGS_PATH,
            encoder=EndpointTextEncoder(endpoint),
        )
    return get_music_description_recall_service()


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    with MUSIC_PROFILE_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    playback_catalog_version, records_by_id, bindings_by_inner_id = _load_playback_catalog(PLAYBACK_UNIT_CATALOG_PATH)
    profiles: list[MusicProfile] = []
    for item in raw.get("profiles", []):
        if not isinstance(item, dict):
            continue
        playback_id = str(item.get("playbackUnitId") or "")
        record = records_by_id.get(playback_id)
        if record:
            item = {**item, "unitKind": record.get("unitKind"), "trackOrder": record.get("trackOrder"), "songId": record.get("songInnerId")}
        binding = bindings_by_inner_id.get(str(item.get("songId") or ""))
        profiles.append(_profile_from_raw(item, binding if isinstance(binding, dict) else None))
    return Catalog(
        profiles,
        str(raw.get("catalogVersion") or "unknown"),
        playback_catalog_version,
    )
