"""Consume recommendation order, then shuffle within the published per-scene song tiers.

Lists come from the 2026-09-23 18-scene draft; the newer 2026-09-24 requirement
replaces weighted tier draws with high -> middle -> low concatenation.
Pressure and other slow-system context do not participate in this selection.
"""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

TIERS = ("high", "middle", "low")
CONFIG_PATH = Path(__file__).parent / "assets/music/v1/FastSceneSongTiers.json"


@lru_cache(maxsize=1)
def tier_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["policy"] == "shuffle-within-high-middle-low-then-concatenate"
    assert len(config["scenes"]) == 18
    return config


def normalize(value: str) -> str:
    return re.sub(r"[\s_-]", "", value).casefold()


def resolve_scene(scene: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    config = tier_config()
    # Stable IDs from the same trusted PlaybackUnitCatalog also support older clients.
    for code, entry in config["scenes"].items():
        if scene.get("sceneId") in entry["sceneIds"]:
            return code, entry
    aliases = {"relax": "UNWIND", "fitness": "WORKOUT", "commute": "TRANSIT", "period": "CYCLES",
               "pet": "PETS", "run": "RUNNING", "meditation": "MINDFULNESS", "emo": "SOLITUDE",
               "运动入口": "RUNNING", "跑步": "RUNNING"}
    for key in ("sceneCode", "sceneName", "sceneId"):
        value = scene.get(key)
        if not isinstance(value, str):
            continue
        value = normalize(value)
        for code, entry in config["scenes"].items():
            if value in {normalize(code), normalize(entry["name"])} or aliases.get(value) == code:
                return code, entry
    return None


def fast_choices(cat: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = [scene for scene in payload.get("scenes", []) if isinstance(scene, dict)]
    selected = payload.get("selectedSceneId")
    if selected:
        scenes.sort(key=lambda scene: scene.get("sceneId") != selected)
    choices = []
    excluded = set(tier_config()["excludedSongIds"])
    for scene in scenes:
        resolved = resolve_scene(scene)
        profiles, tier_by_id, seen = [], {}, set()
        capability_exclusions = []
        if resolved:
            code, entry = resolved
            for tier in TIERS:
                songs = list(entry["tiers"][tier])
                random.shuffle(songs)
                for song_id in songs:
                    for profile in cat.default_units_for_song(song_id):
                        if profile.song_id in excluded or not profile.resource_bindings.get("bindings"):
                            continue
                        if tier == "high" and profile.unit_kind != "touchpad_track":
                            capability_exclusions.append({"songId": profile.song_id, "playbackUnitId": profile.playback_unit_id,
                                "reason": "high_tier_requires_touchpad"})
                            continue
                        if profile.playback_unit_id not in seen:
                            seen.add(profile.playback_unit_id)
                            profiles.append(profile)
                            tier_by_id[profile.playback_unit_id] = tier
        if profiles:
            choices.append({"sceneId": scene["sceneId"], "sceneCode": resolved[0], "profiles": profiles,
                            "tierById": tier_by_id, "selectionPolicy": tier_config()["policyVersion"],
                            "fallbackReason": None, "capabilityExclusions": capability_exclusions})
            # Resource checks happen later. Keep a same-scene legacy fallback after the
            # tiered pool, so dynamic failures also follow the explicit 09-24 fallback rule.
            profiles = []
        # The 09-24 requirement explicitly retains the existing playable fallback when tiers
        # are empty. Unknown/older scene payloads retain their upstream song order and are tagged.
        for playback_id in scene.get("playbackUnitIds", []):
            profile = cat.profile(playback_id)
            if profile and profile.song_id not in excluded and profile.playback_unit_id not in seen:
                profiles.append(profile)
                seen.add(profile.playback_unit_id)
        for song_id in scene.get("songIds", []):
            for profile in cat.default_units_for_song(song_id):
                if profile.song_id not in excluded and profile.playback_unit_id not in seen:
                    profiles.append(profile)
                    seen.add(profile.playback_unit_id)
        if profiles:
            choices.append({"sceneId": scene["sceneId"], "sceneCode": resolved[0] if resolved else None,
                            "profiles": profiles, "tierById": {}, "selectionPolicy": "legacy_scene_song_order",
                            "fallbackReason": ("configured_tiers_resource_unavailable" if tier_by_id else "empty_configured_tiers")
                            if resolved else "unknown_scene_tier_config"})
    if not choices:
        # Preserve the previous API's trusted catalog fallback for legacy empty/single-song
        # requests; final resource validation and the native receipt still own execution truth.
        scene_id, profiles = cat.fast_choice(payload)
        profiles = [profile for profile in profiles if profile.song_id not in excluded]
        if profiles:
            choices.append({"sceneId": scene_id or "auto", "sceneCode": None, "profiles": profiles,
                            "tierById": {}, "selectionPolicy": "legacy_catalog_fallback",
                            "fallbackReason": "no_available_recommended_scene"})
    return choices
