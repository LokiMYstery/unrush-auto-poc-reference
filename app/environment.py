from __future__ import annotations

import os
import math
from datetime import UTC, datetime
from typing import Any

import httpx

from app.amap_poi import enrich_poi


async def enrich_context(context: dict[str, Any]) -> dict[str, Any]:
    enriched = normalize_scene_context(context)
    enriched["serverReceivedAt"] = datetime.now(UTC).isoformat()
    enriched["environmentFreshness"] = {
        "time": "server_current",
        "location": "client_snapshot" if context.get("location") else "missing",
        "weather": "missing",
    }
    enriched["health"] = sanitize_health(context.get("health"))
    location = context.get("location")
    if isinstance(location, dict):
        poi = await enrich_poi(location)
        enriched["poi"] = poi
        enriched["environmentFreshness"]["poi"] = poi["source"] if poi else "missing"
        weather = await open_meteo_weather(location)
        if weather is not None:
            enriched["weather"] = weather
            enriched["environmentFreshness"]["weather"] = weather["source"]
    return enriched


def context_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    """The panel shows the same coarse inputs the agent received, without coordinates."""
    return {
        "weather": {key: value for key, value in (context.get("weather") or {}).items()
                    if key in {"source", "validFreshness", "temperatureC", "weatherCode", "observedAt"}},
        "place": {key: value for key, value in (context.get("poi") or {}).items()
                  if key in {"source", "validFreshness", "placeType"}},
        "pressure": (context.get("health") or {}).get("pressure") or {},
        "locationAvailable": isinstance(context.get("location"), dict),
    }


def normalize_scene_context(context: dict[str, Any]) -> dict[str, Any]:
    """Separate weak recommendations from explicit user scene intent before prompting."""
    result = dict(context)
    legacy_recommendation = result.pop("recommendedScene", None)
    mode = _interaction_mode(result)
    if mode is None:
        result.pop("interactionMode", None)
    else:
        result["interactionMode"] = mode

    if not isinstance(result.get("systemRecommendation"), dict) and isinstance(legacy_recommendation, dict):
        result["systemRecommendation"] = _scene_summary(legacy_recommendation, source="legacy_recommended_scene")
    elif isinstance(result.get("systemRecommendation"), dict):
        result["systemRecommendation"] = _scene_summary(result["systemRecommendation"], source="system_recommendation")

    current_scene = result.get("currentScene")
    if mode == "scene" and isinstance(current_scene, dict) and str(current_scene.get("source") or "").lower() == "user_selected":
        result["currentScene"] = _scene_summary(current_scene, source="user_selected")
    else:
        result.pop("currentScene", None)

    potential_intent = result.get("potentialIntent")
    if mode == "auto" and isinstance(potential_intent, dict) and str(potential_intent.get("source") or "").lower() == "notification":
        result["potentialIntent"] = _intent_summary(potential_intent)
    elif mode == "auto":
        result["potentialIntent"] = None
    else:
        result.pop("potentialIntent", None)

    result["sceneSemantics"] = {
        "currentScene": "explicit_user_selected" if isinstance(result.get("currentScene"), dict) else "not_provided",
        "systemRecommendation": "weak_suggestion" if isinstance(result.get("systemRecommendation"), dict) else "not_provided",
        "potentialIntent": "notification_hint" if isinstance(result.get("potentialIntent"), dict) else "not_provided",
    }
    return result


def _interaction_mode(context: dict[str, Any]) -> str | None:
    raw = context.get("interactionMode")
    if isinstance(raw, str) and raw.lower() in {"auto", "scene", "goal"}:
        return raw.lower()
    if context.get("goalMode") is True:
        return "goal"
    return None


def _scene_summary(value: dict[str, Any], *, source: str) -> dict[str, Any]:
    summary = {
        key: value.get(key)
        for key in ("sceneId", "sceneCode", "sceneName", "title", "reason", "confidence")
        if value.get(key) is not None
    }
    existing_source = str(value.get("source") or "").lower()
    summary["source"] = "legacy_recommended_scene" if existing_source == "legacy_recommended_scene" else source
    return summary


def _intent_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("source", "summary", "title", "reason", "notificationId", "confidence")
        if value.get(key) is not None
    } | {"source": "notification"}


def narration_signature(context: dict[str, Any]) -> dict[str, Any]:
    snapshot = context_snapshot(context)
    weather, pressure = snapshot["weather"], snapshot["pressure"]
    temperature = weather.get("temperatureC")
    # Timestamp refreshes and tiny numeric changes should not restart an environment readout.
    return {"place": snapshot["place"].get("placeType"),
            "weather": [weather.get("weatherCode"), round(temperature) if isinstance(temperature, (int, float)) else None],
            "pressure": [pressure_level(pressure), pressure.get("measurementType"),
                         pressure.get("source"), pressure.get("baselineReady")]}


def pressure_level(pressure: dict[str, Any]) -> str | None:
    """StressLab Models.swift: <=20 / <=60 / <=80 / >80, never default measurements."""
    score = pressure.get("value")
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 100 or pressure.get("source") not in {"recent_measurement", "algorithm_estimate", "historical"}):
        return None
    return "low" if score <= 20 else "normal" if score <= 60 else "elevated" if score <= 80 else "high"


def spoken_context(context: dict[str, Any], language: str) -> dict[str, Any]:
    """Retain provenance but expose StressLab's qualitative labels to narration."""
    result = normalize_scene_context(context)
    health = dict(context.get("health") or {})
    pressure = dict(health.get("pressure") or {})
    level = pressure_level(pressure) if pressure.get("baselineReady") is not False else None
    pressure.pop("value", None)
    pressure["level"] = level
    is_hrv = pressure.get("measurementType") == "RMSSD"
    zh_labels = ["状态优秀", "状态一般", "注意压力", "压力偏高"] if is_hrv else ["较低", "一般", "偏高", "高"]
    en_labels = ["excellent", "typical", "mind your stress", "elevated stress"] if is_hrv else ["low", "normal", "elevated", "high"]
    pressure["levelLabel"] = (en_labels if language.startswith("en") else zh_labels)[
        ["low", "normal", "elevated", "high"].index(level)] if level else None
    health["pressure"] = pressure
    result["health"] = health
    return result


def environment_lead(context: dict[str, Any], language: str) -> str:
    """One factual initial/changed lead; the model chooses the musical energy direction."""
    mode = (context.get("environmentNarration") or {}).get("mode", "initial")
    if mode == "music_only":
        return ""
    fields = set((context.get("environmentNarration") or {}).get("changedFields") or ["place", "weather", "pressure"])
    english = language.startswith("en")
    parts = []
    place = (context.get("poi") or {}).get("placeType")
    if "place" in fields and isinstance(place, str) and place:
        places = {"住宅区":"a residential area", "公园":"the park", "写字楼":"the office", "办公室":"the office",
                  "商场":"the mall", "图书馆":"the library", "在途":"transit", "地铁站":"the subway station",
                  "户外":"outdoors", "机场":"the airport", "海边":"the seaside", "运动场所":"an exercise venue",
                  "酒店":"a hotel", "餐厅":"a restaurant", "高铁站":"the train station", "office":"the office", "park":"the park"}
        if english:
            parts.append("At " + places.get(place, place if place.isascii() else "your location"))
        else:
            parts.append("在" + place)
    weather = context.get("weather") or {}
    if "weather" in fields:
        codes = {0:("clear","晴朗"),1:("clear","晴朗"),2:("partly cloudy","多云"),3:("overcast","阴天"),
                 45:("foggy","有雾"),48:("foggy","有雾"),51:("drizzly","毛毛雨"),53:("drizzly","毛毛雨"),
                 55:("drizzly","毛毛雨"),61:("rainy","下雨"),63:("rainy","下雨"),65:("rainy","下雨"),
                 71:("snowy","下雪"),73:("snowy","下雪"),75:("snowy","下雪"),80:("showery","有阵雨"),
                 81:("showery","有阵雨"),82:("showery","有阵雨"),95:("stormy","有雷雨")}
        description = codes.get(weather.get("weatherCode"))
        temperature = weather.get("temperatureC")
        conditions = [description[0 if english else 1]] if description else []
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and math.isfinite(temperature):
            conditions.append(f"{temperature * 9 / 5 + 32:.0f}°F" if english else f"{temperature:.0f}°C")
        if conditions:
            parts.append(("it's " if english else "外面") + (", ".join(conditions) if english else "、".join(conditions)))
    pressure = spoken_context(context, language)["health"]["pressure"]
    if "pressure" in fields and pressure.get("levelLabel"):
        earlier = pressure.get("source") == "historical"
        if english:
            if pressure.get("measurementType") == "RMSSD":
                interpretations = {"low":"looks excellent", "normal":"looks typical",
                                   "elevated":"suggests keeping an eye on stress", "high":"suggests elevated stress"}
                prefix = "your earlier body reference " if earlier else "your body reference "
                parts.append(prefix + interpretations[pressure["level"]])
                return ", ".join(parts) + ". "
            prefix = "your earlier body-pressure reference is " if earlier else "your body-pressure reference is "
        else:
            prefix = "上次身体参考提示" if earlier else "身体参考提示"
        parts.append(prefix + pressure["levelLabel"])
    return ((", ".join(parts) + ". ") if english else ("，".join(parts) + "。")) if parts else ""


def narration_context(context: dict[str, Any], memories: list[dict[str, Any]]) -> dict[str, Any]:
    signature = narration_signature(context)
    previous = next((item.get("environmentNarrationSignature") for item in reversed(memories)
                     if item.get("spokenAssessment") and isinstance(item.get("environmentNarrationSignature"), dict)), None)
    changed = list(signature) if previous is None else [key for key in signature if signature[key] != previous.get(key)]
    return {"mode": "initial" if previous is None else "changed" if changed else "music_only",
            "changedFields": changed, "signature": signature}


def sanitize_health(value: Any) -> dict[str, Any]:
    health = dict(value) if isinstance(value, dict) else {}
    pressure = health.get("pressure") or health.get("stress")
    sanitized: dict[str, Any] = {}
    if isinstance(pressure, dict):
        sanitized["pressure"] = {
            "value": pressure.get("value"),
            "source": pressure.get("source"),
            "measurementType": pressure.get("measurementType"),
            "measurementOrigin": pressure.get("measurementOrigin"),
            "validUntil": pressure.get("validUntil"),
            "measuredAt": pressure.get("measuredAt"),
            "baselineReady": pressure.get("baselineReady"),
            "qualityReasons": pressure.get("qualityReasons"),
            "algorithmVersion": pressure.get("algorithmVersion"),
        }
    for key in ("sleepHours24h", "workoutMinutes24h"):
        if key in health:
            sanitized[key] = health.get(key)
    for key in ("sleep", "activity"):
        if isinstance(health.get(key), dict):
            sanitized[key] = {
                child_key: child_value
                for child_key, child_value in health[key].items()
                if child_key in {"source", "summary", "value", "validUntil", "measuredAt", "qualityReasons"}
            }
    return sanitized


async def open_meteo_weather(location: dict[str, Any]) -> dict[str, Any] | None:
    lat = location.get("latitude") or location.get("lat")
    lon = location.get("longitude") or location.get("lon")
    if lat is None or lon is None:
        return None
    if os.getenv("AUTO_POC_DISABLE_OPEN_METEO", "").strip().lower() in {"1", "true", "yes"}:
        return None
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("AUTO_OPEN_METEO_TIMEOUT_SECONDS", "2"))) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,precipitation,weather_code,cloud_cover",
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return {
            "source": "open_meteo_unavailable",
            "validFreshness": "missing",
        }
    current = payload.get("current") if isinstance(payload, dict) else None
    if not isinstance(current, dict):
        return None
    return {
        "source": "open_meteo",
        "validFreshness": "current",
        "temperatureC": current.get("temperature_2m"),
        "precipitationMm": current.get("precipitation"),
        "weatherCode": current.get("weather_code"),
        "cloudCover": current.get("cloud_cover"),
        "observedAt": current.get("time"),
    }
