from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import httpx

from app.config import A_MAP_PLACE_KNOWLEDGE_PATH


@lru_cache(maxsize=1)
def knowledge() -> dict[str, Any]:
    return json.loads(A_MAP_PLACE_KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def amap_enabled() -> bool:
    return os.getenv("AUTO_AMAP_POI_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def amap_key() -> str:
    return os.getenv("AUTO_AMAP_API_KEY", "").strip()


async def enrich_poi(location: dict[str, Any]) -> dict[str, Any] | None:
    if not amap_enabled() or not amap_key():
        return None
    lat = location.get("latitude") or location.get("lat")
    lon = location.get("longitude") or location.get("lon")
    if lat is None or lon is None:
        return None
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return None
    coordinate_system = str(location.get("coordinateSystem") or location.get("coordsys") or "gps").lower()
    accuracy = location.get("horizontalAccuracyM") or location.get("horizontalAccuracy") or 0
    try:
        horizontal_accuracy_m = max(0.0, float(accuracy))
    except (TypeError, ValueError):
        horizontal_accuracy_m = 0.0
    if horizontal_accuracy_m > float(os.getenv("AUTO_AMAP_MAX_ACCURACY_M", "1000")):
        return {
            "source": "amap_unavailable",
            "reason": "location_accuracy_too_low",
            "placeType": None,
        }
    timeout = float(os.getenv("AUTO_AMAP_TIMEOUT_SECONDS", "3"))
    radius = min(max(float(os.getenv("AUTO_AMAP_POI_RADIUS_M", "500")), 0), scoring().get("aroundRadiusMaxM", 500))
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if coordinate_system in {"gps", "wgs84", "wgs_84"}:
                longitude, latitude = await convert_to_amap(client, longitude, latitude)
                coordinate_system = "autonavi"
            pois = await fetch_around(client, longitude, latitude, radius)
            mapped = first_mapped_poi(pois)
            if mapped:
                return {
                    "source": mapped["source"],
                    "placeType": mapped["placeType"],
                    "distanceM": mapped.get("distanceM"),
                    "coordinateSystem": coordinate_system,
                    "lookupAvailable": True,
                }
            regeo = await fetch_regeo(client, longitude, latitude, radius)
            mapped = first_mapped_regeo(regeo)
            if mapped:
                return {
                    "source": mapped["source"],
                    "placeType": mapped["placeType"],
                    "distanceM": mapped.get("distanceM"),
                    "coordinateSystem": coordinate_system,
                    "lookupAvailable": True,
                }
        except Exception:
            return {
                "source": "amap_unavailable",
                "reason": "amap_request_failed",
                "placeType": None,
            }
    return {
        "source": "amap_unavailable",
        "reason": "amap_no_usable_poi",
        "placeType": None,
    }


async def convert_to_amap(client: httpx.AsyncClient, longitude: float, latitude: float) -> tuple[float, float]:
    response = await client.get(
        "https://restapi.amap.com/v3/assistant/coordinate/convert",
        params={
            "key": amap_key(),
            "locations": f"{longitude},{latitude}",
            "coordsys": "gps",
        },
    )
    payload = response.json()
    if payload.get("status") != "1":
        raise RuntimeError("amap_convert_failed")
    raw = str(payload.get("locations") or "").split(";")[0]
    lon_text, lat_text = raw.split(",", 1)
    return float(lon_text), float(lat_text)


async def fetch_around(client: httpx.AsyncClient, longitude: float, latitude: float, radius: float) -> list[dict[str, Any]]:
    query = knowledge()["query"]
    response = await client.get(
        "https://restapi.amap.com/v5/place/around",
        params={
            "key": amap_key(),
            "location": f"{longitude},{latitude}",
            "radius": str(int(round(radius))),
            "sortrule": "distance",
            "page_size": str(query.get("aroundPageSize", 25)),
            "show_fields": query.get("aroundShowFields", "business,children,indoor,navi"),
            "types": "|".join(query.get("aroundTypePrefixes", [])),
        },
    )
    payload = response.json()
    if payload.get("status") != "1":
        raise RuntimeError("amap_around_failed")
    pois = payload.get("pois")
    return pois if isinstance(pois, list) else []


async def fetch_regeo(client: httpx.AsyncClient, longitude: float, latitude: float, radius: float) -> dict[str, Any]:
    query = knowledge()["query"]
    response = await client.get(
        "https://restapi.amap.com/v3/geocode/regeo",
        params={
            "key": amap_key(),
            "location": f"{longitude},{latitude}",
            "radius": str(int(round(radius))),
            "extensions": query.get("regeoExtensions", "all"),
            "roadlevel": str(query.get("regeoRoadLevel", 1)),
            "homeorcorp": str(query.get("regeoHomeOrCorp", 1)),
        },
    )
    payload = response.json()
    if payload.get("status") != "1":
        raise RuntimeError("amap_regeo_failed")
    regeo = payload.get("regeocode")
    return regeo if isinstance(regeo, dict) else {}


def first_mapped_poi(pois: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, poi in enumerate(pois):
        place_type = place_type_for(typecode=str(poi.get("typecode") or ""), type_text=str(poi.get("type") or ""), name=str(poi.get("name") or ""))
        if not place_type:
            continue
        return {
            "placeType": place_type,
            "source": "amap_around_typecode",
            "distanceM": parse_distance(poi.get("distance")),
            "rank": index,
        }
    return None


def first_mapped_regeo(regeo: dict[str, Any]) -> dict[str, Any] | None:
    for poi in regeo.get("pois") or []:
        if isinstance(poi, dict):
            place_type = place_type_for(typecode=str(poi.get("typecode") or ""), type_text=str(poi.get("type") or ""), name=str(poi.get("name") or ""))
            if place_type:
                return {
                    "placeType": place_type,
                    "source": "amap_regeo_poi",
                    "distanceM": parse_distance(poi.get("distance")),
                }
    return None


def place_type_for(*, typecode: str, type_text: str, name: str = "") -> str | None:
    data = knowledge()
    provider = next((str(rule["placeType"]) for rule in data.get("providerTypeRules", [])
                     if any(str(segment) and str(segment) in type_text for segment in rule.get("matchSegments", []))), None)
    combined = (name + " " + type_text).lower()
    keywords = data.get("keywords", {})
    strong = {provider} if provider else set()
    weak: set[str] = set()
    for target, key in [(strong, "strongRules"), (weak, "weakRules")]:
        for rule in keywords.get(key, []):
            if any(str(word).lower() in combined for word in rule.get("keywords", [])):
                target.add(str(rule.get("placeType") or ""))
    has_container = any(str(word).lower() in combined for word in keywords.get("containerKeywords", []))
    for rule in data.get("typecodeRules", []):
        prefix = str(rule.get("prefix") or "")
        if not prefix or not typecode.startswith(prefix):
            continue
        if rule.get("requiresStrongKeywordPlaceType") and rule["requiresStrongKeywordPlaceType"] not in strong:
            continue
        if rule.get("requiresWeakKeywordPlaceType") and rule["requiresWeakKeywordPlaceType"] not in weak:
            continue
        if rule.get("requiresProviderTypePlaceType") and rule["requiresProviderTypePlaceType"] != provider:
            continue
        if "requiresContainerEvidence" in rule and bool(rule["requiresContainerEvidence"]) != has_container:
            continue
        return str(rule.get("placeType") or "") or None
    return provider


def parse_distance(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scoring() -> dict[str, Any]:
    value = knowledge().get("scoring")
    return value if isinstance(value, dict) else {}
