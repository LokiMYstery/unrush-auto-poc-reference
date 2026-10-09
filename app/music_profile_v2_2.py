from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.music_recall_v2_2 import (
    ClapTextEncoder,
    FixedTextEncoder,
    MusicToolError,
    TextEmbeddingEncoder,
    dot_product,
    normalize_vector,
    sha256_file,
)


PROFILE_SCHEMA_VERSION = "agent-music-profile-v1"
PROFILE_RANKING_POLICY_VERSION = "profile-structured-v1"
DESCRIPTION_RANKING_POLICY_VERSION = "clap-full-track-cosine-v2"
MUSIC_RETRIEVAL_TOOL_ORDER = (
    "music_profile_search",
    "music_description_recall",
)
_configured_retrieval_mode = os.getenv(
    "AGENTIC_PLAYER_MUSIC_RETRIEVAL_TOOLS", ""
).strip().lower()
if not _configured_retrieval_mode:
    _legacy_description_enabled = os.getenv(
        "AGENTIC_PLAYER_ENABLE_MUSIC_DESCRIPTION_RECALL", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    _configured_retrieval_mode = "both" if _legacy_description_enabled else "profile"
MUSIC_RETRIEVAL_TOOL_MODE = (
    _configured_retrieval_mode
    if _configured_retrieval_mode in {"profile", "description", "both"}
    else "profile"
)
MUSIC_DESCRIPTION_RECALL_ENABLED = MUSIC_RETRIEVAL_TOOL_MODE in {"description", "both"}
DEFAULT_PROFILE_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "config" / "music_profile_fixture.v1.json"
)
DEFAULT_MUSIC_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "music" / "v1"
DEFAULT_PROFILE_ASSET_PATH = DEFAULT_MUSIC_ASSET_DIR / "music_profile_v1.json"
DEFAULT_DESCRIPTION_EMBEDDINGS_PATH = (
    DEFAULT_MUSIC_ASSET_DIR / "music_description_clap_v2.npz"
)
RETRIEVAL_TOOL_NAMES = frozenset(
    tool_name
    for tool_name in MUSIC_RETRIEVAL_TOOL_ORDER
    if MUSIC_RETRIEVAL_TOOL_MODE == "both"
    or (MUSIC_RETRIEVAL_TOOL_MODE == "profile" and tool_name == "music_profile_search")
    or (MUSIC_RETRIEVAL_TOOL_MODE == "description" and tool_name == "music_description_recall")
)


def active_music_retrieval_tool_names() -> tuple[str, ...]:
    return tuple(
        tool_name
        for tool_name in MUSIC_RETRIEVAL_TOOL_ORDER
        if tool_name in RETRIEVAL_TOOL_NAMES
    )


def effective_music_required_tool(configured: Any) -> str | None:
    if configured is None:
        return None
    if configured in RETRIEVAL_TOOL_NAMES:
        return configured
    if configured not in MUSIC_RETRIEVAL_TOOL_ORDER:
        return configured
    return next(iter(active_music_retrieval_tool_names()), None)


def music_retrieval_instruction() -> str:
    from app.instruction_loader_v2_2 import load_instruction_snapshot

    fragment = {
        "description": "retrievalDescription",
        "both": "retrievalBoth",
    }.get(MUSIC_RETRIEVAL_TOOL_MODE, "retrievalProfile")
    return load_instruction_snapshot().fragments.get(fragment, "")
AXIS_NAMES = (
    "arousal",
    "rhythmicDrive",
    "attentionCapture",
    "globalStability",
    "suddenChangeRisk",
    "textureDensity",
    "brightness",
    "warmth",
    "spaciousness",
)
SEARCHABLE_TAG_IDS = (
    "source:piano",
    "source:electric_piano",
    "source:acoustic_guitar",
    "source:electric_guitar",
    "source:strings",
    "source:cello",
    "source:harp",
    "source:flute",
    "source:woodwind",
    "source:saxophone",
    "source:shakuhachi",
    "source:zither",
    "source:brass",
    "source:synth_lead",
    "source:synth_pad",
    "source:drone",
    "source:bell",
    "source:chime",
    "source:bass",
    "source:percussion",
    "source:soft_drum",
    "source:electronic_drum",
    "source:rain_light",
    "source:rain_heavy",
    "source:thunder",
    "source:ocean_wave",
    "source:river_stream",
    "source:wind",
    "source:forest",
    "source:bird",
    "source:insect",
    "source:fire",
    "source:cafe",
    "source:city",
    "source:train",
    "source:vehicle",
    "source:mechanical_hum",
    "source:white_noise",
    "source:pink_noise",
    "source:brown_noise",
    "source:filtered_noise",
    "source:low_frequency_noise",
    "source:spoken_voice",
    "source:lyrical_vocal",
    "source:wordless_vocal",
    "source:choir",
    "source:whisper",
    "source:breath",
    "source:heartbeat",
    "style:ambient",
    "style:classical",
    "style:lullaby",
    "style:folk",
    "style:east_asian_traditional",
    "style:jazz",
    "style:jazz_fusion",
    "style:smooth_jazz",
    "style:bossa_nova",
    "style:latin",
    "style:chillhop",
    "style:neo_soul",
    "style:rnb",
    "style:hip_hop",
    "style:boom_bap",
    "style:trap",
    "style:pop",
    "style:dance_pop",
    "style:synth_pop",
    "style:synth_funk",
    "style:rock",
    "style:edm",
    "style:electronic",
    "style:groove_based",
    "style:minimalist",
    "mood:calm",
    "mood:safe",
    "mood:healing",
    "mood:hopeful",
    "mood:uplifting",
    "mood:joyful",
    "mood:romantic",
    "mood:melancholic",
    "mood:lonely",
    "mood:tense",
    "mood:mysterious",
    "mood:neutral",
    "texture:soft",
    "texture:airy",
    "texture:warm",
    "texture:cool",
    "texture:heavy",
    "texture:light",
    "texture:grainy",
    "texture:spacious",
    "texture:enveloping",
    "texture:intimate",
    "texture:clean",
    "texture:hazy",
    "texture:organic",
    "texture:synthetic",
    "imagery:late_night_alone",
    "imagery:rainy_window",
    "imagery:forest_walk",
    "imagery:ocean_release",
    "imagery:city_night",
    "imagery:cafe_background",
    "imagery:morning_wake",
    "imagery:bedtime_companion",
    "imagery:quiet_reading",
    "imagery:gentle_movement",
)
SearchableTagId = Literal[*SEARCHABLE_TAG_IDS]
TAG_ARRAY_BY_NAMESPACE = {
    "source": "sources",
    "style": "styles",
    "mood": "moods",
    "texture": "textures",
    "imagery": "imagery",
}
TAG_LIMITS = {
    "sources": 4,
    "styles": 4,
    "moods": 3,
    "textures": 4,
    "imagery": 2,
}
NOISE_PLAYBACK_UNIT_IDS = frozenset({
    "pu:0089ZM20260116:o00",
})
GRAPHITE_PLAYBACK_UNIT_ID = "pu:0086ZM20260116:o00"


class ProfileTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=96)
    label: str = Field(min_length=1, max_length=64)


class TagPreferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tagId: SearchableTagId
    importance: float = Field(ge=0, le=1)


class AxisTargetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: float = Field(ge=1, le=5)
    tolerance: float = Field(gt=0, le=4)


class AxisTargetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arousal: AxisTargetInput | None = None
    rhythmicDrive: AxisTargetInput | None = None
    attentionCapture: AxisTargetInput | None = None
    globalStability: AxisTargetInput | None = None
    suddenChangeRisk: AxisTargetInput | None = None
    textureDensity: AxisTargetInput | None = None
    brightness: AxisTargetInput | None = None
    warmth: AxisTargetInput | None = None
    spaciousness: AxisTargetInput | None = None


class AgentAcoustic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    durationSeconds: float = Field(ge=0)
    tempoBpm: float | None = Field(gt=0)
    tempoApplicability: Literal["applicable", "not_applicable", "unknown"]

    @model_validator(mode="after")
    def validate_tempo(self) -> AgentAcoustic:
        if self.tempoApplicability == "not_applicable" and self.tempoBpm is not None:
            raise ValueError("tempoBpm must be null when tempoApplicability is not_applicable")
        return self


class PerceptualAxes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arousal: float | None = Field(ge=1, le=5)
    rhythmicDrive: float | None = Field(ge=1, le=5)
    attentionCapture: float | None = Field(ge=1, le=5)
    globalStability: float | None = Field(ge=1, le=5)
    suddenChangeRisk: float | None = Field(ge=1, le=5)
    textureDensity: float | None = Field(ge=1, le=5)
    brightness: float | None = Field(ge=1, le=5)
    warmth: float | None = Field(ge=1, le=5)
    spaciousness: float | None = Field(ge=1, le=5)


class AgentTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[ProfileTag] = Field(max_length=4)
    styles: list[ProfileTag] = Field(max_length=4)
    moods: list[ProfileTag] = Field(max_length=3)
    textures: list[ProfileTag] = Field(max_length=4)
    imagery: list[ProfileTag] = Field(max_length=2)


class RiskSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^risk:[a-z0-9_]+$")
    severity: float = Field(ge=1, le=5)


class AgentProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["agent-music-profile-v1"]
    summary: str = Field(max_length=320)
    contentType: Literal[
        "music",
        "nature_sound",
        "noise",
        "music_ambience_hybrid",
        "unknown",
    ]
    acoustic: AgentAcoustic
    perceptualAxes: PerceptualAxes
    tags: AgentTags
    riskSignals: list[RiskSignal] = Field(max_length=3)

    @field_validator("summary")
    @classmethod
    def summary_must_be_english_facing(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        if re.search(r"[\u3400-\u9fff]", value):
            raise ValueError("summary must be English-facing and cannot contain CJK text")
        return value.strip()


class CandidateNames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zh: str = Field(min_length=1, max_length=128)
    en: str = Field(min_length=1, max_length=128)


class ProfileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    playbackUnitId: str = Field(pattern=r"^pu:[A-Za-z0-9]+:o[0-9]{2}$")
    songId: str = Field(min_length=1, max_length=64)
    unitKind: Literal["song", "touchpad_track"]
    trackOrder: int = Field(ge=0, le=99)
    names: CandidateNames
    aliases: list[str] = Field(default_factory=list, max_length=24)
    parentNames: list[str] = Field(default_factory=list, max_length=8)
    enabled: bool = True
    agentProfile: AgentProfileV1
    tagStates: dict[str, Literal["accepted", "candidate", "unknown", "not_applicable"]]
    tagSupport: dict[str, float]

    @model_validator(mode="after")
    def validate_published_tags(self) -> ProfileRecord:
        published = {
            tag.id
            for array_name in TAG_LIMITS
            for tag in getattr(self.agentProfile.tags, array_name)
        }
        if any(self.tagStates.get(tag_id) != "accepted" for tag_id in published):
            raise ValueError("agentProfile may publish accepted tags only")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in self.tagSupport.values()):
            raise ValueError("tagSupport values must be finite values from 0 to 1")
        if self.playbackUnitId in NOISE_PLAYBACK_UNIT_IDS:
            if (
                self.agentProfile.contentType != "noise"
                or self.agentProfile.acoustic.tempoBpm is not None
                or self.agentProfile.acoustic.tempoApplicability != "not_applicable"
            ):
                raise ValueError("product-defined noise units must suppress tempo")
        if (
            self.playbackUnitId == GRAPHITE_PLAYBACK_UNIT_ID
            and self.agentProfile.contentType != "music"
        ):
            raise ValueError("Graphite must remain music")
        return self


class OntologyTag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str


class DescriptionEmbeddingProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modelId: str
    modelRevision: str
    pooling: str
    dimension: int = Field(gt=0)
    trackCount: int = Field(gt=0)


class UsageEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    m2d: Literal["production_eligibility_pending"]
    muq_mulan: Literal["research_only"]
    clap: Literal["production_eligibility_pending"]
    music_flamingo: Literal["research_only"]


class MusicProfileAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalogScope: str
    catalogVersion: str
    profileVersion: str
    ontologyVersion: str
    fusionPolicyVersion: str
    expectedProfileCount: int = Field(gt=0)
    artifactDigests: dict[str, str]
    descriptionEmbeddingProvenance: DescriptionEmbeddingProvenance
    usageEligibility: UsageEligibility
    ontologyTags: list[OntologyTag]
    profiles: list[ProfileRecord]
    songContexts: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_asset(self) -> MusicProfileAsset:
        if len(self.profiles) != self.expectedProfileCount:
            raise ValueError("profiles count does not match expectedProfileCount")
        if self.descriptionEmbeddingProvenance.trackCount != self.expectedProfileCount:
            raise ValueError("description embedding trackCount must cover every Profile")
        ids = [profile.playbackUnitId for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("playbackUnitId values must be unique")
        ontology_ids = [tag.id for tag in self.ontologyTags]
        if len(ontology_ids) != len(set(ontology_ids)):
            raise ValueError("ontology tag ids must be unique")
        ontology = set(ontology_ids)
        for profile in self.profiles:
            if not set(profile.tagStates).issubset(ontology):
                raise ValueError("profile tagStates contains an unknown ontology id")
        for key, digest in self.artifactDigests.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"artifactDigests.{key} must be a lowercase SHA-256 digest")
        return self


@dataclass(frozen=True)
class NameMatch:
    tier: int
    score: float
    evidence: str | None


class MusicProfileStore:
    def __init__(self, asset: MusicProfileAsset) -> None:
        self.asset = asset
        self.catalog_scope = asset.catalogScope
        self.catalog_version = asset.catalogVersion
        self.profile_version = asset.profileVersion
        self.ontology_version = asset.ontologyVersion
        self.fusion_policy_version = asset.fusionPolicyVersion
        self._profiles = tuple(profile for profile in asset.profiles if profile.enabled)
        self._by_id = {profile.playbackUnitId: profile for profile in self._profiles}
        self._ontology = {tag.id: tag.label for tag in asset.ontologyTags}
        self._song_contexts = asset.songContexts

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MusicProfileStore:
        try:
            return cls(MusicProfileAsset.model_validate(dict(value)))
        except ValidationError as exc:
            raise MusicToolError(
                "INVALID_MUSIC_PROFILE_ASSET",
                "Music Profile asset failed schema validation.",
            ) from exc

    @classmethod
    def from_path(
            cls,
            path: Path,
            *,
            songs_path: Path | None = None) -> MusicProfileStore:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MusicToolError(
                "MUSIC_PROFILE_ASSET_UNAVAILABLE",
                "Music Profile asset could not be loaded.",
            ) from exc
        if not isinstance(raw, dict):
            raise MusicToolError(
                "INVALID_MUSIC_PROFILE_ASSET",
                "Music Profile asset must be a JSON object.",
            )
        expected_digest = os.getenv("MUSIC_PROFILE_ASSET_SHA256")
        if expected_digest and sha256_file(path) != expected_digest:
            raise MusicToolError(
                "ARTIFACT_DIGEST_MISMATCH",
                "Mounted Music Profile asset does not match MUSIC_PROFILE_ASSET_SHA256.",
            )
        if songs_path is not None:
            raw["songContexts"] = load_song_contexts(songs_path)
        return cls.from_dict(raw)

    def readiness_audit(self) -> dict[str, Any]:
        return {
            "profileRows": len(self.asset.profiles),
            "eligibleRows": len(self._profiles),
            "expectedProfileRows": self.asset.expectedProfileCount,
            "ontologyTags": len(self._ontology),
            "catalogVersion": self.catalog_version,
            "profileVersion": self.profile_version,
            "ontologyVersion": self.ontology_version,
            "usageEligibility": self.asset.usageEligibility.model_dump(),
            "artifactDigests": dict(self.asset.artifactDigests),
        }

    def routeable_playback_unit_ids(self) -> set[str]:
        return set(self._by_id)

    def profile(self, playback_unit_id: str) -> ProfileRecord:
        profile = self._by_id.get(playback_unit_id)
        if profile is None:
            raise MusicToolError(
                "PROFILE_NOT_FOUND",
                "playbackUnitId is absent from the published Music Profile asset.",
                rejected=True,
            )
        return profile

    def search(
            self,
            *,
            name_query: str | None = None,
            tag_preferences: Sequence[Mapping[str, Any]] | None = None,
            axis_targets: Mapping[str, Any] | None = None,
            candidate_kinds: Sequence[str] | None = None,
            exclude_playback_unit_ids: Sequence[str] | None = None,
            top_k: int = 5,
            language: str = "zh",
            include_playback_unit_ids: Sequence[str] | None = None) -> dict[str, Any]:
        name_query = name_query.strip() if isinstance(name_query, str) else ""
        tags = validate_tag_preferences(tag_preferences, self._ontology)
        axes = validate_axis_targets(axis_targets)
        if not name_query and not tags and not axes:
            raise MusicToolError(
                "EMPTY_PROFILE_QUERY",
                "At least one of nameQuery, tagPreferences, or axisTargets is required.",
                rejected=True,
            )
        kinds = validate_candidate_kinds(candidate_kinds)
        scope = validate_exclusions(include_playback_unit_ids) if include_playback_unit_ids is not None else None
        if scope is None:
            validate_top_k(top_k)
        elif not scope or len(scope) > 9 or isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 9:
            # Only the owning song's explicit nine-unit scope may request a full nine-track ranking.
            # Unscoped library search retains the donor 1..5 contract.
            raise MusicToolError("INVALID_INTERNAL_SCOPE", "Internal search requires 1..9 explicit playback units.", rejected=True)
        if language not in {"zh", "en"}:
            raise MusicToolError(
                "UNSUPPORTED_LANGUAGE",
                "language must be zh or en.",
                rejected=True,
            )
        excluded = validate_exclusions(exclude_playback_unit_ids)
        eligible = [
            profile
            for profile in self._profiles
            if profile.unitKind in kinds and profile.playbackUnitId not in excluded
            and (scope is None or profile.playbackUnitId in scope)
        ]
        if not eligible:
            raise MusicToolError("NO_ROUTEABLE_CANDIDATE", "No routeable music candidate is available.")

        weights: list[tuple[str, float]] = []
        if name_query:
            weights.append(("name", 0.5))
        if tags:
            weights.append(("tags", 0.3))
        if axes:
            weights.append(("axes", 0.2))
        weight_total = sum(weight for _, weight in weights)
        normalized_weights = {key: weight / weight_total for key, weight in weights}

        scored: list[tuple[int, float, int, float, ProfileRecord, dict[str, list[str]]]] = []
        any_name_match = False
        for profile in eligible:
            name_match = match_name(profile, name_query) if name_query else NameMatch(0, 0.0, None)
            any_name_match = any_name_match or name_match.tier > 0
            tag_score, tag_support, tag_evidence = score_tags(profile, tags)
            axis_score, axis_evidence = score_axes(profile, axes)
            channel_score = (
                normalized_weights.get("name", 0) * name_match.score
                + normalized_weights.get("tags", 0) * tag_score
                + normalized_weights.get("axes", 0) * axis_score
            )
            evidence = merge_evidence(
                name_match.evidence,
                tag_evidence,
                axis_evidence,
            )
            scored.append((
                name_match.tier,
                channel_score,
                len(evidence["matched"]),
                tag_support,
                profile,
                evidence,
            ))

        if name_query and not any_name_match and not tags and not axes:
            raise MusicToolError(
                "NAME_QUERY_NOT_FOUND",
                "nameQuery did not match a primary name, alias, parent name, prefix, or substring.",
                rejected=True,
            )
        scored.sort(key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            -item[3],
            item[4].playbackUnitId,
        ))
        selected = scored[:top_k]
        candidates = [
            candidate_result(
                profile=item[4],
                rank=rank,
                channel_score=item[1],
                retrieval_channel="profile",
                evidence=item[5],
            )
            for rank, item in enumerate(selected, start=1)
        ]
        warnings: list[dict[str, Any]] = []
        if name_query and not any_name_match:
            warnings.append({
                "code": "NAME_QUERY_NOT_FOUND",
                "message": "nameQuery did not match; other supplied profile criteria were used.",
            })
        if len(candidates) < top_k:
            warnings.append({
                "code": "INSUFFICIENT_ROUTEABLE_CANDIDATES",
                "message": f"Requested {top_k} candidates but only {len(candidates)} are routeable.",
            })
        return retrieval_result(
            store=self,
            retrieval_channel="profile",
            ranking_policy_version=PROFILE_RANKING_POLICY_VERSION,
            candidates=candidates,
            warnings=warnings,
        )

    def get_song_context(self, playback_unit_id: str) -> dict[str, Any]:
        profile = self.profile(playback_unit_id.strip() if isinstance(playback_unit_id, str) else "")
        raw = self._song_contexts.get(profile.songId)
        if not isinstance(raw, dict):
            raise MusicToolError(
                "SONG_CONTEXT_NOT_FOUND",
                "songs_en copy is unavailable for this playback unit.",
                rejected=True,
            )
        return {
            "toolStatus": "ok",
            "playbackUnitId": profile.playbackUnitId,
            "songId": profile.songId,
            "unitKind": profile.unitKind,
            "trackOrder": profile.trackOrder,
            "unitLabel": profile.names.en,
            "catalogScope": self.catalog_scope,
            "songText": {
                "language": "en",
                "internalName": raw["internal_name"],
                "subTitle": raw["sub_title"],
                "introduction": raw["introductions"],
            },
            "acousticTech": raw.get("acoustic_tech") if isinstance(raw.get("acoustic_tech"), dict) else {},
            "catalogVersion": self.catalog_version,
        }


class MusicDescriptionRecallService:
    def __init__(
            self,
            *,
            store: MusicProfileStore,
            playback_unit_ids: Sequence[str],
            embeddings: Sequence[Sequence[float]],
            encoder: TextEmbeddingEncoder) -> None:
        if len(playback_unit_ids) != len(embeddings):
            raise MusicToolError(
                "INVALID_DESCRIPTION_EMBEDDING_BANK",
                "Description embedding ids and rows differ.",
            )
        if not playback_unit_ids:
            raise MusicToolError(
                "INVALID_DESCRIPTION_EMBEDDING_BANK",
                "Description embedding bank is empty.",
            )
        ids = [str(value) for value in playback_unit_ids]
        if len(ids) != len(set(ids)):
            raise MusicToolError(
                "INVALID_DESCRIPTION_EMBEDDING_BANK",
                "Description embedding playbackUnitIds must be unique.",
            )
        routeable = store.routeable_playback_unit_ids()
        if set(ids) != routeable:
            raise MusicToolError(
                "INCOMPLETE_ROUTEABLE_COVERAGE",
                "Description embedding bank must cover the published Profile catalog exactly.",
            )
        self.store = store
        self.encoder = encoder
        self._rows = tuple(
            (playback_unit_id, normalize_vector(embedding))
            for playback_unit_id, embedding in zip(ids, embeddings, strict=True)
        )
        dimensions = {len(row) for _, row in self._rows}
        if len(dimensions) != 1:
            raise MusicToolError(
                "INVALID_DESCRIPTION_EMBEDDING_BANK",
                "Description embedding dimensions are inconsistent.",
            )
        self._dimension = next(iter(dimensions))
        provenance = store.asset.descriptionEmbeddingProvenance
        if self._dimension != provenance.dimension or len(self._rows) != provenance.trackCount:
            raise MusicToolError(
                "DESCRIPTION_EMBEDDING_PROVENANCE_MISMATCH",
                "Description embedding bank dimensions do not match Profile provenance.",
            )
        self._encoder_dimension_validated = False

    @classmethod
    def from_npz(
            cls,
            *,
            store: MusicProfileStore,
            path: Path,
            encoder: TextEmbeddingEncoder) -> MusicDescriptionRecallService:
        expected_digest = store.asset.artifactDigests.get("descriptionEmbeddingsSha256")
        if expected_digest and sha256_file(path) != expected_digest:
            raise MusicToolError(
                "ARTIFACT_DIGEST_MISMATCH",
                "Description embedding bank does not match the Profile asset digest.",
            )
        try:
            import numpy as np
            bank = np.load(path, allow_pickle=False)
            ids = bank["playbackUnitIds"].astype(str).tolist()
            embeddings = bank["clap"].astype("float32").tolist()
        except Exception as exc:
            raise MusicToolError(
                "MUSIC_DESCRIPTION_ASSET_UNAVAILABLE",
                "CLAP full-track embedding bank could not be loaded.",
            ) from exc
        return cls(store=store, playback_unit_ids=ids, embeddings=embeddings, encoder=encoder)

    def coverage_audit(self) -> dict[str, Any]:
        return {
            "embeddingRows": len(self._rows),
            "eligibleRows": len(self._rows),
            "embeddingDimension": self._dimension,
            "rankingPolicyVersion": DESCRIPTION_RANKING_POLICY_VERSION,
        }

    def readiness_audit(self) -> dict[str, Any]:
        if not self._encoder_dimension_validated:
            try:
                probe = normalize_vector(self.encoder.encode(
                    "Warm instrumental ambient music with a calm continuous texture and no vocals."
                ))
            except MusicToolError:
                raise
            except Exception as exc:
                raise MusicToolError(
                    "TEXT_ENCODER_UNAVAILABLE",
                    "CLAP text encoder readiness probe failed.",
                    retryable=True,
                ) from exc
            if len(probe) != self._dimension:
                raise MusicToolError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    "CLAP text encoder and audio embedding dimensions do not match.",
                )
            self._encoder_dimension_validated = True
        audit = self.coverage_audit()
        audit["textEncoder"] = "ready"
        return audit

    def recall(
            self,
            query_text: str,
            *,
            top_k: int = 5,
            candidate_kinds: Sequence[str] | None = None,
            exclude_playback_unit_ids: Sequence[str] | None = None,
            text_language: str = "en") -> dict[str, Any]:
        query_text = query_text.strip() if isinstance(query_text, str) else ""
        if not query_text:
            raise MusicToolError("INVALID_RECALL_QUERY", "queryText is required.", rejected=True)
        validate_top_k(top_k)
        kinds = validate_candidate_kinds(candidate_kinds)
        if text_language != "en":
            raise MusicToolError(
                "UNSUPPORTED_TEXT_LANGUAGE",
                "music_description_recall supports English queries only.",
                rejected=True,
            )
        excluded = validate_exclusions(exclude_playback_unit_ids)
        eligible = [
            (playback_unit_id, embedding)
            for playback_unit_id, embedding in self._rows
            if playback_unit_id not in excluded
            and self.store.profile(playback_unit_id).unitKind in kinds
        ]
        if not eligible:
            raise MusicToolError("NO_ROUTEABLE_CANDIDATE", "No routeable music candidate is available.")
        try:
            query_embedding = normalize_vector(self.encoder.encode(query_text))
        except MusicToolError:
            raise
        except Exception as exc:
            raise MusicToolError("TEXT_ENCODER_UNAVAILABLE", "CLAP text encoding failed.", retryable=True) from exc
        if len(query_embedding) != self._dimension:
            raise MusicToolError(
                "EMBEDDING_DIMENSION_MISMATCH",
                "Query and audio embedding dimensions do not match.",
            )
        ranked = sorted(
            (
                (dot_product(query_embedding, embedding), playback_unit_id)
                for playback_unit_id, embedding in eligible
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        candidates = [
            candidate_result(
                profile=self.store.profile(playback_unit_id),
                rank=rank,
                channel_score=score,
                retrieval_channel="description_embedding",
                evidence={"matched": [], "partial": [], "missing": []},
            )
            for rank, (score, playback_unit_id) in enumerate(ranked, start=1)
        ]
        warnings: list[dict[str, Any]] = []
        if len(candidates) < top_k:
            warnings.append({
                "code": "INSUFFICIENT_ROUTEABLE_CANDIDATES",
                "message": f"Requested {top_k} candidates but only {len(candidates)} are routeable.",
            })
        return retrieval_result(
            store=self.store,
            retrieval_channel="description_embedding",
            ranking_policy_version=DESCRIPTION_RANKING_POLICY_VERSION,
            candidates=candidates,
            warnings=warnings,
        )


def retrieval_result(
        *,
        store: MusicProfileStore,
        retrieval_channel: str,
        ranking_policy_version: str,
        candidates: list[dict[str, Any]],
        warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "toolStatus": "ok",
        "retrievalChannel": retrieval_channel,
        "catalogScope": store.catalog_scope,
        "catalogVersion": store.catalog_version,
        "profileVersion": store.profile_version,
        "ontologyVersion": store.ontology_version,
        "rankingPolicyVersion": ranking_policy_version,
        "candidates": candidates,
        "warnings": warnings,
    }


def candidate_result(
        *,
        profile: ProfileRecord,
        rank: int,
        channel_score: float,
        retrieval_channel: str,
        evidence: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "candidateId": f"candidate_{rank:03d}",
        "rank": rank,
        "channelScore": float(channel_score),
        "playbackUnitId": profile.playbackUnitId,
        "songId": profile.songId,
        "unitKind": profile.unitKind,
        "trackOrder": profile.trackOrder,
        "names": profile.names.model_dump(),
        "agentProfile": profile.agentProfile.model_dump(),
        "matchEvidence": {
            "retrievalChannel": retrieval_channel,
            "channelRank": rank,
            "matched": list(evidence["matched"]),
            "partial": list(evidence["partial"]),
            "missing": list(evidence["missing"]),
        },
    }


def agent_retrieval_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validated retrieval result into the smaller model-visible contract."""
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise MusicToolError(
            "INVALID_MUSIC_RETRIEVAL_OUTPUT",
            "Music retrieval candidates could not be projected safely.",
        )

    projected_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise MusicToolError(
                "INVALID_MUSIC_RETRIEVAL_OUTPUT",
                "Music retrieval candidates could not be projected safely.",
            )
        profile = candidate.get("agentProfile")
        if not isinstance(profile, Mapping):
            raise MusicToolError(
                "INVALID_MUSIC_RETRIEVAL_OUTPUT",
                "Music retrieval candidates could not be projected safely.",
            )
        evidence = candidate.get("matchEvidence")
        if not isinstance(evidence, Mapping):
            raise MusicToolError(
                "INVALID_MUSIC_RETRIEVAL_OUTPUT",
                "Music retrieval candidates could not be projected safely.",
            )

        projected_candidates.append({
            "rank": candidate.get("rank"),
            "channelScore": candidate.get("channelScore"),
            "playbackUnitId": candidate.get("playbackUnitId"),
            "unitKind": candidate.get("unitKind"),
            "names": candidate.get("names"),
            "profileSummary": profile.get("summary"),
            "contentType": profile.get("contentType"),
            "riskSignals": profile.get("riskSignals"),
            "matchedCriteria": evidence.get("matched"),
        })

    warnings = result.get("warnings")
    projected_warnings = [
        {"code": warning.get("code")}
        for warning in warnings
        if isinstance(warning, Mapping) and isinstance(warning.get("code"), str)
    ] if isinstance(warnings, list) else []
    return {
        "toolStatus": result.get("toolStatus"),
        "retrievalChannel": result.get("retrievalChannel"),
        "rankingPolicyVersion": result.get("rankingPolicyVersion"),
        "candidates": projected_candidates,
        "warnings": projected_warnings,
    }


def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 5:
        raise MusicToolError(
            "INVALID_TOP_K",
            "topK must be an integer from 1 to 5.",
            rejected=True,
        )


def validate_candidate_kinds(value: Sequence[str] | None) -> set[str]:
    kinds = list(value or ["song", "touchpad_track"])
    allowed = {"song", "touchpad_track"}
    if not kinds or any(kind not in allowed for kind in kinds) or len(kinds) != len(set(kinds)):
        raise MusicToolError(
            "UNSUPPORTED_CANDIDATE_KIND",
            "candidateKinds must contain unique song and/or touchpad_track values.",
            rejected=True,
        )
    return set(kinds)


def validate_exclusions(value: Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (str, bytes)) or any(not isinstance(item, str) for item in value):
        raise MusicToolError(
            "INVALID_EXCLUSIONS",
            "excludePlaybackUnitIds must be an array of strings.",
            rejected=True,
        )
    return {item.strip() for item in value if item.strip()}


def validate_tag_preferences(
        value: Sequence[Mapping[str, Any]] | None,
        ontology: Mapping[str, str]) -> list[tuple[str, float]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise MusicToolError(
            "INVALID_TAG_PREFERENCES",
            "tagPreferences must be an array.",
            rejected=True,
        )
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise MusicToolError(
                "INVALID_TAG_PREFERENCES",
                f"tagPreferences[{index}] must be an object.",
                rejected=True,
            )
        if set(item) != {"tagId", "importance"}:
            raise MusicToolError(
                "INVALID_TAG_PREFERENCES",
                f"tagPreferences[{index}] must contain tagId and importance only.",
                rejected=True,
            )
        tag_id = item.get("tagId")
        importance = item.get("importance")
        if not isinstance(tag_id, str) or tag_id not in ontology:
            raise MusicToolError(
                "UNKNOWN_TAG_ID",
                f"Unknown ontology tagId at tagPreferences[{index}].",
                rejected=True,
            )
        if tag_id in seen:
            raise MusicToolError(
                "DUPLICATE_TAG_ID",
                f"Duplicate tagId at tagPreferences[{index}].",
                rejected=True,
            )
        if (
            isinstance(importance, bool)
            or not isinstance(importance, (int, float))
            or not math.isfinite(float(importance))
            or float(importance) < 0
            or float(importance) > 1
        ):
            raise MusicToolError(
                "INVALID_TAG_IMPORTANCE",
                f"tagPreferences[{index}].importance must be from 0 to 1.",
                rejected=True,
            )
        seen.add(tag_id)
        result.append((tag_id, float(importance)))
    if result and not any(importance > 0 for _, importance in result):
        raise MusicToolError(
            "ZERO_TAG_IMPORTANCE",
            "At least one tag preference must have importance greater than zero.",
            rejected=True,
        )
    return result


def validate_axis_targets(value: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MusicToolError(
            "INVALID_AXIS_TARGETS",
            "axisTargets must be an object.",
            rejected=True,
        )
    result: dict[str, tuple[float, float]] = {}
    for axis_name, target_spec in value.items():
        if axis_name not in AXIS_NAMES:
            raise MusicToolError(
                "UNKNOWN_PERCEPTUAL_AXIS",
                f"Unknown perceptual axis: {axis_name}.",
                rejected=True,
            )
        if not isinstance(target_spec, Mapping) or set(target_spec) != {"target", "tolerance"}:
            raise MusicToolError(
                "INVALID_AXIS_TARGET",
                f"axisTargets.{axis_name} must contain target and tolerance only.",
                rejected=True,
            )
        target = target_spec.get("target")
        tolerance = target_spec.get("tolerance")
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(float(target))
            or float(target) < 1
            or float(target) > 5
        ):
            raise MusicToolError(
                "INVALID_AXIS_TARGET",
                f"axisTargets.{axis_name}.target must be from 1 to 5.",
                rejected=True,
            )
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0
            or float(tolerance) > 4
        ):
            raise MusicToolError(
                "INVALID_AXIS_TOLERANCE",
                f"axisTargets.{axis_name}.tolerance must be greater than 0 and at most 4.",
                rejected=True,
            )
        result[axis_name] = (float(target), float(tolerance))
    return result


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def match_name(profile: ProfileRecord, query: str) -> NameMatch:
    normalized_query = normalize_name(query)
    if not normalized_query:
        return NameMatch(0, 0.0, None)
    primary = [profile.names.zh, profile.names.en]
    aliases = list(profile.aliases)
    parents = list(profile.parentNames)
    if any(normalize_name(value) == normalized_query for value in primary):
        return NameMatch(3, 1.0, "name:primary_exact")
    if any(normalize_name(value) == normalized_query for value in aliases + parents):
        return NameMatch(2, 0.9, "name:alias_or_parent_exact")
    all_names = primary + aliases + parents
    if any(
        normalized_query in normalized_name or normalized_name in normalized_query
        for value in all_names
        if (normalized_name := normalize_name(value))
    ):
        return NameMatch(1, 0.7, "name:prefix_or_substring")
    return NameMatch(0, 0.0, "name:missing")


def score_tags(
        profile: ProfileRecord,
        preferences: Sequence[tuple[str, float]]) -> tuple[float, float, dict[str, list[str]]]:
    if not preferences:
        return 0.0, 0.0, {"matched": [], "partial": [], "missing": []}
    denominator = sum(importance for _, importance in preferences)
    weighted_fit = 0.0
    supports: list[float] = []
    evidence = {"matched": [], "partial": [], "missing": []}
    for tag_id, importance in preferences:
        state = profile.tagStates.get(tag_id, "unknown")
        fit = 1.0 if state == "accepted" else 0.5 if state == "candidate" else 0.0
        weighted_fit += importance * fit
        supports.append(profile.tagSupport.get(tag_id, 0.0))
        evidence[
            "matched" if state == "accepted" else "partial" if state == "candidate" else "missing"
        ].append(tag_id)
    return (
        weighted_fit / denominator if denominator else 0.0,
        sum(supports) / len(supports) if supports else 0.0,
        evidence,
    )


def score_axes(
        profile: ProfileRecord,
        targets: Mapping[str, tuple[float, float]]) -> tuple[float, dict[str, list[str]]]:
    if not targets:
        return 0.0, {"matched": [], "partial": [], "missing": []}
    scores: list[float] = []
    evidence = {"matched": [], "partial": [], "missing": []}
    axes = profile.agentProfile.perceptualAxes.model_dump()
    for axis_name, (target, tolerance) in targets.items():
        value = axes.get(axis_name)
        evidence_id = f"axis:{axis_name}"
        if value is None:
            scores.append(0.0)
            evidence["missing"].append(evidence_id)
            continue
        difference = abs(float(value) - target)
        scores.append(max(0.0, 1.0 - difference / 4.0))
        evidence[
            "matched" if difference <= tolerance else "partial" if difference <= tolerance * 2 else "missing"
        ].append(evidence_id)
    return sum(scores) / len(scores), evidence


def merge_evidence(
        name_evidence: str | None,
        *parts: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    result = {"matched": [], "partial": [], "missing": []}
    if name_evidence:
        key = "missing" if name_evidence.endswith(":missing") else "matched"
        result[key].append(name_evidence)
    for part in parts:
        for key in result:
            result[key].extend(part.get(key, []))
    return result


def load_song_contexts(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicToolError(
            "SONG_CONTEXT_ASSET_UNAVAILABLE",
            "songs_en copy could not be loaded.",
        ) from exc
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise MusicToolError(
            "INVALID_SONG_METADATA",
            "songs_en.items must be an array.",
        )
    contexts: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        required_values = [item.get(key) for key in ("id", "internal_name", "introductions")]
        sub_title = item.get("sub_title")
        if (
                not all(
                    isinstance(value, str) and value.strip()
                    for value in required_values
                )
                or not isinstance(sub_title, str)):
            continue
        song_id = required_values[0].strip()
        if song_id in contexts:
            raise MusicToolError("INVALID_SONG_METADATA", "songs_en contains duplicate ids.")
        contexts[song_id] = item
    return contexts


def _load_profile_store() -> MusicProfileStore:
    mode = os.getenv("MUSIC_RETRIEVAL_MODE", os.getenv("MUSIC_RECALL_MODE", "strict")).strip().lower()
    if mode == "fixture_for_tests":
        path = Path(os.getenv("MUSIC_PROFILE_FIXTURE_PATH") or DEFAULT_PROFILE_FIXTURE_PATH)
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MusicToolError(
                "INVALID_MUSIC_PROFILE_FIXTURE",
                "The test-only Music Profile fixture is missing or invalid.",
            ) from exc
        if not isinstance(fixture, dict):
            raise MusicToolError(
                "INVALID_MUSIC_PROFILE_FIXTURE",
                "The test-only Music Profile fixture must be an object.",
            )
        fixture = {
            key: value
            for key, value in fixture.items()
            if key not in {"queryEmbedding", "descriptionEmbeddings"}
        }
        return MusicProfileStore.from_dict(fixture)
    if mode != "strict":
        raise MusicToolError(
            "INVALID_MUSIC_RETRIEVAL_MODE",
            "MUSIC_RETRIEVAL_MODE must be strict or fixture_for_tests.",
        )
    asset_value = os.getenv("MUSIC_PROFILE_ASSET_PATH") or str(DEFAULT_PROFILE_ASSET_PATH)
    songs_value = os.getenv("MUSIC_RECALL_SONGS_PATH")
    return MusicProfileStore.from_path(
        Path(asset_value),
        songs_path=Path(songs_value) if songs_value else None,
    )


@lru_cache(maxsize=1)
def get_music_profile_store() -> MusicProfileStore:
    return _load_profile_store()


def _load_description_service() -> MusicDescriptionRecallService:
    store = get_music_profile_store()
    mode = os.getenv("MUSIC_RETRIEVAL_MODE", os.getenv("MUSIC_RECALL_MODE", "strict")).strip().lower()
    if mode == "fixture_for_tests":
        path = Path(os.getenv("MUSIC_PROFILE_FIXTURE_PATH") or DEFAULT_PROFILE_FIXTURE_PATH)
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
            query_embedding = normalize_vector(fixture.pop("queryEmbedding"))
            embeddings = fixture.pop("descriptionEmbeddings")
            ids = [profile["playbackUnitId"] for profile in fixture["profiles"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MusicToolError(
                "INVALID_MUSIC_PROFILE_FIXTURE",
                "The test-only Music Profile embedding fixture is missing or invalid.",
            ) from exc
        return MusicDescriptionRecallService(
            store=store,
            playback_unit_ids=ids,
            embeddings=embeddings,
            encoder=FixedTextEncoder(query_embedding),
        )
    embedding_value = (
        os.getenv("MUSIC_DESCRIPTION_EMBEDDINGS_PATH")
        or os.getenv("MUSIC_RECALL_EMBEDDINGS_PATH")
        or str(DEFAULT_DESCRIPTION_EMBEDDINGS_PATH)
    )
    return MusicDescriptionRecallService.from_npz(
        store=store,
        path=Path(embedding_value),
        encoder=ClapTextEncoder(
            os.getenv("CLAP_MODEL_ID", "laion/clap-htsat-fused"),
            os.getenv("CLAP_DEVICE", "auto"),
            expected_model_id=store.asset.descriptionEmbeddingProvenance.modelId,
            expected_model_revision=store.asset.descriptionEmbeddingProvenance.modelRevision,
            declared_model_revision=os.getenv("CLAP_MODEL_REVISION"),
        ),
    )


@lru_cache(maxsize=1)
def get_music_description_recall_service() -> MusicDescriptionRecallService:
    return _load_description_service()


def clear_music_retrieval_caches() -> None:
    get_music_description_recall_service.cache_clear()
    get_music_profile_store.cache_clear()


def digest_json_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
