from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence


DEFAULT_MODEL_ID = "laion/clap-htsat-fused"
DEFAULT_CATALOG_VERSION = "playback-unit-catalog-v1"
DEFAULT_EMBEDDING_VERSION = "clap-whole-song-v1"
MODEL_IDENTITY_FILENAME = "agentic_player_model_identity.json"
REQUIRED_ARTIFACT_DIGESTS = (
    "playbackCatalogSha256",
    "trackIndexSha256",
    "embeddingsSha256",
    "embeddingMetadataSha256",
    "songsSha256",
    "modelDirectorySha256",
)
MODULE_PATH = Path(__file__).resolve()
if len(MODULE_PATH.parents) > 3:
    WORKSPACE_ROOT = MODULE_PATH.parents[3]
    DEFAULT_SONGS_PATH = WORKSPACE_ROOT / "songs_en.json"
    DEFAULT_CLAP_ROOT = WORKSPACE_ROOT.parent / "clap_retrieval" / "retrieval_framework" / "clap"
else:
    # Container images do not contain the parent workspace. Strict mode must mount
    # these assets or override their paths explicitly.
    WORKSPACE_ROOT = Path("/assets")
    DEFAULT_SONGS_PATH = WORKSPACE_ROOT / "songs_en.json"
    DEFAULT_CLAP_ROOT = WORKSPACE_ROOT / "clap"
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "config" / "music_recall_fixture.v2.2.3.json"


class TextEmbeddingEncoder(Protocol):
    def encode(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class FixedTextEncoder:
    """Explicit test-only encoder used by the full-chain fixture smoke."""

    embedding: tuple[float, ...]

    def encode(self, _text: str) -> Sequence[float]:
        return self.embedding


class MusicToolError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, rejected: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.rejected = rejected

    def as_tool_result(self) -> dict[str, Any]:
        return {
            "toolStatus": "rejected" if self.rejected else "error",
            "issues": [{
                "path": "$",
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }],
        }


@dataclass(frozen=True)
class SongText:
    internal_name: str
    sub_title: str
    introduction: str
    raw: dict[str, Any]

    def summary(self) -> dict[str, str]:
        return {
            "language": "en",
            "internalName": self.internal_name,
            "subTitle": self.sub_title,
            "introduction": self.introduction,
        }


@dataclass(frozen=True)
class RetrievalUnit:
    embedding_row: int
    retrieval_track_id: str
    playback_unit_id: str
    song_id: str
    unit_kind: str
    track_order: int
    unit_label: str
    embedding: tuple[float, ...]
    song_text: SongText


@dataclass(frozen=True)
class EmbeddingProvenance:
    model_id: str
    model_revision: str
    pooling: str
    dimension: int
    track_count: int

    def as_manifest_value(self) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "modelRevision": self.model_revision,
            "pooling": self.pooling,
            "dimension": self.dimension,
            "trackCount": self.track_count,
        }


class ClapTextEncoder:
    """Lazy CLAP text encoder. Imports heavy optional dependencies only when called."""

    def __init__(
            self,
            model_id: str = DEFAULT_MODEL_ID,
            device: str = "auto",
            *,
            expected_model_id: str | None = None,
            expected_model_revision: str | None = None,
            expected_model_directory_sha256: str | None = None,
            declared_model_revision: str | None = None) -> None:
        self.model_id = model_id
        self.device_arg = device
        self.expected_model_id = expected_model_id
        self.expected_model_revision = expected_model_revision
        self.expected_model_directory_sha256 = expected_model_directory_sha256
        self.declared_model_revision = declared_model_revision
        self._processor: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._load_lock = threading.Lock()

    def encode(self, text: str) -> Sequence[float]:
        self._load_once()
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features.detach().cpu().numpy().reshape(-1).astype("float32").tolist()

    def _load_once(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        with self._load_lock:
            if self._processor is not None and self._model is not None:
                return
            try:
                import torch
                from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel, ClapProcessor
            except ImportError as exc:
                raise MusicToolError(
                    "TEXT_ENCODER_UNAVAILABLE",
                    "CLAP text encoder dependencies are not installed in this runtime.",
                    retryable=True,
                ) from exc

            load_path, local_only = resolve_model_path(self.model_id)
            if not local_only and os.getenv("CLAP_ALLOW_REMOTE_MODEL_DOWNLOAD", "false").lower() != "true":
                raise MusicToolError(
                    "TEXT_ENCODER_UNAVAILABLE",
                    "CLAP model is not cached locally; remote model download is disabled by default.",
                    retryable=True,
                )
            if not local_only and self.expected_model_directory_sha256:
                raise MusicToolError(
                    "TEXT_ENCODER_PROVENANCE_MISMATCH",
                    "Strict music recall requires a local CLAP model directory so its artifact digest can be verified.",
                )
            if (
                    local_only
                    and self.expected_model_id
                    and self.expected_model_revision):
                validate_local_model_identity(
                    Path(load_path),
                    expected_model_id=self.expected_model_id,
                    expected_model_revision=self.expected_model_revision,
                    expected_model_directory_sha256=self.expected_model_directory_sha256,
                    declared_model_revision=self.declared_model_revision,
                )
            try:
                feature_extractor = ClapFeatureExtractor.from_pretrained(load_path, local_files_only=local_only)
                tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=local_only)
                processor = ClapProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
                device = resolve_device(self.device_arg, torch)
                model = ClapModel.from_pretrained(load_path, local_files_only=local_only).to(device)
                model.eval()
            except Exception as exc:
                raise MusicToolError(
                    "TEXT_ENCODER_UNAVAILABLE",
                    "CLAP text encoder model could not be loaded.",
                    retryable=True,
                ) from exc
            self._processor = processor
            self._model = model
            self._device = device


class MusicRecallService:
    def __init__(
            self,
            units: Sequence[RetrievalUnit],
            encoder: TextEmbeddingEncoder,
            *,
            catalog_version: str = DEFAULT_CATALOG_VERSION,
            catalog_scope: str = "production_playback",
            embedding_version: str = DEFAULT_EMBEDDING_VERSION,
            audit: dict[str, Any] | None = None) -> None:
        self.units = tuple(units)
        self.encoder = encoder
        self.catalog_version = catalog_version
        self.catalog_scope = catalog_scope
        self.embedding_version = embedding_version
        self._audit = audit or {
            "embeddingRows": len(self.units),
            "manifestRows": len(self.units),
            "eligibleRows": len(self.units),
            "issues": [],
        }
        self._by_playback_unit_id = {unit.playback_unit_id: unit for unit in self.units}
        self._encoder_dimension_validated = False
        if len(self._by_playback_unit_id) != len(self.units):
            raise MusicToolError("INVALID_PLAYBACK_MANIFEST", "playbackUnitId must be unique.")

    @classmethod
    def from_components(
            cls,
            *,
            manifest: dict[str, Any],
            track_index: Sequence[dict[str, Any]],
            embeddings: Sequence[Sequence[float]],
            songs: dict[str, Any],
            encoder: TextEmbeddingEncoder) -> MusicRecallService:
        raw_items = manifest.get("items")
        if not isinstance(raw_items, list):
            raise MusicToolError("INVALID_PLAYBACK_MANIFEST", "manifest.items must be an array.")
        song_items = songs.get("items")
        if not isinstance(song_items, list):
            raise MusicToolError("INVALID_SONG_METADATA", "songs.items must be an array.")
        song_by_id = load_song_texts(song_items)
        index_by_row = validate_track_index(track_index, embeddings)
        manifest_track_ids: set[str] = set()
        units: list[RetrievalUnit] = []
        issues: list[dict[str, Any]] = []

        for manifest_row, item in enumerate(raw_items):
            if not isinstance(item, dict):
                issues.append({"manifestRow": manifest_row, "message": "manifest item must be an object"})
                continue
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                issues.append({"manifestRow": manifest_row, "message": "enabled must be a boolean"})
                continue
            if not enabled:
                continue
            try:
                unit = build_retrieval_unit(item, index_by_row, embeddings, song_by_id)
                if unit.retrieval_track_id in manifest_track_ids:
                    raise ValueError("retrievalTrackId must be unique")
                manifest_track_ids.add(unit.retrieval_track_id)
                units.append(unit)
            except (TypeError, ValueError) as exc:
                issues.append({"manifestRow": manifest_row, "message": str(exc)})

        audit = {
            "embeddingRows": len(embeddings),
            "manifestRows": len(raw_items),
            "eligibleRows": len(units),
            "coverageRatio": len(units) / len(embeddings) if embeddings else 0.0,
            "issues": issues,
        }
        return cls(
            units,
            encoder,
            catalog_version=str(manifest.get("catalogVersion") or DEFAULT_CATALOG_VERSION),
            catalog_scope=str(manifest.get("catalogScope") or "production_playback"),
            embedding_version=str(manifest.get("embeddingVersion") or DEFAULT_EMBEDDING_VERSION),
            audit=audit,
        )

    @classmethod
    def from_paths(
            cls,
            *,
            manifest_path: Path,
            track_index_path: Path,
            embeddings_path: Path,
            songs_path: Path,
            embedding_metadata_path: Path | None = None,
            playback_catalog_path: Path | None = None,
            encoder: TextEmbeddingEncoder | None = None,
            model_path: str | None = None,
            model_revision: str | None = None,
            device: str | None = None) -> MusicRecallService:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            songs = json.loads(songs_path.read_text(encoding="utf-8"))
            embedding_metadata = (
                json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
                if embedding_metadata_path is not None
                else None
            )
            with track_index_path.open("r", encoding="utf-8-sig", newline="") as handle:
                track_index = list(csv.DictReader(handle))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            raise MusicToolError("MUSIC_RECALL_ASSET_UNAVAILABLE", "Music recall assets could not be loaded.") from exc
        if not isinstance(manifest, dict) or not isinstance(songs, dict) or (
                embedding_metadata is not None and not isinstance(embedding_metadata, dict)):
            raise MusicToolError("MUSIC_RECALL_ASSET_UNAVAILABLE", "Music recall JSON assets must be objects.")
        try:
            import numpy as np
        except ImportError as exc:
            raise MusicToolError(
                "MUSIC_RECALL_ASSET_UNAVAILABLE",
                "numpy is required to load the CLAP embedding bank.",
            ) from exc
        try:
            raw_embeddings = np.load(embeddings_path, mmap_mode="r")
            embeddings = raw_embeddings.astype("float32").tolist()
        except Exception as exc:
            raise MusicToolError("MUSIC_RECALL_ASSET_UNAVAILABLE", "CLAP embeddings could not be loaded.") from exc
        provenance = None
        artifact_digests = None
        if embedding_metadata is not None:
            artifact_digests = validate_file_artifact_digests(
                manifest=manifest,
                playback_catalog_path=playback_catalog_path,
                track_index_path=track_index_path,
                embeddings_path=embeddings_path,
                embedding_metadata_path=embedding_metadata_path,
                songs_path=songs_path,
            )
            provenance = validate_embedding_provenance(
                manifest=manifest,
                track_index=track_index,
                embeddings=embeddings,
                metadata=embedding_metadata,
            )
        return cls.from_components(
            manifest=manifest,
            track_index=track_index,
            embeddings=embeddings,
            songs=songs,
            encoder=encoder or ClapTextEncoder(
                model_path or os.getenv("CLAP_MODEL_ID", DEFAULT_MODEL_ID),
                device or os.getenv("CLAP_DEVICE", "auto"),
                expected_model_id=provenance.model_id if provenance else None,
                expected_model_revision=provenance.model_revision if provenance else None,
                expected_model_directory_sha256=(
                    artifact_digests["modelDirectorySha256"] if artifact_digests else None
                ),
                declared_model_revision=model_revision or os.getenv("CLAP_MODEL_REVISION"),
            ),
        )

    def coverage_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._audit))

    def readiness_audit(self) -> dict[str, Any]:
        if (
                self._audit.get("issues")
                or self._audit.get("eligibleRows") != self._audit.get("embeddingRows")
                or self._audit.get("manifestRows") != self._audit.get("embeddingRows")):
            raise MusicToolError(
                "INCOMPLETE_ROUTEABLE_COVERAGE",
                "Strict music recall readiness requires every embedding row to have one valid authoritative playback mapping.",
            )
        if len(self.units) < 3:
            raise MusicToolError(
                "INSUFFICIENT_ROUTEABLE_CANDIDATES",
                "Music recall readiness requires at least three authoritative routeable candidates.",
            )
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
            if len(probe) != len(self.units[0].embedding):
                raise MusicToolError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    "CLAP text encoder and audio embedding dimensions do not match.",
                )
            self._encoder_dimension_validated = True
        audit = self.coverage_audit()
        audit["textEncoder"] = "ready"
        return audit

    def routeable_playback_unit_ids(self) -> set[str]:
        return set(self._by_playback_unit_id)

    def recall(
            self,
            query_text: str,
            *,
            top_k: int = 3,
            candidate_kinds: Sequence[str] | None = None,
            exclude_playback_unit_ids: Sequence[str] | None = None,
            text_language: str = "en") -> dict[str, Any]:
        query_text = query_text.strip() if isinstance(query_text, str) else ""
        if not query_text:
            raise MusicToolError("INVALID_RECALL_QUERY", "queryText is required.", rejected=True)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 3:
            raise MusicToolError("INVALID_TOP_K", "topK must be an integer from 1 to 3.", rejected=True)
        kinds = list(candidate_kinds or ["song"])
        if kinds != ["song"]:
            raise MusicToolError(
                "UNSUPPORTED_CANDIDATE_KIND",
                "V2.2.3 music recall supports candidateKinds=['song'] only.",
                rejected=True,
            )
        if text_language != "en":
            raise MusicToolError("UNSUPPORTED_TEXT_LANGUAGE", "V2.2.3 music recall supports English queries only.", rejected=True)
        excluded = set(exclude_playback_unit_ids or [])
        eligible = [unit for unit in self.units if unit.playback_unit_id not in excluded]
        if not eligible:
            raise MusicToolError("NO_ROUTEABLE_CANDIDATE", "No routeable music candidate is available.")

        try:
            query_embedding = normalize_vector(self.encoder.encode(query_text))
        except MusicToolError:
            raise
        except Exception as exc:
            raise MusicToolError("TEXT_ENCODER_UNAVAILABLE", "CLAP text encoding failed.", retryable=True) from exc
        expected_dim = len(eligible[0].embedding)
        if len(query_embedding) != expected_dim:
            raise MusicToolError("EMBEDDING_DIMENSION_MISMATCH", "Query and audio embedding dimensions do not match.")

        ranked = sorted(
            ((dot_product(query_embedding, unit.embedding), unit) for unit in eligible),
            key=lambda item: (-item[0], item[1].embedding_row),
        )[:top_k]
        candidates = [candidate_result(unit, score, rank) for rank, (score, unit) in enumerate(ranked, start=1)]
        warnings: list[dict[str, Any]] = []
        if len(candidates) < top_k:
            warnings.append({
                "code": "INSUFFICIENT_ROUTEABLE_CANDIDATES",
                "message": f"Requested {top_k} candidates but only {len(candidates)} are routeable.",
            })
        return {
            "toolStatus": "ok",
            "catalogScope": self.catalog_scope,
            "candidates": candidates,
            "catalogVersion": self.catalog_version,
            "embeddingVersion": self.embedding_version,
            "warnings": warnings,
        }

    def get_song_context(self, playback_unit_id: str) -> dict[str, Any]:
        playback_unit_id = playback_unit_id.strip() if isinstance(playback_unit_id, str) else ""
        unit = self._by_playback_unit_id.get(playback_unit_id)
        if unit is None:
            raise MusicToolError(
                "SONG_CONTEXT_NOT_FOUND",
                "playbackUnitId is absent from the authoritative music manifest.",
                rejected=True,
            )
        raw = unit.song_text.raw
        return {
            "toolStatus": "ok",
            "playbackUnitId": unit.playback_unit_id,
            "songId": unit.song_id,
            "unitKind": unit.unit_kind,
            "trackOrder": unit.track_order,
            "unitLabel": unit.unit_label,
            "catalogScope": self.catalog_scope,
            "songText": unit.song_text.summary(),
            "acousticTech": raw.get("acoustic_tech") if isinstance(raw.get("acoustic_tech"), dict) else {},
            "catalogVersion": self.catalog_version,
        }


def load_music_recall_service() -> MusicRecallService:
    mode = os.getenv("MUSIC_RECALL_MODE", "strict").strip().lower()
    if mode == "fixture_for_tests":
        return load_fixture_music_recall_service(
            Path(os.getenv("MUSIC_RECALL_FIXTURE_PATH") or DEFAULT_FIXTURE_PATH)
        )
    if mode != "strict":
        raise MusicToolError(
            "INVALID_MUSIC_RECALL_MODE",
            "MUSIC_RECALL_MODE must be strict or fixture_for_tests.",
        )
    manifest_value = os.getenv("MUSIC_RECALL_MANIFEST_PATH")
    if not manifest_value:
        raise MusicToolError(
            "MUSIC_RECALL_MANIFEST_REQUIRED",
            "MUSIC_RECALL_MANIFEST_PATH must point to an authoritative playback mapping; IDs are never inferred.",
        )
    embedding_metadata_value = os.getenv("MUSIC_RECALL_EMBEDDING_METADATA_PATH")
    if not embedding_metadata_value:
        raise MusicToolError(
            "MUSIC_RECALL_EMBEDDING_METADATA_REQUIRED",
            "MUSIC_RECALL_EMBEDDING_METADATA_PATH is required to bind the text model to the audio embedding bank.",
        )
    playback_catalog_value = os.getenv("MUSIC_RECALL_PLAYBACK_CATALOG_PATH")
    if not playback_catalog_value:
        raise MusicToolError(
            "MUSIC_RECALL_PLAYBACK_CATALOG_REQUIRED",
            "MUSIC_RECALL_PLAYBACK_CATALOG_PATH is required to verify the manifest's authoritative catalog digest.",
        )
    return MusicRecallService.from_paths(
        manifest_path=Path(manifest_value),
        track_index_path=Path(os.getenv("MUSIC_RECALL_TRACK_INDEX_PATH") or DEFAULT_CLAP_ROOT / "track_index.csv"),
        embeddings_path=Path(os.getenv("MUSIC_RECALL_EMBEDDINGS_PATH") or DEFAULT_CLAP_ROOT / "track_embeddings.npy"),
        songs_path=Path(os.getenv("MUSIC_RECALL_SONGS_PATH") or DEFAULT_SONGS_PATH),
        embedding_metadata_path=Path(embedding_metadata_value),
        playback_catalog_path=Path(playback_catalog_value),
    )


def load_fixture_music_recall_service(path: Path) -> MusicRecallService:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        query_embedding = normalize_vector(fixture["queryEmbedding"])
        return MusicRecallService.from_components(
            manifest=fixture["manifest"],
            track_index=fixture["trackIndex"],
            embeddings=fixture["embeddings"],
            songs=fixture["songs"],
            encoder=FixedTextEncoder(query_embedding),
        )
    except MusicToolError:
        raise
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MusicToolError(
            "INVALID_MUSIC_RECALL_FIXTURE",
            "The test-only music recall fixture is missing or invalid.",
        ) from exc


@lru_cache(maxsize=1)
def get_music_recall_service() -> MusicRecallService:
    return load_music_recall_service()


def build_retrieval_unit(
        item: dict[str, Any],
        index_by_row: dict[int, str],
        embeddings: Sequence[Sequence[float]],
        song_by_id: dict[str, SongText]) -> RetrievalUnit:
    required_text = ["retrievalTrackId", "songId", "playbackUnitId", "unitKind"]
    for field in required_text:
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"{field} is required")
    embedding_row = item.get("embeddingRow")
    if isinstance(embedding_row, bool) or not isinstance(embedding_row, int):
        raise ValueError("embeddingRow must be an integer")
    if embedding_row not in index_by_row:
        raise ValueError("embeddingRow is absent from track index")
    retrieval_track_id = item["retrievalTrackId"].strip()
    if not retrieval_track_id.endswith("_A") or len(retrieval_track_id) <= 2:
        raise ValueError("retrievalTrackId must end with the exact whole-song suffix _A")
    if index_by_row[embedding_row] != retrieval_track_id:
        raise ValueError("retrievalTrackId does not match track index row")
    if item["unitKind"] != "song":
        raise ValueError("V2.2.3 manifest supports unitKind=song only")
    song_id = item["songId"].strip()
    if song_id != retrieval_track_id[:-2]:
        raise ValueError("songId must exactly match retrievalTrackId with the _A suffix removed")
    metadata_song_id = item.get("metadataSongId", song_id)
    if metadata_song_id != song_id:
        raise ValueError("metadataSongId must equal songId for song units")
    song_text = song_by_id.get(song_id)
    if song_text is None:
        raise ValueError("songId is absent from songs metadata")
    track_order = item.get("trackOrder", 0)
    if isinstance(track_order, bool) or track_order not in {0, None}:
        raise ValueError("song trackOrder must be 0 or omitted")
    embedding = normalize_vector(embeddings[embedding_row])
    unit_label = item.get("unitLabel") or song_text.internal_name
    if not isinstance(unit_label, str) or not unit_label.strip():
        raise ValueError("unitLabel is required")
    return RetrievalUnit(
        embedding_row=embedding_row,
        retrieval_track_id=retrieval_track_id,
        playback_unit_id=item["playbackUnitId"].strip(),
        song_id=song_id,
        unit_kind="song",
        track_order=0,
        unit_label=unit_label.strip(),
        embedding=embedding,
        song_text=song_text,
    )


def load_song_texts(items: Sequence[Any]) -> dict[str, SongText]:
    result: dict[str, SongText] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        song_id = item.get("id")
        internal_name = item.get("internal_name")
        sub_title = item.get("sub_title")
        introduction = item.get("introductions")
        if not all(isinstance(value, str) and value.strip() for value in [song_id, internal_name, sub_title, introduction]):
            continue
        if song_id in result:
            raise MusicToolError("INVALID_SONG_METADATA", "songs metadata contains duplicate ids.")
        result[song_id] = SongText(
            internal_name=internal_name.strip(),
            sub_title=sub_title.strip(),
            introduction=introduction.strip(),
            raw=item,
        )
    return result


def validate_track_index(
        rows: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]]) -> dict[int, str]:
    if len(rows) != len(embeddings):
        raise MusicToolError("INVALID_EMBEDDING_INDEX", "track index and embedding row counts differ.")
    result: dict[int, str] = {}
    dimension: int | None = None
    for physical_row, row in enumerate(rows):
        try:
            row_index = int(row.get("row_idx"))
        except (TypeError, ValueError) as exc:
            raise MusicToolError("INVALID_EMBEDDING_INDEX", "track index row_idx must be an integer.") from exc
        track_id = row.get("track_id")
        if row_index != physical_row or not isinstance(track_id, str) or not track_id.strip():
            raise MusicToolError("INVALID_EMBEDDING_INDEX", "track index physical order is invalid.")
        if not track_id.strip().endswith("_A") or len(track_id.strip()) <= 2:
            raise MusicToolError(
                "INVALID_EMBEDDING_INDEX",
                "V2.2.3 track index supports whole-song retrievalTrackId values ending in _A only.",
            )
        if row_index in result:
            raise MusicToolError("INVALID_EMBEDDING_INDEX", "track index row_idx must be unique.")
        vector = embeddings[physical_row]
        if not isinstance(vector, Sequence) or not vector:
            raise MusicToolError("INVALID_EMBEDDING_INDEX", "embedding rows must be non-empty vectors.")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise MusicToolError("INVALID_EMBEDDING_INDEX", "embedding dimensions are inconsistent.")
        result[row_index] = track_id.strip()
    return result


def normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding contains a non-numeric value") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding must contain finite numeric values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("embedding norm must be non-zero")
    return tuple(value / norm for value in vector)


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise MusicToolError("EMBEDDING_DIMENSION_MISMATCH", "Embedding dimensions do not match.")
    return sum(a * b for a, b in zip(left, right, strict=True))


def candidate_result(unit: RetrievalUnit, score: float, rank: int) -> dict[str, Any]:
    return {
        "candidateId": f"candidate_{rank:03d}",
        "rank": rank,
        "score": score,
        "playbackUnitId": unit.playback_unit_id,
        "songId": unit.song_id,
        "unitKind": unit.unit_kind,
        "trackOrder": unit.track_order,
        "unitLabel": unit.unit_label,
        "songText": unit.song_text.summary(),
    }


def parse_embedding_provenance(metadata: dict[str, Any]) -> EmbeddingProvenance:
    model_id = metadata.get("model_id")
    pooling = metadata.get("track_pooling")
    dimension = metadata.get("embedding_dim")
    track_count = metadata.get("num_tracks")
    revision = metadata.get("model_revision")
    if not isinstance(revision, str) or not revision.strip():
        resolved_path = metadata.get("resolved_model_path")
        if isinstance(resolved_path, str):
            match = re.search(r"[/\\]snapshots[/\\]([^/\\]+)", resolved_path)
            revision = match.group(1) if match else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise MusicToolError("INVALID_EMBEDDING_METADATA", "audio embedding metadata model_id is required.")
    if not isinstance(revision, str) or not revision.strip():
        raise MusicToolError(
            "INVALID_EMBEDDING_METADATA",
            "audio embedding metadata must declare model_revision or a resolved snapshot path.",
        )
    if not isinstance(pooling, str) or not pooling.strip():
        raise MusicToolError("INVALID_EMBEDDING_METADATA", "audio embedding metadata track_pooling is required.")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise MusicToolError("INVALID_EMBEDDING_METADATA", "audio embedding metadata embedding_dim must be positive.")
    if isinstance(track_count, bool) or not isinstance(track_count, int) or track_count <= 0:
        raise MusicToolError("INVALID_EMBEDDING_METADATA", "audio embedding metadata num_tracks must be positive.")
    return EmbeddingProvenance(
        model_id=model_id.strip(),
        model_revision=revision.strip(),
        pooling=pooling.strip(),
        dimension=dimension,
        track_count=track_count,
    )


def validate_embedding_provenance(
        *,
        manifest: dict[str, Any],
        track_index: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        metadata: dict[str, Any]) -> EmbeddingProvenance:
    provenance = parse_embedding_provenance(metadata)
    if manifest.get("embeddingProvenance") != provenance.as_manifest_value():
        raise MusicToolError(
            "EMBEDDING_PROVENANCE_MISMATCH",
            "Manifest embeddingProvenance does not match audio_embedding_metadata.json.",
        )
    if len(track_index) != provenance.track_count or len(embeddings) != provenance.track_count:
        raise MusicToolError(
            "EMBEDDING_PROVENANCE_MISMATCH",
            "Embedding metadata track count does not match the index and embedding bank.",
        )
    for row, (index_record, embedding) in enumerate(zip(track_index, embeddings, strict=True)):
        pooling = index_record.get("pooling")
        if pooling != provenance.pooling:
            raise MusicToolError(
                "EMBEDDING_PROVENANCE_MISMATCH",
                f"Track index row {row} pooling does not match embedding metadata.",
            )
        if len(embedding) != provenance.dimension:
            raise MusicToolError(
                "EMBEDDING_PROVENANCE_MISMATCH",
                f"Embedding row {row} dimension does not match embedding metadata.",
            )
    return provenance


def normalize_artifact_digests(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise MusicToolError("INVALID_ARTIFACT_DIGESTS", "manifest.artifactDigests must be an object.")
    result: dict[str, str] = {}
    for field in REQUIRED_ARTIFACT_DIGESTS:
        digest = value.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MusicToolError(
                "INVALID_ARTIFACT_DIGESTS",
                f"manifest.artifactDigests.{field} must be a lowercase SHA-256 digest.",
            )
        result[field] = digest
    return result


def validate_file_artifact_digests(
        *,
        manifest: dict[str, Any],
        playback_catalog_path: Path | None,
        track_index_path: Path,
        embeddings_path: Path,
        embedding_metadata_path: Path,
        songs_path: Path) -> dict[str, str]:
    expected = normalize_artifact_digests(manifest.get("artifactDigests"))
    if playback_catalog_path is None:
        raise MusicToolError(
            "MUSIC_RECALL_PLAYBACK_CATALOG_REQUIRED",
            "The authoritative playback catalog is required for artifact digest validation.",
        )
    actual = {
        "playbackCatalogSha256": sha256_file(playback_catalog_path),
        "trackIndexSha256": sha256_file(track_index_path),
        "embeddingsSha256": sha256_file(embeddings_path),
        "embeddingMetadataSha256": sha256_file(embedding_metadata_path),
        "songsSha256": sha256_file(songs_path),
    }
    mismatches = [field for field, digest in actual.items() if expected[field] != digest]
    if mismatches:
        raise MusicToolError(
            "ARTIFACT_DIGEST_MISMATCH",
            "Mounted music recall assets do not match manifest digests: " + ", ".join(mismatches) + ".",
        )
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MusicToolError("MUSIC_RECALL_ASSET_UNAVAILABLE", f"Cannot hash music recall asset: {path}") from exc
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        entries = sorted(
            path.rglob("*"),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
        for candidate in entries:
            if candidate.is_symlink():
                raise MusicToolError(
                    "TEXT_ENCODER_PROVENANCE_MISMATCH",
                    "Local CLAP model directory must be self-contained and cannot contain symlinks.",
                )
            if not candidate.is_dir() and not candidate.is_file():
                raise MusicToolError(
                    "TEXT_ENCODER_PROVENANCE_MISMATCH",
                    "Local CLAP model directory contains an unsupported filesystem entry.",
                )
        files = [candidate for candidate in entries if candidate.is_file()]
        if not files:
            raise MusicToolError("TEXT_ENCODER_PROVENANCE_MISMATCH", "Local CLAP model directory is empty.")
        digest.update(b"agentic-player-model-directory-v1\0")
        for file_path in files:
            relative = file_path.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(sha256_file(file_path)))
    except MusicToolError:
        raise
    except OSError as exc:
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model directory could not be hashed.",
        ) from exc
    return digest.hexdigest()


def validate_local_model_identity(
        model_path: Path,
        *,
        expected_model_id: str,
        expected_model_revision: str,
        expected_model_directory_sha256: str | None,
        declared_model_revision: str | None = None) -> None:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model config.json could not be read for provenance validation.",
        ) from exc
    if not isinstance(config, dict):
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model config.json must be an object.",
        )
    identity: dict[str, Any] = {}
    identity_path = model_path / MODEL_IDENTITY_FILENAME
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MusicToolError(
                "TEXT_ENCODER_PROVENANCE_MISMATCH",
                f"{MODEL_IDENTITY_FILENAME} could not be read.",
            ) from exc
        if not isinstance(identity, dict):
            raise MusicToolError(
                "TEXT_ENCODER_PROVENANCE_MISMATCH",
                f"{MODEL_IDENTITY_FILENAME} must be an object.",
            )
    configured_model_id = identity.get("modelId") or config.get("_name_or_path") or config.get("model_id")
    if configured_model_id != expected_model_id:
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model identity does not match the audio embedding metadata model_id.",
        )
    revision_candidates = {
        value.strip()
        for value in [
            config.get("_commit_hash"),
            identity.get("modelRevision"),
            model_path.name,
            declared_model_revision,
        ]
        if isinstance(value, str) and value.strip()
    }
    if expected_model_revision not in revision_candidates:
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model revision does not match the audio embedding metadata snapshot revision.",
        )
    if (
            expected_model_directory_sha256 is not None
            and sha256_directory(model_path) != expected_model_directory_sha256):
        raise MusicToolError(
            "TEXT_ENCODER_PROVENANCE_MISMATCH",
            "Local CLAP model directory does not match the manifest artifact digest.",
        )


def resolve_model_path(model_id: str) -> tuple[str, bool]:
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve()), True
    if "/" not in model_id:
        return model_id, False
    org, repo = model_id.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}"
    ref_path = cache_root / "refs" / "main"
    if ref_path.exists():
        snapshot = cache_root / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot), True
    return model_id, False


def resolve_device(device_arg: str, torch_module: Any) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch_module.backends.mps.is_available():
            raise MusicToolError("TEXT_ENCODER_UNAVAILABLE", "Requested MPS device is unavailable.", retryable=True)
        return "mps"
    return "mps" if torch_module.backends.mps.is_available() else "cpu"
