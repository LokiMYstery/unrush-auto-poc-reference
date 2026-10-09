from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RunStatus = Literal["running", "waiting_playback", "done", "cancelled", "failed"]
DeliveryPhase = Literal["ASSESSMENT", "SELECTION", "EXPLANATION", "ERROR"]
EnergyGoal = Literal["boost", "maintain", "lower"]


class SceneCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    sceneId: str = Field(min_length=1)
    sceneCode: str | None = None
    sceneName: str | None = None
    priority: Literal["high", "middle", "low"] | None = None
    preordered: bool = False
    playbackUnitIds: list[str] = Field(default_factory=list)
    songIds: list[str] = Field(default_factory=list)


class SongCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    playbackUnitId: str | None = None
    songId: str | None = None
    sceneId: str | None = None
    priority: Literal["high", "middle", "low"] | None = None


class FastRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    clientSessionId: str = Field(min_length=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int = Field(ge=0)
    selectedSceneId: str | None = None
    welcomeOnlyAuto: bool = True
    scenes: list[SceneCandidate] = Field(default_factory=list)
    songCandidates: list[SongCandidate] = Field(default_factory=list)
    language: str | None = "zh"


class Welcome(BaseModel):
    text: str
    voiceId: str | None = None


class FastResponse(BaseModel):
    sceneId: str
    playbackUnitId: str
    orderedPlaybackUnitIds: list[str] = Field(default_factory=list)
    songId: str
    welcome: Welcome | None = None
    status: str = "ok"
    selectionMetadata: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    clientSessionId: str = Field(min_length=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int = Field(ge=0)
    context: dict[str, Any] = Field(default_factory=dict)
    currentPlayback: dict[str, Any] = Field(default_factory=dict)
    catalog: dict[str, Any] = Field(default_factory=dict)
    language: str | None = "zh"


class InitialRun(BaseModel):
    runId: str
    status: RunStatus
    contextSnapshot: dict[str, Any] | None = None


class Delivery(BaseModel):
    sequence: int
    phase: DeliveryPhase
    text: str | None = None
    playbackUnitId: str | None = None
    voiceId: str | None = None
    status: str | None = None
    energyGoal: EnergyGoal | None = None


class PollResponse(BaseModel):
    runId: str
    status: RunStatus
    deliveries: list[Delivery]


class PlaybackReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    clientSessionId: str = Field(min_length=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int = Field(ge=0)
    playbackUnitId: str
    success: bool
    actualPlaybackState: Literal["playing", "paused", "idle", "loading", "failed"] | None = None
    songId: str | None = None
    changeKind: Literal["same_song", "cross_song", "unchanged", "failed"] | None = None


class CancelRequest(BaseModel):
    clientSessionId: str = Field(min_length=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int = Field(ge=0)
