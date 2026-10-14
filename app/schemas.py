from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RunStatus = Literal["running", "waiting_playback", "done", "cancelled", "failed"]
DeliveryPhase = Literal["PLAN", "ASSESSMENT", "SELECTION", "EXPLANATION", "ERROR"]
EnergyGoal = Literal["boost", "lower"]
ENERGY_GOALS = frozenset({"boost", "lower"})
GoalStatus = Literal["planning", "active", "completed", "cancelled", "failed"]
GoalPhaseStatus = Literal["planned", "preparing", "prepared", "waiting_playback", "active", "done", "cancelled", "failed"]


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


class GoalCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    clientSessionId: str = Field(min_length=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int = Field(ge=0)
    input: str = Field(min_length=1)
    split: bool = False
    language: str | None = "zh"
    context: dict[str, Any] = Field(default_factory=dict)
    currentPlayback: dict[str, Any] = Field(default_factory=dict)
    catalog: dict[str, Any] = Field(default_factory=dict)


class GoalControlRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = Field(ge=1)
    clientActionSeq: int = Field(ge=0)
    decisionVersion: int | None = Field(default=None, ge=0)
    phaseId: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    currentPlayback: dict[str, Any] = Field(default_factory=dict)


class GoalPlaybackReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = Field(ge=1)
    phaseId: str
    runId: str
    playbackUnitId: str
    success: bool
    actualPlaybackState: Literal["playing", "paused", "idle", "loading", "failed"] | None = None
    songId: str | None = None
    changeKind: Literal["same_song", "cross_song", "unchanged", "failed"] | None = None


class GoalProgressRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: int = Field(ge=1)
    clientActionSeq: int = Field(ge=0)
    phaseId: str
    runId: str | None = None
    elapsedSeconds: float = Field(ge=0)
    isPlaying: bool = True
    manualOverride: bool = False


class GoalPhase(BaseModel):
    phaseId: str
    index: int
    title: str
    durationMinutes: int
    iconId: str = "scene-music-note"
    status: GoalPhaseStatus
    energyGoal: EnergyGoal | None = None
    preparedPlaybackUnitId: str | None = None
    preparedOpeningText: str | None = None
    preparedOpeningVoiceId: str | None = None
    runId: str | None = None
    elapsedSeconds: float = 0
    isPlaying: bool = False


class GoalDelivery(BaseModel):
    sequence: int
    goalId: str
    version: int
    phaseId: str | None = None
    runId: str | None = None
    phase: DeliveryPhase
    text: str | None = None
    voiceId: str | None = None
    playbackUnitId: str | None = None
    status: str | None = None
    energyGoal: EnergyGoal | None = None


class GoalResponse(BaseModel):
    goalId: str
    version: int
    clientActionSeq: int
    status: GoalStatus
    title: str | None = None
    totalMinutes: int | None = None
    activePhaseId: str | None = None
    phases: list[GoalPhase] = Field(default_factory=list)
    deliveries: list[GoalDelivery] = Field(default_factory=list)
    contextSnapshot: dict[str, Any] | None = None
