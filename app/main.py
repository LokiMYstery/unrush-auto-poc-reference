from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response

from app import store
from app.agent import continue_after_receipt, new_run_id, run_task
from app.catalog import catalog
from app.config import (
    API_PREFIX,
    PHASE_ERROR,
    PHASE_EXPLANATION,
    PHASE_SELECTION,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_WAITING_PLAYBACK,
    configure_retrieval_environment,
    required_token,
)
from app.environment import enrich_context, context_snapshot, narration_context
from app.fast_selection import fast_choices, tier_config
from app.resource_validation import validate_playback_unit
from app import goals
from app.schemas import (
    ENERGY_GOALS,
    CancelRequest,
    FastRequest,
    FastResponse,
    GoalControlRequest,
    GoalCreateRequest,
    GoalPlaybackReceipt,
    GoalProgressRequest,
    GoalResponse,
    InitialRun,
    PlaybackReceipt,
    PollResponse,
    TaskRequest,
    Welcome,
)
from app.tts import WELCOME_TEXT, prebuild_welcome_voices, synthesis_count, welcome_voice_id


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    required_token()
    configure_retrieval_environment()
    store.init_db()
    catalog()
    tier_config()
    await prebuild_welcome_voices()
    await recover_incomplete_runs()
    await goals.recover_goals()
    yield


async def recover_incomplete_runs() -> None:
    for run in store.list_runs_for_recovery():
        if str(run.get("client_session_id") or "").startswith("goal-internal:"):
            continue
        if not store.run_is_session_head(run):
            continue
        run_id = str(run["run_id"])
        context = store.loads(run.get("context_json"), {})
        if context.get("energyGoal") == "maintain":
            # Do not resume an old third-direction choice or replay its speech.
            # Keep historical rows; a fresh task must choose up/down explicitly.
            store.update_run(run_id, status=RUN_STATUS_CANCELLED)
            continue
        status = str(run["status"])
        receipt = store.loads(run.get("receipt_json"), {})
        if receipt:
            if store.delivery_exists(run_id, PHASE_EXPLANATION):
                store.update_run(run_id, status="done")
            else:
                asyncio.create_task(continue_after_receipt(run_id))
            continue
        if status == RUN_STATUS_RUNNING:
            asyncio.create_task(run_task(run_id))
        elif status == RUN_STATUS_WAITING_PLAYBACK and not store.delivery_exists(run_id, PHASE_SELECTION):
            # A restart between the persisted selection state and its delivery must not leave the client waiting.
            selected_id = str(run.get("selected_playback_unit_id") or "")
            if catalog().exists(selected_id):
                context = store.loads(run.get("context_json"), {})
                goal = context.get("energyGoal")
                store.add_delivery_once_phase(run_id=run_id, phase=PHASE_SELECTION,
                                              playback_unit_id=selected_id, status=context.get("selectionStatus") or "model_selection_invalid_keep",
                                              energy_goal=goal if goal in ENERGY_GOALS else None)
            else:
                store.add_delivery(run_id=run_id, phase=PHASE_ERROR,
                                   text="这次音乐选择未能恢复，请重新调整。", status="recovery_selection_missing")
                store.update_run(run_id, status="failed")


app = FastAPI(title="Unrush Auto PoC Backend", lifespan=lifespan)


def auth(authorization: str | None = Header(default=None)) -> None:
    try:
        token = required_token()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="server_token_not_configured")
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    cat = catalog()
    return {
        "status": "ok",
        "catalogVersion": cat.raw_version,
        "catalogUnits": len(cat.profiles),
        "quality": {
            "healthStressSource": "default50_is_reference_not_measurement",
            "rawSamplesLogged": False,
            "ttsSynthesisCount": synthesis_count(),
        },
    }


@app.post(f"{API_PREFIX}/auto/fast", dependencies=[Depends(auth)], response_model=FastResponse)
async def auto_fast(request: FastRequest) -> FastResponse:
    cached = store.fast_decision(request.clientSessionId, request.clientActionSeq, request.decisionVersion)
    if cached:
        return FastResponse(**cached)
    payload = request.model_dump()
    choices = fast_choices(catalog(), payload)
    selected_choice = None
    validation_reason = None
    rejected = []
    deadline = asyncio.get_running_loop().time() + 5
    for choice in choices:
        for index, candidate in enumerate(choice["profiles"]):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise HTTPException(status_code=503, detail="fast_resource_validation_timeout")
            try:
                validation = await asyncio.wait_for(validate_playback_unit(candidate, origin_scene_id=choice.get("sceneId")), timeout=remaining)
            except TimeoutError:
                raise HTTPException(status_code=503, detail="fast_resource_validation_timeout")
            if validation.ok:
                selected_choice = choice
                ordered = choice["profiles"][index:]
                validation_reason = validation.reason
                break
            rejected.append({"playbackUnitId": candidate.playback_unit_id, "reason": validation.reason})
        if selected_choice:
            break
    if not selected_choice:
        raise HTTPException(status_code=422, detail="no_routeable_candidate")
    scene_id = selected_choice["sceneId"]
    profile = ordered[0]
    welcome = None
    logical_auto_requested = request.welcomeOnlyAuto and (request.selectedSceneId is None or request.selectedSceneId.lower() == "auto")
    if logical_auto_requested:
        welcome_text = WELCOME_TEXT["en" if str(request.language or "zh").lower().startswith("en") else "zh"]
        welcome = Welcome(text=welcome_text, voiceId=welcome_voice_id(request.language or "zh"))
    response = FastResponse(
        sceneId=scene_id or "auto",
        playbackUnitId=profile.playback_unit_id,
        orderedPlaybackUnitIds=[item.playback_unit_id for item in ordered],
        songId=profile.song_id,
        welcome=welcome,
        selectionMetadata={"policy": selected_choice["selectionPolicy"], "sceneCode": selected_choice["sceneCode"],
                           "selectedTier": selected_choice["tierById"].get(profile.playback_unit_id),
                           "tierByPlaybackUnitId": selected_choice["tierById"],
                           "fallbackReason": selected_choice["fallbackReason"], "resourceValidation": validation_reason,
                           "nativePlaybackVerified": False, "rejectedCandidates": rejected,
                           "highTierRequiresTouchpad": True, "sourceDocumentRevision": 24,
                           "capabilityExclusions": selected_choice.get("capabilityExclusions", [])},
    )
    stored = store.record_fast_decision(request.clientSessionId, request.clientActionSeq, request.decisionVersion,
                                       [scene.model_dump(include={"sceneId", "sceneCode", "sceneName", "songIds", "playbackUnitIds"}) for scene in request.scenes],
                                       response.model_dump())
    return FastResponse(**stored)


@app.get(f"{API_PREFIX}/catalog", dependencies=[Depends(auth)])
async def get_catalog() -> dict[str, object]:
    cat = catalog()
    return {
        "catalogVersion": cat.raw_version,
        "units": [
            {
                "playbackUnitId": profile.playback_unit_id,
                "songId": profile.song_id,
                "carrierSongId": profile.song_id,
                "businessSongId": profile.song_id,
                "serviceSongId": profile.service_song_id(),
                "unitKind": profile.unit_kind,
                "trackOrder": profile.track_order,
                "resourceBindings": profile.resource_bindings,
                "name": profile.names,
                "summary": profile.summary,
            }
            for profile in cat.profiles
        ],
    }


@app.post(f"{API_PREFIX}/task", dependencies=[Depends(auth)], response_model=InitialRun)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks) -> InitialRun:
    alias = request.currentPlayback.pop("playback_unit_id", None)
    if alias and request.currentPlayback.get("playbackUnitId") and alias != request.currentPlayback["playbackUnitId"]:
        raise HTTPException(status_code=409, detail="conflicting_current_playback_ids")
    if alias:
        request.currentPlayback["playbackUnitId"] = alias
    current_id = str(request.currentPlayback.get("playbackUnitId") or "")
    current = catalog().profile(current_id)
    if current_id and current is None:
        raise HTTPException(status_code=422, detail="unknown_current_playback_unit")
    service_song_id = request.currentPlayback.get("songId")
    if current and service_song_id and not _receipt_song_matches(current, str(service_song_id)):
        raise HTTPException(status_code=409, detail="current_playback_identity_mismatch")
    run_id = new_run_id()
    context = await enrich_context(dict(request.context))
    context.setdefault("language", request.language or "zh")
    context.pop("energyGoal", None)  # Only the server Agent can select the goal.
    context["sessionMemory"] = store.completed_session_summaries(request.clientSessionId, limit=3)
    context["environmentNarration"] = narration_context(context, context["sessionMemory"])
    delivered_narration = store.latest_delivered_environment_narration(request.clientSessionId)
    if delivered_narration:
        context["environmentNarration"] = narration_context(context, [delivered_narration])
    context["currentPlaybackForExplanation"] = request.currentPlayback
    created = store.create_run(
        run_id=run_id,
        client_session_id=request.clientSessionId,
        client_action_seq=request.clientActionSeq,
        decision_version=request.decisionVersion,
        context=context,
        current_playback=request.currentPlayback,
        catalog_payload=request.catalog,
    )
    if created.state == "stale":
        raise HTTPException(status_code=409, detail="stale_action")
    if created.state == "conflict":
        raise HTTPException(status_code=409, detail="conflicting_action")
    if created.state == "duplicate" and created.run_id:
        existing = store.get_run(created.run_id) or {}
        return InitialRun(runId=created.run_id, status=created.status or RUN_STATUS_RUNNING,
                          contextSnapshot=context_snapshot(store.loads(existing.get("context_json"), {})))
    if created.state != "created" or not created.run_id:
        raise HTTPException(status_code=409, detail="action_not_created")
    background_tasks.add_task(run_task, created.run_id)
    return InitialRun(runId=created.run_id, status=RUN_STATUS_RUNNING, contextSnapshot=context_snapshot(context))


@app.get(f"{API_PREFIX}/runs/{{run_id}}/deliveries", dependencies=[Depends(auth)], response_model=PollResponse)
async def poll_deliveries(run_id: str, after: int = 0, waitSeconds: int = 20) -> PollResponse:
    wait_seconds = max(0, min(waitSeconds, 20))
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run_not_found")
        if run["status"] == RUN_STATUS_CANCELLED:
            # Historical deliveries stay in storage, but cancelled work must not
            # replay old speech or selections on a connected client's next poll.
            return PollResponse(runId=run_id, status=RUN_STATUS_CANCELLED, deliveries=[])
        deliveries = store.deliveries_after(run_id, after)
        if deliveries or run["status"] not in {"running", "waiting_playback"} or asyncio.get_running_loop().time() >= deadline:
            return PollResponse(runId=run_id, status=run["status"], deliveries=deliveries)
        await asyncio.sleep(0.25)


@app.post(f"{API_PREFIX}/runs/{{run_id}}/playback-receipt", dependencies=[Depends(auth)], response_model=InitialRun)
async def playback_receipt(run_id: str, receipt: PlaybackReceipt, background_tasks: BackgroundTasks) -> InitialRun:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    receipt_payload = _validated_receipt_payload(run, receipt)
    claim = store.claim_receipt(
        run_id=run_id,
        client_session_id=receipt.clientSessionId,
        client_action_seq=receipt.clientActionSeq,
        decision_version=receipt.decisionVersion,
        receipt_payload=receipt_payload,
    )
    if claim.state == "missing":
        raise HTTPException(status_code=404, detail="run_not_found")
    if claim.state == "cancelled":
        return InitialRun(runId=run_id, status=RUN_STATUS_CANCELLED)
    if claim.state == "stale":
        raise HTTPException(status_code=409, detail="stale_receipt")
    if claim.state == "not_waiting":
        raise HTTPException(status_code=409, detail="receipt_not_awaited")
    if claim.state == "conflict":
        raise HTTPException(status_code=409, detail="conflicting_receipt")
    if claim.state == "claimed":
        background_tasks.add_task(continue_after_receipt, run_id)
    latest = store.get_run(run_id) or run
    return InitialRun(runId=run_id, status=latest["status"])


def _validated_receipt_payload(run: dict[str, object], receipt: PlaybackReceipt) -> dict[str, object]:
    if run["status"] != RUN_STATUS_WAITING_PLAYBACK and not store.loads(str(run.get("receipt_json") or ""), {}):
        raise HTTPException(status_code=409, detail="receipt_not_awaited")
    selected_id = str(run.get("selected_playback_unit_id") or "")
    if not selected_id:
        raise HTTPException(status_code=409, detail="selection_not_recorded")
    current_playback = store.loads(str(run.get("current_playback_json") or "{}"), {})
    current_id = str(current_playback.get("playbackUnitId") or "")
    actual = catalog().profile(receipt.playbackUnitId)
    selected = catalog().profile(selected_id)
    current = catalog().profile(current_id)
    if actual is None:
        raise HTTPException(status_code=422, detail="unknown_playback_unit")
    if selected is None:
        raise HTTPException(status_code=409, detail="selected_unit_missing")
    if receipt.songId and not _receipt_song_matches(actual, receipt.songId):
        raise HTTPException(status_code=409, detail="receipt_song_mismatch")
    if receipt.success and receipt.playbackUnitId != selected_id:
        raise HTTPException(status_code=409, detail="receipt_actual_differs_from_selected")
    expected_change = _expected_change_kind(success=receipt.success, current=current, actual=actual)
    if receipt.changeKind and receipt.changeKind != expected_change:
        raise HTTPException(status_code=409, detail="receipt_change_kind_mismatch")
    payload = receipt.model_dump()
    payload["selectionStatus"] = (store.latest_delivery(str(run["run_id"]), PHASE_SELECTION) or {}).get("status")
    payload["changeKind"] = expected_change
    payload["songId"] = receipt.songId or actual.song_id
    return payload


def _receipt_song_matches(profile, song_id: str) -> bool:
    if song_id == profile.song_id or song_id == profile.service_song_id():
        return True
    for binding in profile.resource_bindings.get("bindings", []):
        if isinstance(binding, dict) and binding.get("serviceSongId") == song_id:
            return True
    return False


def _expected_change_kind(*, success: bool, current, actual) -> str:
    if not success:
        return "failed"
    if current and actual.playback_unit_id == current.playback_unit_id:
        return "unchanged"
    if current and actual.song_id == current.song_id:
        return "same_song"
    return "cross_song"


@app.post(f"{API_PREFIX}/runs/{{run_id}}/cancel", dependencies=[Depends(auth)], response_model=InitialRun)
async def cancel_run(run_id: str, request: CancelRequest) -> InitialRun:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    if not store.is_current(run, request.clientSessionId, request.clientActionSeq, request.decisionVersion) or not store.run_is_session_head(run):
        raise HTTPException(status_code=409, detail="stale_cancel")
    changed = store.update_run(run_id, status=RUN_STATUS_CANCELLED)
    latest = store.get_run(run_id) or run
    if not changed and latest["status"] != RUN_STATUS_CANCELLED:
        return InitialRun(runId=run_id, status=latest["status"])
    return InitialRun(runId=run_id, status=RUN_STATUS_CANCELLED)


@app.post(f"{API_PREFIX}/goals", dependencies=[Depends(auth)], response_model=GoalResponse)
async def create_goal(request: GoalCreateRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    return await goals.create_goal(request, background_tasks)


@app.get(f"{API_PREFIX}/goals/{{goal_id}}", dependencies=[Depends(auth)], response_model=GoalResponse)
async def poll_goal(goal_id: str, after: int = 0, waitSeconds: int = 20) -> GoalResponse:
    wait_seconds = max(0, min(waitSeconds, 20))
    deadline = asyncio.get_running_loop().time() + wait_seconds
    phase_fingerprint = None
    while True:
        response = goals.response_for_goal(goal_id, after=after)
        current_fingerprint = tuple((phase.phaseId, phase.status, phase.preparedPlaybackUnitId,
                                     phase.preparedOpeningVoiceId, phase.runId) for phase in response.phases)
        if response.deliveries or response.status not in {"planning", "active"} or asyncio.get_running_loop().time() >= deadline:
            return response
        if phase_fingerprint is not None and current_fingerprint != phase_fingerprint:
            return response
        phase_fingerprint = current_fingerprint
        await asyncio.sleep(0.25)


@app.post(f"{API_PREFIX}/goals/{{goal_id}}/advance", dependencies=[Depends(auth)], response_model=GoalResponse)
async def advance_goal(goal_id: str, request: GoalControlRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    return await goals.advance_goal(goal_id, request, background_tasks)


@app.post(f"{API_PREFIX}/goals/{{goal_id}}/retry", dependencies=[Depends(auth)], response_model=GoalResponse)
async def retry_goal(goal_id: str, request: GoalControlRequest, background_tasks: BackgroundTasks) -> GoalResponse:
    return await goals.retry_goal_phase(goal_id, request, background_tasks)


@app.post(f"{API_PREFIX}/goals/{{goal_id}}/playback-receipt", dependencies=[Depends(auth)], response_model=GoalResponse)
async def goal_playback_receipt(goal_id: str, receipt: GoalPlaybackReceipt, background_tasks: BackgroundTasks) -> GoalResponse:
    return await goals.goal_playback_receipt(goal_id, receipt, background_tasks)


@app.post(f"{API_PREFIX}/goals/{{goal_id}}/progress", dependencies=[Depends(auth)], response_model=GoalResponse)
async def goal_progress(goal_id: str, request: GoalProgressRequest) -> GoalResponse:
    return goals.progress_goal(goal_id, request)


@app.post(f"{API_PREFIX}/goals/{{goal_id}}/cancel", dependencies=[Depends(auth)], response_model=GoalResponse)
async def cancel_goal(goal_id: str, request: GoalControlRequest) -> GoalResponse:
    return goals.cancel_goal(goal_id, request)


@app.get(f"{API_PREFIX}/voice/{{voice_id}}", dependencies=[Depends(auth)])
async def get_voice(voice_id: str) -> Response:
    voice = store.get_voice(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail="voice_not_found")
    path = Path(voice["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="voice_file_missing")
    return Response(path.read_bytes(), media_type=voice["content_type"])
